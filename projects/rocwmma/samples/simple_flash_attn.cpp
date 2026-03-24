/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/* simple_flash_attn.cpp — Simplified Flash Attention (Tiled SDPA)
 *
 * Implements FlashAttention-1 Algorithm 1:
 *
 *   O = Softmax( Q * K^T / sqrt(D) ) * V
 *
 * Key difference from simple_sdpa.cpp:
 *   - Never materializes the full S*S attention score matrix
 *   - Processes K/V in tiles of TILE_K rows, maintaining running statistics
 *   - Memory: O(TILE_K*D) LDS per block, vs O(S^2) global for naive SDPA
 *
 * Online softmax update for each KV tile j:
 *   m_new = max(m, tile_max(S_j))
 *   O     = exp(m - m_new) * O + sum_k exp(S_j[k] - m_new) * V_j[k,:]
 *   l     = exp(m - m_new) * l + sum_k exp(S_j[k] - m_new)
 *   m     = m_new
 * Final: O[i,:] /= l
 *
 * Kernel design:
 *   Grid : (S, 1)                     one block per query row
 *   Block: (BLOCK_SIZE, 1)            one warp (WAVE_SIZE threads)
 *   Each thread owns HEAD_DIM/BLOCK_SIZE contiguous head-dimension elements.
 *
 *   Inner dot product Q[i] . K[j] uses warp-level reduction (__shfl_xor).
 *   No rocWMMA MFMA is used inside the kernel — this avoids the need to
 *   know the architecture-specific fragment element layout for per-row softmax.
 *
 * Architecture: gfx9 (Wave64), gfx11, gfx12 (Wave32). ELEMS_PER_THREAD
 *   adapts automatically via AMDGCN_WAVE_SIZE compile-time constant.
 *
 * Default: S = SEQ_LEN = 64, D = HEAD_DIM = 64, TILE_K = 16.
 *
 * Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient
 *            Exact Attention with IO-Awareness", NeurIPS 2022, Algorithm 1.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#include "common.hpp"

using rocwmma::float16_t;
using rocwmma::float32_t;

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

// AMDGCN_WAVE_SIZE: 64 for gfx9, 32 for gfx11/gfx12
#if ROCWMMA_ARCH_GFX9
constexpr uint32_t WAVE_SIZE_CT = rocwmma::Constants::AMDGCN_WAVE_SIZE_64;
#else
constexpr uint32_t WAVE_SIZE_CT = rocwmma::Constants::AMDGCN_WAVE_SIZE_32;
#endif

// One warp per block
constexpr uint32_t BLOCK_SIZE = WAVE_SIZE_CT;

// KV tile height: how many K/V rows are processed per inner iteration
constexpr uint32_t TILE_K = 16;

// Problem dimensions (multiples of TILE_K, HEAD_DIM divisible by BLOCK_SIZE)
constexpr uint32_t SEQ_LEN  = 64;
constexpr uint32_t HEAD_DIM = 64;

// Each thread owns this many contiguous head-dimension elements
// Wave32 gfx12: 64/32 = 2   Wave64 gfx9: 64/64 = 1
constexpr uint32_t ELEMS_PER_THREAD = HEAD_DIM / BLOCK_SIZE;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Warp-level sum reduction using __shfl_xor.
// Works for Wave32 (gfx12) and Wave64 (gfx9): loop starts at WAVE_SIZE_CT/2.
ROCWMMA_DEVICE inline float warp_reduce_sum(float val)
{
#pragma unroll
    for(int off = (int)(WAVE_SIZE_CT / 2); off > 0; off >>= 1)
        val += __shfl_xor(val, off);
    return val;
}

// ---------------------------------------------------------------------------
// Flash Attention Kernel
//
// One block per query row i.  Iterates over KV tiles j (step = TILE_K).
// Maintains per-thread unnormalized output o[ELEMS_PER_THREAD] and
// warp-uniform scalars m (running max) and l (running norm).
//
// Shared memory layout (lds_bytes = 2*TILE_K*HEAD_DIM*4 + TILE_K*4 bytes):
//   float lds_k[TILE_K * HEAD_DIM]   K tile
//   float lds_v[TILE_K * HEAD_DIM]   V tile
//   float lds_s[TILE_K]              dot-product scores for this tile
// ---------------------------------------------------------------------------
__global__ void flash_attn_kernel(uint32_t         seq,
                                   uint32_t         head,
                                   float16_t const* q,
                                   float16_t const* k,
                                   float16_t const* v,
                                   float16_t*       out,
                                   uint32_t         ldq,   // = head
                                   uint32_t         ldk,   // = head
                                   uint32_t         ldv,   // = head
                                   uint32_t         ldo,   // = head
                                   float32_t        scale, // = 1/sqrt(head)
                                   uint32_t         tile_k)
{
    uint32_t q_row = blockIdx.x;
    if(q_row >= seq)
        return;

    // Shared memory
    HIP_DYNAMIC_SHARED(float, smem);
    float* lds_k = smem;                          // [TILE_K * head]
    float* lds_v = lds_k + tile_k * head;         // [TILE_K * head]
    float* lds_s = lds_v + tile_k * head;         // [TILE_K]

    // Each thread covers dims [d_start .. d_start+ELEMS_PER_THREAD)
    uint32_t d_start = threadIdx.x * ELEMS_PER_THREAD;

    // Load Q row into registers
    float q_reg[ELEMS_PER_THREAD];
#pragma unroll
    for(uint32_t e = 0; e < ELEMS_PER_THREAD; ++e)
        q_reg[e] = (d_start + e < head)
                   ? static_cast<float>(q[q_row * ldq + d_start + e])
                   : 0.0f;

    // Running online-softmax state (warp-uniform: same m and l for all threads)
    float m = -__builtin_huge_valf();
    float l = 0.0f;

    // Running unnormalized output (per thread, one element per owned dim)
    float o[ELEMS_PER_THREAD];
#pragma unroll
    for(uint32_t e = 0; e < ELEMS_PER_THREAD; ++e)
        o[e] = 0.0f;

    // ---------------------------------------------------------------------------
    // Outer loop: iterate over KV tiles
    // ---------------------------------------------------------------------------
    for(uint32_t j0 = 0; j0 < seq; j0 += tile_k)
    {
        // ----- Phase 1: Cooperative load K tile and V tile into LDS ----------
        uint32_t tile_elems = tile_k * head;
        for(uint32_t e = threadIdx.x; e < tile_elems; e += blockDim.x)
        {
            uint32_t kv_row = j0 + e / head;
            uint32_t kv_col = e % head;
            lds_k[e] = (kv_row < seq)
                       ? static_cast<float>(k[kv_row * ldk + kv_col])
                       : 0.0f;
            lds_v[e] = (kv_row < seq)
                       ? static_cast<float>(v[kv_row * ldv + kv_col])
                       : 0.0f;
        }
        __syncthreads();

        // ----- Phase 2: Compute dot products Q[q_row] . K[j0+kk, :] --------
        // Each thread computes its partial sum over owned dims, then warp-reduce.
        for(uint32_t kk = 0; kk < tile_k; ++kk)
        {
            float partial = 0.0f;
#pragma unroll
            for(uint32_t e = 0; e < ELEMS_PER_THREAD; ++e)
                partial += q_reg[e] * lds_k[kk * head + d_start + e];

            float s = warp_reduce_sum(partial) * scale;

            // Thread 0 broadcasts the reduced score to LDS
            if(threadIdx.x == 0)
                lds_s[kk] = s;
        }
        __syncthreads(); // lds_s now visible to all threads

        // ----- Phase 3: Online softmax + output update -----------------------
        // All threads compute tile_max redundantly (cheap, avoids extra sync)
        float tile_max = lds_s[0];
        for(uint32_t kk = 1; kk < tile_k; ++kk)
            tile_max = fmaxf(tile_max, lds_s[kk]);

        float m_new = fmaxf(m, tile_max);
        float alpha = expf(m - m_new); // correction factor for old O and l

        // Rescale accumulated output and norm
        l *= alpha;
#pragma unroll
        for(uint32_t e = 0; e < ELEMS_PER_THREAD; ++e)
            o[e] *= alpha;

        // Accumulate new contributions: exp(S[kk] - m_new) * V[j0+kk, :]
        for(uint32_t kk = 0; kk < tile_k; ++kk)
        {
            float p_kk = expf(lds_s[kk] - m_new);
            l += p_kk;
#pragma unroll
            for(uint32_t e = 0; e < ELEMS_PER_THREAD; ++e)
                o[e] += p_kk * lds_v[kk * head + d_start + e];
        }

        m = m_new;
        __syncthreads(); // before next tile load overwrites LDS
    }

    // ---------------------------------------------------------------------------
    // Normalize and write output
    // ---------------------------------------------------------------------------
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
    for(uint32_t e = 0; e < ELEMS_PER_THREAD; ++e)
    {
        uint32_t d = d_start + e;
        if(d < head)
            out[q_row * ldo + d] = static_cast<float16_t>(o[e] * inv_l);
    }
}

// ---------------------------------------------------------------------------
// CPU reference (identical algorithm, double precision)
// ---------------------------------------------------------------------------
static void flash_attn_cpu_ref(uint32_t         seq,
                                uint32_t         head,
                                float16_t const* q,
                                float16_t const* k,
                                float16_t const* v,
                                float16_t*       ref_out)
{
    float scale = 1.0f / std::sqrtf((float)head);

#pragma omp parallel for
    for(int i = 0; i < (int)seq; ++i)
    {
        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;
        std::vector<float> o(head, 0.0f);

        for(uint32_t j0 = 0; j0 < seq; j0 += TILE_K)
        {
            // Compute dot products for this tile
            float s[TILE_K];
            uint32_t tile = std::min(TILE_K, seq - j0);
            for(uint32_t kk = 0; kk < tile; ++kk)
            {
                float dot = 0.0f;
                for(uint32_t d = 0; d < head; ++d)
                    dot += (float)q[i * head + d] * (float)k[(j0 + kk) * head + d];
                s[kk] = dot * scale;
            }

            // Online softmax update
            float tile_max = s[0];
            for(uint32_t kk = 1; kk < tile; ++kk)
                tile_max = std::fmaxf(tile_max, s[kk]);

            float m_new = std::fmaxf(m, tile_max);
            float alpha = std::expf(m - m_new);
            l *= alpha;
            for(uint32_t d = 0; d < head; ++d) o[d] *= alpha;

            for(uint32_t kk = 0; kk < tile; ++kk)
            {
                float p_kk = std::expf(s[kk] - m_new);
                l += p_kk;
                for(uint32_t d = 0; d < head; ++d)
                    o[d] += p_kk * (float)v[(j0 + kk) * head + d];
            }
            m = m_new;
        }

        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for(uint32_t d = 0; d < head; ++d)
            ref_out[i * head + d] = static_cast<float16_t>(o[d] * inv_l);
    }
}

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------
static void printDeviceInfo()
{
    hipDevice_t     dev;
    hipDeviceProp_t p;
    CHECK_HIP_ERROR(hipGetDevice(&dev));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&p, dev));

    std::cout << "\n=== GPU Hardware Info ===\n"
              << "  Device name     : " << p.name          << "\n"
              << "  GCN arch        : " << p.gcnArchName    << "\n"
              << "  Compute units   : " << p.multiProcessorCount << "\n"
              << "  Warp size       : " << p.warpSize       << "\n"
              << "  Global memory   : " << (p.totalGlobalMem >> 20) << " MiB\n"
              << "  Shared mem/blk  : " << (p.sharedMemPerBlock >> 10) << " KiB\n"
              << "========================\n\n";
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
__host__ void run_flash_attn_sample(uint32_t seq, uint32_t head)
{
    printDeviceInfo();

    if((seq % TILE_K) || (head % BLOCK_SIZE))
    {
        std::cout << "Unsupported size: seq must be multiple of TILE_K=" << TILE_K
                  << ", head must be multiple of BLOCK_SIZE=" << BLOCK_SIZE << "\n";
        return;
    }

    float32_t scale = 1.0f / std::sqrtf((float)head);

    std::cout << "Flash Attention: S=" << seq << "  D=" << head
              << "  TILE_K=" << TILE_K << "  BLOCK_SIZE=" << BLOCK_SIZE
              << "  ELEMS=" << ELEMS_PER_THREAD
              << "  scale=" << scale << "\n\n";

    // LDS per block (bytes)
    uint32_t lds_bytes = (2u * TILE_K * head + TILE_K) * sizeof(float);
    std::cout << "LDS per block: " << lds_bytes << " bytes\n\n";

    // Host buffers
    std::vector<float16_t> matQ(seq * head);
    std::vector<float16_t> matK(seq * head);
    std::vector<float16_t> matV(seq * head);
    std::vector<float16_t> matO(seq * head, std::numeric_limits<float16_t>::signaling_NaN());

    // Small values to keep outputs in FP16 safe range
    constexpr float kScale = 1.0f / 8.0f;
    std::cout << "Initializing host data...\n";
    fillRand(matQ.data(), seq, head);
    fillRand(matK.data(), seq, head);
    fillRand(matV.data(), seq, head);
    for(auto& x : matQ) x = static_cast<float16_t>(static_cast<float>(x) * kScale);
    for(auto& x : matK) x = static_cast<float16_t>(static_cast<float>(x) * kScale);
    for(auto& x : matV) x = static_cast<float16_t>(static_cast<float>(x) * kScale);

    // Device buffers
    std::cout << "Allocating device memory...\n";
    float16_t *d_q, *d_k, *d_v, *d_o;
    CHECK_HIP_ERROR(hipMalloc(&d_q, seq * head * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_k, seq * head * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_v, seq * head * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_o, seq * head * sizeof(float16_t)));

    CHECK_HIP_ERROR(hipMemcpy(d_q, matQ.data(), seq * head * sizeof(float16_t), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_k, matK.data(), seq * head * sizeof(float16_t), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_v, matV.data(), seq * head * sizeof(float16_t), hipMemcpyHostToDevice));

    // Grid: one block per query row
    dim3 gridDim(seq, 1);
    dim3 blockDim(BLOCK_SIZE, 1);

    std::cout << "grid=(" << gridDim.x << "," << gridDim.y
              << ")  block=(" << blockDim.x << "," << blockDim.y << ")\n\n";

    auto kernelLambda = [&]() {
        hipLaunchKernelGGL(flash_attn_kernel,
                           gridDim, blockDim,
                           lds_bytes, // dynamic shared memory
                           0,
                           seq, head,
                           d_q, d_k, d_v, d_o,
                           head, head, head, head,
                           scale, TILE_K);
    };

    constexpr uint32_t warmups    = 2u;
    constexpr uint32_t recordRuns = 5u;

    std::cout << "Warming up...\n";
    for(uint32_t i = 0; i < warmups; ++i) kernelLambda();

    std::cout << "Benchmarking...\n";
    hipEvent_t evStart, evStop;
    CHECK_HIP_ERROR(hipEventCreate(&evStart));
    CHECK_HIP_ERROR(hipEventCreate(&evStop));
    CHECK_HIP_ERROR(hipEventRecord(evStart));
    for(uint32_t i = 0; i < recordRuns; ++i) kernelLambda();
    CHECK_HIP_ERROR(hipEventRecord(evStop));
    CHECK_HIP_ERROR(hipEventSynchronize(evStop));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, evStart, evStop));
    CHECK_HIP_ERROR(hipEventDestroy(evStart));
    CHECK_HIP_ERROR(hipEventDestroy(evStop));

    // FLOPs: 2 GEMMs (QK^T and AV), each 2*S*S*D
    double gFlops     = 4.0 * (double)seq * (double)seq * (double)head * 1.0e-9;
    double tFlopsPerSec = gFlops * recordRuns / (elapsedMs * 1.0e-3) * 1.0e-3;

    std::cout << std::left
              << std::setw(10) << "S"       << std::setw(10) << "D"
              << std::setw(10) << "TILE_K"  << std::setw(14) << "elapsedMs"
              << std::setw(14) << "GFlops"  << std::setw(14) << "TFlops/s" << "\n";
    std::cout << std::left
              << std::setw(10) << seq       << std::setw(10) << head
              << std::setw(10) << TILE_K    << std::setw(14) << elapsedMs
              << std::setw(14) << gFlops    << std::setw(14) << tFlopsPerSec << "\n\n";

#if !NDEBUG
    std::cout << "Validating against CPU reference (same online softmax algorithm)...\n";

    CHECK_HIP_ERROR(hipMemcpy(matO.data(), d_o, seq * head * sizeof(float16_t), hipMemcpyDeviceToHost));

    std::vector<float16_t> matRef(seq * head, std::numeric_limits<float16_t>::signaling_NaN());
    flash_attn_cpu_ref(seq, head, matQ.data(), matK.data(), matV.data(), matRef.data());

    auto res = compareEqual<float16_t>(matO.data(), matRef.data(), seq * head);
    std::cout << (std::get<0>(res) ? "PASSED!" : "FAILED!") << "\n";
    std::cout << "Max relative error: " << std::get<1>(res) << "\n\n";
#endif

    CHECK_HIP_ERROR(hipFree(d_q));
    CHECK_HIP_ERROR(hipFree(d_k));
    CHECK_HIP_ERROR(hipFree(d_v));
    CHECK_HIP_ERROR(hipFree(d_o));

    std::cout << "Finished!\n";
}

int main()
{
    run_flash_attn_sample(SEQ_LEN, HEAD_DIM);
    return 0;
}
