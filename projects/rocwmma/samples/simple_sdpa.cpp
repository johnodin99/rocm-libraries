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

/* simple_sdpa.cpp — Scaled Dot-Product Attention (Naive Three-Pass)
 *
 *   O = Softmax( Q * K^T / sqrt(d_k) ) * V
 *
 * Three GPU kernels:
 *   sdpa_qk_gemm_d  — Q[S*D] x K^T[D*S] / sqrt(D) -> float32 scores[S*S]
 *   sdpa_softmax_d  — row-wise softmax float32->float16 [S*S]
 *   sdpa_av_gemm_d  — scores_f16[S*S] x V[S*D] -> float16 output[S*D]
 *
 * Key implementation details:
 *   K^T trick: K[S*D] row_major loaded as col_major fragB with ldb=D.
 *     col_major element(r,c) at ptr = k + ki + cCol*D:
 *       ptr[r + c*D] = k[ki+r + (cCol+c)*D] = K[cCol+c][ki+r] = K^T[ki+r][cCol+c]
 *   V loaded as row_major fragB:  ptr = v + ki*D + cCol
 *     row_major element(r,c) = v[(ki+r)*D + cCol+c] = V[ki+r][cCol+c]
 *   Softmax: one thread per row — numerically stable, Wave32/Wave64 safe.
 *
 * Data layouts (all row_major):
 *   Q [S*D]  ldq=D,   K [S*D] ldk=D,  V [S*D] ldv=D,  O [S*D] ldo=D
 *
 * Default: S=SEQ_LEN=64, D=HEAD_DIM=64 (multiples of ROCWMMA_K=16).
 * Architecture: gfx9 (Wave64), gfx11, gfx12 (Wave32). All use 16x16x16 tiles.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#include "common.hpp"

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

// Tile dims: 16x16x16 supported on all rocWMMA targets including gfx12 Wave32
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;
const int ROCWMMA_K = 16;

// Wave size resolved at runtime (Wave32 for gfx11/gfx12, Wave64 for gfx9)
const uint32_t WAVE_SIZE = getWarpSize();

// 4 warps in X x 4 warps in Y per threadblock
const int T_BLOCK_X = 4 * WAVE_SIZE;
const int T_BLOCK_Y = 4;

// Default problem size; must be multiples of 16 and >= one macro-tile
const uint32_t SEQ_LEN  = 64; // S
const uint32_t HEAD_DIM = 64; // D = d_k = d_v

// ---------------------------------------------------------------------------
// Pass 1: Q x K^T / sqrt(D) -> float32 scores [S x S]
//
// fragA = matrix_a + row_major  ->  Q[cRow..][ki..]
// fragB = matrix_b + col_major  ->  K^T[ki..][cCol..] via col_major + ldb=D
//   ptr = k + ki + cCol * lda (lda = D)
// ---------------------------------------------------------------------------
__global__ void sdpa_qk_gemm_d(uint32_t         seq,
                                uint32_t         head,
                                float16_t const* q,
                                float16_t const* k,
                                float32_t*       scores_f32,
                                uint32_t         lda,        // = D (head)
                                uint32_t         ld_scores,  // = S (seq)
                                float32_t        scale)      // = 1/sqrt(D)
{
    auto fragA   = rocwmma::fragment<matrix_a,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t, row_major>();
    auto fragB   = rocwmma::fragment<matrix_b,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t, col_major>();
    auto fragAcc = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float32_t>();

    rocwmma::fill_fragment(fragAcc, 0.0f);

    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x)
                     / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

    uint32_t cRow = majorWarp * ROCWMMA_M;
    uint32_t cCol = minorWarp * ROCWMMA_N;

    if(cRow < seq && cCol < seq)
    {
        // K-loop over head dimension D
        for(uint32_t ki = 0; ki < head; ki += ROCWMMA_K)
        {
            // A: Q[cRow..][ki..] row_major, lda=D
            rocwmma::load_matrix_sync(fragA, q + cRow * lda + ki, lda);

            // B: K^T[ki..][cCol..] via col_major with ldb=D
            //   ptr = k + ki + cCol*D
            //   col_major element(r,c) = ptr[r + c*D] = K[cCol+c][ki+r] = K^T[ki+r][cCol+c]
            rocwmma::load_matrix_sync(fragB, k + ki + cCol * lda, lda);

            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Apply attention scale and store float32 scores
        for(int e = 0; e < (int)fragAcc.num_elements; ++e)
            fragAcc.x[e] *= scale;

        rocwmma::store_matrix_sync(scores_f32 + cRow * ld_scores + cCol,
                                   fragAcc,
                                   ld_scores,
                                   rocwmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Row-wise softmax float32 -> float16
//
// One thread per row: numerically stable (subtract row_max), Wave32/64 safe.
// No warp reduction required — sequential over seq columns.
// ---------------------------------------------------------------------------
__global__ void sdpa_softmax_d(float32_t const* scores_f32,
                               float16_t*       scores_f16,
                               uint32_t         seq)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= seq)
        return;

    float32_t const* src = scores_f32 + row * seq;
    float16_t*       dst = scores_f16 + row * seq;

    // 1. Row maximum for numerical stability
    float32_t row_max = src[0];
    for(uint32_t j = 1; j < seq; ++j)
        row_max = fmaxf(row_max, src[j]);

    // 2. Shifted exponentials and their sum
    float32_t row_sum = 0.0f;
    for(uint32_t j = 0; j < seq; ++j)
        row_sum += expf(src[j] - row_max);

    // 3. Normalize and store as float16 attention weights
    float32_t inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for(uint32_t j = 0; j < seq; ++j)
        dst[j] = static_cast<float16_t>(expf(src[j] - row_max) * inv_sum);
}

// ---------------------------------------------------------------------------
// Pass 3: softmax_scores[S x S] x V[S x D] -> float16 output [S x D]
//
// fragA = matrix_a + row_major  ->  scores[cRow..][ki..]
// fragB = matrix_b + row_major  ->  V[ki..][cCol..]
//   ptr = v + ki * ldv + cCol
//   row_major element(r,c) = v[(ki+r)*D + cCol+c] = V[ki+r][cCol+c]
// ---------------------------------------------------------------------------
__global__ void sdpa_av_gemm_d(uint32_t         seq,
                                uint32_t         head,
                                float16_t const* scores_f16,
                                float16_t const* v,
                                float16_t*       out,
                                uint32_t         ld_scores, // = S (seq)
                                uint32_t         ldv,       // = D (head)
                                uint32_t         ldo)       // = D (head)
{
    auto fragA   = rocwmma::fragment<matrix_a,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t, row_major>();
    auto fragB   = rocwmma::fragment<matrix_b,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t, row_major>();
    auto fragAcc = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float32_t>();
    auto fragOut = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t>();

    rocwmma::fill_fragment(fragAcc, 0.0f);

    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x)
                     / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

    uint32_t cRow = majorWarp * ROCWMMA_M;
    uint32_t cCol = minorWarp * ROCWMMA_N;

    // Output rows are [0..S), output cols are [0..D)
    if(cRow < seq && cCol < head)
    {
        // K-loop over seq dimension S (shared dim between scores cols and V rows)
        for(uint32_t ki = 0; ki < seq; ki += ROCWMMA_K)
        {
            // A: scores[cRow..][ki..] row_major, ld=S
            rocwmma::load_matrix_sync(fragA, scores_f16 + cRow * ld_scores + ki, ld_scores);

            // B: V[ki..][cCol..] row_major, ld=D
            //   ptr = v + ki*D + cCol
            rocwmma::load_matrix_sync(fragB, v + ki * ldv + cCol, ldv);

            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Cast float32 accumulator -> float16 and store
        for(int e = 0; e < (int)fragAcc.num_elements; ++e)
            fragOut.x[e] = static_cast<float16_t>(fragAcc.x[e]);

        rocwmma::store_matrix_sync(out + cRow * ldo + cCol,
                                   fragOut,
                                   ldo,
                                   rocwmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// CPU reference
//   scores[S*S] = Q * K^T * scale     (row_major K, K^T via swapped index)
//   softmax row_major on scores
//   O[S*D] = softmax_scores * V
// ---------------------------------------------------------------------------
static void sdpa_cpu_ref(uint32_t         seq,
                         uint32_t         head,
                         float16_t const* q,
                         float16_t const* k,
                         float16_t const* v,
                         float16_t*       ref_out)
{
    float scale = 1.0f / std::sqrtf((float)head);

    std::vector<float> scores(seq * seq, 0.0f);

    // 1. scores = Q * K^T * scale
#pragma omp parallel for
    for(int i = 0; i < (int)seq; ++i)
        for(int j = 0; j < (int)seq; ++j)
        {
            float s = 0.0f;
            for(uint32_t d = 0; d < head; ++d)
                s += (float)q[i * head + d] * (float)k[j * head + d];
            scores[i * seq + j] = s * scale;
        }

    // 2. Row-wise softmax
    for(int i = 0; i < (int)seq; ++i)
    {
        float row_max = *std::max_element(scores.begin() + i * seq,
                                          scores.begin() + (i + 1) * seq);
        float row_sum = 0.0f;
        for(uint32_t j = 0; j < seq; ++j)
        {
            scores[i * seq + j] = std::expf(scores[i * seq + j] - row_max);
            row_sum += scores[i * seq + j];
        }
        float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for(uint32_t j = 0; j < seq; ++j)
            scores[i * seq + j] *= inv;
    }

    // 3. O = softmax_scores * V
#pragma omp parallel for
    for(int i = 0; i < (int)seq; ++i)
        for(uint32_t d = 0; d < head; ++d)
        {
            float s = 0.0f;
            for(uint32_t j = 0; j < seq; ++j)
                s += scores[i * seq + j] * (float)v[j * head + d];
            ref_out[i * head + d] = static_cast<float16_t>(s);
        }
}

// ---------------------------------------------------------------------------
// GPU device info
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
__host__ void run_sdpa_sample(uint32_t seq, uint32_t head)
{
    printDeviceInfo();

    // Tile compatibility check
    uint32_t tiles_x = T_BLOCK_X / WAVE_SIZE; // warps in X per block
    if((seq  % ROCWMMA_M) || (head % ROCWMMA_K)
       || (seq  < static_cast<uint32_t>(ROCWMMA_M * tiles_x))
       || (head < static_cast<uint32_t>(ROCWMMA_N * T_BLOCK_Y)))
    {
        std::cout << "Unsupported size (need multiples of 16 and >= macro-tile)!\n";
        return;
    }

    const uint32_t ldq        = head; // Q [S x D] row_major
    const uint32_t ldk        = head; // K [S x D] row_major
    const uint32_t ldv        = head; // V [S x D] row_major
    const uint32_t ldo        = head; // O [S x D] row_major
    const uint32_t ld_scores  = seq;  // scores [S x S] row_major
    const float32_t scale     = 1.0f / std::sqrtf((float)head);

    std::cout << "SDPA: S=" << seq << "  D=" << head
              << "  scale=1/sqrt(" << head << ")=" << scale << "\n\n";

    // Host buffers
    std::vector<float16_t> matQ(seq * head);
    std::vector<float16_t> matK(seq * head);
    std::vector<float16_t> matV(seq * head);
    std::vector<float16_t> matO(seq * head, std::numeric_limits<float16_t>::signaling_NaN());

    // Scale inputs to avoid FP16 overflow: max|score| ~ S*(1/8)^2*2 ~ 2, safe
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
    float16_t* d_q;
    float16_t* d_k;
    float16_t* d_v;
    float32_t* d_scores_f32; // Pass 1 output [S x S]
    float16_t* d_scores_f16; // Pass 2 output [S x S]
    float16_t* d_o;

    CHECK_HIP_ERROR(hipMalloc(&d_q,          seq * head * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_k,          seq * head * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_v,          seq * head * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_scores_f32, seq * seq  * sizeof(float32_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_scores_f16, seq * seq  * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_o,          seq * head * sizeof(float16_t)));

    CHECK_HIP_ERROR(hipMemcpy(d_q, matQ.data(), seq * head * sizeof(float16_t), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_k, matK.data(), seq * head * sizeof(float16_t), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_v, matV.data(), seq * head * sizeof(float16_t), hipMemcpyHostToDevice));

    // Grid/block configuration
    // Pass 1 QK: output [S x S], Pass 3 AV: output [S x D]
    auto blockDim    = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gemmGridQK  = dim3(rocwmma::ceil_div(seq,  (uint32_t)(ROCWMMA_M * tiles_x)),
                            rocwmma::ceil_div(seq,  (uint32_t)(ROCWMMA_N * T_BLOCK_Y)));
    auto gemmGridAV  = dim3(rocwmma::ceil_div(seq,  (uint32_t)(ROCWMMA_M * tiles_x)),
                            rocwmma::ceil_div(head, (uint32_t)(ROCWMMA_N * T_BLOCK_Y)));
    constexpr uint32_t smBlockSize = 64u;
    auto softmaxGrid = dim3(rocwmma::ceil_div(seq, smBlockSize));

    std::cout << "QK  grid=(" << gemmGridQK.x << "," << gemmGridQK.y
              << ")  block=(" << blockDim.x   << "," << blockDim.y << ")\n";
    std::cout << "SM  grid=(" << softmaxGrid.x << ")  block=(" << smBlockSize << ")\n";
    std::cout << "AV  grid=(" << gemmGridAV.x << "," << gemmGridAV.y << ")\n\n";

    auto kernelLambda = [&]() {
        // Pass 1: QK^T / sqrt(D) -> float32 scores
        hipExtLaunchKernelGGL(sdpa_qk_gemm_d,
                              gemmGridQK, blockDim, 0, 0, nullptr, nullptr, 0,
                              seq, head, d_q, d_k, d_scores_f32, ldq, ld_scores, scale);
        // Pass 2: Row-wise softmax -> float16 attention weights
        hipLaunchKernelGGL(sdpa_softmax_d,
                           softmaxGrid, dim3(smBlockSize), 0, 0,
                           d_scores_f32, d_scores_f16, seq);
        // Pass 3: attention * V -> float16 output
        hipExtLaunchKernelGGL(sdpa_av_gemm_d,
                              gemmGridAV, blockDim, 0, 0, nullptr, nullptr, 0,
                              seq, head, d_scores_f16, d_v, d_o,
                              ld_scores, ldv, ldo);
    };

    constexpr uint32_t warmups    = 2u;
    constexpr uint32_t recordRuns = 5u;

    std::cout << "Warming up...\n";
    for(uint32_t i = 0; i < warmups; ++i)
        kernelLambda();

    std::cout << "Benchmarking...\n";
    hipEvent_t evStart, evStop;
    CHECK_HIP_ERROR(hipEventCreate(&evStart));
    CHECK_HIP_ERROR(hipEventCreate(&evStop));
    CHECK_HIP_ERROR(hipEventRecord(evStart));
    for(uint32_t i = 0; i < recordRuns; ++i)
        kernelLambda();
    CHECK_HIP_ERROR(hipEventRecord(evStop));
    CHECK_HIP_ERROR(hipEventSynchronize(evStop));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, evStart, evStop));
    CHECK_HIP_ERROR(hipEventDestroy(evStart));
    CHECK_HIP_ERROR(hipEventDestroy(evStop));

    // FLOPs: 2 GEMMs each 2*S*S*D -> 4*S^2*D total + softmax (negligible)
    double gFlops     = 4.0 * (double)seq * (double)seq * (double)head * 1.0e-9;
    double tFlopsPerSec = gFlops * recordRuns / (static_cast<double>(elapsedMs) * 1.0e-3) * 1.0e-3;

    std::cout << std::left
              << std::setw(8)  << "BlkM"    << std::setw(8)  << "BlkN"
              << std::setw(8)  << "BlkK"    << std::setw(8)  << "S"
              << std::setw(8)  << "D"       << std::setw(14) << "elapsedMs"
              << std::setw(14) << "GFlops"  << std::setw(14) << "TFlops/s" << "\n";
    std::cout << std::left
              << std::setw(8)  << ROCWMMA_M  << std::setw(8)  << ROCWMMA_N
              << std::setw(8)  << ROCWMMA_K  << std::setw(8)  << seq
              << std::setw(8)  << head        << std::setw(14) << elapsedMs
              << std::setw(14) << gFlops      << std::setw(14) << tFlopsPerSec << "\n\n";

#if !NDEBUG
    std::cout << "Validating against CPU reference...\n";

    CHECK_HIP_ERROR(hipMemcpy(matO.data(), d_o,
                              seq * head * sizeof(float16_t), hipMemcpyDeviceToHost));

    std::vector<float16_t> matRef(seq * head, std::numeric_limits<float16_t>::signaling_NaN());
    sdpa_cpu_ref(seq, head, matQ.data(), matK.data(), matV.data(), matRef.data());

    auto res = compareEqual<float16_t>(matO.data(), matRef.data(), seq * head);
    std::cout << (std::get<0>(res) ? "PASSED!" : "FAILED!") << "\n";
    std::cout << "Max relative error: " << std::get<1>(res) << "\n\n";
#endif

    CHECK_HIP_ERROR(hipFree(d_q));
    CHECK_HIP_ERROR(hipFree(d_k));
    CHECK_HIP_ERROR(hipFree(d_v));
    CHECK_HIP_ERROR(hipFree(d_scores_f32));
    CHECK_HIP_ERROR(hipFree(d_scores_f16));
    CHECK_HIP_ERROR(hipFree(d_o));

    std::cout << "Finished!\n";
}

int main()
{
    // Single-head SDPA: S=64 tokens, D=64 head dimension
    run_sdpa_sample(SEQ_LEN, HEAD_DIM);
    return 0;
}
