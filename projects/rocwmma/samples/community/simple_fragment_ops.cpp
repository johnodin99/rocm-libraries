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

/* Fragment Operations Tutorial Sample
 *
 * Purpose: teach the fundamental rocWMMA fragment APIs that every user needs
 * to know before building fused LLM operator kernels.  Six self-contained
 * demos, each exercising a different operation on accumulator fragments.
 *
 * Key concept — fragment element ownership
 * ─────────────────────────────────────────
 * A 16×16 rocWMMA tile has 256 elements.  These are distributed evenly across
 * all threads in the warp:
 *
 *   gfx9  (Wave64): 256 / 64 = 4  elements per thread  → frag.num_elements == 4
 *   gfx12 (Wave32): 256 / 32 = 8  elements per thread  → frag.num_elements == 8
 *
 * Each thread accesses its own elements via frag.x[0] … frag.x[num_elements-1].
 * The mapping from (thread, element_index) → (row, col) in the tile is
 * architecture-dependent and managed by rocWMMA internally.
 *
 * Demo overview
 * ─────────────
 *  Demo 1  fill_fragment          fill all elements with a scalar constant
 *  Demo 2  element-wise scale     frag.x[i] *= scale  (read-modify-write .x[])
 *  Demo 3  GEMM + ReLU            mma_sync → max(0, frag.x[i])
 *  Demo 4  GEMM + scale + bias    mma_sync → alpha * frag.x[i] + beta
 *  Demo 5  GEMM + reduce (sum)    mma_sync → sum of all frag.x[i] per tile
 *  Demo 6  print layout           show lane_id / num_elements via device printf
 *
 * Matrix dimensions (all demos):  M=64, N=64, K=64  (4×4 warp tiles of 16×16)
 * A layout: row_major [M×K],  B layout: col_major [K×N]
 * Output D: row_major [M×N],  type: float32
 *
 * Launch config (all GEMM demos):
 *   Grid  = (TILES_M, TILES_N) = (4, 4)  — one block per output tile
 *   Block = (WAVE_SIZE, 1)               — one warp per block
 *
 * CPU validation is active when NDEBUG is NOT defined.
 */

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

using namespace rocwmma;

// ---------------------------------------------------------------------------
// Architecture: compile-time wave size
// ---------------------------------------------------------------------------
namespace gfx9Params  { constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE_64; }
namespace gfx11Params { constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE_32; }

#if ROCWMMA_ARCH_GFX9
using namespace gfx9Params;
#else
using namespace gfx11Params;
#endif

// ---------------------------------------------------------------------------
// Tile and matrix sizes
// ---------------------------------------------------------------------------
constexpr uint32_t ROCWMMA_M = 16u;
constexpr uint32_t ROCWMMA_N = 16u;
constexpr uint32_t ROCWMMA_K = 16u;

constexpr uint32_t MATRIX_M = 64u;
constexpr uint32_t MATRIX_N = 64u;
constexpr uint32_t MATRIX_K = 64u;

constexpr uint32_t TILES_M = MATRIX_M / ROCWMMA_M; // 4
constexpr uint32_t TILES_N = MATRIX_N / ROCWMMA_N; // 4

// Leading dimensions
constexpr uint32_t LDA = MATRIX_K; // A [M x K] row_major
constexpr uint32_t LDB = MATRIX_K; // B [K x N] col_major  (ldb = #rows = K)
constexpr uint32_t LDD = MATRIX_N; // D [M x N] row_major

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using InputT   = float16_t;
using ComputeT = float32_t;

// Fragment type aliases
//   FragA  : A sub-tile for one mma_sync call
//   FragB  : B sub-tile for one mma_sync call
//   FragAcc: float32 accumulator (result of mma_sync)
using FragA   = fragment<matrix_a,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT,   row_major>;
using FragB   = fragment<matrix_b,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT,   col_major>;
using FragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;

// ---------------------------------------------------------------------------
// Device helpers shared by all kernels
// ---------------------------------------------------------------------------

// Warp tile origin in the output matrix.
// Grid=(TILES_M, TILES_N), Block=(WAVE_SIZE,1): blockIdx directly gives tile index.
ROCWMMA_DEVICE static inline void warp_tile_origin(uint32_t& row, uint32_t& col)
{
    row = blockIdx.x * ROCWMMA_M;
    col = blockIdx.y * ROCWMMA_N;
}

// Compute A×B into fragAcc for the current tile.
// A: row_major [M×K],  lda=K
// B: col_major [K×N],  ldb=K  (col_major: B element (r,c) = b[c*ldb+r])
ROCWMMA_DEVICE static inline void
    gemm_tile(FragAcc& fragAcc, InputT const* a, InputT const* b, uint32_t tileRow, uint32_t tileCol)
{
    fill_fragment(fragAcc, ComputeT(0));
    for(uint32_t k = 0; k < MATRIX_K; k += ROCWMMA_K)
    {
        FragA fragA;
        FragB fragB;
        load_matrix_sync(fragA, a + tileRow * LDA + k,    LDA);
        load_matrix_sync(fragB, b + tileCol * LDB + k,    LDB);
        mma_sync(fragAcc, fragA, fragB, fragAcc);
    }
}

// ===========================================================================
// Demo 1 — fill_fragment
//
// Shows: fill_fragment(frag, scalar) → sets every element to that scalar.
//
// Expected output D[i,j] = FILL_VAL for all (i,j).
// ===========================================================================
ROCWMMA_KERNEL void demo1_fill(ComputeT* d_out, ComputeT fill_val)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        uint32_t tileRow, tileCol;
        warp_tile_origin(tileRow, tileCol);

        FragAcc frag;

        // ── Key API ─────────────────────────────────────────────────────────
        fill_fragment(frag, fill_val);
        // ────────────────────────────────────────────────────────────────────

        store_matrix_sync(d_out + tileRow * LDD + tileCol, frag, LDD, mem_row_major);
    }
}

// ===========================================================================
// Demo 2 — element-wise scale via .x[i] loop
//
// Shows:
//   1. fill_fragment(frag, init)  — set all elements to init_val
//   2. frag.x[i] *= scale         — read/write each owned element
//   3. frag.num_elements           — how many elements this thread owns
//
// Expected output D[i,j] = init_val * scale for all (i,j).
// ===========================================================================
ROCWMMA_KERNEL void demo2_element_scale(ComputeT* d_out, ComputeT init_val, ComputeT scale)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        uint32_t tileRow, tileCol;
        warp_tile_origin(tileRow, tileCol);

        FragAcc frag;
        fill_fragment(frag, init_val);

        // ── Key API ─────────────────────────────────────────────────────────
        // frag.num_elements: number of matrix elements owned by this thread.
        // frag.x[i]        : direct read/write access to the i-th element.
        for(int i = 0; i < (int)frag.num_elements; i++)
            frag.x[i] = frag.x[i] * scale;
        // ────────────────────────────────────────────────────────────────────

        store_matrix_sync(d_out + tileRow * LDD + tileCol, frag, LDD, mem_row_major);
    }
}

// ===========================================================================
// Demo 3 — GEMM + element-wise ReLU (map with conditional)
//
// Shows:
//   1. mma_sync accumulates A×B into fragAcc
//   2. fragAcc.x[i] = max(0, fragAcc.x[i])   — element-wise activation
//
// Expected output D[i,j] = max(0, (A×B)[i,j]).
// ===========================================================================
ROCWMMA_KERNEL void demo3_gemm_relu(InputT const* a, InputT const* b, ComputeT* d_out)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        uint32_t tileRow, tileCol;
        warp_tile_origin(tileRow, tileCol);

        FragAcc fragAcc;
        gemm_tile(fragAcc, a, b, tileRow, tileCol);

        // ── Key API ─────────────────────────────────────────────────────────
        for(int i = 0; i < (int)fragAcc.num_elements; i++)
            fragAcc.x[i] = fragAcc.x[i] > ComputeT(0) ? fragAcc.x[i] : ComputeT(0);
        // ────────────────────────────────────────────────────────────────────

        store_matrix_sync(d_out + tileRow * LDD + tileCol, fragAcc, LDD, mem_row_major);
    }
}

// ===========================================================================
// Demo 4 — GEMM + element-wise scale + bias (compound transform)
//
// Shows:
//   fragAcc.x[i] = alpha * fragAcc.x[i] + beta
//
// Expected output D[i,j] = alpha * (A×B)[i,j] + beta.
// Same pattern used in the final step of GEMM+RMSNorm, GEMM+Softmax, etc.
// ===========================================================================
ROCWMMA_KERNEL void demo4_gemm_scalebias(InputT const* a,
                                          InputT const* b,
                                          ComputeT*     d_out,
                                          ComputeT      alpha,
                                          ComputeT      beta)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        uint32_t tileRow, tileCol;
        warp_tile_origin(tileRow, tileCol);

        FragAcc fragAcc;
        gemm_tile(fragAcc, a, b, tileRow, tileCol);

        // ── Key API ─────────────────────────────────────────────────────────
        for(int i = 0; i < (int)fragAcc.num_elements; i++)
            fragAcc.x[i] = alpha * fragAcc.x[i] + beta;
        // ────────────────────────────────────────────────────────────────────

        store_matrix_sync(d_out + tileRow * LDD + tileCol, fragAcc, LDD, mem_row_major);
    }
}

// ===========================================================================
// Demo 5 — GEMM + fragment reduce (sum all owned elements)
//
// Shows:
//   ComputeT partial = 0;
//   for(int i ...) partial += fragAcc.x[i];   ← per-thread partial sum
//   atomicAdd(&d_sums[tileIdx], partial);      ← warp-level reduction via atomic
//
// Each tile produces one float in d_sums[tileIdx].
// For production, use warp-shuffle reduction instead of atomic.
//
// This pattern is the building block for:
//   - Softmax (sum of exp values)
//   - RMSNorm (sum of squares)
//   - Mean pooling
// ===========================================================================
ROCWMMA_KERNEL void demo5_gemm_reduce(InputT const* a, InputT const* b, ComputeT* d_sums)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        uint32_t tileRow, tileCol;
        warp_tile_origin(tileRow, tileCol);
        uint32_t tileIdx = blockIdx.x * TILES_N + blockIdx.y;

        FragAcc fragAcc;
        gemm_tile(fragAcc, a, b, tileRow, tileCol);

        // ── Key API ─────────────────────────────────────────────────────────
        // Per-thread partial sum over its owned elements
        ComputeT partial = ComputeT(0);
        for(int i = 0; i < (int)fragAcc.num_elements; i++)
            partial += fragAcc.x[i];

        // Accumulate across all threads in the warp (simple; use shuffle for perf)
        atomicAdd(&d_sums[tileIdx], partial);
        // ────────────────────────────────────────────────────────────────────
    }
}

// ===========================================================================
// Demo 6 — Print fragment element layout (device printf)
//
// Only runs for the first tile (blockIdx == (0,0)).
// Prints: lane_id, num_elements, and each element value.
//
// This demonstrates that the 16×16 = 256 tile elements are split evenly
// across all threads in the warp.  Every thread "owns" a subset.
//
//   gfx9  (Wave64): 256/64 = 4  elements per thread
//   gfx12 (Wave32): 256/32 = 8  elements per thread
// ===========================================================================
ROCWMMA_KERNEL void demo6_print_layout(ComputeT fill_val)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        // Only print for the very first tile
        if(blockIdx.x != 0 || blockIdx.y != 0)
            return;

        FragAcc frag;

        // Assign each element a unique value: lane * 100 + element_index
        // so readers can trace which thread owns which element slot.
        uint32_t lane = threadIdx.x;
        for(int i = 0; i < (int)frag.num_elements; i++)
            frag.x[i] = static_cast<ComputeT>(lane * 100 + i);

        // ── Key API ─────────────────────────────────────────────────────────
        // Print from every lane in the warp. Output may be interleaved by the
        // hardware; sort by 'lane' when reading the output.
        printf("  lane[%2u]  num_elements=%u  values=",
               lane,
               frag.num_elements);
        for(int i = 0; i < (int)frag.num_elements; i++)
            printf("%.0f ", static_cast<float>(frag.x[i]));
        printf("\n");
        // ────────────────────────────────────────────────────────────────────

        // Also show that fill_fragment overrides .x[] assignments
        if(lane == 0)
        {
            fill_fragment(frag, fill_val);
            printf("  After fill_fragment(frag, %.1f): "
                   "frag.x[0]=%.1f  frag.num_elements=%u\n",
                   static_cast<float>(fill_val),
                   static_cast<float>(frag.x[0]),
                   frag.num_elements);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference for GEMM: C = A x B
//   A row_major [M×K],  B col_major [K×N]  → C row_major [M×N]
// ---------------------------------------------------------------------------
static void cpu_gemm(uint32_t      m,
                     uint32_t      n,
                     uint32_t      k,
                     InputT const* a,
                     InputT const* b,
                     ComputeT*     c)
{
#pragma omp parallel for
    for(int i = 0; i < (int)m; i++)
    {
        for(int j = 0; j < (int)n; j++)
        {
            ComputeT sum = ComputeT(0);
            for(int h = 0; h < (int)k; h++)
            {
                // A[i,h] row_major   : a[i*K + h]
                // B[h,j] col_major   : b[j*K + h]
                sum += static_cast<ComputeT>(a[i * k + h])
                       * static_cast<ComputeT>(b[j * k + h]);
            }
            c[i * n + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Print GPU hardware info
// ---------------------------------------------------------------------------
static void printDeviceInfo()
{
    hipDevice_t     dev;
    hipDeviceProp_t props;
    CHECK_HIP_ERROR(hipGetDevice(&dev));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&props, dev));

    std::cout << "\n=== GPU Hardware Info ===\n"
              << "  Device name     : " << props.name << "\n"
              << "  GCN arch        : " << props.gcnArchName << "\n"
              << "  Warp size       : " << props.warpSize << "\n"
              << "  Global memory   : " << (props.totalGlobalMem >> 20) << " MiB\n"
              << "  Shared mem/blk  : " << (props.sharedMemPerBlock >> 10) << " KiB\n"
              << "========================\n\n";
}

// ---------------------------------------------------------------------------
// Helper: report demo result
// ---------------------------------------------------------------------------
static void report(const char* name, bool passed, double max_err)
{
    std::cout << std::left << std::setw(36) << name
              << (passed ? "PASSED" : "FAILED")
              << "  max_rel_err=" << max_err << "\n";
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
ROCWMMA_HOST void run_fragment_ops_sample()
{
    printDeviceInfo();

    auto warpSz = getWarpSize();
    std::cout << "Fragment element count per thread (num_elements):\n"
              << "  TILE = " << ROCWMMA_M << "x" << ROCWMMA_N << " = "
              << ROCWMMA_M * ROCWMMA_N << " elements / "
              << warpSz << " threads = "
              << ROCWMMA_M * ROCWMMA_N / warpSz << " per thread\n\n";

    // -------------------------------------------------------------------
    // Allocate and fill host input matrices
    // -------------------------------------------------------------------
    std::vector<InputT> matA(MATRIX_M * MATRIX_K);
    std::vector<InputT> matB(MATRIX_K * MATRIX_N);

    fillRand(matA.data(), MATRIX_M, MATRIX_K);
    fillRand(matB.data(), MATRIX_K, MATRIX_N);

    // Scale to keep GEMM outputs in a moderate range for FP32
    constexpr float kScale = 1.0f / 16.0f;
    for(auto& x : matA) x = static_cast<InputT>(static_cast<float>(x) * kScale);
    for(auto& x : matB) x = static_cast<InputT>(static_cast<float>(x) * kScale);

    // -------------------------------------------------------------------
    // Device allocations
    // -------------------------------------------------------------------
    InputT*   d_a;
    InputT*   d_b;
    ComputeT* d_out1;   // Demo 1: fill
    ComputeT* d_out2;   // Demo 2: scale
    ComputeT* d_out3;   // Demo 3: ReLU
    ComputeT* d_out4;   // Demo 4: scale+bias
    ComputeT* d_sums;   // Demo 5: per-tile reduce sums

    const size_t bytesAB  = MATRIX_M * MATRIX_K * sizeof(InputT);
    const size_t bytesOut = MATRIX_M * MATRIX_N  * sizeof(ComputeT);
    const size_t bytesSums = TILES_M  * TILES_N   * sizeof(ComputeT);

    CHECK_HIP_ERROR(hipMalloc(&d_a,    bytesAB));
    CHECK_HIP_ERROR(hipMalloc(&d_b,    bytesAB));
    CHECK_HIP_ERROR(hipMalloc(&d_out1, bytesOut));
    CHECK_HIP_ERROR(hipMalloc(&d_out2, bytesOut));
    CHECK_HIP_ERROR(hipMalloc(&d_out3, bytesOut));
    CHECK_HIP_ERROR(hipMalloc(&d_out4, bytesOut));
    CHECK_HIP_ERROR(hipMalloc(&d_sums, bytesSums));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matA.data(), bytesAB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matB.data(), bytesAB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_sums, 0, bytesSums)); // Demo 5 uses atomicAdd

    // -------------------------------------------------------------------
    // Launch config: one warp per tile
    //   grid  = (TILES_M, TILES_N) = (4, 4)
    //   block = (WAVE_SIZE, 1)
    // -------------------------------------------------------------------
    dim3 gridDim(TILES_M, TILES_N);
    dim3 blockDim(warpSz, 1);

    std::cout << "Grid (" << gridDim.x << "x" << gridDim.y << ")  "
              << "Block (" << blockDim.x << "x" << blockDim.y << ")  "
              << "Tiles=" << TILES_M * TILES_N << "\n\n";

    // -------------------------------------------------------------------
    // Constants for demos 2 and 4
    // -------------------------------------------------------------------
    constexpr ComputeT FILL_VAL  = 3.0f;
    constexpr ComputeT SCALE_VAL = 4.0f;
    constexpr ComputeT INIT_VAL  = 2.0f;
    constexpr ComputeT ALPHA     = 2.0f;
    constexpr ComputeT BETA      = 1.0f;

    // -------------------------------------------------------------------
    // Launch all demos
    // -------------------------------------------------------------------
    std::cout << "Launching demos...\n";

    hipLaunchKernelGGL(demo1_fill,
                       gridDim, blockDim, 0, 0,
                       d_out1, FILL_VAL);

    hipLaunchKernelGGL(demo2_element_scale,
                       gridDim, blockDim, 0, 0,
                       d_out2, INIT_VAL, SCALE_VAL);

    hipLaunchKernelGGL(demo3_gemm_relu,
                       gridDim, blockDim, 0, 0,
                       d_a, d_b, d_out3);

    hipLaunchKernelGGL(demo4_gemm_scalebias,
                       gridDim, blockDim, 0, 0,
                       d_a, d_b, d_out4, ALPHA, BETA);

    hipLaunchKernelGGL(demo5_gemm_reduce,
                       gridDim, blockDim, 0, 0,
                       d_a, d_b, d_sums);

    // Demo 6: print layout (only first tile, uses device printf)
    std::cout << "\n--- Demo 6: Fragment layout (first tile, all warp lanes) ---\n";
    hipLaunchKernelGGL(demo6_print_layout,
                       gridDim, blockDim, 0, 0,
                       FILL_VAL);
    CHECK_HIP_ERROR(hipDeviceSynchronize()); // flush device printf

    std::cout << "--- End Demo 6 ---\n\n";

    // -------------------------------------------------------------------
    // Copy results back
    // -------------------------------------------------------------------
    std::vector<ComputeT> resOut1(MATRIX_M * MATRIX_N);
    std::vector<ComputeT> resOut2(MATRIX_M * MATRIX_N);
    std::vector<ComputeT> resOut3(MATRIX_M * MATRIX_N);
    std::vector<ComputeT> resOut4(MATRIX_M * MATRIX_N);
    std::vector<ComputeT> resSums(TILES_M * TILES_N);

    CHECK_HIP_ERROR(hipMemcpy(resOut1.data(), d_out1, bytesOut,  hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(resOut2.data(), d_out2, bytesOut,  hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(resOut3.data(), d_out3, bytesOut,  hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(resOut4.data(), d_out4, bytesOut,  hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(resSums.data(), d_sums, bytesSums, hipMemcpyDeviceToHost));

    // -------------------------------------------------------------------
    // CPU references and validation
    // -------------------------------------------------------------------

    // Demo 1 reference: constant FILL_VAL
    std::vector<ComputeT> refOut1(MATRIX_M * MATRIX_N, FILL_VAL);

    // Demo 2 reference: INIT_VAL * SCALE_VAL
    std::vector<ComputeT> refOut2(MATRIX_M * MATRIX_N, INIT_VAL * SCALE_VAL);

    // Compute GEMM for Demos 3, 4, 5
    std::vector<ComputeT> gemm_result(MATRIX_M * MATRIX_N);
    cpu_gemm(MATRIX_M, MATRIX_N, MATRIX_K,
             matA.data(), matB.data(), gemm_result.data());

    // Demo 3 reference: ReLU(GEMM)
    std::vector<ComputeT> refOut3(MATRIX_M * MATRIX_N);
    for(int i = 0; i < (int)(MATRIX_M * MATRIX_N); i++)
        refOut3[i] = gemm_result[i] > 0.f ? gemm_result[i] : 0.f;

    // Demo 4 reference: alpha * GEMM + beta
    std::vector<ComputeT> refOut4(MATRIX_M * MATRIX_N);
    for(int i = 0; i < (int)(MATRIX_M * MATRIX_N); i++)
        refOut4[i] = ALPHA * gemm_result[i] + BETA;

    // Demo 5 reference: per-tile sum of GEMM
    std::vector<ComputeT> refSums(TILES_M * TILES_N, 0.f);
    for(uint32_t ti = 0; ti < TILES_M; ti++)
        for(uint32_t tj = 0; tj < TILES_N; tj++)
        {
            float tsum = 0.f;
            for(uint32_t i = ti * ROCWMMA_M; i < (ti + 1) * ROCWMMA_M; i++)
                for(uint32_t j = tj * ROCWMMA_N; j < (tj + 1) * ROCWMMA_N; j++)
                    tsum += gemm_result[i * MATRIX_N + j];
            refSums[ti * TILES_N + tj] = tsum;
        }

    std::cout << "Validation results:\n";

    auto r1 = compareEqual(resOut1.data(), refOut1.data(), MATRIX_M * MATRIX_N);
    report("Demo 1  fill_fragment",         std::get<0>(r1), std::get<1>(r1));

    auto r2 = compareEqual(resOut2.data(), refOut2.data(), MATRIX_M * MATRIX_N);
    report("Demo 2  element-wise scale",    std::get<0>(r2), std::get<1>(r2));

    auto r3 = compareEqual(resOut3.data(), refOut3.data(), MATRIX_M * MATRIX_N);
    report("Demo 3  GEMM + ReLU",           std::get<0>(r3), std::get<1>(r3));

    auto r4 = compareEqual(resOut4.data(), refOut4.data(), MATRIX_M * MATRIX_N);
    report("Demo 4  GEMM + scale+bias",     std::get<0>(r4), std::get<1>(r4));

    auto r5 = compareEqual(resSums.data(), refSums.data(), TILES_M * TILES_N);
    report("Demo 5  GEMM + reduce (sum)",   std::get<0>(r5), std::get<1>(r5));

    std::cout << "Demo 6  print layout        (see printf output above)\n";

    // -------------------------------------------------------------------
    // Free device memory
    // -------------------------------------------------------------------
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_out1));
    CHECK_HIP_ERROR(hipFree(d_out2));
    CHECK_HIP_ERROR(hipFree(d_out3));
    CHECK_HIP_ERROR(hipFree(d_out4));
    CHECK_HIP_ERROR(hipFree(d_sums));

    std::cout << "\nFinished!\n";
}

int main()
{
    run_fragment_ops_sample();
    return 0;
}
