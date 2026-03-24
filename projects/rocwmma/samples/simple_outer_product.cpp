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

/* Outer Product (Rank-1 Update) Sample
 *
 * Computes the outer product of two vectors using rocWMMA:
 *
 *   D[M x N] = alpha * (u[M] x v[N]^T) + beta * C[M x N]
 *
 * where u is an M-dimensional column vector and v is an N-dimensional row
 * vector. This is equivalent to a rank-1 GEMM (K=1).
 *
 * Application:
 *   LoRA (Low-Rank Adaptation) delta-weight update:
 *     delta_W = A_lora x B_lora   (rank r, each "slice" is an outer product)
 *
 * Implementation strategy:
 *   rocWMMA requires K to be a multiple of ROCWMMA_K=16. We represent the
 *   outer product as a K=ROCWMMA_K GEMM by padding the K dimension:
 *     - matA [M x ROCWMMA_K] : row_major, only column 0 holds u[0..M-1]
 *     - matB [ROCWMMA_K x N] : col_major, only row    0 holds v[0..N-1]
 *   All other K-slot entries are zero, so the GEMM collapses to a rank-1
 *   result: D[i][j] = alpha * u[i] * v[j] + beta * C[i][j].
 *
 * Data layouts:
 *   A    : row_major  [M x ROCWMMA_K]
 *   B    : col_major  [ROCWMMA_K x N]  (padding rows 1..ROCWMMA_K-1 to zero)
 *   C, D : row_major  [M x N]
 *
 * Note: This is a simplified implementation for demonstrating API usage.
 *       It is not optimized for performance.
 */

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

// ---------------------------------------------------------------------------
// Tile dimensions — 16x16 supported on all rocWMMA architectures including
// gfx12 (RDNA4 Wave32) and gfx9 (CDNA Wave64).
// ---------------------------------------------------------------------------
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;
const int ROCWMMA_K = 16; // Minimum required K tile; also the padded K size

// Device warp size (Wave32 for gfx11/12, Wave64 for gfx9)
const uint32_t WAVE_SIZE = getWarpSize();

// Thread block: each wave computes one ROCWMMA_M x ROCWMMA_N output tile.
// T_BLOCK_X must be a multiple of WAVE_SIZE.
const int T_BLOCK_X = 4 * WAVE_SIZE;
const int T_BLOCK_Y = 4;

// ---------------------------------------------------------------------------
// Kernel: outer_product_rocwmma_d
//
//   D = alpha * A * B + beta * C
//
//   A [M x K] row_major  — encodes column vector u in column-0 (K-padded)
//   B [K x N] col_major  — encodes row    vector v in row-0    (K-padded)
//   C [M x N] row_major  — accumulation bias
//   D [M x N] row_major  — output
//
// One wave per ROCWMMA_M x ROCWMMA_N output block.
// ---------------------------------------------------------------------------
__global__ void outer_product_rocwmma_d(uint32_t         m,
                                        uint32_t         n,
                                        uint32_t         k,
                                        float16_t const* a,
                                        float16_t const* b,
                                        float16_t const* c,
                                        float16_t*       d,
                                        uint32_t         lda,
                                        uint32_t         ldb,
                                        uint32_t         ldc,
                                        uint32_t         ldd,
                                        float32_t        alpha,
                                        float32_t        beta)
{
    // Create fragments
    auto fragA   = rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t, row_major>();
    auto fragB   = rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t, col_major>();
    auto fragC   = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float16_t>();
    auto fragAcc = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                     float32_t>();

    rocwmma::fill_fragment(fragAcc, 0.0f);

    // Warp tile coordinates (2-D grid of waves)
    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x)
                     / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

    auto cRow = majorWarp * ROCWMMA_M;
    auto cCol = minorWarp * ROCWMMA_N;

    if(cRow < m && cCol < n)
    {
        // K-loop: only one iteration for the padded K dimension.
        // Only K-slot 0 carries data (u[i] * v[j]); all other slots are zero.
        for(uint32_t ki = 0; ki < k; ki += ROCWMMA_K)
        {
            rocwmma::load_matrix_sync(fragA, a + (cRow * lda + ki), lda);
            rocwmma::load_matrix_sync(fragB, b + (ki + cCol * ldb), ldb);
            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Load C, scale, and store D
        rocwmma::load_matrix_sync(fragC,
                                   c + (cRow * ldc + cCol),
                                   ldc,
                                   rocwmma::mem_row_major);

        for(int e = 0; e < (int)fragC.num_elements; ++e)
            fragC.x[e] = static_cast<float16_t>(
                alpha * fragAcc.x[e] + beta * static_cast<float32_t>(fragC.x[e]));

        rocwmma::store_matrix_sync(d + (cRow * ldd + cCol),
                                    fragC,
                                    ldd,
                                    rocwmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// CPU reference
//
//   D[i][j] = alpha * A[i][k=0] * B[k=0][j] + beta * C[i][j]
//           = alpha * u[i]       * v[j]       + beta * C[i][j]
//
// Only k=0 contributes; remaining K-slots of matA/matB are zero by design.
// The loop over all k is written out in full for generality.
// ---------------------------------------------------------------------------
static void outer_product_cpu_ref(uint32_t         m,
                                   uint32_t         n,
                                   uint32_t         k,
                                   float16_t const* a,
                                   float16_t const* b,
                                   float16_t const* c,
                                   float16_t*       d,
                                   uint32_t         lda,
                                   uint32_t         ldb,
                                   uint32_t         ldc,
                                   uint32_t         ldd,
                                   float32_t        alpha,
                                   float32_t        beta)
{
    // A : row_major  -> a[i*lda + ki]
    // B : col_major  -> b[ki + ldb*j]  (ldb == k == ROCWMMA_K for col_major [k x n])
    // C, D : row_major -> c[i*ldc + j]
#pragma omp parallel for
    for(int i = 0; i < (int)m; ++i)
    {
        for(int j = 0; j < (int)n; ++j)
        {
            float32_t acc = 0.0f;
            for(uint32_t ki = 0; ki < k; ++ki)
                acc += static_cast<float32_t>(a[i * lda + ki])
                       * static_cast<float32_t>(b[ki + ldb * j]);

            d[i * ldd + j] = static_cast<float16_t>(
                alpha * acc + beta * static_cast<float32_t>(c[i * ldc + j]));
        }
    }
}

// ---------------------------------------------------------------------------
// Print GPU device info
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
              << "  Compute units   : " << props.multiProcessorCount << "\n"
              << "  Warp size       : " << props.warpSize << "\n"
              << "  Global memory   : " << (props.totalGlobalMem >> 20) << " MiB\n"
              << "  Shared mem/blk  : " << (props.sharedMemPerBlock >> 10) << " KiB\n"
              << "========================\n\n";
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
__host__ void outer_product_test(uint32_t m, uint32_t n, float32_t alpha, float32_t beta)
{
    printDeviceInfo();

    // For this sample the "K" dimension is exactly ROCWMMA_K (padded rank-1).
    const uint32_t k = ROCWMMA_K;

    // Tile geometry bounds check
    if((m < (ROCWMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y))
        || (m % ROCWMMA_M || n % ROCWMMA_N))
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    // Leading dimensions
    // A : row_major [M x K]    -> lda = K
    // B : col_major [K x N]    -> ldb = K  (leading dim = number of rows = K)
    // C, D : row_major [M x N] -> ldc = ldd = N
    const uint32_t lda = k; // row_major A: stride = K
    const uint32_t ldb = k; // col_major B: leading dim = K (rows)
    const uint32_t ldc = n;
    const uint32_t ldd = n;

    std::cout << "Outer product test: M=" << m << " N=" << n
              << " K(padded)=" << k << "\n";
    std::cout << "  Semantics: D[M x N] = alpha * u[M] x v[N]^T + beta * C[M x N]\n";
    std::cout << "  alpha=" << alpha << "  beta=" << beta << "\n\n";

    // -----------------------------------------------------------------------
    // Initialize host matrices
    //
    // matA [M x K] row_major: u[i] lives in column 0, columns 1..K-1 = 0.
    // matB [K x N] col_major: v[j] lives in row    0, rows    1..K-1 = 0.
    //   col_major storage: b[ki + k*j]  ->  b[0 + k*j] = v[j]
    // -----------------------------------------------------------------------
    std::vector<float16_t> matA(m * k, float16_t(0));
    std::vector<float16_t> matB(k * n, float16_t(0));
    std::vector<float16_t> matC(m * n);
    std::vector<float16_t> matD(m * n, std::numeric_limits<float16_t>::signaling_NaN());

    // Fill the "real" vector u into column 0 of matA (row_major)
    // u[i] = small random value to avoid FP16 overflow
    std::cout << "Initializing host data...\n";
    auto seed = static_cast<unsigned>(time(nullptr));
    srand(seed);
    for(uint32_t i = 0; i < m; ++i)
    {
        float val   = (static_cast<float>(rand() % 9) - 4.0f) / 8.0f; // [-0.5, 0.5]
        matA[i * lda + 0] = static_cast<float16_t>(val);
        // columns 1..K-1 remain 0 (vector is already initialized above)
    }

    // Fill the "real" vector v into row 0 of matB (col_major: b[0 + k*j] = v[j])
    for(uint32_t j = 0; j < n; ++j)
    {
        float val        = (static_cast<float>(rand() % 9) - 4.0f) / 8.0f;
        matB[0 + ldb * j] = static_cast<float16_t>(val);
        // rows 1..K-1 remain 0
    }

    // Fill C with small random values
    fillRand(matC.data(), m, n);
    for(auto& x : matC)
        x = static_cast<float16_t>(static_cast<float>(x) / 16.0f);

    // -----------------------------------------------------------------------
    // Allocate and copy device memory
    // -----------------------------------------------------------------------
    std::cout << "Allocating device memory...\n";
    float16_t* d_a;
    float16_t* d_b;
    float16_t* d_c;
    float16_t* d_d;

    CHECK_HIP_ERROR(hipMalloc(&d_a, matA.size() * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, matB.size() * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, matC.size() * sizeof(float16_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, matD.size() * sizeof(float16_t)));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matA.data(), matA.size() * sizeof(float16_t),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matB.data(), matB.size() * sizeof(float16_t),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matC.data(), matC.size() * sizeof(float16_t),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matD.data(), matD.size() * sizeof(float16_t),
                              hipMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Launch kernel
    // -----------------------------------------------------------------------
    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceil_div(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                         rocwmma::ceil_div(n, ROCWMMA_N * T_BLOCK_Y));

    std::cout << "Launching outer product kernel...\n";
    std::cout << "  gridDim=(" << gridDim.x << "," << gridDim.y
              << ")  blockDim=(" << blockDim.x << "," << blockDim.y << ")\n";

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    hipExtLaunchKernelGGL(outer_product_rocwmma_d,
                          gridDim,
                          blockDim,
                          0,          // sharedMemBytes
                          0,          // stream
                          startEvent, // event start
                          stopEvent,  // event stop
                          0,          // flags
                          m, n, k,
                          d_a, d_b, d_c, d_d,
                          lda, ldb, ldc, ldd,
                          alpha, beta);

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Effective FLOPs: 2*M*N*K (rank-1 GEMM) + M*N (scale/add)
    // For the outer product, K=1 conceptually but K=ROCWMMA_K here.
    // We report the "true" rank-1 cost: 2*M*N*1 multiply-adds + M*N for beta.
    double gFlops     = 2.0 * static_cast<double>(m) * static_cast<double>(n) * 1.0e-9;
    double tFlopsPerSec = gFlops / static_cast<double>(elapsedMs);

    std::cout << "\n";
    std::cout << std::left
              << std::setw(10) << "BlkM"   << std::setw(10) << "BlkN"
              << std::setw(10) << "BlkK"   << std::setw(8)  << "MatM"
              << std::setw(8)  << "MatN"   << std::setw(8)  << "alpha"
              << std::setw(8)  << "beta"   << std::setw(14) << "elapsedMs"
              << std::setw(14) << "GFlops" << std::setw(14) << "TFlops/s" << "\n";

    std::cout << std::left
              << std::setw(10) << ROCWMMA_M  << std::setw(10) << ROCWMMA_N
              << std::setw(10) << ROCWMMA_K  << std::setw(8)  << m
              << std::setw(8)  << n          << std::setw(8)  << alpha
              << std::setw(8)  << beta       << std::setw(14) << elapsedMs
              << std::setw(14) << gFlops     << std::setw(14) << tFlopsPerSec << "\n\n";

#if !NDEBUG
    std::cout << "Validating result with CPU reference...\n";

    CHECK_HIP_ERROR(hipMemcpy(matD.data(), d_d,
                              matD.size() * sizeof(float16_t),
                              hipMemcpyDeviceToHost));

    std::vector<float16_t> matD_ref(m * n,
                                    std::numeric_limits<float16_t>::signaling_NaN());

    outer_product_cpu_ref(m, n, k,
                          matA.data(), matB.data(), matC.data(), matD_ref.data(),
                          lda, ldb, ldc, ldd,
                          alpha, beta);

    auto res = compareEqual<float16_t>(matD.data(), matD_ref.data(), m * n);

    if(!std::get<0>(res))
        std::cout << "FAILED!\n";
    else
        std::cout << "PASSED!\n";

    std::cout << "Max relative error: " << std::get<1>(res) << "\n\n";

    // Print a small 8x8 corner of the result for visual inspection
    const uint32_t PRINT_ROWS = std::min(m, 8u);
    const uint32_t PRINT_COLS = std::min(n, 8u);
    std::cout << "GPU result D[0.." << PRINT_ROWS-1
              << "][0.." << PRINT_COLS-1 << "]:\n";
    for(uint32_t i = 0; i < PRINT_ROWS; ++i)
    {
        for(uint32_t j = 0; j < PRINT_COLS; ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << static_cast<float>(matD[i * ldd + j]) << " ";
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "CPU reference D[0.." << PRINT_ROWS-1
              << "][0.." << PRINT_COLS-1 << "]:\n";
    for(uint32_t i = 0; i < PRINT_ROWS; ++i)
    {
        for(uint32_t j = 0; j < PRINT_COLS; ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << static_cast<float>(matD_ref[i * ldd + j]) << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
#endif // !NDEBUG

    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!\n";
}

int main()
{
    outer_product_test(256, 256, 1.0f, 1.0f);
    return 0;
}
