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

#include <iostream>
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

// Tile dimensions: 16x16 works on all supported architectures (gfx9 and gfx12).
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;
const int ROCWMMA_K = 16;

// Device warp size (Wave64 on gfx9, Wave32 on gfx12).
const uint32_t WAVE_SIZE = getWarpSize();

// Thread block layout.
// T_BLOCK_X must be a multiple of WAVE_SIZE.
// Each wave computes one ROCWMMA_M x ROCWMMA_N output tile.
// Workgroup computes (T_BLOCK_X / WAVE_SIZE) x T_BLOCK_Y tiles per batch slice.
const int T_BLOCK_X = 4 * WAVE_SIZE;
const int T_BLOCK_Y = 4;

// The following device kernel is a naive implementation of Batched GEMM:
//   D[b] = alpha * (A[b] x B[b]) + beta * C[b],  for b = 0 .. batchCount-1
//
// Each batch slice is an independent M x N x K GEMM.
// The batch index is mapped to blockIdx.z so that all B batches are launched
// in a single kernel call without any host-side loop.
//
// Memory layout assumptions:
//   A[b]: row-major,  stride = M * K  (shape: M x K)
//   B[b]: col-major,  stride = K * N  (shape: K x N)
//   C[b]: row-major,  stride = M * N  (shape: M x N)
//   D[b]: row-major,  stride = M * N  (shape: M x N)
//
// Note: This is a simplified implementation to demonstrate API usage in
// context of wave-level batched GEMM computation, and is not optimized.
__global__ void batched_hgemm_rocwmma_d(uint32_t         m,
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
                                        uint32_t         strideA,
                                        uint32_t         strideB,
                                        uint32_t         strideC,
                                        uint32_t         strideD,
                                        float32_t        alpha,
                                        float32_t        beta)
{
    // Each batch slice is selected by blockIdx.z
    auto batchIdx = blockIdx.z;

    auto const* batchA = a + batchIdx * strideA;
    auto const* batchB = b + batchIdx * strideB;
    auto const* batchC = c + batchIdx * strideC;
    auto*       batchD = d + batchIdx * strideD;

    // Create fragments
    auto fragA
        = rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, row_major>();
    auto fragB
        = rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, col_major>();
    auto fragC   = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t>();
    auto fragAcc = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();

    rocwmma::fill_fragment(fragAcc, 0.0f);

    // Map waves to a 2-D tile grid within the M x N output.
    // blockIdx.x / blockIdx.y cover the M and N dimensions respectively.
    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

    auto cRow = majorWarp * ROCWMMA_M;
    auto cCol = minorWarp * ROCWMMA_N;

    if(cRow < m && cCol < n)
    {
        // Accumulate A[b] x B[b] over K tiles
        for(uint32_t ki = 0; ki < k; ki += ROCWMMA_K)
        {
            rocwmma::load_matrix_sync(fragA, batchA + (cRow * lda + ki), lda);
            rocwmma::load_matrix_sync(fragB, batchB + (ki + cCol * ldb), ldb);
            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Load C tile and apply alpha/beta scaling: D = alpha * A*B + beta * C
        rocwmma::load_matrix_sync(fragC, batchC + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

        for(int i = 0; i < fragC.num_elements; ++i)
        {
            fragC.x[i] = static_cast<float16_t>(alpha * fragAcc.x[i]
                                                 + beta * static_cast<float32_t>(fragC.x[i]));
        }

        rocwmma::store_matrix_sync(
            batchD + (cRow * ldd + cCol), fragC, ldd, rocwmma::mem_row_major);
    }
}

__host__ void batched_gemm_test(uint32_t  m,
                                uint32_t  n,
                                uint32_t  k,
                                uint32_t  batchCount,
                                float32_t alpha,
                                float32_t beta)
{
    // Validate that dimensions are multiples of tile sizes and large enough for the grid
    if((m < (ROCWMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y) || k < ROCWMMA_K)
       || (m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K))
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    const uint32_t lda = k;
    const uint32_t ldb = k;
    const uint32_t ldc = n;
    const uint32_t ldd = n;

    const uint32_t strideA = m * k;
    const uint32_t strideB = k * n;
    const uint32_t strideC = m * n;
    const uint32_t strideD = m * n;

    std::cout << "Initializing host data..." << std::endl;

    std::vector<float16_t> matA(strideA * batchCount);
    std::vector<float16_t> matB(strideB * batchCount);
    std::vector<float16_t> matC(strideC * batchCount);
    std::vector<float16_t> matD(strideD * batchCount,
                                std::numeric_limits<float16_t>::signaling_NaN());

    // fill() from common.hpp initialises m*k elements per batch
    fill<float16_t>(matA.data(), m, k, batchCount);
    fill<float16_t>(matB.data(), k, n, batchCount);
    fill<float16_t>(matC.data(), m, n, batchCount);

    std::cout << "Initializing device data..." << std::endl;

    float16_t *d_a, *d_b, *d_c, *d_d;

    const size_t bytesA = matA.size() * sizeof(float16_t);
    const size_t bytesB = matB.size() * sizeof(float16_t);
    const size_t bytesC = matC.size() * sizeof(float16_t);
    const size_t bytesD = matD.size() * sizeof(float16_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matD.data(), bytesD, hipMemcpyHostToDevice));

    // Grid: x covers M-tiles, y covers N-tiles, z covers batches
    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceil_div(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                        rocwmma::ceil_div(n, ROCWMMA_N * T_BLOCK_Y),
                        batchCount);

    std::cout << "Launching Batched GEMM kernel..." << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    hipExtLaunchKernelGGL(batched_hgemm_rocwmma_d,
                          gridDim,
                          blockDim,
                          0, // sharedMemBytes
                          0, // stream
                          startEvent,
                          stopEvent,
                          0, // flags
                          m,
                          n,
                          k,
                          d_a,
                          d_b,
                          d_c,
                          d_d,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          strideA,
                          strideB,
                          strideC,
                          strideD,
                          alpha,
                          beta);

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Batched GEMM flops: batchCount * 2*m*n*k
    auto gFlops       = static_cast<double>(batchCount) * calculateGFlops(m, n, k);
    auto tFlopsPerSec = gFlops / static_cast<double>(elapsedTimeMs);

    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, BatchCount, "
              << "alpha, beta, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << ROCWMMA_M << ", " << ROCWMMA_N << ", " << ROCWMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << batchCount << ", " << alpha << ", " << beta << ", "
              << elapsedTimeMs << ", " << gFlops << ", " << tFlopsPerSec << std::endl;

#if !NDEBUG

    std::cout << "Validating result with reference..." << std::endl;

    CHECK_HIP_ERROR(hipMemcpy(matD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    std::vector<float16_t> matD_ref(strideD * batchCount,
                                    std::numeric_limits<float16_t>::signaling_NaN());

    for(uint32_t b = 0; b < batchCount; ++b)
    {
        gemm_cpu_h<float16_t, float16_t, float32_t, row_major, col_major, row_major>(
            m,
            n,
            k,
            matA.data() + b * strideA,
            matB.data() + b * strideB,
            matC.data() + b * strideC,
            matD_ref.data() + b * strideD,
            lda,
            ldb,
            ldc,
            ldd,
            alpha,
            beta);
    }

    auto res = compareEqual<float16_t>(
        matD.data(), matD_ref.data(), strideD * batchCount);

    if(std::get<0>(res) == false)
    {
        std::cout << "FAILED!\n";
    }
    else
    {
        std::cout << "PASSED!\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

#endif // !NDEBUG

    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    // M=N=K=256, 8 batches — representative of multi-head attention projections
    batched_gemm_test(256, 256, 256, 8, 1.0f, 0.0f);
    return 0;
}
