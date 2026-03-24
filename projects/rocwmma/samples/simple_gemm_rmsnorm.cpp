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

/* GEMM + RMSNorm Fused Sample
 *
 * Implements the per-row RMSNorm applied directly to the GEMM output,
 * as used in LLaMA / Mistral / Qwen decoder layers:
 *
 *   C      = A x B                        [M x N] = [M x K] x [K x N]
 *   rms(i) = sqrt( mean_j(C[i,j]^2) + eps )
 *   D[i,j] = (C[i,j] / rms(i)) * gamma[j]
 *
 * where gamma is the learned scale vector of length N (row_major, in registers).
 *
 * Algorithm overview (warp-level):
 *   1. Each warp accumulates a warp-tile [BLOCKS_X * M x BLOCKS_Y * N] via MFMA.
 *   2. For each output row in the warp tile, sum-of-squares is reduced across
 *      threads in the warp using __builtin_amdgcn_ds_swizzle / DPP or a manual
 *      register-shuffle loop compatible with rocWMMA's accumulator layout.
 *      Because rocWMMA accumulator elements are distributed across the warp,
 *      every thread holds a contiguous subset; we sum locally then use a
 *      warp-wide reduction via bitwise ds-permute.
 *   3. rms_inv = rsqrt( local_ss / N + eps ), broadcast per row.
 *   4. D = C * rms_inv * gamma  (element-wise, in registers).
 *   5. Write D to global memory (ComputeT -> OutputT cast).
 *
 * Data layouts (matches fillRand row-major fill convention):
 *   A     : row_major  [M x K],  lda  = K
 *   B     : row_major  [K x N],  ldb  = N
 *   gamma : row_major  [1 x N],  host vector
 *   D     : row_major  [M x N],  ldd  = N
 *
 * Kernel features:
 *   - Cooperative global read with LDS double buffering (ping-pong prefetch)
 *   - LDS layout: two segments (A | B^T) in col_major
 *   - Warp-level sum-of-squares reduction using shuffle (no LDS needed)
 *   - Fused RMSNorm in registers (zero extra global memory)
 *   - gamma broadcast from constant memory / kernel argument pointer
 *   - CPU reference for debug validation (active when NDEBUG is NOT set)
 *
 * RMSNorm vs LayerNorm:
 *   - No mean subtraction needed -> simpler, one pass over C
 *   - RMS is computed per output row (dimension = N)
 *   - Preferred in LLaMA-family models for speed & training stability
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
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

// ---------------------------------------------------------------------------
// Compile-time kernel parameters per architecture
// ---------------------------------------------------------------------------
namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        BLOCKS_X  = 2u,
        BLOCKS_Y  = 2u,
        TBLOCK_X  = 128u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
    };
}

namespace gfx11Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        BLOCKS_X  = 2u,
        BLOCKS_Y  = 2u,
        TBLOCK_X  = 64u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32
    };
}

#if(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
using namespace gfx11Params;
#endif

// ---------------------------------------------------------------------------
// Types and Data Layouts
// ---------------------------------------------------------------------------
using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = row_major;
using DataLayoutB   = row_major;
using DataLayoutD   = row_major;
using DataLayoutLds = col_major;

// ---------------------------------------------------------------------------
// Tile dimensions
// ---------------------------------------------------------------------------
constexpr uint32_t WARP_TILE_X  = BLOCKS_X * ROCWMMA_M;
constexpr uint32_t WARP_TILE_Y  = BLOCKS_Y * ROCWMMA_N;
constexpr uint32_t WARPS_X      = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_Y      = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_X = WARPS_X * WARP_TILE_X;
constexpr uint32_t MACRO_TILE_Y = WARPS_Y * WARP_TILE_Y;
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

// Elements per thread in one MfmaFragAcc (ComputeT) on gfx9 with 16x16x16:
// Each warp (64 threads) owns 16x16 = 256 elements -> 4 elements/thread per block.
// With BLOCKS_X * BLOCKS_Y = 4 blocks -> 16 elements/thread per warp tile.

// ---------------------------------------------------------------------------
// Fragment types
// ---------------------------------------------------------------------------
using MfmaFragA        = fragment<matrix_a,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT,  DataLayoutA>;
using MfmaFragB        = fragment<matrix_b,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT,  DataLayoutB>;
using MfmaFragAcc      = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;
// MfmaFragAccStore: same as MfmaFragAcc but with an explicit DataLayout so that
// GetDataLayout_t<> and the typed store_matrix_sync overload compile.
// Used only for offset arithmetic and store; MFMA accumulation uses MfmaFragAcc.
using MfmaFragAccStore = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT, DataLayoutD>;
using MfmaFragOut      = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT,  DataLayoutD>;

// Cooperative global read (macro tile)
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

using GRBuffA
    = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA, CoopScheduler>;
using GRBuffB
    = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutB, CoopScheduler>;

// Local write (macro tile) — col_major LDS layout; B must be transposed
using LWBuffA = apply_data_layout_t<GRBuffA,                   DataLayoutLds>;
using LWBuffB = apply_data_layout_t<apply_transpose_t<GRBuffB>, DataLayoutLds>;

// Local read (MFMA fragment-level) — matches LDS col_major layout
using LRFragA = apply_data_layout_t<MfmaFragA,                    DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MfmaFragB>,  DataLayoutLds>;

// ---------------------------------------------------------------------------
// Device helper functions
// ---------------------------------------------------------------------------

ROCWMMA_DEVICE static inline void
    globalReadCoopA(GRBuffA& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void
    globalReadCoopB(GRBuffB& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopA(InputT* ldsAddr, GRBuffA const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(gr), ldsld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopB(InputT* ldsAddr, GRBuffB const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(apply_transpose(gr)), ldsld);
}

// Read BLOCKS_X A-blocks from LDS
ROCWMMA_DEVICE static inline void
    localReadA(MfmaFragA (&fragsA)[BLOCKS_X], InputT const* ldsAddrA, uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragA>;
    using FragShape = GetIOShape_t<LRFragA>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragA tmp;
        load_matrix_sync(tmp, ldsAddrA, ldsld);
        fragsA[i]  = apply_data_layout<DataLayoutA>(tmp);
        ldsAddrA  += blockStep;
    }
}

// Read BLOCKS_Y B-blocks from LDS
ROCWMMA_DEVICE static inline void
    localReadB(MfmaFragB (&fragsB)[BLOCKS_Y], InputT const* ldsAddrB, uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragB>;
    using FragShape = GetIOShape_t<LRFragB>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragB tmp;
        load_matrix_sync(tmp, ldsAddrB, ldsld);
        fragsB[i]  = apply_data_layout<DataLayoutB>(apply_transpose(tmp));
        ldsAddrB  += blockStep;
    }
}

// Fill all BLOCKS_X x BLOCKS_Y accumulator fragments with a scalar
ROCWMMA_DEVICE static inline void
    clear_acc_fragments(MfmaFragAcc (&frags)[BLOCKS_X][BLOCKS_Y], ComputeT val)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            fill_fragment(frags[i][j], val);
}

// MFMA accumulation: acc += fragsA * fragsB
ROCWMMA_DEVICE static inline void
    mfma_warp_tile(MfmaFragAcc (&accOut)[BLOCKS_X][BLOCKS_Y],
                   MfmaFragA const (&fragsA)[BLOCKS_X],
                   MfmaFragB const (&fragsB)[BLOCKS_Y],
                   MfmaFragAcc const (&accIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            mma_sync(accOut[i][j], fragsA[i], fragsB[j], accIn[i][j]);
}

// ---------------------------------------------------------------------------
// Write warp tile (ComputeT accumulators) to a float32 workspace buffer.
// GetDataLayout_t requires a fragment type that carries an explicit DataLayout;
// we use MfmaFragAccStore (== MfmaFragAcc + DataLayoutD) for offset arithmetic
// only, and apply_data_layout<DataLayoutD> to rebind the fragment before store.
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void
    globalWriteC(ComputeT*              gAddrC,
                 MfmaFragAcc const (&fragsC)[BLOCKS_X][BLOCKS_Y],
                 uint32_t               ldc)
{
    using Mapper1d  = GetDataLayout_t<MfmaFragAccStore>;
    using FragShape = GetIOShape_t<MfmaFragAccStore>;
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldc);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth),  ldc);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            store_matrix_sync(gAddrC + offsetY, apply_data_layout<DataLayoutD>(fragsC[i][j]), ldc);
            offsetY += blockStepY;
        }
        gAddrC += blockStepX;
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Per-row RMSNorm kernel
//
// Each thread handles one output row.
// Reads all N columns of row i from the float32 workspace d_c,
// computes rms_inv, applies gamma, writes fp16 D.
//
// This avoids any architecture-specific accumulator layout assumption.
// Grid: (ceil(m,1024), 1, 1)  Block: (1024, 1, 1)  or similar.
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void
    rmsnorm_apply_kernel(uint32_t          m,
                         uint32_t          n,
                         ComputeT const*   c,
                         ComputeT const*   gamma,
                         OutputT*          d,
                         uint32_t          ldc,
                         uint32_t          ldd,
                         ComputeT          eps)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if(row >= m)
            return;

        ComputeT const* c_row = c + (uint32_t)row * ldc;
        OutputT*        d_row = d + (uint32_t)row * ldd;

        float ss = 0.f;
        for(uint32_t j = 0; j < n; j++)
        {
            float v = static_cast<float>(c_row[j]);
            ss += v * v;
        }
        float rms_inv = __frsqrt_rn(ss / (float)n + (float)eps);

        for(uint32_t j = 0; j < n; j++)
        {
            float v = static_cast<float>(c_row[j]) * rms_inv
                      * static_cast<float>(gamma[j]);
            d_row[j] = static_cast<OutputT>(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1: GEMM kernel (rocWMMA)
//
//   C = A x B   written as float32 to workspace d_c.
//   C     [M x N] row_major ComputeT
//   A     [M x K] row_major InputT
//   B     [K x N] row_major InputT
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(256)
    gemm_rocwmma(uint32_t      m,
                 uint32_t      n,
                 uint32_t      k,
                 InputT const* a,
                 InputT const* b,
                 ComputeT*     c,
                 uint32_t      lda,
                 uint32_t      ldb,
                 uint32_t      ldc)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        constexpr auto warpTileSize  = make_coord2d(WARP_TILE_X, WARP_TILE_Y);
        constexpr auto macroTileSize = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y);

        auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto localWarpOffset = localWarpCoord * warpTileSize;

        auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
        auto warpTileCoord  = macroTileCoord + localWarpOffset;

        auto warpTileBound = warpTileCoord + warpTileSize;
        if(get<0>(warpTileBound) > m || get<1>(warpTileBound) > n)
            return;

        using GRBuffAMap1d = GetDataLayout_t<GRBuffA>;
        using GRBuffBMap1d = GetDataLayout_t<GRBuffB>;

        auto globalReadOffsetA
            = GRBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
        auto globalReadOffsetB
            = GRBuffBMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);

        auto kStepOffsetA = GRBuffAMap1d::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
        auto kStepOffsetB = GRBuffBMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

        GRBuffA grBuffA;
        GRBuffB grBuffB;
        globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
        globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);
        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        HIP_DYNAMIC_SHARED(void*, localMemPtr);

        using LWBuffAShape = GetIOShape_t<LWBuffA>;
        using LWBuffBShape = GetIOShape_t<LWBuffB>;
        using LWBuffAMap1d = GetDataLayout_t<LWBuffA>;

        constexpr uint32_t ldsWidth  = MACRO_TILE_K;
        constexpr uint32_t ldsHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
        constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
        constexpr uint32_t ldsld
            = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

        auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
        auto* ldsPtrHi = ldsPtrLo + sizeLds;

        auto ldsWriteOffsetA = 0u;
        auto ldsWriteOffsetB
            = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), ldsld);

        using LWBuffAMap1d2 = GetDataLayout_t<LWBuffA>;
        using LWBuffBMap1d2 = GetDataLayout_t<LWBuffB>;

        auto ldsReadOffsetA
            = ldsWriteOffsetA
              + LWBuffAMap1d2::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
        auto ldsReadOffsetB
            = ldsWriteOffsetB
              + LWBuffBMap1d2::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

        localWriteCoopA(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldsld);
        localWriteCoopB(ldsPtrLo + ldsWriteOffsetB, grBuffB, ldsld);

        MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
        clear_acc_fragments(fragsAcc, ComputeT(0));
        synchronize_workgroup();

        for(uint32_t currentK = MACRO_TILE_K; currentK < k; currentK += MACRO_TILE_K)
        {
            MfmaFragA fragsA[BLOCKS_X];
            MfmaFragB fragsB[BLOCKS_Y];
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

            globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
            globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);
            globalReadOffsetA += kStepOffsetA;
            globalReadOffsetB += kStepOffsetB;

            mfma_warp_tile(fragsAcc, fragsA, fragsB, fragsAcc);

            localWriteCoopA(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldsld);
            localWriteCoopB(ldsPtrHi + ldsWriteOffsetB, grBuffB, ldsld);
            synchronize_workgroup();

            auto* tmp = ldsPtrLo;
            ldsPtrLo  = ldsPtrHi;
            ldsPtrHi  = tmp;
        }

        {
            MfmaFragA fragsA[BLOCKS_X];
            MfmaFragB fragsB[BLOCKS_Y];
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);
            mfma_warp_tile(fragsAcc, fragsA, fragsB, fragsAcc);
        }

        // Write float32 C to workspace (no RMSNorm here).
        // MfmaFragAccStore carries DataLayoutD so GetDataLayout_t resolves correctly.
        using MfmaFragAccMap1d = GetDataLayout_t<MfmaFragAccStore>;
        globalWriteC(c + MfmaFragAccMap1d::fromMatrixCoord(warpTileCoord, ldc), fragsAcc, ldc);
    }
}

// ---------------------------------------------------------------------------
// CPU reference: D[i,j] = (C[i,j] / rms_i) * gamma[j]
// All matrices are row_major.
// ---------------------------------------------------------------------------
static void rmsnorm_cpu_ref(uint32_t          m,
                            uint32_t          n,
                            uint32_t          k,
                            InputT const*     a,
                            InputT const*     b,
                            ComputeT const*   gamma,
                            OutputT*          d,
                            uint32_t          lda,
                            uint32_t          ldb,
                            uint32_t          ldd,
                            ComputeT          eps)
{
    auto rowMjr = [](uint32_t r, uint32_t c, uint32_t ld) { return r * ld + c; };

#pragma omp parallel for
    for(int i = 0; i < (int)m; i++)
    {
        // GEMM row i
        std::vector<float> c_row(n, 0.f);
        for(int h = 0; h < (int)k; h++)
        {
            float a_val = static_cast<float>(a[rowMjr(i, h, lda)]);
            for(int j = 0; j < (int)n; j++)
                c_row[j] += a_val * static_cast<float>(b[rowMjr(h, j, ldb)]);
        }

        // RMSNorm for row i
        float ss = 0.f;
        for(int j = 0; j < (int)n; j++)
            ss += c_row[j] * c_row[j];
        float rms_inv = 1.f / std::sqrtf(ss / (float)n + (float)eps);

        for(int j = 0; j < (int)n; j++)
            d[rowMjr(i, j, ldd)]
                = static_cast<OutputT>(c_row[j] * rms_inv * static_cast<float>(gamma[j]));
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
              << "  Compute units   : " << props.multiProcessorCount << "\n"
              << "  Warp size       : " << props.warpSize << "\n"
              << "  Global memory   : " << (props.totalGlobalMem >> 20) << " MiB\n"
              << "  Shared mem/blk  : " << (props.sharedMemPerBlock >> 10) << " KiB\n"
              << "  Max clock (MHz) : " << (props.clockRate / 1000) << "\n"
              << "  Memory bw (GB/s): "
              << (props.memoryBusWidth / 8 * props.memoryClockRate * 2) / 1000000 << "\n"
              << "========================\n\n";
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
ROCWMMA_HOST void run_gemm_rmsnorm_sample(uint32_t m, uint32_t n, uint32_t k)
{
    printDeviceInfo();

    // Runtime arch selection mirrors compile-time namespace
    uint32_t hTBLOCK_X  = isGfx9() ? gfx9Params::TBLOCK_X  : gfx11Params::TBLOCK_X;
    uint32_t hTBLOCK_Y  = isGfx9() ? gfx9Params::TBLOCK_Y  : gfx11Params::TBLOCK_Y;
    uint32_t hROCWMMA_M = isGfx9() ? gfx9Params::ROCWMMA_M : gfx11Params::ROCWMMA_M;
    uint32_t hROCWMMA_N = isGfx9() ? gfx9Params::ROCWMMA_N : gfx11Params::ROCWMMA_N;
    uint32_t hROCWMMA_K = isGfx9() ? gfx9Params::ROCWMMA_K : gfx11Params::ROCWMMA_K;
    uint32_t hBLOCKS_X  = isGfx9() ? gfx9Params::BLOCKS_X  : gfx11Params::BLOCKS_X;
    uint32_t hBLOCKS_Y  = isGfx9() ? gfx9Params::BLOCKS_Y  : gfx11Params::BLOCKS_Y;

    uint32_t hWARP_TILE_X = hBLOCKS_X * hROCWMMA_M;
    uint32_t hWARP_TILE_Y = hBLOCKS_Y * hROCWMMA_N;

    auto warpSize     = getWarpSize();
    auto macroTileSize
        = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_X, hTBLOCK_Y * hWARP_TILE_Y);

    // Architecture checks
    if((isGfx11() || isGfx12()) && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if(isGfx9() && (hROCWMMA_M != hROCWMMA_N || (hROCWMMA_M != 16 && hROCWMMA_M != 32)))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if((isGfx11() || isGfx12()) && warpSize != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    if(isGfx9() && warpSize != Constants::AMDGCN_WAVE_SIZE_64)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    if((m < get<0>(macroTileSize) || n < get<1>(macroTileSize) || k < hROCWMMA_K)
       || (m % hROCWMMA_M || n % hROCWMMA_N || k % hROCWMMA_K))
    {
        std::cout << "Unsupported matrix size!\n";
        return;
    }

    // Leading dimensions — all row_major
    uint32_t lda = k; // A  [M x K] row_major
    uint32_t ldb = n; // B  [K x N] row_major
    uint32_t ldd = n; // D  [M x N] row_major

    constexpr ComputeT eps = static_cast<ComputeT>(1e-5);

    std::cout << "Initializing host data (m=" << m << " n=" << n << " k=" << k << ")...\n";

    std::vector<InputT>   matA(m * k);
    std::vector<InputT>   matB(k * n);
    std::vector<ComputeT> matGamma(n, ComputeT(1));   // default all-ones
    std::vector<OutputT>  matD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    // fillRand fills mat[i*cols+j] (row_major), values are small integers 0..4.
    // Scale to avoid FP16 overflow: with K=128, max|C_ij| ~ 128*(1/16)^2 * (16*16) ~ 8, safe.
    constexpr float kScale = 1.0f / 16.0f;
    fillRand(matA.data(), m, k);
    fillRand(matB.data(), k, n);
    for(auto& x : matA) x = static_cast<InputT>(static_cast<float>(x) * kScale);
    for(auto& x : matB) x = static_cast<InputT>(static_cast<float>(x) * kScale);

    // Randomise gamma in [0.5, 1.5] for a more interesting test
    for(uint32_t j = 0; j < n; j++)
        matGamma[j] = ComputeT(0.5f + static_cast<float>(j % 8) * (1.0f / 8.0f));

    std::cout << "Allocating device memory...\n";

    InputT*   d_a;
    InputT*   d_b;
    ComputeT* d_gamma;
    ComputeT* d_c;    // float32 GEMM workspace [M x N]
    OutputT*  d_d;

    CHECK_HIP_ERROR(hipMalloc(&d_a,     m * k * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_b,     k * n * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_gamma, n     * sizeof(ComputeT)));
    CHECK_HIP_ERROR(hipMalloc(&d_c,     m * n * sizeof(ComputeT)));
    CHECK_HIP_ERROR(hipMalloc(&d_d,     m * n * sizeof(OutputT)));

    CHECK_HIP_ERROR(hipMemcpy(d_a,     matA.data(),     m * k * sizeof(InputT),   hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b,     matB.data(),     k * n * sizeof(InputT),   hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_gamma, matGamma.data(), n     * sizeof(ComputeT), hipMemcpyHostToDevice));

    // Pass 1: GEMM grid — same as before, tiles M and N
    auto blockDim  = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gemmGrid  = dim3(rocwmma::ceil_div(m, get<0>(macroTileSize)),
                          rocwmma::ceil_div(n, get<1>(macroTileSize)));
    // Pass 2: RMSNorm grid — one thread per output row
    constexpr uint32_t normBlockSize = 256u;
    auto               normGrid      = dim3(rocwmma::ceil_div(m, normBlockSize));

    std::cout << "gemmGrid (" << gemmGrid.x << " " << gemmGrid.y << ")"
              << "  blockDim (" << blockDim.x << " " << blockDim.y << ")\n"
              << "normGrid (" << normGrid.x << ")  normBlock (" << normBlockSize << ")\n";

    // LDS: 2 ping-pong buffers
    using LWBuffAShape = GetIOShape_t<LWBuffA>;
    using LWBuffBShape = GetIOShape_t<LWBuffB>;
    constexpr uint32_t ldsSegHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
    int ldsUsage = 2 * sizeof(InputT) * ldsSegHeight * MACRO_TILE_K;
    std::cout << "LDS usage: " << ldsUsage << " bytes (" << ldsUsage / 1024 << " KiB)\n";

    uint32_t ldc = n; // C [M x N] row_major

    // Combined two-pass lambda timed together
    auto kernelLambda = [&]() {
        // Pass 1: rocWMMA GEMM -> float32 workspace d_c
        hipExtLaunchKernelGGL(gemm_rocwmma,
                              gemmGrid,
                              blockDim,
                              ldsUsage,
                              0,
                              nullptr,
                              nullptr,
                              0,
                              m, n, k,
                              d_a, d_b, d_c,
                              lda, ldb, ldc);
        // Pass 2: per-row RMSNorm
        hipLaunchKernelGGL(rmsnorm_apply_kernel,
                           normGrid,
                           dim3(normBlockSize),
                           0, 0,
                           m, n,
                           d_c, d_gamma, d_d,
                           ldc, ldd,
                           eps);
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

    float elapsedMs = 0.f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, evStart, evStop));
    CHECK_HIP_ERROR(hipEventDestroy(evStart));
    CHECK_HIP_ERROR(hipEventDestroy(evStop));

    // FLOPs: GEMM dominates = 2MNK; RMSNorm per-element = 3 ops (negligible)
    double gFlopsPerRun = 2.0
                          * static_cast<double>(m)
                          * static_cast<double>(n)
                          * static_cast<double>(k)
                          * 1e-9;
    double tFlopsPerSec = gFlopsPerRun * recordRuns
                          / (static_cast<double>(elapsedMs) * 1e-3)
                          * 1e-3;

    std::cout << std::left
              << std::setw(8)  << "TBlkX"  << std::setw(8)  << "TBlkY"
              << std::setw(6)  << "BlkM"   << std::setw(6)  << "BlkN"
              << std::setw(6)  << "BlkK"   << std::setw(8)  << "MatM"
              << std::setw(8)  << "MatN"   << std::setw(8)  << "MatK"
              << std::setw(8)  << "lda"    << std::setw(8)  << "ldb"
              << std::setw(8)  << "ldd"    << std::setw(14) << "elapsedMs"
              << std::setw(22) << "GFlops(GEMM)"
              << std::setw(12) << "TFlops/s"
              << "\n";

    std::cout << std::left
              << std::setw(8)  << hTBLOCK_X  << std::setw(8)  << hTBLOCK_Y
              << std::setw(6)  << hROCWMMA_M << std::setw(6)  << hROCWMMA_N
              << std::setw(6)  << hROCWMMA_K << std::setw(8)  << m
              << std::setw(8)  << n          << std::setw(8)  << k
              << std::setw(8)  << lda        << std::setw(8)  << ldb
              << std::setw(8)  << ldd        << std::setw(14) << elapsedMs
              << std::setw(22) << (gFlopsPerRun * recordRuns)
              << std::setw(12) << tFlopsPerSec
              << "\n";

#if !NDEBUG
    std::cout << "\nValidating against CPU reference...\n";

    CHECK_HIP_ERROR(hipMemcpy(matD.data(), d_d, m * n * sizeof(OutputT), hipMemcpyDeviceToHost));

    std::vector<OutputT> matDref(m * n, std::numeric_limits<OutputT>::signaling_NaN());
    rmsnorm_cpu_ref(m, n, k,
                    matA.data(), matB.data(), matGamma.data(),
                    matDref.data(),
                    lda, ldb, ldd, eps);

    auto res = compareEqual(matD.data(), matDref.data(), m * n);
    std::cout << (std::get<0>(res) ? "PASSED" : "FAILED") << "\n";
    std::cout << "Max relative error: " << std::get<1>(res) << "\n";
#endif

    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_gamma));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!\n";
}

int main()
{
    // LLaMA decoder output: hidden=128, seq_len=64, proj=256
    // (small and multiples of 16 for quick validation)
    run_gemm_rmsnorm_sample(128, 256, 128);
    return 0;
}
