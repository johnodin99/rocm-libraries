# simple_fusion_gemm: Architecture Porting Notes (CDNA & RDNA)

## Bug Fix 1: WARP_SIZE Mismatch

### Problem
Kernel was hardcoded for gfx9 (CDNA, wave64):
```cpp
WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
```
gfx1201 (RDNA4) uses wave32, causing all geometry calculations to be wrong.

### Impact (WARP_SIZE=64 on wave32 hardware)

| Calculation | Expected (wave32) | Actual (wrong) | Effect |
|---|---|---|---|
| `WARPS_X = TBLOCK_X / WARP_SIZE` | `128/32 = 4` | `128/64 = 2` | Wrong warp count |
| `MACRO_TILE_X = WARPS_X * WARP_TILE_X` | `4*32 = 128` | `2*32 = 64` | Wrong tile geometry |
| `localWarpCoord.x = threadIdx.x / WARP_SIZE` | 0,1,2,3 | 0,0,1,1 | Warps overlap, LDS corrupted |
| `num_elements` per thread (16x16 acc) | `16*16/32 = 8` | `16*16/64 = 4` | Fragment data misinterpreted |

### Root Cause
`WARP_SIZE` is a **compile-time constant** baked into all kernel geometry.
Host-side `getWarpSize()` check validates hardware but doesn't fix kernel constants.

### Fix (Unified CDNA & RDNA Support)
To support both CDNA (wave64) and RDNA (wave32) architectures seamlessly, we introduced architecture-specific configuration namespaces (`gfx9Params` and `gfx11Params`). The device code uses preprocessor directives (`#if(ROCWMMA_ARCH_GFX9)`) to ensure the correct `WARP_SIZE` is statically compiled, while the host code uses `isGfx9()` to dynamically pick parameters at runtime.

```cpp
namespace gfx9Params {
    // ...
    WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
};

namespace gfx11Params {
    // ...
    WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32
};

#if(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
using namespace gfx11Params;
#endif
```

And on the host side:
```cpp
uint32_t hWARP_TILE_X = isGfx9() ? gfx9Params::TBLOCK_X : gfx11Params::TBLOCK_X;
// ...
```

---

## Bug Fix 2: Missing `synchronize_workgroup()` (Race Condition)

### Problem
Between Phase 1 tail (LDS reads) and Phase 2 (LDS writes), there was no sync barrier.

### Where in the code (around line 897-913)

```
Phase 1 tail:
  localReadA(fragsA, ldsPtrLo + ...)   // <-- reads from LDS
  localReadB(fragsB, ldsPtrLo + ...)   // <-- reads from LDS
  mfma(fragsAcc, fragsA, fragsB, ...)  // <-- register operation

  convertI32toI8(fragsAcc, fragsTmp)   // <-- register operation

  ldsPtr = reinterpret_cast<InputTV*>(localMemPtr)  // <-- reuse same LDS!

  *** NO SYNC HERE ***  <-- BUG: fast warps overwrite LDS while slow warps still reading

  localWriteAcc(fragsTmp, ldsPtr + ...) // <-- writes to LDS (same memory!)
  synchronize_workgroup()               // <-- too late, damage already done
```

### Why this matters on wave32 (gfx12) more than wave64 (gfx9)

```
                    wave64 (gfx9)          wave32 (gfx12)
  ---------------------------------------------------------------
  Total warps:      4                      8
  Warp scheduling:  Less contention        More contention
  Race likelihood:  Low (may appear OK)    High (corruption visible)
```

With 4 warps (wave64), all warps often reach `localWriteAcc` before any warp's
LDS data becomes stale -- the race exists but rarely triggers.

With 8 warps (wave32), the scheduling window is wider. Warp 0 can start overwriting
`localMemPtr` (Phase 2 S-matrix region) while Warp 7 is still executing `localReadA`
from `ldsPtrLo` which points to the **same physical LDS memory** (after buffer swaps,
`ldsPtrLo` may alias `localMemPtr`).

### Fix
```diff
+ // Ensure all warps finished Phase 1 LDS reads before Phase 2 overwrites
+ synchronize_workgroup();
+
  localWriteAcc(fragsTmp, ldsPtr + ldsReadOffsetAcc, ldsld_new);
  synchronize_workgroup();
-
- //load S like fragA
- synchronize_workgroup();   // this duplicate sync was redundant
```

### GPU Barrier Semantics
`synchronize_workgroup()` is equivalent to `__syncthreads()` in CUDA.
It guarantees:
1. All threads in the workgroup reach the barrier before any proceed
2. All prior LDS writes are visible to all threads
3. All prior LDS reads are complete before any new writes

Without this barrier, the GPU's out-of-order warp scheduler can interleave
Phase 1 reads and Phase 2 writes across different warps, causing data corruption.
