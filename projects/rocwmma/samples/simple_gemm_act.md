# simple_gemm_act: Architecture Porting Notes (CDNA & RDNA)

## Bug Fix: WARP_SIZE Mismatch

### Problem
Kernel was hardcoded for gfx9 (CDNA, wave64):
```cpp
WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
```
gfx1201 (RDNA4) uses wave32, causing all geometry calculations to be wrong.

### Symptom
```
FAILED
Max relative error: 0.992188
```
`0.992188 = 127/128` -- GPU outputs 0 where reference expects 127 (or vice versa),
indicating the GEMM accumulation is completely wrong (not just precision drift).

### Impact (WARP_SIZE=64 on wave32 hardware)

| Calculation | Expected (wave32) | Actual (wrong) | Effect |
|---|---|---|---|
| `WARPS_X = TBLOCK_X / WARP_SIZE` | `128/32 = 4` | `128/64 = 2` | Only 2 warps recognized, 4 warps exist |
| `MACRO_TILE_X = WARPS_X * WARP_TILE_X` | `4*32 = 128` | `2*32 = 64` | Half the macro tile |
| `localWarpCoord.x = threadIdx.x / WARP_SIZE` | 0,1,2,3 (unique) | 0,0,1,1 (overlap) | Warps 2,3 compute same coord as 0,1 |
| `num_elements` per thread (16x16 acc) | `16*16/32 = 8` | `16*16/64 = 4` | Fragment data misinterpreted |

### Why warps overlap causes wrong results

```
wave32 actual warps:     wave64 assumed mapping:
  Warp 0: threads 0-31     threadIdx.x / 64 = 0  --> warp coord 0
  Warp 1: threads 32-63    threadIdx.x / 64 = 0  --> warp coord 0  (COLLISION!)
  Warp 2: threads 64-95    threadIdx.x / 64 = 1  --> warp coord 1
  Warp 3: threads 96-127   threadIdx.x / 64 = 1  --> warp coord 1  (COLLISION!)
```

Warps 0 and 1 both think they are warp (0, y), so they:
- Read the SAME A/B input tiles from LDS
- Write their GEMM results to the SAME D output location
- One warp's output overwrites the other's --> data corruption

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

### Result after fix
```
PASSED
Max relative error: 0
```

### Note
The host-side `getWarpSize()` check (line 789-793) validates that hardware warp size
matches expectations, but it does NOT fix the **compile-time constant** `WARP_SIZE`
used in all kernel geometry calculations. Both must agree.
