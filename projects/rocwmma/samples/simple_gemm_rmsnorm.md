# simple_gemm_rmsnorm — GEMM + RMSNorm Fused Sample

## Overview

This sample demonstrates fusing a GEMM operation with RMSNorm directly in GPU
registers using rocWMMA.  The pattern appears in every decoder layer of
LLaMA / Mistral / Qwen-family models.

```
C      = A x B                        [M x N] = [M x K] x [K x N]
rms_i  = sqrt( (1/N) * sum_j C[i,j]^2  + eps )
D[i,j] = C[i,j] / rms_i * gamma[j]
```

**No intermediate global write of C is required** — the normalisation happens
inside registers immediately after the K-loop accumulation.

---

## Algorithm Details

### GEMM Phase

| Feature | Detail |
|---|---|
| Fragment API | rocWMMA `mma_sync` (MFMA) |
| LDS layout | col\_major, two segments: A \| B^T |
| Prefetch | Ping-pong double buffering |
| Tile sizes (gfx9) | MACRO\_TILE\_X=64, MACRO\_TILE\_Y=64, K=16 |
| Tile sizes (gfx11) | MACRO\_TILE\_X=64, MACRO\_TILE\_Y=64, K=16 |

### RMSNorm Phase (warp-level, in-register)

1. **Local reduction** — each thread sums `c^2` over all accumulator elements
   it owns across all BLOCKS\_Y column blocks (row-block loop).
2. **Warp butterfly reduction** — `__shfl_xor` halving loop reduces partial
   sums across all lanes; final value = `sum_j C[row,j]^2` for the rows
   covered by this warp tile row-group.
3. **Normalisation** — `rms_inv = __frsqrt_rn(ss/N + eps)` (hardware fast
   reciprocal square root).
4. **Scale** — each element multiplied by `rms_inv * gamma[col]`.
5. **Store** — `ComputeT (float32) -> OutputT (float16)` cast at store time.

### RMSNorm vs LayerNorm

| Property | LayerNorm | RMSNorm |
|---|---|---|
| Mean subtraction | Yes | No |
| Variance | `E[x^2] - E[x]^2` | `E[x^2]` |
| Parameters | scale + bias | scale only |
| Passes over C | 2 | 1 |
| LLM usage | BERT, GPT-2 | LLaMA, Mistral, Qwen |

---

## Data Layouts

| Matrix | Shape | Layout | Leading dim |
|---|---|---|---|
| A | M x K | row\_major | K |
| B | K x N | row\_major | N |
| gamma | 1 x N | row\_major | — |
| D (output) | M x N | row\_major | N |

---

## File Structure

```
simple_gemm_rmsnorm.cpp   - kernel + host driver
simple_gemm_rmsnorm.md    - this document
```

---

## Building

```bash
# From the rocwmma build directory
cmake --build . --target simple_gemm_rmsnorm
```

Or together with all samples:

```bash
cmake --build . --target rocwmma_samples
```

---

## Running

```bash
./samples/simple_gemm_rmsnorm
```

Default problem size: `M=128, N=256, K=128` (LLaMA-style, multiples of tile).

Debug validation (CPU reference enabled when built without `-DNDEBUG`):

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --target simple_gemm_rmsnorm
./samples/simple_gemm_rmsnorm
# prints PASSED / FAILED and max relative error
```

---

## Expected Output (gfx942 example)

```
=== GPU Hardware Info ===
  Device name     : gfx942
  ...

Initializing host data (m=128 n=256 k=128)...
Allocating device memory...
gridDim  (2 4)  blockDim (128 2)
LDS usage: 12288 bytes (12 KiB)
Warming up...
Benchmarking...
TBlkX   TBlkY   BlkM  BlkN  BlkK  MatM    MatN    MatK    lda     ldb     ldd     elapsedMs     GFlops(GEMM)          TFlops/s
128     2       16    16    16    128     256     128     128     256     256     0.012         4.2950272             ...

Validating against CPU reference...
PASSED
Max relative error: 2.5e-03
Finished!
```

---

## Key Kernel Parameters

```cpp
// gfx9 (gfx908, gfx90a, gfx942 ...)
ROCWMMA_M = 16, ROCWMMA_N = 16, ROCWMMA_K = 16
BLOCKS_X  = 2,  BLOCKS_Y  = 2
TBLOCK_X  = 128, TBLOCK_Y = 2   // 4 warps x 2 rows = 8 warps/block

// gfx11 (gfx1100 ...)
TBLOCK_X  = 64,  TBLOCK_Y = 2   // 2 warps x 2 rows = 4 warps/block
```

LDS usage per block = `2 buffers * (MACRO_TILE_X + MACRO_TILE_Y) * MACRO_TILE_K * sizeof(InputT)`
= `2 * 128 * 16 * 2 = 8192 bytes` (A only) + B segment = ~12 KiB total.

---

## Validation Notes

The CPU reference computes the exact double-precision GEMM + RMSNorm.
GPU result uses FP16 accumulation via MFMA (FP32 accumulator) so maximum
relative error is typically `< 1%` for the default small-integer inputs.

---

## Application Context

```
Decoder layer (LLaMA-style)
  input_hidden  [seq, hidden]
       |
    Linear W_proj   <-- this GEMM
       |
    RMSNorm         <-- fused here
       |
    (Attention / FFN next layer)
```

The fusion saves one global memory round-trip for the C matrix
(`M*N*sizeof(fp16)` bytes read + written) and one kernel launch overhead.


---
Nice — this confirms the **two-pass version is correct on RX 9070 / gfx1201**. In your current code, **Pass 1** is `gemm_rocwmma()` writing `float32` workspace `d_c`, and **Pass 2** is `rmsnorm_apply_kernel()` reading the full row from `d_c`, applying RMSNorm with `gamma`, then writing `fp16` output `d_d`. The host launches them back-to-back inside `kernelLambda()`, and validation compares against a CPU reference.   

Below is a repo-ready **flow chart** and **README.md** you can paste directly.

---

## Flow chart

```mermaid
flowchart TD
    A[Start] --> B[Print GPU device info]
    B --> C[Select arch params<br/>gfx9 -> wave64<br/>gfx11/gfx12 -> wave32]
    C --> D[Check matrix size / tile compatibility]
    D --> E[Allocate host tensors<br/>A B gamma D]
    E --> F[Initialize random A/B<br/>scale inputs<br/>initialize gamma]
    F --> G[Allocate device buffers<br/>d_a d_b d_gamma d_c d_d]
    G --> H[Copy A B gamma to device]

    H --> I[Launch Pass 1: gemm_rocwmma]
    I --> J[Block computes macro tile]
    J --> K[Cooperative global read A/B]
    K --> L[Store A and transposed B into LDS]
    L --> M[Double-buffered K-loop]
    M --> N[localReadA/localReadB from LDS]
    N --> O[mma_sync accumulate into MfmaFragAcc]
    O --> P[Write float32 C tile to workspace d_c]

    P --> Q[Launch Pass 2: rmsnorm_apply_kernel]
    Q --> R[One thread handles one output row]
    R --> S[Load full row C[i,:] from d_c]
    S --> T[Compute sum of squares]
    T --> U[Compute rms_inv = rsqrt(mean + eps)]
    U --> V[Apply D[i,j] = C[i,j] * rms_inv * gamma[j]]
    V --> W[Write fp16 output to d_d]

    W --> X[Benchmark timing]
    X --> Y[Copy d_d back to host]
    Y --> Z[CPU reference compare]
    Z --> AA[Print PASS / FAIL]
    AA --> AB[Free device memory]
    AB --> AC[End]
```

---

## `README.md`

````md
# rocWMMA GEMM + RMSNorm (Two-Pass) Sample

This sample implements a **two-pass GEMM + RMSNorm pipeline** using **HIP + rocWMMA**.

It targets AMD GPUs and is validated on:

- **AMD Radeon RX 9070**
- **gfx1201**
- **wave32**

## Overview

The computation matches the common decoder-layer style RMSNorm pattern used in LLaMA-family models:

\[
C = A \times B
\]

\[
rms(i) = \sqrt{\frac{1}{N}\sum_j C[i,j]^2 + \epsilon}
\]

\[
D[i,j] = C[i,j] \times \frac{1}{rms(i)} \times \gamma[j]
\]

Where:

- `A` is `[M x K]`
- `B` is `[K x N]`
- `C` is the intermediate GEMM result `[M x N]`
- `gamma` is the RMSNorm scale vector of length `N`
- `D` is the final output `[M x N]`

---

## Why this version uses two passes

An earlier fused approach attempted to apply RMSNorm directly inside the rocWMMA accumulator path.

That approach is fragile on newer architectures because RMSNorm needs the **full row across all `N` columns**, while each warp-level tile only owns a **partial tile** of the output. In addition, architecture-specific accumulator lane mapping can make in-register row-wise normalization error-prone.

This implementation avoids those issues by splitting the work into:

1. **Pass 1: GEMM**
   - rocWMMA computes `C = A x B`
   - output is stored in a **float32 workspace** `d_c`

2. **Pass 2: RMSNorm**
   - one thread processes one full row
   - reads the complete row from `d_c`
   - computes RMS over the full normalization dimension
   - applies `gamma`
   - writes final `fp16` output `d_d`

This is the current correctness-first design used by the sample. The code clearly separates `gemm_rocwmma()` and `rmsnorm_apply_kernel()`, and launches them sequentially from the host driver. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## Data types and layouts

### Data types

- `InputT   = float16_t`
- `ComputeT = float32_t`
- `OutputT  = float16_t`

### Layouts

All matrices are stored in **row-major** form:

- `A`: `[M x K]`, `lda = K`
- `B`: `[K x N]`, `ldb = N`
- `C`: `[M x N]`, `ldc = N`
- `D`: `[M x N]`, `ldd = N`

The GEMM workspace `d_c` is kept in **float32** to preserve numerical stability before RMSNorm is applied. :contentReference[oaicite:5]{index=5}

---

## Kernel structure

## Pass 1: `gemm_rocwmma`

This kernel performs tiled GEMM using rocWMMA.

### Key features

- rocWMMA fragment-based matrix multiply
- cooperative global load
- LDS staging
- double-buffered K-loop
- macro-tile / warp-tile decomposition
- float32 accumulator output written to workspace

### Internal flow

1. Compute warp tile and macro tile coordinates
2. Cooperative read of A and B from global memory
3. Store A and transposed B into LDS
4. Loop over K dimension with ping-pong LDS buffers
5. Load matrix fragments from LDS
6. Run `mma_sync` into `MfmaFragAcc`
7. Store accumulated float32 tile to `d_c`

The helper `globalWriteC()` is used to write the accumulator fragments into the global float32 workspace. :contentReference[oaicite:6]{index=6}

---

## Pass 2: `rmsnorm_apply_kernel`

This kernel applies RMSNorm row-by-row.

### Mapping

- **one thread = one output row**

### Steps

For each row `i`:

1. read full row `C[i, :]` from workspace
2. accumulate sum of squares
3. compute:

   \[
   rms\_inv = \frac{1}{\sqrt{\frac{1}{N}\sum_j C[i,j]^2 + \epsilon}}
   \]

4. apply scale:

   \[
   D[i,j] = C[i,j] \times rms\_inv \times \gamma[j]
   \]

5. write output to `d_d`

This design avoids any dependence on architecture-specific WMMA accumulator element mapping. :contentReference[oaicite:7]{index=7}

---

## Architecture-dependent parameters

The sample uses different compile-time parameters depending on architecture family.

### gfx9-style path

- `ROCWMMA_M = 16`
- `ROCWMMA_N = 16`
- `ROCWMMA_K = 16`
- `BLOCKS_X  = 2`
- `BLOCKS_Y  = 2`
- `TBLOCK_X  = 128`
- `TBLOCK_Y  = 2`
- `WARP_SIZE = 64`

### gfx11/gfx12-style path

- `ROCWMMA_M = 16`
- `ROCWMMA_N = 16`
- `ROCWMMA_K = 16`
- `BLOCKS_X  = 2`
- `BLOCKS_Y  = 2`
- `TBLOCK_X  = 64`
- `TBLOCK_Y  = 2`
- `WARP_SIZE = 32`

At runtime, the sample checks:

- supported block sizes
- supported wave size
- matrix dimension divisibility
- macro tile compatibility

These checks are performed before kernel launch in `run_gemm_rmsnorm_sample()`. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

---

## Host-side execution flow

The host driver does the following:

1. print device info
2. choose runtime parameters by detected architecture
3. allocate and initialize host tensors
4. allocate device buffers
5. copy `A`, `B`, and `gamma` to device
6. launch GEMM kernel
7. launch RMSNorm kernel
8. benchmark the combined two-pass execution
9. copy output back
10. compare with CPU reference
11. free device memory

The sample uses:

- warmup runs
- timed benchmark runs
- CPU reference validation in non-`NDEBUG` builds

:contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

---

## Validation

The CPU reference computes:

1. GEMM on CPU into a temporary row buffer
2. per-row RMSNorm
3. gamma scaling
4. output comparison against GPU result

This is implemented in `rmsnorm_cpu_ref()`. :contentReference[oaicite:12]{index=12}

---

## Current verified result

For the configuration:

- `M = 128`
- `N = 256`
- `K = 128`

the updated two-pass version reports:

- `PASSED`
- `Max relative error: 0`

This indicates the correctness issue from the earlier fused attempt has been resolved.

---

## Example launch configuration

From the current passing run:

- `gemmGrid = (2, 4)`
- `blockDim = (64, 2)`
- `normGrid = (1)`
- `normBlock = (256)`
- `LDS usage = 8192 bytes`

This matches the wave32 path used on RX 9070 / gfx1201.

---

## Build notes

This sample depends on:

- HIP
- rocWMMA
- a working ROCm toolchain
- `common.hpp` from the sample environment

Typical requirements:

- compile for the correct target architecture
- ensure rocWMMA headers are visible
- link against HIP/ROCm libraries as needed by your environment

If you are integrating this into another project, make sure:

- matrix sizes are multiples of `ROCWMMA_M`, `ROCWMMA_N`, and `ROCWMMA_K`
- row-major leading dimensions are consistent
- output workspace `d_c` is allocated as `float32`

---

## Important implementation note

Although the file header may still describe the sample as "fused", the **current implementation is logically two-pass**:

- GEMM is done first
- RMSNorm is applied in a second kernel

So this version should be described as:

> **Correctness-first two-pass GEMM + RMSNorm sample using rocWMMA**

That wording is more accurate than calling it a fully fused implementation.

---

## Future work

Possible next steps:

- optimize the RMSNorm kernel
- vectorize row reads/writes
- fuse more epilogue work safely
- explore block-level row reduction for a future fused design
- benchmark larger problem sizes
- compare RX 9070 vs MI-series behavior

---

## File structure suggestion

```text
.
├── gemm_rmsnorm.cpp
├── common.hpp
└── README.md
````

---

## Summary

This sample demonstrates a practical and correct way to combine:

* rocWMMA GEMM
* float32 accumulation workspace
* row-wise RMSNorm
* fp16 output

The main design choice is to prioritize correctness and portability by separating GEMM and RMSNorm into two kernels.

```

---

If you want, I can next turn this into a **more polished GitHub-style README** with sections like **Build / Run / Sample Output / Known Issues / Future Optimization**.

| Key Point | Summary |
|---|---|
| Current status | Your updated two-pass implementation now passes validation on RX 9070 / gfx1201 |
| Pass 1 | `gemm_rocwmma()` computes GEMM and writes float32 workspace `d_c` |
| Pass 2 | `rmsnorm_apply_kernel()` reads full rows from `d_c`, applies RMSNorm and `gamma`, writes `d_d` |
| Why it works | Full-row RMSNorm is computed correctly, without relying on fragile accumulator lane mapping |
| Flow chart | A Mermaid flow chart was provided for repo documentation |
| README.md | A full repo-ready README draft was provided |
| Important wording | The current code should be described as **two-pass**, not truly fused |
| Validation basis | Host launches both kernels sequentially and compares against a CPU reference |
```
---

Here is the ASCII flow chart for your current **two-pass GEMM + RMSNorm** version:

```text
+--------------------------------------------------+
|                    Program Start                 |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Print GPU Hardware Info                          |
| - device name                                    |
| - gfx arch                                       |
| - warp size                                      |
| - memory / CU / clock / bandwidth                |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Select Architecture Parameters                   |
| if gfx9    -> wave64 path                        |
| else       -> gfx11/gfx12 wave32 path            |
| RX 9070    -> gfx1201 -> wave32 path             |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Check Problem Size / Tile Compatibility          |
| - M,N,K divisibility                             |
| - block/tile constraints                         |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Allocate Host Buffers                            |
| hA, hB, hGamma, hD                               |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Initialize Host Data                             |
| - random A                                       |
| - random B                                       |
| - gamma                                          |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Allocate Device Buffers                          |
| dA, dB, dGamma, dC, dD                           |
| dC = float32 GEMM workspace                      |
| dD = fp16 final output                           |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Copy Inputs to Device                            |
| hA -> dA                                         |
| hB -> dB                                         |
| hGamma -> dGamma                                 |
+--------------------------------------------------+
                      |
                      v
+==================================================+
|             Pass 1 : GEMM rocWMMA                |
+==================================================+
                      |
                      v
+--------------------------------------------------+
| Launch gemm_rocwmma kernel                       |
| gemmGrid = (2, 4)                                |
| blockDim = (64, 2)                               |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Each Block Computes a Macro Tile                 |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Global Load A/B Tiles                            |
| -> stage into LDS                                |
| -> B is arranged for rocWMMA consumption         |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| K-loop / Double Buffering in LDS                 |
| - local read A fragment                          |
| - local read B fragment                          |
| - mma_sync accumulate into float32 accumulators  |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Store GEMM Result Tile                           |
| accumulator -> global workspace dC (float32)     |
+--------------------------------------------------+
                      |
                      v
+==================================================+
|           Pass 2 : RMSNorm Apply Kernel          |
+==================================================+
                      |
                      v
+--------------------------------------------------+
| Launch rmsnorm_apply_kernel                      |
| normGrid  = (1)                                  |
| normBlock = (256)                                |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| One Thread Handles One Output Row                |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Read Full Row from dC                            |
| C[i, 0 ... N-1]                                  |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Compute Sum of Squares                           |
| row_ss = sum( C[i,j]^2 )                         |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Compute RMS Inverse                              |
| rms_inv = rsqrt( row_ss / N + eps )              |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Apply RMSNorm + Gamma                            |
| D[i,j] = C[i,j] * rms_inv * gamma[j]             |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Write Final Output                               |
| -> dD (fp16)                                     |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Benchmark Timing                                 |
| elapsedMs / GFLOPS / TFLOPS                      |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| Copy dD Back to Host                             |
| dD -> hD                                         |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
| CPU Reference Validation                         |
| - CPU GEMM                                       |
| - CPU RMSNorm                                    |
| - compare with GPU result                        |
+--------------------------------------------------+
                      |
                      v
+---------------------------+      +---------------------------+
|         PASSED            |  or  |          FAILED           |
| max relative error = 0    |      | print max relative error  |
+---------------------------+      +---------------------------+
                      |
                      v
+--------------------------------------------------+
| Free Device Memory                               |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
|                    Program End                   |
+--------------------------------------------------+
```

A more compact version:

```text
Start
  |
  v
Print GPU info
  |
  v
Select gfx params
  |
  v
Check tile compatibility
  |
  v
Allocate host/device buffers
  |
  v
Copy A/B/gamma to GPU
  |
  v
+---------------------------+
| Pass 1: gemm_rocwmma      |
| A x B -> dC (float32)     |
+---------------------------+
  |
  v
+---------------------------+
| Pass 2: rmsnorm_apply     |
| dC + gamma -> dD (fp16)   |
+---------------------------+
  |
  v
Benchmark
  |
  v
Copy dD back
  |
  v
CPU reference compare
  |
  v
PASS / FAIL
  |
  v
End
```

| Key Point         | Summary                                                                             |
| ----------------- | ----------------------------------------------------------------------------------- |
| Flow type         | Current implementation is a two-pass flow                                           |
| Pass 1            | rocWMMA GEMM computes `A x B` into `dC` as float32 workspace                        |
| Pass 2            | RMSNorm kernel reads each full row from `dC` and writes final fp16 `dD`             |
| Why correct       | RMSNorm now sees the full row dimension, so normalization is mathematically correct |
| RX 9070 path      | RX 9070 / gfx1201 uses the wave32 parameter path                                    |
| Validation result | Your latest run shows `PASSED` with `Max relative error: 0`                         |
