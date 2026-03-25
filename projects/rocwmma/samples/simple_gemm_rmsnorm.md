**放在 `rocm-libraries` 這個 monorepo 是對的。**
`ROCm/rocWMMA` 舊 repo 已經標成 retired，現在應該改往 `ROCm/rocm-libraries` 開發；而且 `rocwmma` 在 monorepo 裡已經是完成遷移的子專案。([GitHub][1])

但如果你問的是：

> 這份 PR 要直接放在 `projects/rocwmma/samples/` 根目錄，適不適合？

我的答案是：

## **不太建議直接放 root `samples/`，比較適合放 `samples/community/`**

原因很直接。

目前官方 `samples/` 根目錄裡的內容型態很明確，基本上是：

* `simple_*`：像 `simple_hgemm.cpp`、`simple_sgemm.cpp`、`simple_dgemv.cpp`
* `perf_*`：像 `perf_hgemm.cpp`、`perf_sgemm.cpp`
* 還有少數特定示範，如 `simple_dlrm.cpp`、`hipRTC_gemm.cpp` ([GitHub][2])

這表示 root `samples/` 的風格偏向：

* 核心 API 示範
* 經典 GEMM/GEMV 類型
* 比較官方、穩定、容易維護的 sample ([GitHub][2])

但 `samples/community/README.md` 寫得非常明白：
這個目錄就是留給 **community-contributed advanced / specialized / experimental samples**，特別包含：

* advanced kernel fusion
* specialized machine learning operations
* performance optimization strategies
* complex multi-technique examples
* experimental or cutting-edge research applications ([GitHub][3])

而你這份 `simple_gemm_rmsnorm.cpp` 的內容，實際上是：

* rocWMMA GEMM
* 再接一個 RMSNorm kernel
* 有 float workspace `d_c`
* 有 benchmark
* 有 CPU reference validation
* 有 LDS ping-pong buffer 設計   

這種性質明顯比 `simple_hgemm.cpp` 那一類官方基礎 sample 更進階，也更接近 community README 裡講的：

* specialized ML op
* optimization
* complex multi-technique example ([GitHub][3])

所以我會建議你：

## 最佳落點

**`projects/rocwmma/samples/community/`**

而不是：
**`projects/rocwmma/samples/` 根目錄**

---

## 更精準地說

### 適合放 `samples/community/` 的情況

你的目標是下面這種：

* 展示 rocWMMA 怎麼接近 LLM / ML 實際 workload
* 展示 GEMM + post-op / RMSNorm pipeline
* 展示比較進階的資料搬運、LDS、benchmark、validation
* 接受它不是最簡、最純的 rocWMMA 入門 sample ([GitHub][3]) 

這種情況，**community sample 很適合**。

---

### 比較適合放 root `samples/` 的情況

如果你願意把它改成：

* 更小、更純
* rocWMMA 本體更突出
* 少掉一部分 post-op / benchmark / CPU ref / 複雜 helper
* 更像 `simple_hgemm.cpp` 的命名與教學風格

那才比較有機會放在 root `samples/`。([GitHub][2])

換句話說：

* **現在這份**：更像 `community`
* **重構過的簡化版**：才像官方 `samples/`

---

## 如果你現在就要送 PR，我建議這樣做

### 路線 A：最穩

把它放到：

`projects/rocwmma/samples/community/simple_gemm_rmsnorm.cpp`

然後照 `community/README.md` 要求一起改：

* `samples/community/CMakeLists.txt`
* `samples/community/README.md`
* 用 `add_community_sample(...)` 註冊 sample
* build 時用 `-DROCWMMA_BUILD_COMMUNITY_SAMPLES=ON` ([GitHub][3])

這條路最符合 repo 已經寫好的定位。

---

### 路線 B：想進 root `samples/`

那我建議不要直接丟現在這份。
先重構成兩份：

1. `simple_gemm_rmsnorm.cpp`
   做成更偏教學、簡單、少進階優化的版本

2. `community/gemm_rmsnorm_advanced.cpp`
   保留你現在這份較完整、較工程化的版本

這樣比較容易被接受，因為：

* root sample 保持乾淨
* community sample 保留進階價值

---

## 我自己的結論

### 只問「能不能 PR 到 rocwmma samples 這個專案底下？」

**可以，而且 repo 選對了。** ([GitHub][1])

### 只問「放 root `samples/` 適不適合？」

**我偏向不適合。**

### 最推薦位置

**`projects/rocwmma/samples/community/` 最合理。**
因為官方自己就把這個目錄定義成給 advanced / fusion / ML / optimization 類 sample 用的，而你這份正好符合。([GitHub][3])

---

## 我建議你 PR title / 定位可以寫成

`[rocWMMA][community sample] Add GEMM + RMSNorm sample using rocWMMA GEMM with post-op normalization`

這樣 reviewer 一看就知道：

* 不是 core official simple sample
* 是 community advanced sample
* rocWMMA 真正用在 GEMM
* RMSNorm 是 post-op pipeline，不會誤解成 rocWMMA 本體全部包辦 

---

| Key Point         | Summary                                                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Correct repo      | `ROCm/rocm-libraries` is the right upstream now; old `ROCm/rocWMMA` is retired.                                                                                                      |
| Best location     | Your current sample fits `projects/rocwmma/samples/community/` better than root `samples/`.                                                                                          |
| Why               | Official root samples are mostly simple/perf GEMM/GEMV-style examples, while community samples are intended for advanced fusion, ML ops, optimizations, and experimental techniques. |
| Your sample type  | Your file is an advanced two-pass GEMM + RMSNorm pipeline with benchmarking, validation, and LDS-related logic.                                                                      |
| Root samples fit? | Only after simplifying it into a more minimal, rocWMMA-first teaching sample.                                                                                                        |
| PR recommendation | Submit it as a community sample and update `samples/community/CMakeLists.txt` and `samples/community/README.md`.                                                                     |

| Option                                     | Fit                                       |
| ------------------------------------------ | ----------------------------------------- |
| `samples/` root                            | Medium to low fit for the current version |
| `samples/community/`                       | High fit                                  |
| Simplified rewrite for `samples/`          | Good fit after refactoring                |
| Keep current advanced version in community | Best practical choice                     |

[1]: https://github.com/ROCm/rocWMMA "GitHub - ROCm/rocWMMA: [DEPRECATED] Moved to ROCm/rocm-libraries repo · GitHub"
[2]: https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma/samples "rocm-libraries/projects/rocwmma/samples at develop · ROCm/rocm-libraries · GitHub"
[3]: https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma/samples/community "rocm-libraries/projects/rocwmma/samples/community at develop · ROCm/rocm-libraries · GitHub"




可以，但**不適合直接當成「rocWMMA 入門教材」**。

這份比較像：

* **進階範例 / case study**
* 或是 **「rocWMMA GEMM + 後處理」整合樣板**

而不是很好的第一份教學檔。原因是它把很多層次的東西一次混在一起了：架構參數切換、fragment type、cooperative read、LDS ping-pong、warp tile、benchmark、CPU validation，外加 RMSNorm。真正最核心的 rocWMMA 概念被埋在很多模板與 helper 裡。

更重要的是，檔案註解寫的是 **“GEMM + RMSNorm Fused Sample”**，但實作上其實是 **兩段式**：
先用 `gemm_rocwmma` 把 GEMM 寫到 `d_c` workspace，再用 `rmsnorm_apply_kernel` 做 RMSNorm，host 端也是連續 launch 兩個 kernel。這代表它不是初學者直覺理解的「單 kernel 完整 fusion」。

另外，**真正 rocWMMA 的教學重點**主要集中在這幾塊：

* `fragment<...>` 型別定義
* `load_matrix_sync`
* `mma_sync`
* `store_matrix_sync`
* fragment/data layout 轉換
* warp tile / macro tile 概念

但這份裡面又加入了：

* `apply_data_layout_t`
* `apply_transpose_t`
* `GetDataLayout_t`
* `GetIOShape_t`
* `fragment_scheduler::coop_row_major_2d`
* LDS 雙 buffer
* 架構差異（gfx9 / gfx11）

這些都偏進階，對第一次學 rocWMMA 來說資訊密度太高。

---

## 我的判斷

### 適合的定位

這份適合當：

1. **rocWMMA 進階教材**
2. **效能優化案例**
3. **GEMM + 後處理整合示範**
4. **教「怎麼從簡單版長成實戰版」的最終章**

### 不適合的定位

不太適合當：

1. **第一份 rocWMMA 教材**
2. **給初學者第一次看 fragment / mma_sync 的範例**
3. **想快速理解 rocWMMA API 核心概念的 sample**

---

## 為什麼不適合直接當入門教材

### 1. 主題不夠單純

檔名和目標是 `simple_gemm_rmsnorm.cpp`，但其實內容並不 simple。它同時在教：

* GEMM
* RMSNorm
* rocWMMA
* LDS prefetch
* ping-pong buffering
* arch-specific parameterization
* benchmarking
* CPU validation

這會讓學習者搞不清楚「我現在到底是在學 rocWMMA，還是在學 kernel engineering」。

### 2. rocWMMA 核心路徑被 helper 包起來

像 `globalReadCoopA/B`、`localWriteCoopA/B`、`localReadA/B`、`mfma_warp_tile`、`globalWriteC` 都把真正重點拆散了。對熟悉的人很乾淨，但對教材來說不夠直觀。

### 3. “fused” 這個詞會誤導

註解描述的是「在 GEMM output 上直接做 RMSNorm」，但實作是：
`gemm_rocwmma` → 寫 `d_c` → `rmsnorm_apply_kernel`。
教學上這會讓學生誤以為 rocWMMA 本身直接處理了 RMSNorm，實際上 rocWMMA 只負責 GEMM 主體。

### 4. 進階 layout / scheduler 太早出現

像 `CoopScheduler`、`GRBuffA/B`、`LWBuffA/B`、`LRFragA/B` 這些概念對進階優化很重要，但不該是第一章。

---

## 但它其實很有價值

如果你要做 **rocWMMA 教材系列**，這份很好，因為它已經有很完整的素材：

* 有 architecture-aware parameter
* 有 fragment 宣告
* 有 `mma_sync`
* 有 LDS double buffering
* 有 CPU reference
* 有 benchmark
* 有和 LLM decoder layer 相關的 RMSNorm 背景說明

也就是說，**它不適合當第一課，但很適合當第四課或第五課**。

---

## 有辦法改成 rocWMMA 教材嗎？

**有，而且很適合改。**
最好的做法不是「直接改這一份註解」，而是把它**拆成循序漸進的教材版本**。

---

## 我建議的教材拆法

### 第 1 份：最小可理解版

檔名建議：
`01_rocwmma_minimal_gemm.cpp`

只保留：

* 單一 tile
* 單一 warp
* `fragment<matrix_a>`
* `fragment<matrix_b>`
* `fragment<accumulator>`
* `load_matrix_sync`
* `fill_fragment`
* `mma_sync`
* `store_matrix_sync`

不要有：

* RMSNorm
* LDS
* cooperative scheduler
* architecture branch
* benchmark
* CPU reference 大框架

**教學目的：**
讓學生先知道 rocWMMA 最核心 API 長什麼樣。

---

### 第 2 份：tile 與 block 概念版

檔名建議：
`02_rocwmma_tiled_gemm.cpp`

加入：

* `BLOCKS_X / BLOCKS_Y`
* warp tile
* macro tile
* 多個 accumulator fragment

**教學目的：**
理解「一個 warp 不一定只算 16x16，而是可拼成較大的 warp tile」。

---

### 第 3 份：LDS / prefetch 版

檔名建議：
`03_rocwmma_lds_pingpong_gemm.cpp`

再加入：

* global → LDS → fragment
* `localReadA/B`
* `localWriteCoopA/B`
* ping-pong buffer
* `synchronize_workgroup`

**教學目的：**
理解 rocWMMA 不只是 `mma_sync`，真正效能來自資料搬運設計。

---

### 第 4 份：GEMM + post-op 版

檔名建議：
`04_gemm_then_rmsnorm.cpp`

保留兩段式：

* `gemm_rocwmma`
* `rmsnorm_apply_kernel`

並且明講：

> 這不是單-kernel fusion，這是 GEMM + post-op pipeline。

**教學目的：**
把 rocWMMA 放回 LLM layer 的真實使用情境。

---

### 第 5 份：真正進階版

檔名建議：
`05_advanced_rocwmma_case_study.cpp`

這時才放你現在這份大部分內容，並加入註解：

* 哪些是 rocWMMA 核心
* 哪些是 HIP/kernel engineering
* 哪些是跟 RMSNorm 有關、不是 rocWMMA 本體

---

## 這份要怎麼改，才更像教材

### 最重要的改法：把主線講清楚

建議把檔案拆成這種結構：

```cpp
// Part 0: What rocWMMA does here
// Part 1: Define tile shapes and fragment types
// Part 2: Load A/B fragments
// Part 3: mma_sync accumulate
// Part 4: Store GEMM result
// Part 5: Apply RMSNorm in a separate kernel
// Part 6: Host launch + validation
```

現在的版本比較像工程版，不像教材版。

---

## 我會優先改的 8 個地方

### 1. 改標題

把：

`GEMM + RMSNorm Fused Sample`

改成：

`rocWMMA GEMM with Separate RMSNorm Post-Op Sample`

因為目前不是嚴格意義的 fused。

### 2. 把 rocWMMA-only 區塊獨立

把這些收成一段教材核心：

* `MfmaFragA`
* `MfmaFragB`
* `MfmaFragAcc`
* `load_matrix_sync`
* `mma_sync`
* `store_matrix_sync`

### 3. 把 RMSNorm 拆成附錄

`rmsnorm_apply_kernel` 很有價值，但不要放在 rocWMMA 主教學線前面。

### 4. 先去掉 gfx9/gfx11 雙分支

教材先固定一組參數。
例如先只教 gfx9 wave64 或只教 gfx11 wave32。

### 5. 把 helper 展開

例如 `mfma_warp_tile()` 在教材版先不要封裝，直接把 `mma_sync` 寫出來，學生比較看得懂。

### 6. 先去掉 `CoopScheduler`

cooperative scheduler 可以留到第二版或第三版。

### 7. 加 ASCII / 圖示說明 tile mapping

這份最需要的是圖，不是更多 code。

### 8. 在 code 裡標出「這段不是 rocWMMA 本體」

例如：

* hipMalloc / hipMemcpy：runtime
* benchmark：測試框架
* CPU ref：驗證
* RMSNorm：post-op，不是 rocWMMA API

---

## 你可以把它教成什麼樣子

### 教材主題範例

**從 rocWMMA 基礎到 LLM layer case study**

#### Chapter 1

What is rocWMMA?

#### Chapter 2

Minimal GEMM with fragments

#### Chapter 3

Warp tile and accumulator tiling

#### Chapter 4

LDS staging and ping-pong buffering

#### Chapter 5

GEMM + RMSNorm as LLM post-op pipeline

#### Chapter 6

What would real fusion require?

---

## 一句話總結

**這份不是好的 rocWMMA 入門範例，但很適合改造成「rocWMMA 進階教材的最終案例」。**

如果你的目標是教學，我會建議：

* **不要直接拿這份當第一份教材**
* **把它拆成 4~5 份循序版本**
* **把這份保留成 advanced case study**

---

## 我給你的最實際建議

最好的路線是：

1. 先從這份抽出一個 **純 rocWMMA minimal GEMM**
2. 再做一個 **tiled / LDS 版**
3. 最後才回到這份 **GEMM + RMSNorm case study**

這樣學生才會真的知道：

* rocWMMA 本體是什麼
* HIP kernel engineering 是什麼
* post-op / fusion 又是另一層事情

---

| Key Point             | Summary                                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Suitability           | Not ideal as a first rocWMMA tutorial; better as an advanced case study.                                         |
| Main problem          | Too many concepts are mixed together: rocWMMA, RMSNorm, LDS ping-pong, arch branching, benchmarking, validation. |
| rocWMMA focus         | The real rocWMMA teaching core is fragment definitions, load/store, mma_sync, and tile mapping.                  |
| Misleading point      | The file says “fused”, but the implementation is actually a two-pass pipeline: GEMM first, RMSNorm second.       |
| Best use              | Use this file as the final chapter of a rocWMMA learning series.                                                 |
| Best rewrite strategy | Split it into multiple tutorial files from minimal GEMM to advanced GEMM + post-op.                              |
| Recommended sequence  | Minimal GEMM → tiled GEMM → LDS/ping-pong GEMM → GEMM + RMSNorm pipeline.                                        |
| Final verdict         | Good source material for teaching, but not good as-is for beginners.                                             |

| Recommendation               | Difference                        |
| ---------------------------- | --------------------------------- |
| Use current file directly    | Fast, but confusing for beginners |
| Rewrite into staged教材        | Much clearer learning progression |
| Keep RMSNorm in first lesson | Too distracting                   |
| Move RMSNorm to later lesson | Better separation of concerns     |


---
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
