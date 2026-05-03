# 04. cuBLAS · cuDNN 의 활용

## 🎯 핵심 질문

- cuBLAS GEMM 의 수식 $C = \alpha AB + \beta C$ 에서 각 매개변수의 역할은?
- FLOP 계산 $\text{FLOP} = 2MNK$ 가 어디서 나오는가?
- cuDNN convolution 이 "implicit GEMM", "Winograd", "FFT-based" 3가지 알고리즘 중 어떤 것을 선택하는가?
- `torch.backends.cudnn.benchmark = True` 가 구체적으로 무엇을 하는가?
- Tensor Core (A100+) 의 자동 가속은 어떻게 작동하고 TF32 무엇인가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 사용자 99% 는 **직접 CUDA kernel 을 작성하지 않습니다**. 대신 `torch.mm()`, `F.conv2d()` 등을 호출하는데, 이들의 뒤에는 **NVIDIA 가 5년간 최적화한 cuBLAS · cuDNN library** 가 숨어있습니다. 이들을 이해하면:

1. **kernel selection**: 같은 conv 라도 input shape 에 따라 100배 성능 차이
2. **Tensor Core 자동 활용**: BF16 + Tensor Core = 8배 가속 (automatic)
3. **Profiling**: why is my model slow? cuBLAS bottleneck? memory bandwidth? → pinpoint 가능
4. **Memory layout**: NCHW vs NHWC 가 cuDNN kernel 선택을 바꾸는 메커니즘

---

## 📐 수학적 선행 조건

- **Linear Algebra**: Matrix multiplication, BLAS operations
- **Chapter 4**: Memory hierarchy, bandwidth, arithmetic intensity, roofline
- **Convolution**: im2col, direct convolution, Winograd algorithm
- **Tensor Core**: matrix multiplication unit (특정 크기의 GEMM)

---

## 📖 직관적 이해

### cuBLAS: Library 의 역할

```
Python: torch.mm(A, B)
           ↓
PyTorch dispatcher
           ↓
cuBLAS library (pre-compiled, tuned)
           ↓
GPU: Tensor Core (if FP16/TF32) or FP32 unit
           ↓
Result: C
```

**핵심**: NVIDIA 가 모든 (M, N, K) 에 대해 최적의 kernel 을 미리 컴파일해둠
- Small GEMM (M < 64): 특화 kernel
- Large GEMM (M > 10000): tiling, blocking 최적화

### cuDNN: 컨볼루션의 여러 구현

```
Conv forward: y = conv(x, w)

Algorithm choice:
  1. Implicit GEMM: x → im2col → y' = GEMM → y'
  2. Winograd: x → transform → element-wise mul → inverse
  3. FFT: x → FFT → w → FFT → multiply → iFFT
  
Input shape 에 따라:
  - 작은 kernel, 작은 input: Implicit GEMM 빠름
  - 큰 kernel, 큰 input: Winograd 좋음
  - 특정 크기: FFT 최적
```

---

## ✏️ 엄밀한 정의

### 정의 5.9 — GEMM (General Matrix Multiplication)

cuBLAS 기본 연산:

$$C \leftarrow \alpha A B + \beta C$$

where:
- $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$
- $\alpha, \beta \in \mathbb{R}$ (scalar)

**FLOP count**:
- Matrix multiplication: $2MNK$ (M×N 개 output, 각각 K 번 multiply-add)
- Scaling: $2MN$ (negligible)
- Total: $\approx 2MNK$ FLOPs

**Throughput**: $\frac{2MNK}{t_{\text{exec}}}$ [TFLOPS]

### 정의 5.10 — cuDNN Convolution Algorithm

Forward convolution 을 위한 3가지 알고리즘:

1. **Implicit GEMM** (General):
   - $x_{\text{col}} \in \mathbb{R}^{C \times K_h K_w \times H'W'}$ (im2col)
   - $w_{\text{mat}} \in \mathbb{R}^{C_{\text{out}} \times CK_hK_w}$
   - $y = w_{\text{mat}} \times x_{\text{col}}^T$ (standard GEMM)

2. **Winograd** (Fast for small filters):
   - $\mathcal{F}$ = Winograd transform matrix
   - $\hat{y} = (\mathcal{F}^{-1} \hat{x}) \circ (\mathcal{F}^{-1} \hat{w})$
   - Fewer multiplications (fast fourier decomposition)

3. **FFT** (Fast for large spatial):
   - $\hat{x} = \text{FFT}(x)$, $\hat{w} = \text{FFT}(w)$
   - Pointwise multiply: $\hat{y} = \hat{x} \circ \hat{w}$
   - $y = \text{iFFT}(\hat{y})$

### 정의 5.11 — Tensor Core (NVIDIA A100+)

Specialized hardware unit:

$$\text{Tensor Core: } D + A \times B \to D$$

16×16×16 matrix operations (mixed precision):
- $A, B$ : FP16 (input)
- Accumulate: FP32 (for stability)
- $D$ : FP32 output

**Throughput**: 125 TFLOPS (FP16, A100), 312 TFLOPS (TF32, H100)

### 정의 5.12 — TensorFloat32 (TF32)

Hybrid format (A100+):

$$\text{TF32: } 1 \text{ sign} + 8 \text{ exponent} + 10 \text{ mantissa}$$

- FP32 와 동일 range ($\pm 10^{\pm 38}$)
- FP16 보다 정확 (16 vs 10 bits)
- Tensor Core 에 **자동으로 반올림** (transparent)
- **성능**: FP32 와 같은 이름, FP16 과 같은 속도

---

## 🔬 정리와 증명

### 정리 5.7 — cuBLAS Roofline (arithmetic intensity vs bandwidth)

GEMM $C = AB$ 의 arithmetic intensity:

$$I = \frac{\text{FLOP}}{\text{bytes transferred}} = \frac{2MNK}{(MK + KN + MN) \times 4}$$

for large $M, N, K$:

$$I \approx \frac{2MNK}{4(MK + KN)} \approx \frac{K}{2} \quad \text{(when } M \approx N \gg K\text{)}$$

**Implication**: K 가 크면 compute-bound (좋음), K 가 작으면 memory-bound (나쁨)

**증명**: By definition $\square$

### 정리 5.8 — cuDNN Algorithm Selection Rule

Input/kernel size 에 따라 최적 알고리즘:

$$\text{Algorithm}^* = \arg\max_{\text{algo}} \text{throughput}(\text{input shape, kernel size})$$

**Benchmark** (`benchmark=True`):
- $2^5$ ~ $2^7$ 개의 (algorithm, tile size) pair 에 대해 실제 execution time 측정
- 최적값을 kernel 별로 cache (same input shape 시 재사용)

**비용**: 초기 warmup 시 100ms ~ 1s 소비, 이후 zero overhead

**증명**: NVIDIA cuDNN autotuning 구현 $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — cuBLAS GEMM 벤치마크

```python
import torch
import time

def benchmark_gemm(m, n, k, dtype=torch.float32, runs=100):
    A = torch.randn(m, k, dtype=dtype, device='cuda')
    B = torch.randn(k, n, dtype=dtype, device='cuda')
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    t_total = time.time() - t0
    
    flops = 2 * m * n * k * runs
    tflops = flops / (t_total * 1e12)
    return tflops

# Memory-bound (K small)
print(f"M=100,  K=10:   {benchmark_gemm(100, 100, 10):.2f} TFLOPS")
# Compute-bound (K large)
print(f"M=1000, K=1000: {benchmark_gemm(1000, 1000, 1000):.2f} TFLOPS")
print(f"M=4096, K=4096: {benchmark_gemm(4096, 4096, 4096):.2f} TFLOPS")
```

**예상 출력** (A100):
```
M=100,  K=10:   15.4 TFLOPS    (memory-bound)
M=1000, K=1000: 95.2 TFLOPS    (balanced)
M=4096, K=4096: 312.5 TFLOPS   (compute-bound, TF32)
```

### 실험 2 — cuDNN Algorithm Selection

```python
import torch
import torch.nn as nn

# Enable/disable benchmarking
torch.backends.cudnn.benchmark = False
model_noopt = nn.Conv2d(3, 64, 3, padding=1).cuda()

x = torch.randn(128, 3, 224, 224, device='cuda')

# Timing without benchmark
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    y = model_noopt(x)
torch.cuda.synchronize()
t_noopt = time.time() - t0

# Enable benchmarking
torch.backends.cudnn.benchmark = True
model_opt = nn.Conv2d(3, 64, 3, padding=1).cuda()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    y = model_opt(x)
torch.cuda.synchronize()
t_opt = time.time() - t0

print(f"Without benchmark: {t_noopt:.3f} s")
print(f"With benchmark:    {t_opt:.3f} s")
print(f"Speedup: {t_noopt / t_opt:.2f}x")
```

**예상 출력**:
```
Without benchmark: 0.456 s   (conservative algorithm)
With benchmark:    0.289 s   (tuned algorithm)
Speedup: 1.58x
```

### 실험 3 — Tensor Core 자동 활용 (FP16 vs FP32)

```python
import torch

def benchmark_dtype(m, n, k, dtype, runs=100):
    A = torch.randn(m, k, dtype=dtype, device='cuda')
    B = torch.randn(k, n, dtype=dtype, device='cuda')
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    t_total = time.time() - t0
    
    flops = 2 * m * n * k * runs
    tflops = flops / (t_total * 1e12)
    return tflops

M, N, K = 4096, 4096, 4096

# FP32
tflops_fp32 = benchmark_dtype(M, N, K, torch.float32)
print(f"FP32:    {tflops_fp32:.2f} TFLOPS")

# FP16 (Tensor Core)
tflops_fp16 = benchmark_dtype(M, N, K, torch.float16)
print(f"FP16:    {tflops_fp16:.2f} TFLOPS")

# TF32 (automatic, transparent)
with torch.cuda.amp.autocast(dtype=torch.float32):
    tflops_tf32 = benchmark_dtype(M, N, K, torch.float32)
    print(f"TF32:    {tflops_tf32:.2f} TFLOPS")

print(f"Speedup (FP16 vs FP32): {tflops_fp16 / tflops_fp32:.1f}x")
print(f"Speedup (TF32 vs FP32): {tflops_tf32 / tflops_fp32:.1f}x")
```

**예상 출력** (A100):
```
FP32:    312.5 TFLOPS
FP16:    625.0 TFLOPS      (2x faster, Tensor Core)
TF32:    625.0 TFLOPS      (2x faster, auto-tuned)
Speedup (FP16 vs FP32): 2.0x
Speedup (TF32 vs FP32): 2.0x
```

### 실험 4 — Memory Layout 이 cuDNN Algorithm 선택에 미치는 영향

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
conv.eval()

# NCHW (default)
x_nchw = torch.randn(128, 3, 224, 224, device='cuda', memory_format=torch.contiguous_format)

# NHWC (channels last)
x_nhwc = x_nchw.to(memory_format=torch.channels_last)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    y_nchw = conv(x_nchw)
torch.cuda.synchronize()
t_nchw = time.time() - t0

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    y_nhwc = conv(x_nhwc)
torch.cuda.synchronize()
t_nhwc = time.time() - t0

print(f"NCHW:  {t_nchw:.3f} s")
print(f"NHWC:  {t_nhwc:.3f} s")
print(f"Ratio: {t_nhwc / t_nchw:.2f}x")
```

---

## 🔗 실전 활용

### 1. cuBLAS 직접 호출 (torch.linalg)

```python
# torch 는 대부분 cuBLAS wrapper
A = torch.randn(1000, 1000, device='cuda')
B = torch.randn(1000, 1000, device='cuda')

# cuBLAS GEMM
C = torch.mm(A, B)

# cuBLAS TRSM (triangular solve)
L = torch.linalg.cholesky(A @ A.T)
x = torch.linalg.solve_triangular(L, B)
```

### 2. cudnnBatchNorm vs Fused (torch.nn.functional)

```python
import torch
import torch.nn.functional as F

x = torch.randn(128, 256, 56, 56, device='cuda')
weight = torch.randn(256, device='cuda')
bias = torch.randn(256, device='cuda')
running_mean = torch.zeros(256, device='cuda')
running_var = torch.ones(256, device='cuda')

# cuDNN BatchNorm
y = F.batch_norm(x, running_mean, running_var, weight, bias, training=False)

# Fused (faster in some cases)
with torch.cuda.amp.autocast(dtype=torch.float16):
    y_fused = F.batch_norm(x, running_mean, running_var, weight, bias)
```

### 3. 성능 프로파일링

```python
import torch
import torch.profiler as profiler

model = torch.nn.ResNet50().cuda()
x = torch.randn(128, 3, 224, 224, device='cuda')

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    y = model(x)

prof.export_chrome_trace("profile.json")
# Chrome DevTools 에서 열어서 kernel names 확인
# (cudnn::batch_norm, cuBLAS::gemm, ...)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| cuBLAS/cuDNN 설치됨 | CUDA 설치 시 기본 포함 |
| Input shape 크기 합리적 | 매우 큰 M/N/K 시 메모리 부족 |
| Benchmark cache 유효 | Input shape 자주 변하면 오버헤드 |
| Tensor Core 활용 가능 | 낮은 NVIDIA GPU (GTX 1080 등) 는 지원 안 함 |

---

## 📌 핵심 정리

$$\boxed{C = \alpha AB + \beta C, \quad \text{FLOP} = 2MNK, \quad I = \frac{K}{2}}$$

| 라이브러리 | 주요 연산 | 선택 |
|-----------|----------|------|
| **cuBLAS** | GEMM, TRSM, dot | 자동 (dispatcher) |
| **cuDNN** | Conv, BatchNorm, Softmax | benchmark 또는 heuristic |
| **Tensor Core** | FP16/TF32 GEMM | 자동 (A100+ 에서 transparent) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GEMM $C = AB$ 에서 $M=100, N=100, K=10$ 일 때 arithmetic intensity $I$ 를 계산하라. A100 (312 TFLOPS, 2 TB/s) 의 roofline 에서 기대 throughput 은?

<details>
<summary>해설</summary>

$$I = \frac{2 \times 100 \times 100 \times 10}{4(100 \times 10 + 10 \times 100 + 100 \times 100)} = \frac{200000}{4(1000 + 1000 + 10000)} = \frac{200000}{48000} \approx 4.17 \text{ FLOP/byte}$$

Roofline: $\min(312, 2000 \times 4.17) = \min(312, 8340) = 312$ TFLOPS

**결론**: Compute-bound 는 아니고 (312 TFLOP << 8340), 실제로는 ~50-100 TFLOPS 정도 나올 것 (memory-latency 때문)

$\square$

</details>

**문제 2** (심화): Winograd algorithm (Conv) 이 왜 fast convolution 이라고 불리는가? Implicit GEMM 과 FLOPs 를 비교하라 (hint: Cooley-Tukey FFT, Cook 1966).

<details>
<summary>해설</summary>

**Implicit GEMM** (direct convolution):
- Input: $H \times W$, Kernel: $K_h \times K_w$, Output: $H' \times W'$
- FLOPs: $2 \times H' \times W' \times K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}$

**Winograd** (예: 3×3 kernel on 6×6 patch):
- Transform x, w: $O(N^2 \log N)$ (FFT-like)
- Element-wise multiply: $O(N^2)$
- Inverse transform: $O(N^2 \log N)$
- 총 FLOPs: $\approx 36$ multiplies per 6×6 output (vs 54 for direct)

→ 약 $36/54 \approx 67\%$ FLOPs (30% 절감)

**문제**: Winograd 는 kernel size 고정 (3×3) → 다양한 kernel 에 부적합

**현대**: Deep learning 은 3×3 주로 사용 → Winograd 유리. 1×1 은 Implicit GEMM $\square$

</details>

**문제 3** (논문 비평): TensorFloat32 (TF32) 는 "무료 가속" 이라 불리지만, 정확도 손실이 있다. FP32 (23-bit mantissa) vs TF32 (10-bit mantissa) 의 정확도 차이를 IEEE 754 관점에서 분석하라.

<details>
<summary>해설</summary>

**Precision** (유효 자리수):
- FP32: $\log_{10}(2^{23}) \approx 6.9$ decimal digits
- TF32: $\log_{10}(2^{10}) \approx 3$ decimal digits

**영향**:
1. **Accumulation error**: GEMM 에서 K 번 더할 때, 매번 TF32 반올림 → 큰 K 에서 오차 누적
2. **Gradient 계산**: backward pass 에서도 오차 전파
3. **실제 영향**: ResNet, BERT 등 대부분 기존 모델과 동등한 정확도 유지 (by empirical validation, NVIDIA benchmarks)

**해결책**:
- Critical operations (loss computation) 은 FP32 유지
- GEMM 만 TF32 적용 → mixed precision (Ch6)

Current practice: TF32 ON by default, 문제 생기면 명시적 disable $\square$

</details>

---

<div align="center">

[◀ 이전](./03-triton.md) | [📚 README](../README.md) | [다음 ▶](./05-kernel-fusion.md)

</div>
