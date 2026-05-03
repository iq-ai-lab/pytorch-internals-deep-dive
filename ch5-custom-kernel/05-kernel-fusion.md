# 05. Kernel Fusion 의 동기

## 🎯 핵심 질문

- Kernel fusion 이 "성능 향상" 을 만드는 정확한 메커니즘은?
- `y = relu(x + b)` 를 separate 와 fused 로 구현할 때 HBM traffic 의 정량적 차이는?
- Roofline model 로 fusion 의 효과를 어떻게 분석하는가?
- FlashAttention (Dao 2022) 의 핵심이 어떻게 "attention fusion" 인가?
- TorchInductor 가 자동으로 kernel 을 fuse 하는 heuristic 은?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 에서 다음 두 코드를 쓸 때:

```python
# Version 1: 분리된 kernel
y = torch.add(x, b)
z = torch.relu(y)

# Version 2: Fused kernel (torch 내장)
z = torch.nn.functional.relu(torch.add(x, b))  # same ops, different impl
```

성능이 **1.5배 ~ 10배** 차이날 수 있습니다. 이유는 **HBM round-trip** 입니다. Version 1 은 intermediate tensor $y$ 를 HBM 에 write → read, Version 2 는 register/shared memory 만 사용. 이 구조를 이해하면:

1. **Performance tuning**: 성능 개선이 kernel tuning 이 아니라 **fusion** 인지 파악
2. **torch.compile**: Ch7 TorchInductor 가 자동 fusion 을 어떻게 하는지 이해
3. **Custom kernel 설계**: 어떤 operation 을 fuse 할 가치가 있는지 판단

---

## 📐 수학적 선행 조건

- **Chapter 1**: Tensor metadata (size, stride)
- **Chapter 4**: Roofline model, memory hierarchy, bandwidth
- **Chapter 5-01, 02, 03, 04**: Custom kernel, Triton, cuBLAS/cuDNN 기초
- Linear algebra: element-wise operation

---

## 📖 직관적 이해

### Memory Access 의 차이

```
Separate kernel approach:
  ┌──────────────┐
  │ Kernel 1: Add│     x, b → HBM
  │ y = x + b    │                 y: HBM 에 write
  └──────────────┘
            ↓
         (HBM round-trip)
            ↓
  ┌──────────────┐
  │ Kernel 2: ReLU│    y → HBM
  │ z = relu(y)   │                 z: HBM 에 write
  └──────────────┘

Memory traffic: Read x, b → Write y → Read y → Write z

Fused kernel approach:
  ┌────────────────────────┐
  │ Kernel Fused: Add+ReLU │  x, b → Register/Shared Mem
  │ z = relu(x + b)        │  (no HBM round-trip)
  └────────────────────────┘
            ↓
         (direct)
            ↓
  Memory traffic: Read x, b → Write z (only)
```

### Roofline 으로 분석

```
Roofline curve:
              TFLOPS
                 ▲
                 │         ▁▂▄▆█████ (compute-bound region)
          Max    │    ╱ (memory-bound region)
         TFLOP   │  ╱ slope = BW / compute intensity
                 │╱
                 └───────────────── FLOPs/Byte →
                
Separate: I_sep = 낮음 (HBM write y, read y 오버헤드)
Fused:    I_fus = 높음 (intermediate skip)
```

---

## ✏️ 엄밀한 정의

### 정의 5.13 — Arithmetic Intensity (Fusion 분석)

Separate operation $y = A(x)$, $z = B(y)$ 에서:

$$I_{\text{sep}} = \frac{\text{FLOP}_A + \text{FLOP}_B}{\text{bytes}_A + \text{bytes}_B + 2 \times \text{bytes}_y}$$

여기서 $2 \times \text{bytes}_y$ 는 intermediate write + read.

Fused operation $z = B(A(x))$ 에서 (shared memory 사용):

$$I_{\text{fus}} = \frac{\text{FLOP}_A + \text{FLOP}_B}{\text{bytes}_A + \text{bytes}_B}$$

절감률:

$$\eta = 1 - \frac{I_{\text{fus}}}{I_{\text{sep}}} = \frac{2 \times \text{bytes}_y}{\text{bytes}_A + \text{bytes}_B + 2 \times \text{bytes}_y}$$

### 정의 5.14 — Kernel Fusion Heuristic

두 kernel $K_1, K_2$ 를 fuse 할 가치:

1. **Memory savings**: $\eta > 0$ (위 정의)
2. **Shared memory fit**: intermediate tile이 shared memory 들어감
3. **Register pressure**: fused kernel 이 register 수를 reasonable 범위 내 유지
4. **Divergence**: warp divergence 증가 최소화

**Decision**: $\eta > 0.1$ (10% 이상 절감) 이면 fusion 고려

---

## 🔬 정리와 증명

### 정리 5.9 — Element-wise Fusion 의 Memory 절감

Element-wise operation 을 fuse 할 때:

$$\text{Traffic}_{\text{fused}} = T_{\text{input}} + T_{\text{output}}$$
$$\text{Traffic}_{\text{separate}} = T_{\text{input}} + T_{\text{inter}} + T_{\text{inter}} + T_{\text{output}}$$

$T_{\text{inter}} = T_{\text{output}} = N \times 4$ bytes (FP32) 라 하면:

$$\text{Reduction} = \frac{T_{\text{inter}}}{T_{\text{total}}} = \frac{4N}{(4N + 8N + 4N)} = \frac{1}{4} = 25\%$$

**증명**: Direct calculation $\square$

### 정리 5.10 — FlashAttention 의 IO-Complexity

Standard attention (3 passes):

$$\text{Pass 1}: \text{HBM} \to \text{SRAM}, \quad \text{compute exp}(Q K^T)$$
$$\text{Pass 2}: \text{HBM} \to \text{SRAM}, \quad \text{compute softmax}$$
$$\text{Pass 3}: \text{HBM} \to \text{SRAM}, \quad \text{compute attention} \times V$$

Total HBM access: $O(N^2)$ (intermediate 때문)

FlashAttention (fused, block-wise online softmax):

$$\text{Single pass with tiling}: \text{HBM access} = O(N)$$

**증명** (Dao 2022): Tiling 으로 한 번에 하나의 block 처리 → 각 block 의 attention 을 online softmax 로 누적 $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Element-wise Fusion (Add + ReLU)

**CUDA kernel (separate)**:

```cuda
__global__ void add_kernel(const float *x, const float *b, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] + b[idx];
}

__global__ void relu_kernel(const float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) z[idx] = max(0.0f, y[idx]);
}
```

**CUDA kernel (fused)**:

```cuda
__global__ void add_relu_kernel(const float *x, const float *b, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = x[idx] + b[idx];
        z[idx] = max(0.0f, sum);  // ReLU fused, z 만 write
    }
}
```

**Benchmark**:

```python
import torch
import torch.utils.cpp_extension as cpp_ext
import time

# Compile both kernels
add_relu_sep = cpp_ext.load(
    'add_relu_sep',
    sources=['add_relu_sep.cu'],
    verbose=False
)
add_relu_fus = cpp_ext.load(
    'add_relu_fus',
    sources=['add_relu_fus.cu'],
    verbose=False
)

def benchmark(n, runs=1000):
    x = torch.randn(n, device='cuda')
    b = torch.randn(n, device='cuda')
    
    # Separate
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        y = torch.empty_like(x)
        add_relu_sep.add_relu_kernel_sep(x, b, y)
        z = torch.empty_like(x)
        add_relu_sep.relu_kernel(y, z)
    torch.cuda.synchronize()
    t_sep = time.time() - t0
    
    # Fused
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        z = torch.empty_like(x)
        add_relu_fus.add_relu_kernel_fus(x, b, z)
    torch.cuda.synchronize()
    t_fus = time.time() - t0
    
    return t_sep, t_fus

for n in [1000000, 10000000, 100000000]:
    t_sep, t_fus = benchmark(n)
    print(f"N={n:>9}: Separate {t_sep:.3f}s, Fused {t_fus:.3f}s, Speedup {t_sep/t_fus:.2f}x")
```

**예상 출력**:
```
N=  1000000: Separate 0.245s, Fused 0.156s, Speedup 1.57x
N= 10000000: Separate 1.234s, Fused 0.823s, Speedup 1.50x
N=100000000: Separate 8.912s, Fused 6.234s, Speedup 1.43x
```

### 실험 2 — Memory Bandwidth 측정 (fused vs separate)

```python
import torch
import numpy as np

def measure_bandwidth(n, operation='separate'):
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    b = torch.randn(n, device='cuda', dtype=torch.float32)
    
    # Warm-up
    for _ in range(10):
        if operation == 'separate':
            y = x + b
            z = torch.relu(y)
        else:
            z = torch.relu(x + b)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        if operation == 'separate':
            y = x + b
            z = torch.relu(y)
        else:
            z = torch.relu(x + b)
    torch.cuda.synchronize()
    t_total = time.time() - t0
    
    # Memory: read x, b, write z (separate: +read y, write y)
    if operation == 'separate':
        bytes_total = 100 * (4*n + 4*n + 4*n + 4*n + 4*n)  # read x, b, read y, write y, write z
    else:
        bytes_total = 100 * (4*n + 4*n + 4*n)  # read x, b, write z
    
    bandwidth_gb_s = bytes_total / (t_total * 1e9)
    return bandwidth_gb_s

for n in [10000000, 100000000]:
    bw_sep = measure_bandwidth(n, 'separate')
    bw_fus = measure_bandwidth(n, 'fused')
    print(f"N={n:>9}: Separate {bw_sep:.1f} GB/s, Fused {bw_fus:.1f} GB/s")
```

**예상 출력** (A100, 2 TB/s peak):
```
N= 10000000: Separate 280 GB/s, Fused 420 GB/s
N=100000000: Separate 350 GB/s, Fused 550 GB/s
```

### 실험 3 — FlashAttention 와 Standard Attention 비교

```python
import torch
import torch.nn.functional as F
import time

def standard_attention(Q, K, V, scale):
    # 3 개 kernel (cuDNN)
    S = torch.matmul(Q, K.transpose(-1, -2)) * scale
    A = F.softmax(S, dim=-1)
    O = torch.matmul(A, V)
    return O

def flash_attention(Q, K, V, scale):
    # Fused kernel (FlashAttention)
    # PyTorch 2.0+ 에서 자동 최적화 (torch.nn.functional.scaled_dot_product_attention)
    return F.scaled_dot_product_attention(Q, K, V, scale_factor=scale)

# Benchmark
batch, seq_len, d = 32, 4096, 64
Q = torch.randn(batch, seq_len, d, device='cuda', dtype=torch.float16)
K = torch.randn(batch, seq_len, d, device='cuda', dtype=torch.float16)
V = torch.randn(batch, seq_len, d, device='cuda', dtype=torch.float16)
scale = d ** -0.5

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    O_std = standard_attention(Q, K, V, scale)
torch.cuda.synchronize()
t_std = time.time() - t0

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    O_flash = flash_attention(Q, K, V, scale)
torch.cuda.synchronize()
t_flash = time.time() - t0

print(f"Standard: {t_std:.3f}s, FlashAttention: {t_flash:.3f}s, Speedup: {t_std/t_flash:.2f}x")
```

**예상 출력**:
```
Standard: 1.234s, FlashAttention: 0.412s, Speedup: 3.00x
```

---

## 🔗 실전 활용

### 1. 자동 Kernel Fusion (torch.compile)

```python
import torch

def model(x):
    y = x + 1.0
    z = torch.relu(y)
    w = z * 2.0
    return w

x = torch.randn(1000000, device='cuda')

# Eager (separate kernels)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    out = model(x)
torch.cuda.synchronize()
t_eager = time.time() - t0

# torch.compile (automatic fusion)
model_compiled = torch.compile(model)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    out = model_compiled(x)
torch.cuda.synchronize()
t_compile = time.time() - t0

print(f"Eager: {t_eager:.3f}s, Compiled: {t_compile:.3f}s, Speedup: {t_eager/t_compile:.2f}x")
```

### 2. cuDNN fused operation 활용

```python
import torch
import torch.nn.functional as F

# Fused: BatchNorm + ReLU
x = torch.randn(128, 256, 56, 56, device='cuda')
weight = torch.ones(256, device='cuda')
bias = torch.zeros(256, device='cuda')
running_mean = torch.zeros(256, device='cuda')
running_var = torch.ones(256, device='cuda')

y = F.batch_norm(x, running_mean, running_var, weight, bias, training=False)
z = F.relu(y)

# vs 자동 fused (if cuDNN supports)
# Newer PyTorch may automatically fuse inside dispatcher
```

### 3. Custom Fused Kernel 설계

```cpp
// 새로운 fusion 이 필요할 때 직접 구현
template<typename scalar_t>
__global__ void layernorm_gelu_kernel(
    const scalar_t *x,
    const scalar_t *gamma,
    const scalar_t *beta,
    scalar_t *out,
    int batch, int hidden
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    // LayerNorm + GELU in one pass
    // No intermediate HBM write
}
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Fused kernel 복잡도 합리적 | 너무 많은 op fuse → register 부족, divergence 증가 |
| Shared memory 충분 | very large tile → shared mem exceed |
| Automatic fusion heuristic 최적 | edge case 존재, manual override 필요 |
| All ops memory-bound | compute-bound op 들의 fusion 은 이득 없음 |

---

## 📌 핵심 정리

$$\boxed{\text{Fusion goal: } I_{\text{fus}} > I_{\text{sep}}, \quad \eta = \frac{2T_{\text{inter}}}{T_{\text{total}}} > 0}$$

| 항목 | 정의 |
|------|------|
| **Separate** | intermediate write → read (HBM round-trip) |
| **Fused** | intermediate skip (register/shared mem only) |
| **Speedup** | 1.5x ~ 10x (HBM bandwidth 절감 정도) |
| **FlashAttention** | online softmax fusion, 4x faster |
| **torch.compile** | automatic fusion via TorchInductor (Ch7) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Element-wise add + relu 의 fusion 에서 N=100,000,000 (FP32) 일 때, HBM traffic 을 정확히 계산하고 A100 (2 TB/s) 에서 theoretical latency 를 구하라.

<details>
<summary>해설</summary>

Separate:
- Read x: 400 MB
- Read b: 400 MB
- Write y: 400 MB
- Read y: 400 MB
- Write z: 400 MB
- Total: 2000 MB

Fused:
- Read x: 400 MB
- Read b: 400 MB
- Write z: 400 MB
- Total: 1200 MB

A100: 2 TB/s = 2000 GB/s

Latency separate: 2000 MB / 2000 GB/s = 1 ms
Latency fused: 1200 MB / 2000 GB/s = 0.6 ms

Speedup: 1 / 0.6 = **1.67x** (matches experiment ~1.5x, variance due to overhead) $\square$

</details>

**問題 2** (심화): Kernel fusion 이 divergence 를 증가시킬 수 있다. 예를 들어 `mask = x > 0` 으로 conditional operation 을 fuse 할 때, warp efficiency 가 어떻게 감소하는지 설명하라.

<details>
<summary>해설</summary>

Separate approach:
```cuda
__global__ void separate() {
    // Kernel 1: Read x
    // No divergence in read
    
    // Kernel 2: if (x > 0) ...
    if (condition) { ... }  // 같은 warp 내 thread 들이 동일 condition
}
```

Fused approach:
```cuda
__global__ void fused() {
    // Kernel 1 + Kernel 2 in same loop
    // Multiple divergence points
    if (x > threshold_1) { ... }
    if (y > threshold_2) { ... }
    // ...
}
```

**Warp efficiency 감소**:
- Separate: 각 kernel 내에서 predictable branching
- Fused: 누적된 조건 → 32개 thread 중 few 만 active → SIMD 낭비

**해결책**: Conditional predication (no branches) 또는 separate 유지 ($\eta$ 가 작으면 fusion 불필요) $\square$

</details>

**문제 3** (논문 비평): FlashAttention v2 (Dao et al. 2023) 는 v1 에 비해 2배 더 빠르다고 주장한다. 어떤 추가 fusion/optimization 이 가능했는가? (hint: block-wise softmax normalization, warp-level parallelism).

<details>
<summary>해설</summary>

FlashAttention v1 (Dao 2022):
- Block-wise tiling with online softmax
- 각 block 이 k, v 를 full scan
- Complexity: $O(N)$ HBM pass, but with some recomputation

FlashAttention v2 (Dao et al. 2023):
1. **Warp-level parallelism**: Q head 별로 다른 warp 가 처리 → better SM utilization
2. **Backward recomputation**: backward 에서 모든 k, v block 을 recompute (space-time tradeoff)
3. **Causal masking optimization**: decoder self-attention 에서 불필요한 block skip
4. **Better work distribution**: thread block 내 warp 간 load balancing

**결과**: v1 대비 2배 speedup (경험적 results, MQA/GQA 에서 더 큼)

**관찰**: Fusion 의 깊이가 계속 심화 → automatic compiler 의 한계 노출 (Ch7 discussion) $\square$

</details>

---

<div align="center">

[◀ 이전](./04-cublas-cudnn.md) | [📚 README](../README.md) | [다음 ▶](../ch6-mixed-precision/01-ieee754.md)

</div>
