# 03. Memory Coalescing 의 수학

## 🎯 핵심 질문

- Memory coalescing 의 정확한 정의는 무엇인가 — "같은 warp 의 32 thread 가 연속 128 byte 접근 → 1 transaction" 은 정확히 어떤 메커니즘인가?
- Non-coalesced access 의 비용은 정확히 얼마인가 — 왜 32 transaction 이 필요하고, 따라서 bandwidth 가 1/32 이 되는가?
- Stride access 에서, stride = $s$ 일 때 transaction 수는 정확히 몇 개인가? $s = 1$ (coalesced) vs $s = 2$ vs $s = 256$ (completely scattered) 의 차이는?
- PyTorch tensor 의 stride 와 memory layout (NCHW vs NHWC) 이 coalescing 에 어떻게 영향을 주는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 에서 `tensor.sum(dim=...)` 의 backend kernel 은 어느 차원을 reduce 하는지에 따라 **coalescing 여부가 달라지고**, 이것이 **2–3배 성능 차이** 를 만듭니다. 또한 custom kernel 을 작성할 때, thread 들이 메모리를 어떻게 접근하는지가 전체 throughput 을 좌우합니다.

더 깊이, `tensor.permute()` 나 `tensor.view()` 후 contiguous 여부를 확인하는 것은 다음 op 이 coalesced memory access 를 할 수 있는지 판단하기 위함입니다. NCHW 와 NHWC 중 cuDNN kernel 이 어느 것을 선호하는지도 coalescing 과 직결됩니다.

---

## 📐 수학적 선행 조건

- Stride, contiguous memory layout (Ch1-02 기반)
- Modular arithmetic (byte address modulo)
- Warp 의 구조 (32 thread, lane ID)

---

## 📖 직관적 이해

### Coalesced Access: "Perfect Alignment"

같은 warp 의 32 개 thread 가 메모리를 다음과 같이 접근:

```
Thread 0, 1, ..., 31이 다음 주소로부터 각각 4 byte (FP32) 를 읽음:

Address:  0    4    8   12   16  ...  124  (128 addresses, 128 bytes)

Thread 0: [0]
Thread 1:   [4]
Thread 2:       [8]
...
Thread 31:                      [124]

→ 모두 연속된 128 byte 범위 내 → **1 transaction**
```

### Non-Coalesced Access: "Scattered"

stride = 256 (non-contiguous):

```
Address:  0              256            512            ...
         ↑               ↑              ↑
Thread 0 Thread 1        Thread 2       ...            Thread 31

→ 32 개 주소가 서로 멀리 떨어짐 → **32 separate transactions**

Bandwidth = 128 / 32 = 4 bytes/transaction × 32 = 128 bytes (coalesced 와 동일 크기)
하지만 transaction 수 = 32배 증가 → latency 증가, 대역폭 활용률 하락
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Memory Transaction 과 Alignment

GPU 메모리 subsystem 은 **access patterns** 를 32 thread 단위 (warp) 로 분석하여 transaction 으로 변환합니다:

각 thread $i \in [0, 32)$ 가 주소 $a_i$ (byte address) 에서 크기 $s_i$ bytes 를 접근할 때,

**Transaction** $t$ 는 연속된 128 byte 범위 $[128k, 128k+127]$ 에 포함된 모든 요청을 처리합니다 (aligned 64-byte chunks 로 세분).

### 정의 3.2 — Memory Coalescing

**Coalesced access**: 같은 warp 의 모든 thread 의 요청이 **단 1–2 개의 연속된 128-byte 범위** 에 포함되는 경우.

예:
- Thread $i$ 가 주소 $a_i = 4i$ (stride 4) → 주소 범위 $[0, 128)$ → **1 transaction**
- Thread $i$ 가 주소 $a_i = 256i$ (stride 256) → 주소 범위 $[0, 256) \cup [256, 512) \cup ...$ → **32 transaction**

### 정의 3.3 — Stride 와 Transaction 수

Thread $i$ 가 주소 $a_i = \text{base} + i \cdot s$ (stride $s$, FP32 = $s \in \{4, 8, 16, ...\}$) 를 접근할 때:

$$\text{# transactions} = \min\left(32, \left\lceil \frac{\text{max addr} - \text{min addr}}{128} \right\rceil \right)$$

더 정확하게, addressing pattern 이 어느 cache line (보통 128 byte) 에 걸쳐 있는지에 따라 결정.

---

## 🔬 정리와 증명

### 정리 3.1 — Contiguous Coalesced Access

Thread $i$ (lane ID) 가 주소 $a_i = a_0 + i \cdot 4$ (FP32, stride = 4) 로부터 4 bytes 를 읽을 때:

$$\text{Address range} = [a_0, a_0 + 31 \times 4] = [a_0, a_0 + 124]$$

이 범위가 연속된 128 bytes 이므로:

$$\text{# transactions} = 1 \quad \checkmark$$

**증명**: Warp 의 32 thread 가 각각 4 bytes (FP32) 를 읽으면, 총 128 bytes. $a_0$ 이 4-byte aligned 이면, 요청 범위는 $[a_0, a_0 + 127]$ → 정확히 1 개의 128-byte cache line 에 fit $\square$.

### 정리 3.2 — Strided Access 의 Transaction Cost

Thread $i$ 가 주소 $a_i = a_0 + i \cdot (s \times 4)$ (stride factor $s$, 단위 = 4 bytes) 로 접근할 때:

$$\text{# transactions} = \min\left(32, \left\lceil \frac{s}{32} \right\rceil \times 32 \right) = \min(32, s)$$

**증명**:
- $s = 1$: stride = 4 bytes, range = $[a_0, a_0 + 124]$ → 1 transaction
- $s = 2$: stride = 8 bytes, range = $[a_0, a_0 + 248]$ → 2 transaction (또는 1–2 depending on alignment)
- $s = 32$: stride = 128 bytes, range = $[a_0, a_0 + 31 \times 128]$ → 32 transaction (각 thread 가 다른 128-byte line)
- $s = 256$: stride = 1024 bytes, 훨씬 sparse → 32 transaction (scattered)

일반적으로, **stride $s$ 가 커질수록 transaction 수 증가** (최대 32) $\square$.

### 정리 3.3 — NCHW vs NHWC 의 Coalescing

Tensor shape $(N, C, H, W)$ 에서, 다음 순서로 메모리를 순회한다고 가정 (row-major):

**NCHW layout** (PyTorch default):
```
memory: [N:0,C:0,H:0,W:0] [N:0,C:0,H:0,W:1] ... [N:0,C:0,H:0,W:W-1] [N:0,C:1,H:0,W:0] ...
stride:  1 (W dimension이 가장 내부 → stride = 4 bytes for FP32)
```

**NHWC layout** (cuDNN preferred):
```
memory: [N:0,H:0,W:0,C:0] [N:0,H:0,W:0,C:1] ... [N:0,H:0,W:0,C:C-1] [N:0,H:0,W:1,C:0] ...
stride:  1 (C dimension이 가장 내부)
```

**예**: $C = 64$ 채널 reduce (합)
- **NCHW**: W 방향으로 sequential access → stride = 1 → **coalesced** (1 transaction)
- **NHWC**: H 차원 reduce 시, thread 들이 서로 다른 position (H, W) 에서 같은 C 를 읽음 → stride = large → **scattered** (multiple transactions)

**결론**: reduction 차원에 따라 coalescing 여부가 결정 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Contiguous vs Non-Contiguous Stride 측정

```python
import torch
import time

def measure_bandwidth(tensor, num_iter=100):
    """Measure effective bandwidth of tensor access."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iter):
        # Simple copy to force memory access
        result = tensor.clone()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Bandwidth = (num elements * element size) / time
    num_elements = tensor.numel()
    element_size = 4  # FP32
    total_bytes = num_elements * element_size * num_iter
    
    bandwidth_gb_s = total_bytes / elapsed / 1e9
    return bandwidth_gb_s

# Test different layouts
size = (1024, 1024)

# Contiguous (row-major, stride = 1)
x_contiguous = torch.randn(*size, device='cuda').contiguous()
bw_contiguous = measure_bandwidth(x_contiguous)

# Non-contiguous (transposed, stride = large)
x_transposed = torch.randn(*size, device='cuda').T
bw_transposed = measure_bandwidth(x_transposed)

# Strided (every other element)
x_strided = torch.randn(2048, 2048, device='cuda')[::2, ::2]
bw_strided = measure_bandwidth(x_strided)

print(f"Contiguous (stride=1):    {bw_contiguous:.1f} GB/s")
print(f"Transposed (stride=large): {bw_transposed:.1f} GB/s")
print(f"Strided (stride=2):        {bw_strided:.1f} GB/s")
print(f"Ratio (contiguous/strided): {bw_contiguous / bw_strided:.2f}x")
```

**예상 출력**:
```
Contiguous (stride=1):     250.0 GB/s
Transposed (stride=large):  50.0 GB/s
Strided (stride=2):        150.0 GB/s
Ratio (contiguous/strided): 5.00x
```

### 실험 2 — Reduction 에서의 Coalescing 분석

```python
import torch
import torch.nn.functional as F

def benchmark_reduction(shape, dim, layout='NCHW'):
    """Benchmark tensor.sum() along a specific dimension."""
    if layout == 'NCHW':
        x = torch.randn(shape, device='cuda')  # default: NCHW
    else:
        # Convert to NHWC
        x = torch.randn(shape, device='cuda').permute(0, 2, 3, 1).contiguous()
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    num_iter = 100
    start.record()
    for _ in range(num_iter):
        result = x.sum(dim=dim)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / num_iter
    return elapsed_ms

# Test NCHW vs NHWC for different reduction dimensions
shape = (32, 64, 128, 128)  # N, C, H, W

print("Reduction time (ms), lower is better:")
print("-" * 50)
print(f"{'Dim':<5} {'NCHW (coalesced)':<20} {'NHWC':<20}")
print("-" * 50)

for dim in [0, 1, 2, 3]:
    time_nchw = benchmark_reduction(shape, dim, 'NCHW')
    time_nhwc = benchmark_reduction(shape, dim, 'NHWC')
    ratio = time_nchw / time_nhwc
    
    dim_name = ['N', 'C', 'H', 'W'][dim]
    print(f"dim={dim_name:<2} {time_nchw:<20.3f} {time_nhwc:<20.3f}")
```

**예상 출력**:
```
Reduction time (ms), lower is better:
--------------------------------------------------
Dim    NCHW (coalesced)   NHWC
--------------------------------------------------
dim=N  0.100              0.150
dim=C  0.050              0.200
dim=H  0.060              0.180
dim=W  0.040              0.120
```

### 실험 3 — Coalescing 직접 확인 (Nsight Compute)

```bash
# CUDA kernel with coalesced vs non-coalesced access
cat > coalesce_demo.cu << 'EOF'
#include <cuda_runtime.h>

__global__ void coalesced_kernel(float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * 2.0f;  // stride = 1, coalesced
    }
}

__global__ void noncoalesced_kernel(float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i * 256] * 2.0f;  // stride = 256, scattered
    }
}

int main() {
    int n = 1024 * 1024;
    float *x, *y;
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&y, n * sizeof(float));
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // Profile coalesced
    coalesced_kernel<<<grid, block>>>(x, y, n);
    
    // Profile non-coalesced
    noncoalesced_kernel<<<grid, block>>>(x, y, n);
    
    cudaFree(x);
    cudaFree(y);
    return 0;
}
EOF

# Compile and profile
nvcc -o coalesce_demo coalesce_demo.cu
nv-nsight-compute -o coalesce_demo.ncu-rep ./coalesce_demo
# Viewer: check "Memory" → "L1 Cache" → "Global Accesses" 등
```

---

## 🔗 실전 활용

### 1. PyTorch Tensor Layout 최적화

```python
import torch

def optimal_layout_for_reduce(x, dim):
    """선택한 차원에 맞는 layout 으로 변환."""
    # dim 을 가장 내부 (stride = 1) 로 이동
    dims = list(range(x.ndim))
    dims.pop(dim)
    dims.append(dim)  # 마지막으로 이동
    
    x_permuted = x.permute(*dims).contiguous()
    # 이제 x_permuted[..., 0] 에서 reduce 는 coalesced
    return x_permuted

x = torch.randn(32, 64, 128, 128, device='cuda')
# Reduce along dim=1 (channels)
y_optimized = optimal_layout_for_reduce(x, 1).sum(dim=-1)
```

### 2. Nsight Compute 로 coalescing 확인

```bash
nv-nsight-compute --set full -o report.ncu-rep ./app
# Report 에서 "Memory" 섹션 → "Global Load/Store Throughput"
# (coalesced 는 높은 throughtput, scattered 는 낮음)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 모든 thread 가 같은 크기 bytes 접근 | 가변 크기 (벡터화) 시 복잡해짐 |
| 128-byte cache line = 고정 | GPU 모델에 따라 64–256 bytes 범위 |
| Row-major layout 만 고려 | F-order (column-major) 도 지원 (Fortran) |
| Coalescing 후 L2 cache 도달 | L1 cache miss 후 L2 access latency 추가 |

---

## 📌 핵심 정리

$$\boxed{\text{Coalesced} : 1 \text{ transaction}, \quad \text{Non-coalesced} : 32 \text{ transactions} \quad \rightarrow \quad \text{BW ratio} = 1:32}$$

| 패턴 | Stride | Transactions | Bandwidth |
|------|--------|--------------|-----------|
| Contiguous | 1 | 1 | 100% |
| Strided (s=2) | 2 | 2–4 | 50–100% |
| Strided (s=32) | 32 | 32 | ~3% |
| Scattered | >>32 | 32 | ~3% |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 같은 warp 의 32 개 thread 가 주소 0, 1, 2, ..., 31 (byte address, stride = 1 byte) 로부터 1 byte씩 읽으면 몇 개의 transaction 이 필요한가?

<details>
<summary>해설</summary>

주소 범위: $[0, 31]$ → 32 bytes, 한 cache line (128 bytes) 에 fit. 따라서 **1 transaction** $\square$.

</details>

**문제 2** (심화): Matrix transpose 이후 각 행의 합 (row-sum) 을 계산한다고 하자. Transpose 전후로 coalescing 이 어떻게 변하는가? Transpose 후 `.contiguous()` 를 호출해야 하는 이유는?

<details>
<summary>해설</summary>

**Before transpose**: shape (N, M), row-major layout
- Row-sum (dim=1 reduce): W 방향 sequential → stride = 1 → **coalesced**

**After transpose**: shape (M, N), stride = (1, N) → non-contiguous
- Row-sum 을 다시 하려면, stide = N (large) → **scattered**

**Solution**: `.contiguous()` 로 메모리를 다시 정렬 → new layout 에서 row = W 방향 다시 sequential → coalescing 복원 $\square$.

</details>

**문제 3** (논문 비평): 일부 최근 연구 (NVIDIA CUTLASS library) 는 "coalescing-aware tiling" 을 제안한다 — tile size 와 layout 을 함께 설계하여 warp-level 과 block-level coalescing 을 동시에 최적화한다. 이것이 전통적 coalescing 분석을 어떻게 확장하는가?

<details>
<summary>해설</summary>

전통적 coalescing = warp 내 32 thread 의 메모리 접근 분석 (thread-level).

**Warp-aware tiling**: 여러 warp 가 같은 tile 을 process. 예:
- Block size = 256 thread = 8 warp
- Tile shape = $8 \times 32$ (8 warp, 각각 32 thread)
- 각 warp 의 내부 thread 들 + warp 간 다른 warp 들의 접근 패턴 동시 최적화

결과: thread-level + warp-level coalescing = "2-level coalescing", roofline 상의 effective bandwidth 가 더 높음.

CUTLASS 는 이를 자동화하는 C++ template library $\square$.

</details>

---

<div align="center">

[◀ 이전](./02-memory-hierarchy.md) | [📚 README](../README.md) | [다음 ▶](./04-bank-conflict.md)

</div>
