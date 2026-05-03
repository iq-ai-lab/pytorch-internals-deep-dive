# 02. Memory Hierarchy — HBM/L2/L1/Shared/Register

## 🎯 핵심 질문

- GPU 의 메모리 계층 (Register → Shared → L1/L2 → HBM) 의 latency, bandwidth, capacity 의 정확한 수치는?
- Roofline model 에서 arithmetic intensity $I = \mathrm{FLOP}/\text{bytes}$ 를 어떻게 정의하고, compute-bound vs memory-bound 의 분기는?
- 피크 성능 (peak FLOP/s) 와 메모리 대역폭 (peak GB/s) 이 주어졌을 때, 어느 kernel 이 compute-bound 인지 memory-bound 인지 판단하는 정확한 기준은?
- 같은 연산을 register 에서 수행하는 것과 HBM 에서 수행하는 것의 latency 차이는 정확히 몇 배인가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 성능 측정 및 최적화는 **메모리 계층을 이해하지 않고는 불가능** 합니다. 예를 들어:

- `tensor.sum()` 연산이 GPU 에서 빠른지 느린지는 reduction kernel 이 shared memory 를 얼마나 효율적으로 사용하는지에 달려 있습니다.
- Nsight Compute 에서 "achieved memory throughput" 을 봐도, 그것이 peak bandwidth 의 몇 %인지 아는 것 자체가 병목 진단의 핵심입니다.
- custom kernel 을 작성할 때, shared memory 를 쓸지 말지 결정하는 것은 latency/bandwidth/capacity 의 trade-off 이며, roofline model 은 이 결정을 정량화합니다.

더 깊이, HBM 과 L2 캐시의 대역폭 차이 (50× 이상), shared memory 의 bank conflict 구조 (Ch4-04) 는 모두 여기서 비롯됩니다.

---

## 📐 수학적 선행 조건

- 부동소수점 연산 (FLOP, floating point operation)
- 메모리 대역폭 개념 (bytes/second)
- Log scale, 선형 함수 (roofline 그래프)
- (선택) 캐시 이론 (locality, working set)

---

## 📖 직관적 이해

### Memory Hierarchy 의 계층

```
┌─────────────────────────────────────────────────────┐
│  Latency (cycle)  │  Bandwidth      │  Capacity     │
├────────────────────────────────────────────────────┤
│  Register                                           │
│  1 cycle          │  ~3000 GB/s     │  ~256 KB      │
│  (per thread)     │  (이론)          │  (per 32 thread)│
│                                                     │
├────────────────────────────────────────────────────┤
│  Shared Memory (per block)                         │
│  4-10 cycle       │  ~1000 GB/s     │  48-100 KB    │
│  (depends on bank)│  (with collision)│  (per SM)     │
│                                                     │
├────────────────────────────────────────────────────┤
│  L1 Cache (per SM)                                 │
│  ~30 cycle        │  ~500 GB/s      │  32-64 KB     │
│  (worst case)     │  (miss → L2)     │  (per SM)     │
│                                                     │
├────────────────────────────────────────────────────┤
│  L2 Cache (shared)                                 │
│  ~100-200 cycle   │  ~100-200 GB/s  │  40-80 MB     │
│  (L2 hit)         │  (vs HBM 100:1) │  (all SM)     │
│                                                     │
├────────────────────────────────────────────────────┤
│  HBM (Global Memory)                               │
│  ~200-400 cycle   │  1-3 TB/s       │  24-80 GB     │
│  (cold access)    │  (peak, rare)    │  (all SM)     │
│                                                     │
│  A100: 2 TB/s, H100: 3 TB/s                       │
└─────────────────────────────────────────────────────┘
```

### Roofline Model 의 직관

일반적으로:

$$\text{성능} = \min(\text{Peak FLOP/s}, \text{Peak BW} \times I^*)$$

여기서 $I^* = \text{Peak FLOP/s} / \text{Peak BW}$ 는 **roofline 의 전환점** (arithmetic intensity).

```
         Performance (FLOP/s)
         ↑
         │
  Peak   │  ╱────────────── Compute-bound region
  FLOP/s │ ╱
         │╱
         │        (I*, Peak FLOP/s) 교점
         │
         │    Memory-bound region
         │   (선형, BW × I)
         │
         └─────────────────────→ Arithmetic Intensity I
                    I*
```

- **I < I***: memory-bound. 성능은 메모리 BW 에 의해 제한
- **I > I***: compute-bound. 성능은 연산 처리량에 의해 제한

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Memory Hierarchy 의 계층

각 계층 $L \in \{\text{Register}, \text{Shared}, \text{L1}, \text{L2}, \text{HBM}\}$ 에 대해:

$$\text{Latency}(L) = \text{cycle} \text{ to access}$$
$$\text{Bandwidth}(L) = \text{bytes/sec}$$
$$\text{Capacity}(L) = \text{bytes}$$

### 정의 2.2 — Arithmetic Intensity

임의의 kernel 에 대해, 수행된 연산과 메모리 접근의 비율:

$$I = \frac{\text{FLOPs performed}}{\text{Bytes transferred to/from main memory (HBM)}}$$

단위: FLOP/byte.

### 정의 2.3 — Roofline Model

두 constraint 의 최소값:

$$\text{Achieved throughput} = \min\left(\frac{\text{Peak FLOP/s}}{1}, \text{Peak BW} \times I\right)$$

**전환점 (knee point)**:

$$I^* = \frac{\text{Peak FLOP/s}}{\text{Peak BW}}$$

단위: FLOP/byte.

### 정의 2.4 — Memory Bandwidth 효율

실제 달성 대역폭과 이론 대역폭의 비:

$$\eta_{\text{BW}} = \frac{\text{Achieved throughput (bytes/s)}}{\text{Peak throughput (bytes/s)}}$$

일반적으로 0–100% 범위. Coalesced access = 85–95%, uncoalesced = 5–10% (Ch4-03).

---

## 🔬 정리와 증명

### 정리 2.1 — Memory Latency Hiding 과 Roofline 의 관계

Global memory access 의 latency $L$ (cycle) 을 hide 하려면, 충분한 arithmetic intensity 가 필요합니다:

$$I > \frac{L}{P}$$

여기서 $P$ = instruction throughput (cycle/instruction, 보통 1–4).

**증명**: 한 memory request 의 latency 를 fully hide 하려면, 그 동안 다른 warp 들이 $L$ 개의 독립적인 연산을 수행해야 합니다. 각 연산이 1 byte 의 데이터를 필요로 한다면,

$$I > \frac{L \text{ FLOP}}{1 \text{ byte}} = L \, \text{ FLOP/byte}$$

$L \approx 400$ cycle (HBM cold access) 이면, $I > 400$ FLOP/byte 가 필요. 대부분의 kernel 은 $I < 100$ 이므로, **HBM latency 를 완전히 hide 하기는 어려움** $\square$.

### 정리 2.2 — Roofline 의 이분 성질

주어진 kernel 의 arithmetic intensity $I$ 가 $I^*$ 와 비교했을 때:

1. **$I < I^*$ (Memory-bound)**:
   $$\text{성능} = \text{Peak BW} \times I$$

2. **$I > I^*$ (Compute-bound)**:
   $$\text{성능} = \text{Peak FLOP/s}$$

**증명**: 첫 번째, $I < I^*$ 일 때:
$$\text{Peak BW} \times I < \text{Peak BW} \times I^* = \text{Peak FLOP/s}$$

따라서 메모리가 bottleneck.

두 번째, $I > I^*$ 일 때:
$$\text{Peak BW} \times I > \text{Peak FLOP/s}$$

연산 처리량이 bottleneck $\square$.

### 정리 2.3 — Shared Memory 의 Latency Benefit

Shared memory 를 사용하여 데이터를 HBM 에서 한 번 로드하고 여러 번 재사용하면, effective arithmetic intensity 가 증가합니다:

$$I_{\text{eff}} = \frac{\text{FLOPs}}{\text{Bytes from HBM}} > I_{\text{original}}$$

**증명**: 예를 들어, shared memory tiling 으로 tile size $T \times T$ 를 HBM 에서 로드 ($T^2$ bytes) 하고, $T^2$ 개의 점적을 계산 ($2T^2$ FLOP) 하면:

$$I_{\text{eff}} = \frac{2T^2}{T^2} = 2 \, \text{ FLOP/byte}$$

하지만 나이브 구현은:

$$I_{\text{original}} = \frac{2T^2}{2T^2} = 1 \, \text{ FLOP/byte}$$

따라서 shared memory tiling 으로 **arithmetic intensity 2배 향상** $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Roofline Model 그래프 그리기

```python
import numpy as np
import matplotlib.pyplot as plt

# H100 기준
peak_flops = 141e12  # 141 TFLOP/s (tensor core, FP32)
peak_bw = 3e12       # 3 TB/s

# Roofline knee point
I_star = peak_flops / peak_bw  # FLOP/byte

print(f"Peak FLOPS: {peak_flops/1e12:.1f} TFLOP/s")
print(f"Peak BW: {peak_bw/1e12:.1f} TB/s")
print(f"I*: {I_star:.1f} FLOP/byte")

# Arithmetic intensity range
I = np.logspace(-1, 3, 100)  # 0.1 ~ 1000 FLOP/byte

# Roofline: min(peak_flops, peak_bw * I)
performance = np.minimum(peak_flops, peak_bw * I)

plt.figure(figsize=(10, 6))
plt.loglog(I, performance / 1e12, 'b-', linewidth=2, label='Roofline')
plt.axvline(I_star, color='r', linestyle='--', alpha=0.5, label=f'I* = {I_star:.1f}')
plt.axhline(peak_flops / 1e12, color='g', linestyle='--', alpha=0.5)
plt.axhline(peak_bw * I_star / 1e12, color='g', linestyle='--', alpha=0.5)

# 일반적인 kernel들의 위치
kernels = {
    'SAXPY (y += a*x)': (1/4, 141),      # 1 FLOP per 4 bytes (load x, y, store y)
    'Matrix mult (dense)': (100, 141),   # 매우 높은 arithmetic intensity
    'Reduction': (5, 100),               # 중간 정도
    'Gather/scatter': (0.5, 50),         # 메모리 바운드
}

for name, (I_val, perf) in kernels.items():
    perf_actual = min(peak_flops, peak_bw * I_val) / 1e12
    plt.scatter(I_val, perf_actual, s=100, alpha=0.6)
    plt.annotate(name, (I_val, perf_actual), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
plt.ylabel('Performance (TFLOP/s)', fontsize=12)
plt.title('H100 Roofline Model', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('h100_roofline.png', dpi=150)
plt.close()

print("Roofline plot saved to h100_roofline.png")
```

### 실험 2 — Memory Hierarchy Latency/Bandwidth 측정

```python
import torch
import time

def measure_bandwidth(size_mb, num_iter=100, location='hbm'):
    """
    Measure memory bandwidth by doing simple copy operations.
    """
    size_bytes = size_mb * 1024 * 1024
    num_floats = size_bytes // 4
    
    x = torch.randn(num_floats, device='cuda')
    y = torch.empty_like(x)
    
    # Warm up
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    
    # Timed copy
    start = time.perf_counter()
    for _ in range(num_iter):
        y.copy_(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Bandwidth = bytes transferred / time
    # Each copy is 2 * size_bytes (read x, write y)
    bandwidth_gb_s = (2 * num_iter * size_mb) / elapsed
    return bandwidth_gb_s

# Test different sizes (rough proxy for memory hierarchy)
print("Memory Bandwidth Measurement:")
print("-" * 50)
for size_mb in [1, 4, 16, 64, 256]:
    bw = measure_bandwidth(size_mb, num_iter=100)
    print(f"Size {size_mb:3d} MB: {bw:7.1f} GB/s")

# Expected: smaller sizes fit in cache (slower due to overhead)
# larger sizes use HBM (saturate at ~2-3 TB/s on H100)
```

### 실험 3 — Roofline Analysis of Custom Kernel

```python
import torch
import torch.nn.functional as F

def measure_arithmetic_intensity(kernel_name, input_size, flops, bytes_accessed):
    """
    Measure actual kernel performance and compare to roofline.
    """
    x = torch.randn(input_size, device='cuda')
    
    # Warm up
    for _ in range(10):
        if kernel_name == 'relu':
            y = F.relu(x)
        elif kernel_name == 'softmax':
            y = F.softmax(x, dim=-1)
        elif kernel_name == 'linear':
            w = torch.randn(input_size, input_size, device='cuda')
            y = torch.matmul(x.unsqueeze(0), w)
    torch.cuda.synchronize()
    
    # Timed execution
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    num_iter = 100
    start.record()
    for _ in range(num_iter):
        if kernel_name == 'relu':
            y = F.relu(x)
        elif kernel_name == 'softmax':
            y = F.softmax(x, dim=-1)
        elif kernel_name == 'linear':
            w = torch.randn(input_size, input_size, device='cuda')
            y = torch.matmul(x.unsqueeze(0), w)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000 / num_iter
    
    achieved_flops = flops / elapsed_s
    achieved_bw = bytes_accessed / elapsed_s
    arithmetic_intensity = flops / bytes_accessed
    
    print(f"\n{kernel_name} (size {input_size}):")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.2f} FLOP/byte")
    print(f"  Achieved FLOP/s: {achieved_flops/1e9:.1f} GFLOP/s")
    print(f"  Achieved BW: {achieved_bw/1e9:.1f} GB/s")
    
    # Compare to roofline
    peak_flops = 141e12
    peak_bw = 3e12
    I_star = peak_flops / peak_bw
    
    if arithmetic_intensity < I_star:
        print(f"  → Memory-bound (I < I* = {I_star:.1f})")
    else:
        print(f"  → Compute-bound (I > I* = {I_star:.1f})")

# Test various kernels
measure_arithmetic_intensity('relu', 10000000, 10000000, 40000000)  # read + write
measure_arithmetic_intensity('softmax', 10000000, 40000000, 40000000)  # more computation
```

---

## 🔗 실전 활용

### 1. Nsight Compute 에서 Roofline 생성

```bash
nv-nsight-compute -o report.ncu-rep <binary>
# Viewer 에서 "Roofline" 차트 확인 — actual kernel 이 어디에 위치하는지 시각화
```

### 2. PyTorch Profiler 로 Memory Statistics 확인

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)

print(prof.key_averages().table(sort_by='cuda_memory_usage', row_limit=10))
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Roofline = 이상적 performance bound | Actual kernel 은 instruction cache miss, register spilling, synchronization overhead 등으로 더 낮음 |
| Peak BW = 이론 최대값 | Actual BW = peak × efficiency (coalescing, cache hit 등) |
| Arithmetic intensity = HBM 바이트만 계산 | Shared mem 재사용 시 다른 계층의 bandwidth 고려 필요 |
| 선형 roofline = 메모리 바운드 영역 | 실제로는 prefetching, out-of-order execution 등으로 편차 |

---

## 📌 핵심 정리

$$\boxed{I^* = \frac{\text{Peak FLOP/s}}{\text{Peak BW}}, \quad \text{Performance} = \min(\text{Peak FLOP/s}, \text{Peak BW} \times I)}$$

| 메모리 | Latency | Bandwidth | Capacity |
|--------|---------|-----------|----------|
| Register | 1 cycle | ~3 TB/s | ~256 KB |
| Shared | 4–10 cycle | ~1 TB/s | 48–100 KB |
| L1/L2 | 30–200 cycle | 100–500 GB/s | 32 KB / 40 MB |
| HBM | 200–400 cycle | 2–3 TB/s | 24–80 GB |

---

## 🤔 생각해볼 문제

**문제 1** (기초): H100 의 roofline knee point 를 계산하라 (peak FLOPS = 141 TFLOP/s, peak BW = 3 TB/s). 일반적인 elementwise operation (SAXPY: y += a*x) 의 arithmetic intensity 는? 이 operation 은 compute-bound 인지 memory-bound 인지?

<details>
<summary>해설</summary>

$$I^* = \frac{141 \times 10^{12}}{3 \times 10^{12}} = 47 \, \text{FLOP/byte}$$

SAXPY: 각 element 에 대해 1 FLOP (multiply-add) 을 수행하고, 3 번 메모리 접근 (load a, load x, load y, write y). 따라서:

$$I = \frac{1}{4} = 0.25 \, \text{FLOP/byte}$$

$0.25 < 47$ 이므로 **memory-bound**. 성능은 다음과 같이 제한:

$$\text{Performance} = 3 \times 10^{12} \times 0.25 = 750 \times 10^9 = 750 \, \text{GFLOP/s}$$

(peak 141 TFLOP/s 의 0.5% 에 불과) $\square$

</details>

**문제 2** (심화): 공유 메모리 를 이용한 tiling 에서, $T \times T$ tile 을 HBM 에서 로드하고 $T$ 번 재사용한다면 arithmetic intensity 는 몇 배 향상되는가? 이것이 $T$ 의 함수로서 scalable 한가?

<details>
<summary>해설</summary>

원래:
$$I_0 = \frac{\text{FLOP}}{\text{Bytes from HBM}}$$

Tiling 으로 $T$ 회 재사용:
$$I_T = \frac{T \times \text{FLOP}}{\text{Bytes from HBM}} = T \times I_0$$

즉, **$T$ 배 향상**. 하지만 제약:
- Shared memory 크기: $T^2 \times 4$ bytes (FP32) $\leq 48$ KB
  - H100: $T \leq 110$ (대략)
- Register pressure: $T$ 가 크면 spilling → HBM 접근 증가

따라서 $T \lesssim 100$ 범위에서만 scalable. Beyond this, diminishing returns $\square$.

</details>

**문제 3** (논문 비평): 일부 최근 GPU (H100+) 에서 **L2 cache prefetching** 을 도입했다고 한다 (NVIDIA H100 datasheet). 이것이 memory-bound kernel 의 roofline 을 어떻게 바꾸는가? Roofline model 은 여전히 유효한가?

<details>
<summary>해설</summary>

L2 prefetching = 다음 iteration 의 메모리 접근을 미리 L2 로 가져옴. 결과:
1. **Effective latency 감소**: HBM 접근이 L2 cache hit 으로 변환 → latency ~100 cycle 줄어듦
2. **Effective BW 증가**: L2 hit 은 HBM miss 보다 빠름

따라서:
- **Old roofline** (prefetching 없음): 메모리 바운드 영역 선형
- **New roofline** (prefetching 있음): 메모리 바운드 영역 steeper (더 높은 BW)

Roofline 은 여전히 유효하지만, **prefetching 을 활성화한 버전으로 재측정 필요**. 실제로 Nsight Compute 의 "with prefetching" 옵션이 이것을 반영 $\square$.

</details>

---

<div align="center">

[◀ 이전](./01-gpu-architecture.md) | [📚 README](../README.md) | [다음 ▶](./03-memory-coalescing.md)

</div>
