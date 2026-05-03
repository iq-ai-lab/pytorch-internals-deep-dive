# 05. Device 와 CUDA Context

## 🎯 핵심 질문

- Device 란 무엇인가? (`cpu`, `cuda:0`, `mps`, `xla`) 의 차이는?
- CUDA Stream 이란 무엇이고, 왜 async execution 이 중요한가?
- `torch.cuda.synchronize()` 는 언제 호출하고, 비용은?
- `tensor.to('cuda')` 가 내부적으로 수행하는 H2D copy 메커니즘은?
- Pinned memory (`tensor.pin_memory()`) 가 H2D transfer 를 왜 2-3배 가속하는가?

---

## 🔍 왜 Device Abstraction 이 PyTorch 사용자에게 중요한가

PyTorch 는 CPU 와 GPU 를 **동일한 API** 로 추상화합니다. 대부분의 코드는 `.to('cpu')` 또는 `.to('cuda')` 한 줄로 device 를 전환합니다. 그러나 이 안에 숨어 있는 메커니즘을 모르면:

1. **성능 병목**: Host-to-Device (H2D) memory transfer 가 느린 이유 (PCIe bandwidth limited)
2. **Async 버그**: CUDA kernel 이 비동기 실행되는데 `.cpu()` 호출 전 synchronize 하지 않아 invalid memory access
3. **Multi-GPU 복잡성**: 여러 GPU 에서 실행할 때 각각 다른 CUDA context 를 가져야 함
4. **Stream overlap 최적화**: A100 에서 H2D copy 와 computation 을 동시에 실행해 성능 2배 향상 가능

이 구조를 모르면 "왜 GPU 로 옮기는데 시간이 오래 걸려?" 또는 "왜 multi-GPU 가 느려?" 하는 문제를 분석 불가능합니다.

---

## 📐 수학적 선행 조건

- **Computer Architecture**: PCIe topology, memory hierarchy
- **Parallel Computing**: CUDA stream, GPU command queue, synchronization barrier
- **Ch1-01**: Tensor metadata (device 포함)

---

## 📖 직관적 이해

### Device 종류와 메모리 배치

```
Logical view:
┌──────────────┐
│ CPU Memory   │ (host)
│ (virtual)    │
└──────────────┘
       ↕ (PCIe transfer)
┌──────────────┐
│ GPU Memory   │ (device)
│ (HBM)        │
└──────────────┘

PyTorch tensor:
- device='cpu': host DRAM 에 할당
- device='cuda:0': GPU 0 의 HBM 에 할당
```

### CUDA Stream 과 Async Execution

**Sequential (eager) execution**:
```
Host:     Call kernel_A → Wait → Call kernel_B → Wait
GPU:      [kernel_A] (10ms) → [kernel_B] (5ms) → idle
Total:    15ms
```

**Async with stream**:
```
Host:     Call kernel_A (return immediately)
          Call kernel_B (return immediately)
GPU:      [kernel_A] (10ms) → [kernel_B] (5ms)
Total:    15ms (computation overlap 가능)
          BUT: H2D copy 와 kernel 을 overlap 하면 
          20ms + 5ms = 25ms instead of 20+5=25ms sequential...
```

**H2D copy + Kernel overlap**:
```
Stream 0: [H2D copy: 10ms]
Stream 1:              [Kernel] (5ms)
Total:    15ms (copy 와 kernel 동시 실행)
```

### Pinned Memory 의 역할

```
Pageable memory (default):
Host → PCIe → GPU
CPU 가 IOMMU 를 통해 간접 access → slow

Pinned memory:
Host (locked page) → PCIe → GPU
직접 DMA (Direct Memory Access) → fast (2-3배)
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Device

PyTorch 의 **device** 는 tensor memory 가 할당되는 실제 위치를 지정:

$$\text{device} \in \{\text{cpu}, \text{cuda:i}, \text{mps}, \text{xla:i}, \ldots\}$$

각 device 는:
- **Memory pool**: 독립적 할당 구간
- **CUDA context** (GPU only): device-specific state
- **Stream queue**: async op 들의 실행 순서

### 정의 5.2 — CUDA Stream

CUDA stream 은 GPU 에서 연속적으로 실행되는 command 들의 queue:

$$\text{Stream} = \text{queue of } (Kernel, Params, Dependencies)$$

- **Default stream** (stream 0): 동기적, 모든 operation 이 순차 실행
- **Explicit stream**: 비동기, 여러 stream 간 overlap 가능

### 정의 5.3 — H2D Transfer 비용

Tensor `x` (shape, dtype) 를 host 에서 device 로 복사하는 비용:

$$T_{\text{H2D}} = \frac{|\text{numel}(x) \times \text{sizeof(dtype)}|}{B_{\text{PCIe}}}$$

여기서 $B_{\text{PCIe}}$ 는 PCIe bandwidth (보통 16 GB/s for PCIe 4.0 x16).

### 정의 5.4 — Pinned Memory

Tensor 의 host memory 가 **page-locked** (OS swap 불가):

$$\text{pinned}(x) \iff x \text{ 는 pinned page pool 에 할당됨}$$

결과: direct DMA access 가능 → H2D transfer 2-3배 fast.

---

## 🔬 정리와 증명

### 정리 5.1 — Synchronize 의 필요조건

Tensor `x` 를 GPU 에서 Host 로 전달하기 전 (`.cpu()` 호출):

$$\text{필요}: \text{torch.cuda.synchronize()}$$

**증명**:
GPU 는 host-issued command 를 비동기로 실행. `.cpu()` 호출 시 x 의 수정이 아직 GPU 메모리에서 완료되지 않았을 수 있음 → invalid memory access. Synchronize 는 host 를 모든 GPU operation 이 완료될 때까지 block $\square$

### 정리 5.2 — PCIe Transfer vs Computation

H2D transfer time $T_H$ 와 kernel execution time $T_K$ 에 대해, 두 stream 을 사용하면 total time:

$$T_{\text{parallel}} = \max(T_H, T_K) \quad (\approx T_H + T_K \text{ if overlappable})$$

**증명**:
H2D (stream 0) 와 kernel (stream 1) 이 독립적이면 GPU scheduler 가 동시 실행. Total time 은 두 operation 의 max (또는 dependency 있으면 합) $\square$

### 정리 5.3 — Pinned Memory 의 Bandwidth 향상

Pinned memory H2D transfer 의 bandwidth $B_{\text{pinned}}$ 와 pageable memory 의 $B_{\text{paged}}$:

$$\frac{B_{\text{pinned}}}{B_{\text{paged}}} \approx 2 \text{-} 3$$

**증명**:
- Pageable: IOMMU 를 통한 indirect translation → latency overhead
- Pinned: direct DMA → IOMMU bypass $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Device 종류 확인

```python
import torch

# 사용 가능한 device 확인
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

# Device 생성
x_cpu = torch.randn(1000, 1000)
print(f"x_cpu device: {x_cpu.device}")

if torch.cuda.is_available():
    x_cuda = torch.randn(1000, 1000, device='cuda')
    print(f"x_cuda device: {x_cuda.device}")
    print(f"x_cuda is_cuda: {x_cuda.is_cuda}")
    
    # Device 이동
    x_cuda2 = x_cpu.to('cuda')
    print(f"x_cpu → cuda: {x_cuda2.device}")
```

**예상 출력** (GPU 있을 때):
```
CUDA available: True
Device count: 1
Current device: 0
x_cpu device: cpu
x_cuda device: cuda:0
x_cuda is_cuda: True
x_cpu → cuda: cuda:0
```

### 실험 2 — H2D Transfer 비용 측정

```python
import torch
import time

if torch.cuda.is_available():
    # 다양한 크기의 tensor
    sizes = [10**6, 10**7, 10**8]
    
    for n in sizes:
        x_cpu = torch.randn(n, dtype=torch.float32)
        
        # Warmup
        _ = x_cpu.to('cuda')
        torch.cuda.synchronize()
        
        # H2D transfer 측정
        start = time.time()
        x_cuda = x_cpu.to('cuda')
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Bandwidth 계산
        bytes_transferred = n * 4  # float32 = 4 bytes
        bandwidth = bytes_transferred / elapsed / 1e9  # GB/s
        
        print(f"Size {n}: {elapsed*1000:.2f} ms, Bandwidth {bandwidth:.1f} GB/s")
```

**예상 출력** (PCIe 4.0 x16, ~16 GB/s):
```
Size 1000000: 0.10 ms, Bandwidth 40.0 GB/s (overhead 큼)
Size 10000000: 0.51 ms, Bandwidth 15.6 GB/s (more realistic)
Size 100000000: 5.12 ms, Bandwidth 15.6 GB/s
```

### 실험 3 — Synchronize 의 필요성

```python
import torch
import time

if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    
    # Async operation
    start = time.time()
    y = x.sum()  # GPU kernel (async)
    elapsed_no_sync = time.time() - start
    print(f"Without synchronize: {elapsed_no_sync*1000:.4f} ms")
    
    # With synchronize
    start = time.time()
    y = x.sum()  # GPU kernel
    torch.cuda.synchronize()  # wait for GPU
    elapsed_with_sync = time.time() - start
    print(f"With synchronize: {elapsed_with_sync*1000:.4f} ms")
    
    # synchronize 는 훨씬 느림 (실제 computation time)
    print(f"Synchronize overhead: {(elapsed_with_sync - elapsed_no_sync)*1000:.4f} ms")
```

**예상 출력**:
```
Without synchronize: 0.0100 ms (거의 no-op)
With synchronize: 0.5000 ms (actual computation)
Synchronize overhead: 0.4900 ms
```

### 실험 4 — Stream 을 사용한 Async Execution

```python
import torch
import time

if torch.cuda.is_available():
    # Case 1: Sequential (default stream)
    x = torch.randn(1000, 1000, device='cuda')
    
    start = time.time()
    y = x.sum()
    z = x.mean()
    torch.cuda.synchronize()
    seq_time = time.time() - start
    
    # Case 2: Async with streams (이론적으로)
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    start = time.time()
    with torch.cuda.stream(stream1):
        y = x.sum()
    with torch.cuda.stream(stream2):
        z = x.mean()
    torch.cuda.synchronize()
    async_time = time.time() - start
    
    print(f"Sequential: {seq_time*1000:.4f} ms")
    print(f"Async (with streams): {async_time*1000:.4f} ms")
    print(f"두 operation 이 overlap 하면 async 가 빠름")
```

**예상 출력**:
```
Sequential: 0.5000 ms
Async (with streams): 0.5500 ms (또는 비슷, 단순 op 는 overlap 어려움)
```

### 실험 5 — Pinned Memory 효과

```python
import torch
import time

if torch.cuda.is_available():
    n = 10**8  # 400 MB
    
    # Case 1: Pageable memory
    x_paged = torch.randn(n, dtype=torch.float32)
    torch.cuda.synchronize()
    start = time.time()
    x_cuda_paged = x_paged.to('cuda')
    torch.cuda.synchronize()
    time_paged = time.time() - start
    
    # Case 2: Pinned memory
    x_pinned = torch.randn(n, dtype=torch.float32).pin_memory()
    torch.cuda.synchronize()
    start = time.time()
    x_cuda_pinned = x_pinned.to('cuda')
    torch.cuda.synchronize()
    time_pinned = time.time() - start
    
    print(f"Pageable memory: {time_paged*1000:.2f} ms")
    print(f"Pinned memory: {time_pinned*1000:.2f} ms")
    print(f"Speedup: {time_paged/time_pinned:.2f}x")
```

**예상 출력** (GPU 에 따라 다름):
```
Pageable memory: 15.00 ms
Pinned memory: 5.00 ms
Speedup: 3.00x
```

---

## 🔗 실전 활용

### 1. Multi-GPU 에서 Device 명시

```python
import torch

# GPU 여러 개 사용
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x = torch.randn(1000, 1000, device=device)
model = MyModel().to(device)

# 또는
x = torch.randn(1000, 1000)
x = x.to(device)
```

### 2. H2D Transfer 최적화

```python
import torch
from torch.utils.data import DataLoader

# Pinned memory 사용
dataset = MyDataset()
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True  # Host memory pinning
)

# Async H2D 와 computation 을 overlap
for batch in dataloader:
    batch = batch.to('cuda', non_blocking=True)  # async transfer
    # GPU 가 이전 batch 를 처리하는 동안 H2D 진행
```

### 3. Synchronize 의 명시적 사용

```python
import torch

x = torch.randn(1000, 1000, device='cuda')

# GPU 에서 계산
y = x.sum()

# Host 로 가져오기 전에 동기화 (명시적)
torch.cuda.synchronize()
result = y.item()  # safe now
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Device 간 transfer 는 PCIe bottleneck | NVLink (고가 GPU) 는 더 빠름 |
| Single default stream | 여러 stream 의 관리는 복잡 |
| Synchronize 는 blocking | Non-blocking alternative 는 제한적 |
| Pinned memory 는 항상 좋음 | Host memory 부족 시 문제 (OS swap 불가) |

---

## 📌 핵심 정리

$$\boxed{T_{\text{H2D}} = \frac{\text{numel} \times \text{sizeof(dtype)}}{B_{\text{PCIe}}}}$$

| 개념 | 설명 | 특징 |
|------|------|------|
| device | Tensor memory 위치 (cpu/cuda/...) | metadata 의 일부 |
| CUDA Stream | GPU command queue | async execution 가능 |
| H2D transfer | Host → Device memory copy | PCIe bandwidth limited |
| Pinned memory | Page-locked host memory | 2-3배 빠른 transfer |
| Synchronize | Host-device sync barrier | blocking, 필요할 때만 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): `torch.cuda.synchronize()` 를 호출하지 않고 `.cpu()` 하면 어떻게 되는가?

<details>
<summary>해설</summary>

`.cpu()` 호출 시 host 는 GPU 가 메모리 수정을 완료했다고 가정. 그러나 GPU kernel 이 아직 실행 중이면, host 가 불완전한 데이터를 읽을 수 있음 (race condition) → undefined behavior / crash.

따라서 synchronize 후 `.cpu()` 가 안전 $\square$

</details>

**문제 2** (심화): 크기 N 인 tensor 를 H2D 로 옮기는 비용이 16 GB/s PCIe 에서 10ms 라면, N 은 대략 몇인가?

<details>
<summary>해설</summary>

$$T = \frac{N \times 4}{16 \times 10^9} = 10 \times 10^{-3}$$

$$N = \frac{10 \times 10^{-3} \times 16 \times 10^9}{4} = 40 \times 10^6$$

따라서 약 4천만개 float32 (160 MB) $\square$

</details>

**문题 3** (논문 비평): NVIDIA 의 CUDA 아키텍처에서 stream 을 여러 개 사용하는 것이 항상 성능을 향상시키는가? Stream scheduling 의 한계는? (NVIDIA CUDA programming guide, ch. 5)

<details>
<summary>해설</summary>

**Stream 의 이점**:
- H2D + kernel overlap 가능
- Multiple independent kernels 동시 실행

**한계**:
1. **Memory bottleneck**: 여러 kernel 이 같은 memory 접근 → serialization
2. **Hardware limits**: Kernel 은 동일 SM 을 점유 → limited concurrency
3. **Scheduling overhead**: Stream context switching 자체가 비용

**현실**:
- A100: 최대 128 concurrent kernel (8 SM 당 1 kernel)
- 대부분의 경우 3-4 stream 만 해도 충분
- Over-subscription 은 성능 저하

(NVIDIA CUDA optimization guide, "Occupancy" section) $\square$

</details>

---

<div align="center">

[◀ 이전](./04-dtype-and-promotion.md) | [📚 README](../README.md) | [다음 ▶](../ch2-autograd/01-forward-vs-reverse-ad.md)

</div>
