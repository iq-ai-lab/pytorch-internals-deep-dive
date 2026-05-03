# 06. Reduction 최적화 — Tree vs Warp Shuffle

## 🎯 핵심 질문

- Mark Harris 의 7-step optimization (2007, GTC) 은 정확히 무엇인가 — naive interleaved → strided → sequential → first add during load → unroll last warp → template BLOCK_SIZE → warp shuffle, 각 단계의 개선은?
- Tree-based reduction 은 언제부터 warp shuffle 로 전환되는가? 마지막 몇 step 에서만 가능한가?
- Warp shuffle (\_\_shfl\_down\_sync) 를 사용했을 때 latency 는 얼마나 줄어드는가 — shared memory bypass 의 이점은?
- Reduction 의 final bandwidth 와 peak bandwidth 의 비율은? Roofline 에서 어디에 위치하는가?
- PyTorch 의 `tensor.sum()` backend 에서 이 최적화들이 어떻게 적용되는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 aggregation operations (`tensor.sum()`, `tensor.mean()`, `tensor.max()`) 는 모두 내부적으로 reduction kernel 을 실행합니다. 이 kernel 의 성능은 위 7-step optimization 의 응용이며, 초기 단계 (coalescing 최적화) 부터 최종 단계 (warp shuffle) 까지 모두 포함됩니다.

더 깊이, custom reduction (예: grouped sum) 을 작성할 때, 같은 최적화 기법을 직접 구현해야 합니다. Nsight Compute 에서 "achieved throughput" 이 peak 에 못 미치는 이유를 진단하려면, 각 step 의 병목을 이해해야 합니다.

---

## 📐 수학적 선행 조건

- Reduction 의 기본 수학: $\sum_{i=0}^{n-1} x_i$ 의 tree decomposition
- Ch4-01 ~ Ch4-05 의 모든 최적화 (coalescing, bank conflict, divergence)
- Binary tree 의 높이와 level

---

## 📖 직관적 이해

### Reduction 의 기본 아이디어

$$\sum_{i=0}^{n-1} x_i = \sum_{j=0}^{n/2-1} (x_{2j} + x_{2j+1})$$

이를 tree 로 표현:

```
Level 0 (n elements):    [x0] [x1] [x2] [x3] ... [xn-1]
                          |    |    |    |         |
Level 1 (n/2 elements):  [+]  [+]  ... [+]
                          |    |         |
Level 2 (n/4 elements):  [+]  ... [+]
                          |         |
...
Level log(n):            [final sum]
```

각 level 은 parallel 하게 실행 가능, 하지만 level 간은 synchronization (\_\_syncthreads) 필수.

### Mark Harris 의 7-Step Optimization

```
Step 1: Interleaved Index
  stride = 1, 2, 4, 8, 16, 32 (shared memory bank conflict!)
  
Step 2: Strided Index (memory coalescing improvement)
  thread i loads from addr[stride], addr[2*stride], ...
  
Step 3: Sequential Addressing
  stride starts large, gets smaller (reverse order)
  → coalesced in early stages
  
Step 4: First Add During Load
  Load + compute + store 를 하나의 kernel loop 으로
  → register reuse, memory access 감소
  
Step 5: Unroll Last Warp
  마지막 stride = 32 에서 warp shuffle 사용
  → shared memory bypass, no synchronization
  
Step 6: Template Reduction
  BLOCK_SIZE 를 template parameter 화
  → compiler가 loop unroll 가능
  
Step 7: Complete Loop Unroll
  모든 iteration 을 inline 전개
  → pipeline efficiency 최대
```

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Reduction 의 Computational Pattern

$n$ 개 원소를 $m$ 개 thread block 으로 reduce:

$$\text{Level } k : \text{each thread computes } 2^k \text{ elements}$$

$$\text{Synchronization at end of each level}$$

Total work: $O(n)$, depth: $O(\log n)$.

### 정의 6.2 — Step 1: Interleaved Index

```cpp
stride = 1;
for (int step = 0; step < log2(BLOCK_SIZE); step++) {
    if (tid + stride < BLOCK_SIZE)
        sdata[tid] += sdata[tid + stride];
    stride *= 2;
    __syncthreads();
}
```

**문제**: stride = 1, 2, 4, ..., 32 일 때:
- stride = 32 → bank conflict ($2^5 = 32$)
- 초기 stride (< 32) → coalescing poor (비연속 memory access)

### 정의 6.3 — Step 3: Sequential Addressing

```cpp
for (int step = BLOCK_SIZE / 2; step > 0; step >>= 1) {
    if (tid < step)
        sdata[tid] += sdata[tid + step];
    __syncthreads();
}
```

**개선**: stride 가 큼 → 우에서 작아짐 → 초기 단계 coalescing 향상.

### 정의 6.4 — Warp Shuffle

Shared memory 대신 register 간 직접 전달 (Volta+):

```cpp
volatile int warpSum = sdata[tid];

// Warp shuffle (no shared memory, no __syncthreads__)
warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 16);
warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 8);
warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 4);
warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 2);
warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 1);

if (tid == 0) atomicAdd(output, warpSum);
```

**이점**:
1. Shared memory bank conflict 제거
2. `__syncthreads()` 불필요 (warp 내부)
3. Register latency < shared memory latency

---

## 🔬 정리와 증명

### 정리 6.1 — Interleaved Index 의 Bank Conflict

stride $s$ 에 대해, thread $i$ 와 thread $i + s$ 가 bank conflict 할 확률:

$$P(\text{conflict}) = \begin{cases} 0 & s < 32 \\ 1 & s = 32 \end{cases}$$

정확히, $s = 32$ 일 때:

$$B_i = (i / 4) \mod 32, \quad B_{i+32} = ((i + 32) / 4) \mod 32 = (i/4) \mod 32 = B_i$$

즉, **32-way conflict** (같은 warp 의 모든 thread 가 같은 bank 에 접근) $\square$.

### 정리 6.2 — Sequential Addressing 의 개선

Sequential addressing (stride 큼 → 작아짐) 에서:

**초기 단계** (stride $s >> 32$): thread i 와 i+s 가 완전히 다른 128-byte cache line → coalesced (1–2 transactions)

**후기 단계** (stride $s < 32$): 다시 같은 shared memory bank 에 접근 가능, 하지만 thread 수가 줄어들어 (if 조건) active thread 가 적음 → divergence 적음

**결과**: 초기 단계 coalescing 향상 + 후기 단계 divergence 감소 → overall throughput 향상 $\square$.

### 정리 6.3 — Warp Shuffle 의 Latency Hiding

Warp shuffle `__shfl_down_sync(mask, var, delta)` 는:

$$\text{Latency} = 1 \text{ cycle (또는 2)}$$

Shared memory reduction (마지막 few step):

$$\text{Latency} \geq 4 \text{ cycles (bank 고려)}$$

따라서 **마지막 $\log_2(32) = 5$ step** 에서 shuffle 사용 시:

$$\text{Latency saved} = 5 \times (4 - 2) = 10 \text{ cycles}$$

Total reduction latency $\approx 100$ cycles 에서 **10% 절감** $\square$.

### 정리 6.4 — Reduction 의 메모리 대역폭

각 level $k$ 에서:

$$\text{Data moved} = n \text{ bytes}$$

(각 element 가 매번 load+add+store)

Total: $\log_2(n)$ level × $n$ bytes = $n \log_2 n$ bytes 전송.

Achieved throughput:

$$\eta = \frac{\text{Peak FLOP/s} \times \text{FLOP per element}}{\text{Bytes moved} \times \text{Clock rate}}$$

Typical: 30–50% of peak bandwidth (coalescing loss, synchronization overhead).

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — 7-Step Optimization 의 성능 비교

```cpp
// reduction_7steps.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// Step 1: Interleaved Index
__global__ void reduce_interleaved(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        if (tid + stride < BLOCK_SIZE)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// Step 3: Sequential Addressing
__global__ void reduce_sequential(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// Step 5: Unroll Last Warp + Shuffle
__global__ void reduce_warp_shuffle(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Shared memory reduction (stride > 32)
    for (int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    
    // Load into register
    float warpSum = sdata[tid];
    
    // Warp shuffle (stride <= 32)
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 16);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 8);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 4);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 2);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 1);
    
    if (tid == 0) atomicAdd(output, warpSum);
}

int main() {
    int n = 100000000;  // 100M floats
    float *input, *output;
    cudaMalloc(&input, n * sizeof(float));
    cudaMalloc(&output, sizeof(float));
    
    // Initialize
    cudaMemset(output, 0, sizeof(float));
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float elapsed;
    
    // Benchmark each version
    cudaEventRecord(start);
    reduce_interleaved<<<grid_size, BLOCK_SIZE>>>(input, output, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Interleaved: %.3f ms\n", elapsed);
    
    cudaMemset(output, 0, sizeof(float));
    
    cudaEventRecord(start);
    reduce_sequential<<<grid_size, BLOCK_SIZE>>>(input, output, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Sequential: %.3f ms\n", elapsed);
    
    cudaMemset(output, 0, sizeof(float));
    
    cudaEventRecord(start);
    reduce_warp_shuffle<<<grid_size, BLOCK_SIZE>>>(input, output, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Warp shuffle: %.3f ms\n", elapsed);
    
    cudaFree(input);
    cudaFree(output);
    return 0;
}
```

**예상 출력**:
```
Interleaved: 45.2 ms (baseline, bank conflicts)
Sequential: 28.1 ms  (coalescing + divergence improvement)
Warp shuffle: 24.3 ms (shuffle replaces last warp sync)
```

### 실험 2 — Template BLOCK_SIZE 의 효과

```cpp
template <int BLOCK_SIZE>
__global__ void reduce_template(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Compiler can unroll this loop if BLOCK_SIZE is compile-time constant
    for (int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    
    float warpSum = sdata[tid];
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 16);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 8);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 4);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 2);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 1);
    
    if (tid == 0) atomicAdd(output, warpSum);
}

// Instantiate for different block sizes
template __global__ void reduce_template<64>(float*, float*, int);
template __global__ void reduce_template<128>(float*, float*, int);
template __global__ void reduce_template<256>(float*, float*, int);
template __global__ void reduce_template<512>(float*, float*, int);
```

### 실험 3 — PyTorch 의 내장 reduction

```python
import torch
import time

def benchmark_sum(x, dim):
    """Benchmark PyTorch's built-in sum."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(100):
        result = x.sum(dim=dim)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / 100

# Test different tensor shapes and reduce dimensions
shape = (10000, 1000, 100)
x = torch.randn(*shape, device='cuda')

print("Reduction performance (PyTorch backend):")
print("-" * 50)
print(f"Reduce dim 0 (coalesced): {benchmark_sum(x, 0)*1000:.3f} ms")
print(f"Reduce dim 1 (strided):   {benchmark_sum(x, 1)*1000:.3f} ms")
print(f"Reduce dim 2 (scattered): {benchmark_sum(x, 2)*1000:.3f} ms")

# PyTorch uses optimized CUDA kernel (likely similar to 7-step optimization)
```

---

## 🔗 실전 활용

### 1. Custom Reduction Kernel 작성 Template

```cpp
template <int BLOCK_SIZE>
__global__ void custom_reduce(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load and first add
    float value = 0.0f;
    if (idx < n) value = input[idx];
    if (idx + gridDim.x * BLOCK_SIZE < n)
        value += input[idx + gridDim.x * BLOCK_SIZE];
    
    sdata[tid] = value;
    __syncthreads();
    
    // Shared memory reduction
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads();
    }
    
    // Warp shuffle for last 32
    float warpSum = sdata[tid];
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 32);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 16);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 8);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 4);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 2);
    warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, 1);
    
    if (tid == 0) atomicAdd(output, warpSum);
}
```

### 2. Nsight Compute 에서 성능 분석

```bash
nv-nsight-compute -o report.ncu-rep ./reduce_app
# Report: Memory, SM Efficiency, Warp Utilization 등 확인
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Tree-based reduction 이 optimal | Multi-GPU reduction (gradient all-reduce) 는 다른 알고리즘 (ring, butterfly) 사용 |
| Warp shuffle = free | Register live range 증가 → occupancy 감소 가능 |
| Atomicadd 무시 | Global memory atomic 의 serialization, HBM contention |
| Block 내만 고려 | Grid-level reduction 시 여러 block 의 결과 합치기 필요 (2-phase) |

---

## 📌 핵심 정리

$$\boxed{\text{Step 3 (Sequential)} : 2\times \text{ faster}, \quad \text{Step 5 (Shuffle)} : 3-4\times \text{ faster than naive}}$$

| Step | Optimization | Speedup | Notes |
|------|--------------|---------|-------|
| 1 | Interleaved | 1.0x | Bank conflict |
| 2 | Strided | 1.3x | Coalescing improve |
| 3 | Sequential | 2.0x | Stride order reversed |
| 4 | First add load | 2.5x | Load fusion |
| 5 | Unroll warp | 3.2x | Shuffle, no sync |
| 6 | Template | 3.5x | Compiler unroll |
| 7 | Full unroll | 3.8x | Ultimate optimization |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 1024 개 원소를 block size 256 으로 reduce 할 때, tree depth 는 몇인가? 각 level 의 active thread 수는?

<details>
<summary>해설</summary>

$$\log_2(256) = 8 \text{ levels}$$

$$\text{Level } k : 2^k \text{ thread 에서 시작, } 2^k \text{ thread 진행, output } 2^{k-1} \text{ element}$$

총 $256 + 128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 511$ thread iterations $\square$.

</details>

**문제 2** (심화): Warp shuffle 은 step 몇부터 사용하는 것이 최적인가? 즉, stride 가 언제부터 shuffle 로 바꾸면 좋은가? 근거는?

<details>
<summary>해설</summary>

Warp shuffle 은 within-warp (32 thread 이내) 에서만 작동.

따라서 stride $\leq 16$ 부터 (last 5 steps = log_2(32)) shuffle 사용 권장.

**이유**:
1. stride < 32 = warp 내부 only
2. Bank conflict 가능 (stride >= 32)
3. Shuffle latency < shared mem latency

**결론**: stride 부터 = 32 에서 16, 8, 4, 2, 1 로 내려갈 때 shuffle 시작 $\square$.

</details>

**문제 3** (논문 비평): CUB library (NVIDIA CUDA Utility library) 의 block-level reduction 은 "cooperative_groups" 를 사용한다고 한다. 이것이 Harris 의 7-step optimization 을 어떻게 확장하는가? Modern GPU (SM > 8.0) 의 추가 최적화는?

<details>
<summary>해설</summary>

**Cooperative Groups (CUDA 9.0+)**:
- Abstraction for flexible synchronization patterns
- Sub-warp synchronization + warp groups
- Cross-block synchronization (limited)

**Application to reduction**:
- Multiple warp-level reduction (각 warp 는 independent)
- 그 후 warp 간 synchronization
- Shuffle 뿐 아니라 "warp aggregation" 가능

**Modern GPU (Ampere/H100+)**:
- **Tensor memory accelerator** (TMA) — asynchronous memcpy
- **Flexible scheduling** — Independent thread scheduling (Volta+)
- **Strand executor** — 더 정교한 scheduling

따라서 future reduction = cooperative_groups + TMA 조합 $\square$.

</details>

---

<div align="center">

[◀ 이전](./05-warp-divergence.md) | [📚 README](../README.md) | [다음 ▶](../ch5-custom-kernel/01-cpp-extension.md)

</div>
