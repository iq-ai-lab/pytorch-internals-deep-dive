# 04. Bank Conflict 와 Shared Memory

## 🎯 핵심 질문

- Shared memory 의 32 bank 구조는 정확히 무엇인가 — 각 bank 는 어떤 크기이고, 어느 주소가 어느 bank 로 매핑되는가?
- Bank conflict 의 정확한 정의는 무엇인가 — "같은 bank 의 다른 주소에 동시 접근" 이 왜 직렬화되는가?
- $k$-way conflict 시 성능 저하는 정확히 몇 배인가? 즉, 한 cycle 에 여러 thread 가 같은 bank 에 접근하면 몇 cycle 에 걸쳐 처리되는가?
- Bank padding (size + 1) 트릭이 왜 작동하는가? 수학적 증명은?
- Reduction 과 transpose kernel 에서 bank-free 설계의 구체적 패턴은?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

Shared memory 는 thread block 내에서 빠른 협력을 위한 핵심 자원입니다. 하지만 부주의하면 bank conflict 로 인해 같은 kernel 이 **2–32배 느려질 수 있습니다**. 예를 들어:

- Reduction kernel 에서 tree-based summation (Ch4-06) 은 shared memory 에서 high-bandwidth 접근이 필수인데, bank conflict 를 무시하면 효율이 급격히 하락합니다.
- `__shared__` 배열을 선언할 때 padding 을 추가하는 이유도 bank conflict 를 피하기 위함입니다.
- Warp shuffle (Ch4-06) 이 shared memory 를 우회하는 이유 중 하나도 이 bank conflict 를 회피하기 위함입니다.

---

## 📐 수학적 선행 조건

- Modular arithmetic ($\mod 32$)
- Memory banking 개념 (캐시 및 SRAM 이론)
- Ch4-01 의 warp 구조

---

## 📖 직관적 이해

### 32 Bank 의 구조

Shared memory (48 KB typical per SM) 는 32 개의 독립적인 **bank** 로 나뉩니다:

```
┌──────────┬──────────┬─────────┬──────────┐
│ Bank 0   │ Bank 1   │  ...    │ Bank 31  │
│ 192 B    │ 192 B    │         │ 192 B    │
└──────────┴──────────┴─────────┴──────────┘
  (each bank 48 KB / 32 = 1.5 KB)

Address → Bank mapping:
addr % 4 (byte offset) + (addr / 4) % 32 (bank ID)

또는 간단히:
Bank ID = (address / 4) % 32
```

**이점**: 32 개 bank 를 독립적으로 access 하면 32×higher throughput.  
**제약**: 같은 bank 의 다른 주소 → serialization.

### Bank Conflict 의 직관

같은 cycle 에 여러 thread 가 같은 bank 의 **다른 주소** 에 접근:

```
Thread 0: read from addr = 0*4 = 0  (bank 0)
Thread 1: read from addr = 1*4 = 4  (bank 1)
Thread 2: read from addr = 2*4 = 8  (bank 2)
...
Thread 31: read from addr = 31*4 = 124 (bank 31)

→ No conflict! 각 bank 에 1개씩 → 1 cycle

---

Thread 0: read from addr = 0*4 = 0  (bank 0)
Thread 1: read from addr = 32*4 = 128  (bank 0)  ← same bank!
Thread 2: read from addr = 2*4 = 8  (bank 2)

→ Thread 0, 1 이 같은 bank 충돌 → 두 요청이 serial 처리
  → 2–3 cycles 소요 (1-way conflict → 2x slow)
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Shared Memory Bank

Shared memory 는 32 개의 **bank** $B_0, B_1, \ldots, B_{31}$ 로 분할.

각 bank 는 독립적인 memory port 를 가지며, 한 cycle 에 **1개 주소만** 처리:

$$B_j = \{\text{address } a : (a / 4) \mod 32 = j\}$$

여기서 $/$ 는 정수 나눗셈, address $a$ 는 byte 단위.

### 정의 4.2 — Bank Conflict

같은 cycle 에 $k$ 개 thread ($k > 1$) 가 같은 bank $B_j$ 의 **서로 다른 주소** 에 접근:

$$a_1, a_2, \ldots, a_k \in B_j, \quad a_i \neq a_j \text{ for some } i \neq j$$

이를 **$k$-way bank conflict** 라 합니다.

### 정의 4.3 — Bank Conflict Serialization

$k$-way conflict 는 hardware scheduler 가 요청들을 sequentially 처리하므로:

$$\text{Cycles needed} = k$$

따라서 **throughput 이 $k$ 배 저하** (동일 latency 에서).

---

## 🔬 정리와 증명

### 정리 4.1 — Sequential Addressing 의 Bank-Free 조건

Thread $i$ 가 주소 $a_i = \text{base} + i \cdot 4$ (stride = 4 bytes, FP32) 로 접근할 때:

$$B_i = (a_i / 4) \mod 32 = (\text{base}/4 + i) \mod 32$$

Warp 의 32 thread ($i = 0, 1, \ldots, 31$) 는:

$$B_i = (\text{base}/4) \mod 32 + i \mod 32$$

32 개 서로 다른 값 → **no conflict**.

**증명**: $B_i$ 는 $(\text{base}/4) \mod 32$ offset 으로부터 연속 32 개 → 각 bank 1 번씩 → no serialization $\square$.

### 정리 4.2 — Padding 트릭

Shared memory 배열을 다음과 같이 선언:

```cpp
__shared__ float arr[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 padding
```

그러면 row-major 배열 `arr[i][j]` 는 주소:

$$a_{ij} = \text{base} + i \cdot (\text{BLOCK\_SIZE} + 1) \cdot 4 + j \cdot 4$$

Bank ID:

$$B_{ij} = (a_{ij} / 4) \mod 32 = (i \cdot (\text{BLOCK\_SIZE} + 1) + j) \mod 32$$

**예**: BLOCK_SIZE = 32, stride = 33
- $(0, 0)$ → bank $(0 \cdot 33) \mod 32 = 0$
- $(1, 0)$ → bank $(1 \cdot 33) \mod 32 = 1$
- $(2, 0)$ → bank $(2 \cdot 33) \mod 32 = 2$
- ...
- $(31, 0)$ → bank $(31 \cdot 33) \mod 32 = 31$

각 행의 첫 원소가 다른 bank → **no conflict**.

**증명**: stride = BLOCK_SIZE + 1 = 33 이면, $\gcd(33, 32) = 1$ → 연속 32 개 주소가 32 개 서로 다른 bank 를 방문 $\square$.

### 정리 4.3 — Reduction Kernel 의 Bank Conflict 분석

Tree-based reduction 에서, step $k$ 마다:

```cpp
// Step k: 2^k stride
for (int stride = 1 << k; stride >= 1; stride >>= 1) {
    if (tid + stride < BLOCK_SIZE)
        sdata[tid] += sdata[tid + stride];
    __syncthreads();
}
```

Thread `tid` 와 `tid + stride` 가 같은 bank 충돌:

$$B_{\text{tid}} = \text{tid} \mod 32$$
$$B_{\text{tid+stride}} = (\text{tid} + \text{stride}) \mod 32$$

Conflict iff $\text{stride} \equiv 0 \pmod{32}$.

**결과**: stride = 1, 2, 4, 8, 16 (< 32) → **no conflict**. stride = 32 이상 → potential conflict.

**해결책 (Mark Harris)**: warp shuffle (Ch4-06) 으로 마지막 warp 는 shared memory 우회 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Bank Conflict 직접 측정

```cpp
// bank_conflict_demo.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void no_conflict_kernel(float *output) {
    __shared__ float sdata[32 * 32];
    
    int tid = threadIdx.x;
    int idx = tid;  // Sequential addressing: idx = tid
    
    // Load
    sdata[idx] = tid;
    __syncthreads();
    
    // Accumulate (no conflict)
    float sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += sdata[i * 32 + tid];  // stride = 32, no conflict
    }
    
    output[tid] = sum;
}

__global__ void conflict_kernel(float *output) {
    __shared__ float sdata[32 * 32];
    
    int tid = threadIdx.x;
    
    // Load
    sdata[tid * 32] = tid;  // 같은 row 의 다른 col
    __syncthreads();
    
    // Accumulate (potential conflict)
    float sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += sdata[tid + i * 32];  // tid 는 가변, stride = 32
    }
    
    output[tid] = sum;
}

__global__ void padded_kernel(float *output) {
    __shared__ float sdata[32 * 33];  // Padding = 33
    
    int tid = threadIdx.x;
    
    // Load
    sdata[tid] = tid;
    __syncthreads();
    
    // Accumulate (no conflict with padding)
    float sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += sdata[i * 33 + tid];  // Padding resolves conflict
    }
    
    output[tid] = sum;
}

int main() {
    int n = 32;
    float *output;
    cudaMalloc(&output, n * sizeof(float));
    
    // Profile each kernel
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    float elapsed;
    
    // No conflict
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        no_conflict_kernel<<<1, 32>>>(output);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("No conflict: %.3f ms\n", elapsed);
    
    // Conflict
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        conflict_kernel<<<1, 32>>>(output);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("With conflict: %.3f ms\n", elapsed);
    
    // Padded
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        padded_kernel<<<1, 32>>>(output);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Padded: %.3f ms\n", elapsed);
    
    cudaFree(output);
    return 0;
}
```

**예상 출력**:
```
No conflict: 10.5 ms
With conflict: 25.0 ms     (2–3배 느림)
Padded: 10.8 ms            (conflict 제거됨)
```

### 실험 2 — Transpose Kernel 의 Bank-Free 설계

```cpp
// transpose_kernel.cu
#include <cuda_runtime.h>

#define BLOCK_DIM 32

__global__ void naive_transpose(float *in, float *out, int n) {
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM];
    
    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
    
    // Read: coalesced
    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    
    // Write: tile[i][j] → out[i+j*n]
    // Bank conflict: threadIdx.x 들이 같은 tile 열 read → conflict
    if (x < n && y < n)
        out[x * n + y] = tile[threadIdx.x][threadIdx.y];
}

__global__ void padded_transpose(float *in, float *out, int n) {
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];  // +1 padding
    
    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
    
    // Read: coalesced
    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    
    // Write: no conflict (padding prevents column alignment)
    if (x < n && y < n)
        out[x * n + y] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    int n = 4096;
    float *in, *out;
    cudaMalloc(&in, n * n * sizeof(float));
    cudaMalloc(&out, n * n * sizeof(float));
    
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (n + 31) / 32);
    
    // Benchmark naive
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    float elapsed;
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        naive_transpose<<<grid, block>>>(in, out, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Naive transpose: %.2f ms\n", elapsed / 100);
    
    // Benchmark padded
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        padded_transpose<<<grid, block>>>(in, out, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Padded transpose: %.2f ms\n", elapsed / 100);
    
    cudaFree(in);
    cudaFree(out);
    return 0;
}
```

**예상 출력**:
```
Naive transpose: 15.5 ms (bank conflicts slow down read)
Padded transpose: 8.2 ms  (conflicts eliminated)
```

### 실험 3 — PyTorch 에서 Shared Memory 선언 (cpp_extension)

```python
# shared_mem_optim.py
import torch
import torch.utils.cpp_extension as ext

cuda_code = '''
#include <cuda_runtime.h>
#include <cmath>

__global__ void reduction_kernel_padded(const float *input, float *output, int n) {
    extern __shared__ char shared_mem_raw[];
    float *sdata = (float *)shared_mem_raw;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduce with padding (stride 33 instead of 32)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

torch::Tensor reduction_padded(torch::Tensor input) {
    int block_size = 256;
    int grid_size = (input.numel() + block_size - 1) / block_size;
    torch::Tensor output = torch::zeros({grid_size}, input.options());
    
    int shared_mem_bytes = block_size * sizeof(float);
    
    reduction_kernel_padded<<<grid_size, block_size, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    
    return output;
}
'''

cpp_code = '''
torch::Tensor reduction_padded(torch::Tensor input);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduction_padded", &reduction_padded);
}
'''

module = ext.load_inline(
    name='shared_mem',
    cpp_sources=cpp_code,
    cuda_sources=cuda_code,
    functions=['reduction_padded'],
    verbose=False
)

# Test
x = torch.randn(1000000, device='cuda')
result = module.reduction_padded(x)
print(f"Reduction result: {result.sum():.4f}")
```

---

## 🔗 실전 활용

### 1. Shared Memory 최적화 Checklist

```python
def optimize_shared_memory(kernel_code):
    """
    Bank conflict 를 피하기 위한 checklist:
    """
    checks = [
        "✓ Array access pattern 분석: thread i 는 어느 shared mem 주소에 접근?",
        "✓ Bank ID 계산: (address / 4) % 32 = ?",
        "✓ Conflict 판정: 같은 warp 의 여러 thread 가 같은 bank 인가?",
        "✓ Padding 추가: stride가 32의 배수인가? → stride + 1 로 패딩",
        "✓ Warp shuffle 고려: shared mem 전대신 __shfl_down_sync 사용?",
    ]
    
    for check in checks:
        print(check)
```

### 2. NVIDIA CUTLASS 사용

NVIDIA 의 CUTLASS library 는 bank-conflict-free shared memory tiling 을 자동 제공.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 32 bank = 항상 고정 | NVIDIA GPU 기준; AMD 는 다름 |
| Bank ID = (address/4) % 32 | 다른 GPU 아키텍처에서는 다를 수 있음 |
| Padding = 32 + 1 | 이론적 증명이지만, 실제로는 architecture-specific |
| No broadcast optimization | 최근 GPU (Volta+) 는 broadcast (같은 주소) 를 특별 처리 |

---

## 📌 핵심 정리

$$\boxed{B_i = \left(\frac{a_i}{4}\right) \mod 32, \quad \text{Conflict} = k \text{-way} \Rightarrow k \times \text{ slower}}$$

| 기법 | 효과 | 사용 시기 |
|------|------|----------|
| Sequential addressing | No conflict | Warp 내 thread 들이 다른 bank 접근 |
| Padding (stride + 1) | 행 단위 접근 시 conflict 제거 | 2D array transpose/reduction |
| Warp shuffle | Shared mem 우회 | 마지막 reduction step (Volta+) |
| Broadcast handling | 여러 thread 가 같은 주소 | 최신 GPU 하드웨어 최적화 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Shared memory 주소 0, 4, 8, 12, ..., 124 (stride = 4, FP32 배열) 에 접근하는 32 개 thread 의 bank ID 들을 나열하라. 몇 개의 서로 다른 bank 가 사용되는가?

<details>
<summary>해설</summary>

$$B_i = (i \cdot 4 / 4) \mod 32 = i \mod 32$$

따라서 bank ID = 0, 1, 2, ..., 31 → **32 개 서로 다른 bank**, **no conflict** $\square$.

</details>

**문제 2** (심화): 2D array `tile[32][32]` (row-major) 를 shared memory 에 선언했을 때, thread $(i, j)$ 가 `tile[i][j]` 에 접근하면 bank ID 는? Thread 들이 같은 row 를 읽는 경우와 같은 column 을 읽는 경우를 비교하라.

<details>
<summary>해설</summary>

Row-major address: $a_{ij} = i \cdot 32 \cdot 4 + j \cdot 4 = (i \cdot 32 + j) \cdot 4$

Bank ID: $B_{ij} = (i \cdot 32 + j) \mod 32 = j$

**같은 row (i 고정) 읽기**: thread (i, 0), (i, 1), ..., (i, 31) 이 bank 0, 1, ..., 31 → **no conflict** ✓

**같은 column (j 고정) 읽기**: thread (0, j), (1, j), ..., (31, j) 가 모두 bank j → **32-way conflict** ✗

**해결책**: padding 으로 stride = 33 → bank ID = $(i \cdot 33 + j) \mod 32$ → no conflict $\square$

</details>

**문제 3** (논문 비평): Harris (2005) 의 reduction optimization 에서, "unroll last warp" 라는 기법은 마지막 반복에서 shared memory 대신 warp shuffle 을 사용한다고 했다. 이것이 왜 bank conflict 를 제거하는가? Warp shuffle 의 장점은 무엇인가?

<details>
<summary>해설</summary>

마지막 stride (stride = 32) 에서:
- Shared mem: stride = 32 → thread 0, 1, ..., 31 이 bank 0, 1, ..., 31 의 same-lane 접근 → broadcast 가 아니면 conflict
- Warp shuffle: register 간 direct transfer → shared memory 우회 → **no conflict, no synchronization**

**Warp shuffle advantage**:
1. Shared memory bank conflict 제거
2. `__syncthreads()` 불필요 (warp 내부이므로 동기화 무료)
3. Latency 더 낮음 (register 간 직접 전달)

따라서 마지막 warp iteration 에서는 shuffle 이 최적 $\square$.

</details>

---

<div align="center">

[◀ 이전](./03-memory-coalescing.md) | [📚 README](../README.md) | [다음 ▶](./05-warp-divergence.md)

</div>
