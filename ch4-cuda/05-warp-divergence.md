# 05. Warp Divergence 와 SIMT 병목

## 🎯 핵심 질문

- SIMT 에서 warp divergence 란 정확히 무엇인가 — "같은 warp 의 32 thread 가 다른 branch 를 선택" 할 때 무엇이 일어나는가?
- Divergence 의 비용은 정확히 얼마인가 — warp 효율이 얼마나 떨어지는가? (50% vs 100% utilization)
- Volta (2017) 이후의 "Independent Thread Scheduling" 이 divergence 를 어떻게 완화하는가? 하지만 여전히 완전히 제거되지 않는 이유는?
- Divergence 를 회피하는 구체적 패턴은 무엇인가 — warp 단위 균일 분기, branch-free trick (predication, BALLOT, `__all_sync`)?
- Real-world example: reduction tree 의 step iteration `if (threadIdx.x < active)` 에서 divergence 는 어떻게 관리되는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 많은 kernel (조건부 연산, masking, dynamic shape 처리) 은 implicit 하게 warp divergence 를 만듭니다. 예를 들어:

- `torch.where(condition, x, y)` 는 condition 이 warp 내에서 heterogeneous 하면 divergence
- Custom kernel 에서 `if (tid % 2 == 0)` 같은 조건은 warp 를 정확히 절반으로 나누어 **50% 성능 저하** 를 만듭니다.

이를 피하려면 warp 단위 균일 분기 (`if (blockIdx.x % 2 == 0)` 같이 warp 전체가 같은 조건) 또는 branch-free programming (predication, BALLOT) 을 알아야 합니다.

---

## 📐 수학적 선행 조건

- Ch4-01 의 warp 와 SIMT
- Control flow 와 instruction flow 개념
- Bitmask 와 ballot 연산

---

## 📖 직관적 이해

### SIMT 의 근본: 모든 thread 가 같은 instruction 실행

```
같은 warp 의 32 thread:

if (condition[tid]) {
    path_A();
} else {
    path_B();
}

SIMT 동작 (pre-Volta):
1. condition 평가 (각 thread 의 값)
2. Active mask = {i : condition[i] == true}
3. path_A 실행 (mask 에 따라 각 thread selective write-back)
4. Active mask = {i : condition[i] == false}
5. path_B 실행 (mask 에 따라 각 thread selective write-back)
6. PC synchronize

→ 최악의 경우, instruction count 가 2배 (두 path 를 sequential 실행)
```

### Divergence 의 비용

**Uniform branching** (모든 thread 가 같은 branch):
```
if (blockIdx.x % 2 == 0) {  // warp 전체가 true/false
    // 모든 32 thread 가 같은 path
}
→ No divergence, 1x throughput
```

**Divergent branching** (thread 마다 다른 branch):
```
if (threadIdx.x % 2 == 0) {  // warp 내에서 1/2 thread 가 true, 1/2 false
    // path_A
} else {
    // path_B
}
→ 2-way divergence, path 를 sequential 실행 → 2x latency, 50% 효율
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Warp 와 Execution Mask

Warp 의 PC 에서 branch instruction 이 fetch & decode 될 때:

$$\text{Active mask} = \{i \in [0, 32) : \text{thread } i \text{ can execute instruction at PC}\}$$

Branch 후:
- Path A: mask_A = {i : condition[i] == true}
- Path B: mask_B = {i : condition[i] == false}

각 path 의 instruction 들은 해당 mask 에 따라 selective 실행.

### 정의 5.2 — Warp Divergence

$k$-way divergence 는 $k$ 개의 disjoint active masks 가 서로 다른 path 를 따른다는 의미:

$$\text{mask}_1, \ldots, \text{mask}_k \text{ are disjoint and non-empty}$$

각 path 는 **sequential** 하게 실행되어야 함.

### 정의 5.3 — Divergence Cost

$k$-way divergence 의 경우, instruction latency 는 $k$ 배 증가:

$$\text{Effective cycle} = \sum_{i=1}^{k} \text{cycles in path } i$$

따라서 **throughput 이 $1/k$ 로 저하**.

### 정의 5.4 — Independent Thread Scheduling (Volta+)

Volta 아키텍처부터, 각 thread 는 독립적인 PC 를 유지할 수 있음:

$$\text{PC}_i \neq \text{PC}_j \text{ for some } i, j \in \text{same warp}$$

이론적으로는 divergence 비용이 제거되지만, 실제로는:
1. Register pressure (per-thread state) 증가
2. Synchronization barrier (`__syncthreads()`) 필요
3. Compiler complexity 증가

따라서 **divergence 회피 여전히 권장**.

---

## 🔬 정리와 증명

### 정리 5.1 — Uniform Branching 의 No-Divergence 조건

모든 thread 가 같은 branch 를 택할 필요충분조건:

$$\forall i, j \in \text{warp}: \text{condition}[i] = \text{condition}[j]$$

또는, condition 이 warp 내 모든 thread 에 대해 상수:

$$\text{condition} = f(\text{blockIdx}), \quad \text{not } f(\text{threadIdx})$$

**증명**: Condition 이 block index 만 의존하면, 같은 block 의 모든 thread 는 같은 값 → uniform branching → no divergence $\square$.

### 정리 5.2 — Maximally Divergent Branching 의 Cost

Thread $i$ 가 조건 `(i % 2 == 0)` 를 따를 때, 2-way divergence:
- Mask A (i even): 16 thread
- Mask B (i odd): 16 thread

Warp efficiency = 50% (한 번에 절반 thread 만 active).

**증명**: 각 path 를 sequential 실행하므로 total time = path_A_time + path_B_time. Parallel 하게 하면 max(path_A, path_B) 이므로, speedup factor = $1 / 2$ $\square$.

### 정리 5.3 — BALLOT 를 이용한 Branch-Free Prediction

조건 `cond` 에 대해:

```cpp
int ballot = __ballot_sync(0xFFFFFFFF, cond);
// ballot = bitmask of which threads have cond == true
int count = __popc(ballot);  // popcount

if (count > 16) {  // Majority: do A
    result = A;
} else {           // Minority: do B
    result = B;
}
```

이렇게 하면 **모든 thread 가 같은 path 를 실행** (predication 으로 selective write) → **no divergence**.

**증명**: 모든 32 thread 가 같은 instruction sequence (majority path) 를 fetch & decode 하므로, warp 는 truly uniform. Minority thread 들은 write-back 만 skip $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Divergence 의 비용 직접 측정

```cpp
// divergence_demo.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void uniform_branch(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    
    // Uniform branching: block 단위 조건 (warp 단위도 uniform)
    if (blockIdx.x % 2 == 0) {
        // Path A: heavy computation
        for (int i = 0; i < 100; i++)
            result += sinf(result);
    } else {
        // Path B: different heavy computation
        for (int i = 0; i < 100; i++)
            result += cosf(result);
    }
    
    if (idx < n)
        output[idx] = result;
}

__global__ void divergent_branch(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    
    // Divergent branching: thread 단위 조건 (warp 내 divergence)
    if (threadIdx.x % 2 == 0) {  // ← This creates divergence within warp
        // Path A: heavy computation
        for (int i = 0; i < 100; i++)
            result += sinf(result);
    } else {
        // Path B: different heavy computation
        for (int i = 0; i < 100; i++)
            result += cosf(result);
    }
    
    if (idx < n)
        output[idx] = result;
}

int main() {
    int n = 1024 * 1024;
    float *output;
    cudaMalloc(&output, n * sizeof(float));
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    float elapsed;
    
    // Benchmark uniform
    cudaEventRecord(start);
    uniform_branch<<<(n + 255) / 256, 256>>>(output, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Uniform branch: %.3f ms\n", elapsed);
    
    // Benchmark divergent
    cudaEventRecord(start);
    divergent_branch<<<(n + 255) / 256, 256>>>(output, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Divergent branch: %.3f ms\n", elapsed);
    
    cudaFree(output);
    return 0;
}
```

**예상 출력**:
```
Uniform branch: 15.2 ms
Divergent branch: 28.5 ms  (약 2배 느림)
```

### 실험 2 — BALLOT 를 이용한 Branch-Free 구현

```cpp
// branch_free_demo.cu
#include <cuda_runtime.h>

__global__ void ballot_branch(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float value = (idx < n) ? input[idx] : 0.0f;
    bool cond = (value > 0.5f);  // condition
    
    // BALLOT to check predicate across warp
    int ballot = __ballot_sync(0xFFFFFFFF, cond);
    int count = __popc(ballot);
    
    float result;
    if (count >= 16) {
        // Majority path: All threads compute, some mask write
        result = sinf(value) * value;
    } else {
        // Minority path: All threads compute, some mask write
        result = cosf(value) / (value + 1e-6f);
    }
    
    if (idx < n)
        output[idx] = result;
}

// 비교: 순진한 divergent implementation
__global__ void naive_branch(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float value = (idx < n) ? input[idx] : 0.0f;
    float result;
    
    if (value > 0.5f) {
        result = sinf(value) * value;
    } else {
        result = cosf(value) / (value + 1e-6f);
    }
    
    if (idx < n)
        output[idx] = result;
}
```

### 실험 3 — PyTorch 에서 Divergence 측정

```python
import torch

def test_warp_divergence():
    """
    PyTorch 에서의 divergence 는 usually implicit
    하지만 custom CUDA kernel 은 직접 영향을 받음
    """
    
    # Uniform condition: 같은 결과 (예: batch 단위)
    x1 = torch.randn(1024, device='cuda')
    y1 = torch.where(x1 > 0, x1 * 2, x1 / 2)
    
    # Divergent condition: thread 단위 heterogeneous
    x2 = torch.randn(1024, device='cuda')
    condition = torch.arange(1024, device='cuda') % 2 == 0
    y2 = torch.where(condition, x2 * 2, x2 / 2)
    
    # PyTorch 는 divergence 를 자동으로 처리하지만,
    # custom kernel 은 수동으로 관리해야 함
    
    print("PyTorch embedding divergence: hidden in backend kernel")
    print("Custom CUDA kernel: explicit divergence management required")

test_warp_divergence()
```

---

## 🔗 실전 활용

### 1. Divergence 회피 Patterns

```cuda
// ✗ Bad: thread-level divergence
if (threadIdx.x % 2 == 0) {
    // Path A
} else {
    // Path B
}

// ✓ Good: block-level uniform branching
if (blockIdx.x % 2 == 0) {
    // Entire block divergence, but no warp-internal divergence
}

// ✓ Best: branch-free, predicate instead
int mask = (threadIdx.x % 2 == 0) ? 1 : 0;
result_a = compute_a(x);
result_b = compute_b(x);
result = mask ? result_a : result_b;  // No divergence, just predicate
```

### 2. 동적 배열 접근에서 divergence

```cuda
// ✗ Dynamic array access causes divergence
for (int i = 0; i < dynamic_size[threadIdx.x]; i++) {  // per-thread loop bound
    // Different threads iterate different counts → divergence
}

// ✓ Better: warp-level synchronization
int max_iter = __shfl_sync(0xFFFFFFFF, dynamic_size[threadIdx.x], 0);  // broadcast max
for (int i = 0; i < max_iter; i++) {
    if (i < dynamic_size[threadIdx.x]) {
        // process
    }
}
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Pre-Volta: warp 내 PC synchronization 필수 | Volta+: independent PC, 하지만 비용 여전함 |
| SIMT = lockstep execution | 실제로는 scheduling 복잡성으로 인한 오버헤드 |
| Branch prediction 없음 | 최신 GPU: branch prediction unit (주의 필요) |
| Ballot/SHFL 가능 (SM 3.0+) | 구 GPU (SM < 3.0) 는 지원 안 함 |

---

## 📌 핵심 정리

$$\boxed{\text{Uniform branching} : 1 \times \text{ throughput}, \quad k\text{-way divergence} : \frac{1}{k} \times \text{ throughput}}$$

| 기법 | Divergence | 비용 |
|------|-----------|------|
| Thread-level condition | k-way | $k \times$ slower |
| Block-level condition | No warp-internal | No divergence |
| BALLOT predication | No | Branch-free |
| Warp shuffle | No | Ultra-low latency |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 32 개 thread 의 warp 에서, `if (tid < 16)` 조건이 만드는 divergence 수준을 계산하라. 이 경우 warp efficiency 는 몇 %인가?

<details>
<summary>해설</summary>

**Condition**: thread 0–15 는 true, 16–31 은 false → 2-way divergence

**Warp efficiency**: 한 번에 16 개 thread 만 active (각 path 마다) → 평균 16/32 = 50% $\square$.

</details>

**문제 2** (심화): Reduction kernel 에서, step $k$ 마다 `if (tid < (1 << k))` 조건을 사용한다고 하자. 이 조건이 만드는 divergence 는? Step = 0, 1, 2, ..., 5 에 대해 각각 계산하라.

<details>
<summary>해설</summary>

$$\text{Active thread} = 2^k$$

- Step 0: $2^0 = 1$ → 31-way divergence (1 vs 31)
- Step 1: $2^1 = 2$ → 2-way divergence (2 vs 30)
- Step 2: $2^2 = 4$ → 2-way divergence (4 vs 28)
- ...
- Step 5: $2^5 = 32$ → No divergence (all 32)

**결론**: 초기 step 은 심각한 divergence, 진행할수록 개선. 마지막 step 에서 warp shuffle 을 사용하면 synchronization cost 를 완전히 제거 $\square$.

</details>

**문제 3** (논문 비평): Harris 의 reduction optimization 에서, "마지막 warp 에만 shuffle 을 사용" 한다고 했는데, 왜 모든 step 에서 shuffle 을 사용하지 않는가? 각 단계에서 shuffle 의 비용은 무엇인가?

<details>
<summary>해설</summary>

**Warp shuffle latency**: $\text{__shfl\_down\_sync}$ 는 각 단계마다 1–2 cycles (register 간 직접 전달).

**Shared memory latency**: 4–10 cycles (bank conflict 없을 때).

따라서 shuffle 이 더 빠르지만, **초기 단계** (active thread 많음) 에서는:
- Shuffle 으로 합을 전달 → thread 간 synchronization (warp 내 동기화는 자유)
- Shared memory 로 합을 저장 → 다음 block iteration 에 재사용 가능

**절충**: 초기–중반 단계는 shared memory (재사용성, batch 처리), 마지막 단계는 shuffle (divergence 제거, no synchronization) $\square$.

</details>

---

<div align="center">

[◀ 이전](./04-bank-conflict.md) | [📚 README](../README.md) | [다음 ▶](./06-reduction-optimization.md)

</div>
