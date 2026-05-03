# 04. TorchInductor — Code Generation

## 🎯 핵심 질문

- **FX graph → backend code** 의 과정은? GPU 의 경우 Triton, CPU 의 경우 C++/OpenMP.
- **Fusion heuristic**: pointwise + reduction + pointwise 를 single kernel 로 합치는 규칙은?
- **Reduction kernel** 의 split-K, persistent kernel, atomic-free 패턴은 무엇인가?
- **Matmul decision**: cuBLAS (성능 최고) vs Triton (fusion 가능) 를 자동으로 선택하는 로직은?
- `TORCH_LOGS="output_code"` 로 생성된 kernel 코드를 직접 읽으려면?

---

## 🔍 왜 Code Generation 이 중요한가

AOTAutograd 가 forward + backward graph 를 생성한 후, **실행 가능한 코드** 로 변환해야 합니다.

Inductor 는:

1. **FX graph 분석**: Operation topology, data dependencies
2. **Kernel fusion**: 여러 ops 를 single kernel 로 merge
3. **Backend selection**: 각 op 에 맞는 kernel (cuBLAS, Triton, C++) 선택
4. **Code generation**: Triton 소스 코드 생성 및 JIT compile
5. **Optimization**: Loop ordering, tiling, vectorization

이를 통해 **eager mode 대비 2~5배 throughput** 달성.

---

## 📐 수학적 선행 조건

- **Roofline model** (Ch4-02): compute intensity, bandwidth
- **Kernel fusion** (Ch5-05): 여러 ops 의 memory access pattern
- **GEMM** (Ch4, Ch5): matrix multiplication
- **Triton** (Ch5-03): block-level GPU programming
- **Parallelism**: thread, block, warp

---

## 📖 직관적 이해

### Fusion 의 원리

```
EAGER MODE: Multiple kernels, HBM roundtrips
┌────────────────────────────────────────┐
│ Kernel 1: matmul(x, w) → y             │
│ ┌──────────────────────────────────┐   │
│ │ Read y from HBM, write to HBM    │   │ ← HBM roundtrip (큼)
│ └──────────────────────────────────┘   │
└────────────────────────────────────────┘
         │ y ∈ HBM
         ↓
┌────────────────────────────────────────┐
│ Kernel 2: add(y, bias) → z             │
│ ┌──────────────────────────────────┐   │
│ │ Read y from HBM, write to HBM    │   │ ← HBM roundtrip
│ └──────────────────────────────────┘   │
└────────────────────────────────────────┘
         │ z ∈ HBM
         ↓
         Total bandwidth: 3 × data_size (read y, write y, read y, write z)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FUSED MODE: Single kernel, SRAM only
┌────────────────────────────────────────┐
│ Fused kernel: matmul(x, w) + add(bias) │
│ ┌──────────────────────────────────┐   │
│ │ Load x from HBM (큼)              │   │ ← only 1 HBM load
│ │ Load w from HBM (큼)              │   │
│ │ Compute matmul → y (register/SRAM) │ │
│ │ Add bias to y (no HBM roundtrip)  │ │
│ │ Write z to HBM (작음, final output) │ │
│ └──────────────────────────────────┘   │
└────────────────────────────────────────┘
         │ z ∈ HBM
         
         Total bandwidth: 1 × input_size + 1 × output_size
         Much smaller than eager (no intermediate HBM roundtrip)
```

### Reduction Kernel Pattern

```
Naive sum reduction: All threads sum their part, synchronize O(log n)
─────────────────────────────────────────────────────────────────

x = [1, 2, 3, 4, 5, 6, 7, 8]
Block size: 8 threads

Step 1:  [1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]  (4 threads active, 4 sync)
         [3+7, 11+15]          = [10, 26]        (2 threads active, 2 sync)
         [10+26]               = [36]            (1 thread active, 1 sync)

Problem: Many sync points, many idle threads

─────────────────────────────────────────────────────────────────
Split-K pattern:
─────────────────────────────────────────────────────────────────

Instead of one large reduction, split into K smaller reductions:
- Chunk 1: thread_0, 1, 2 sum their part → local_sum_1
- Chunk 2: thread_3, 4, 5 sum their part → local_sum_2
- Chunk 3: thread_6, 7 sum their part → local_sum_3
Final: atomic_add(global_sum, local_sum_i) for i=1,2,3

Benefits:
- Fewer sync points within each chunk
- Atomic operations 은 hardware native (no manual sync)
- Better occupancy (many blocks can run in parallel)
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Fusion Group

FX graph 의 여러 operation nodes 를 single kernel 로 합칠 fusion group:

$$
F = \{ n_i : n_i \in G, \text{canFuse}(n_i, n_{i+1}) \}
$$

여기서 canFuse 조건:

$$
\text{canFuse}(n_i, n_{i+1}) \iff \begin{cases}
\text{shape}(n_i.\text{output}) = \text{shape}(n_{i+1}.\text{input}) & \text{(broadcasting 없음)} \\
\text{n}_{i+1} \text{이 } n_i \text{의 유일한 consumer} & \text{(single consumer)} \\
n_i, n_{i+1} \notin \{\text{unsupported for fusion}\} & \text{(backend constraint)}
\end{cases}
$$

### 정의 4.2 — Reduction Pattern

Reduction op $\text{reduce}(x, \text{axis})$ 의 kernel type:

- **Pointwise reduction**: 각 element 가 independent (예: sum 의 각 summand)
- **Split-K reduction**: Global reduction 을 local chunks 로 분해
  - Block $b$ 가 자신의 chunk compute
  - Atomic add 로 global accumulate
- **Persistent kernel**: Single kernel 이 모든 reduction step 수행
  - Thread 가 multiple chunks 를 sequential 처리
  - Occupancy 높음

### 정의 4.3 — Matmul Selection Decision

주어진 matrix multiplication $C = A B$ (shape: $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$):

**cuBLAS 선택 조건**:
$$
\text{is\_large\_matmul}(M, N, K) \land \neg \text{enable\_fusion}
$$

**Triton 선택 조건**:
$$
\text{is\_fused\_with\_bias\_activation} \lor \text{K small}
$$

---

## 🔬 정리와 증명

### 정리 4.1 — Fusion 에 의한 bandwidth 감소

Single fusion group 에서 pointwise + reduction + pointwise pattern:

$$
\text{BW}_{\text{fused}} = \text{BW}_{\text{input}} + \text{BW}_{\text{output}}
$$

Eager (3 individual kernels):

$$
\text{BW}_{\text{eager}} = 3 \times (\text{BW}_{\text{input}} + \text{BW}_{\text{intermediate}})
$$

따라서:

$$
\boxed{\text{speedup} \geq \frac{\text{BW}_{\text{eager}}}{\text{BW}_{\text{fused}}} = \frac{3 \times (D_I + D_M)}{D_I + D_O} \approx 3 \times \left(1 - \frac{D_O}{D_I + D_M}\right)}
$$

최악의 경우 (output 이 input 과 같은 크기): $\approx 1.5 \times$ speedup. 최선의 경우: $\approx 3 \times$.

**증명**: Roofline model (Ch4-02). Kernel 간 synchronization overhead 무시하면, bandwidth 가 dominant 한 경우 이득이 fusion.  $\square$

### 정리 4.2 — Split-K Reduction 의 Atomic 오버헤드

Split-K reduction 에서 K 개 blocks 이 각각 local reduction 후 atomic add:

$$
t_{\text{atomic}} = K \cdot t_{\text{single\_atomic}}
$$

전체 시간:

$$
t_{\text{split-K}} = t_{\text{local\_reduce}} + t_{\text{atomic}} + t_{\text{write}}
$$

Naive tree reduction (single block):

$$
t_{\text{tree}} = t_{\text{reduce}} + \sum_{i=1}^{\log n} t_{\text{sync}}
$$

**조건**: $K \cdot t_{\text{atomic}} < \sum \log t_{\text{sync}}$ 이면 split-K 더 빠름.

일반적으로: **K 가 수십~수백 정도면** atomic overhead 가 많은 sync 를 상쇄하여 split-K 가 더 빠름. $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — 기본 Inductor code generation 추적

```python
import torch
import torch.nn as nn
import os

# Enable Inductor code output
os.environ['TORCH_LOGS'] = 'output_code'

class SimpleModel(nn.Module):
    def forward(self, x):
        x = x @ torch.randn(10, 20, device=x.device)  # matmul
        x = x + 1  # add (should fuse with matmul)
        x = torch.relu(x)  # relu (should fuse)
        return x

model = SimpleModel()
x = torch.randn(8, 10, device='cuda')

compiled = torch.compile(model)

print("=== First forward (triggers code generation) ===")
_ = compiled(x)

# Console will print generated Triton code:
# @triton.jit
# def _kernel_0(...)
#   ...Triton IR for fused matmul + add + relu...
```

### 실험 2 — Fusion 효과 측정

```python
import torch
import time

class UnfusedOps(torch.nn.Module):
    def forward(self, x):
        # Three separate operations (eager: 3 kernels)
        x = x + 1
        x = torch.relu(x)
        x = x * 2
        return x

class FusableOps(torch.nn.Module):
    def forward(self, x):
        # Same ops, but structured for fusion
        return ((x + 1).relu()) * 2

x = torch.randn(1024, 1024, device='cuda')

# Eager
unfused_eager = UnfusedOps().cuda()
fusable_eager = FusableOps().cuda()

# Compiled (Inductor fuses)
unfused_compiled = torch.compile(unfused_eager)
fusable_compiled = torch.compile(fusable_eager)

def benchmark(fn, x_, name):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = fn(x_)
    torch.cuda.synchronize()
    t = time.time() - t0
    print(f"{name:25s}: {t*1000:7.2f}ms")

print("=== Fusion Impact ===")
benchmark(unfused_eager, x, "Unfused (eager)")
benchmark(unfused_compiled, x, "Unfused (compiled)")
benchmark(fusable_eager, x, "Fusable (eager)")
benchmark(fusable_compiled, x, "Fusable (compiled)")

# Expected:
# Unfused (eager)        :  100.00ms  (3 kernels)
# Unfused (compiled)     :   80.00ms  (still 3 kernels? or fused by Inductor?)
# Fusable (eager)        :  100.00ms  (still 3 ops, but structure different)
# Fusable (compiled)     :   40.00ms  (fused to 1 kernel)
```

### 실험 3 — Reduction kernel 의 split-K vs tree pattern

```python
import torch

def sum_reduction_simple(x):
    # Will use tree-based or split-K depending on size
    return x.sum(dim=-1)

x_small = torch.randn(8, 1024, device='cuda')
x_large = torch.randn(1024, 1024*1024, device='cuda')

compiled = torch.compile(sum_reduction_simple)

print("=== Reduction Kernel Pattern ===")

# Small sum: tree reduction likely
print(f"Small sum: shape {x_small.shape}")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    _ = compiled(x_small)
torch.cuda.synchronize()
t_small = time.time() - t0
print(f"  Time: {t_small*1000:.2f}ms")

# Large sum: split-K + atomic
print(f"Large sum: shape {x_large.shape}")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    _ = compiled(x_large)
torch.cuda.synchronize()
t_large = time.time() - t0
print(f"  Time: {t_large*1000:.2f}ms")

# Inductor automatically selects pattern based on data size
```

### 실험 4 — Matmul backend selection (cuBLAS vs Triton)

```python
import torch
import torch.nn as nn

class MatmulModel(nn.Module):
    def __init__(self, fused_with_bias=False):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)
        self.fused_with_bias = fused_with_bias
    
    def forward(self, x):
        x = self.linear(x)  # matmul + bias
        if self.fused_with_bias:
            x = torch.relu(x)  # fuse with matmul
        return x

model_matmul_only = MatmulModel(fused_with_bias=False)
model_matmul_fused = MatmulModel(fused_with_bias=True)

compiled_only = torch.compile(model_matmul_only)
compiled_fused = torch.compile(model_matmul_fused)

x = torch.randn(256, 1024, device='cuda')

print("=== Matmul Backend Selection ===")

# Pure matmul: likely uses cuBLAS
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    _ = compiled_only(x)
torch.cuda.synchronize()
t_only = time.time() - t0

# Matmul + activation: Triton (cuBLAS can't fuse)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    _ = compiled_fused(x)
torch.cuda.synchronize()
t_fused = time.time() - t0

print(f"Pure matmul (likely cuBLAS): {t_only*1000:.2f}ms")
print(f"Matmul + ReLU (likely Triton): {t_fused*1000:.2f}ms")
# Fused might be ~20-30% slower due to Triton vs cuBLAS,
# but saves kernel launch overhead
```

### 실험 5 — Generated kernel 검사 (TORCH_LOGS)

```python
import torch
import os
import tempfile

# Redirect TORCH_LOGS to file
log_dir = tempfile.mkdtemp()
os.environ['TORCH_LOGS'] = f'output_code={log_dir}'

model = torch.nn.Linear(10, 20).cuda()
compiled = torch.compile(model)

x = torch.randn(8, 10, device='cuda')
_ = compiled(x)

# Read generated kernel code
import glob
kernel_files = glob.glob(f'{log_dir}/**/output_code*.py', recursive=True)
if kernel_files:
    with open(kernel_files[0]) as f:
        print("=== Generated Triton Kernel (Sample) ===")
        print(f.read()[:1000])  # First 1000 chars

# Output shows:
# @triton.jit
# def triton___(...)
#     ...Triton IR...
#     tl.store(out_ptr, ...)
```

---

## 🔗 실전 활용

### 1. Fusion-friendly 코드 작성

```python
# ❌ Unfusable: intermediate 가 여러 consumers
x = x.relu()
y = x * 2
z = x + 1  # x 의 second consumer, fusion breaks

# ✅ Fusable: single consumer 패턴
x = (x.relu() * 2) + 1
```

### 2. Reduction 성능 최적화

```python
# ❌ Slow: 불필요한 reshape/transpose
x = x.permute(0, 2, 1, 3)
result = x.sum(dim=-1)

# ✅ Fast: reduction 을 memory-friendly 한 차원으로
x = x.reshape(batch, height*width, channels)
result = x.sum(dim=-1)  # reduction over contiguous dim
```

### 3. Code generation debugging

```python
import os
os.environ['TORCH_LOGS'] = 'output_code'
os.environ['TORCH_VERBOSE'] = '1'

# Run compiled code, check console for kernel IR
model = torch.compile(model)
_ = model(x)

# If Inductor generated inefficient code:
# - Check with TORCH_LOGS output
# - Consider explicit op ordering to help fusion heuristic
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Statically-known shapes | Dynamic shapes: may skip fusion (conservatively) |
| Memory-bound operations | Compute-bound matmul: Triton may underperform vs cuBLAS |
| No dependencies between fused ops | In-place ops with aliases: fusion breaks |
| Deterministic kernels | Random ops: Inductor conservatively doesn't fuse |
| Small number of operations | Large graphs: code generation time O(n) |

---

## 📌 핵심 정리

$$\boxed{\text{Inductor} = \text{FX graph} \to \text{fusion groups} \to \text{backend selection} \to \text{Triton/C++}}$$

| 단계 | 역할 | 결과 |
|------|------|------|
| **Fusion group** | Ops 합치기 | Multiple → single kernel |
| **Backend select** | Op 별 kernel 선택 | cuBLAS / Triton / C++ |
| **Code generation** | IR → Triton source | triton_*.py |
| **JIT compile** | Triton → GPU binary | .cubin file |
| **Execution** | Single kernel run | ~20-50% faster |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Fusion 이 bandwidth 를 감소시키는 원리를 roofline model 로 설명하라. Pointwise + reduction + pointwise fusion 이 가능한 이유는?

<details>
<summary>해설</summary>

**Roofline 관점**:

Eager: 3 kernels → 3 × (load + store intermediate)
- Intermediate (HBM ↔) roundtrip 비용 $D_M$ (메모리 크기)

Fused: 1 kernel → single load-compute-store
- Intermediate 가 register/SRAM 에서 처리 (HBM 우회)

**Point-Red-Point pattern**:
- Pointwise 1: per-element ops, shape 동일
- Reduction: shape 감소, 하지만 intermediate keep 필요
- Pointwise 2: reduced shape 에서 per-element ops

이 세 가지가 stack 되면, reduction 의 output 이 Pointwise 2 의 input 으로 곧바로 사용되므로 intermediate materialization 불필요. 따라서 fusion 가능.  $\square$

</details>

**문제 2** (심화): Split-K reduction 이 왜 atomic operation 을 사용하는가? Tree-based reduction 보다 나은 이유는?

<details>
<summary>해설</summary>

**Tree-based**:
```
Block 0: threads sum first half
Block 1: threads sum second half
...
All blocks must synchronize at global level
→ __global__ sync (expensive, grid-level)
```

**Split-K**:
```
Block 0: sum chunk 0 locally → atomic_add(global_result)
Block 1: sum chunk 1 locally → atomic_add(global_result)
...
No global sync needed, blocks independent
```

**Why atomic is better**:
1. **Occupancy**: Blocks can run independently (device scheduler more flexible)
2. **No explicit sync**: Hardware atomic is fast on modern GPUs
3. **Scalability**: Many blocks can run in parallel (different SMs)

Atomic contention 은 data size 가 작아서 (single element 또는 few elements) 문제 없음.  $\square$

</details>

**문제 3** (논문 비평): Inductor 의 fusion heuristic (Horace He et al., MLSYS 2023) 이 모든 경우 최적인가? Counter-example 을 제시하고 해결책을 제안하라.

<details>
<summary>해설</summary>

**Counter-example**: Matmul 앞뒤로 pointwise ops

```python
x = x @ w              # cuBLAS (매우 빠름)
x = x.relu()           # Triton (느림)
x = x @ w2             # cuBLAS (매우 빠름)
```

Inductor heuristic 이 first two ops 를 fuse 하려 하면:
- Triton kernel (cuBLAS 보다 느림) for matmul + relu
- 전체 성능 저하 가능

**Optimal strategy**:
- cuBLAS matmul (빠름)
- Triton relu (작은 op, overhead 무시)
- cuBLAS matmul (빠름)

**해결책**:
- Profiling-guided fusion: 실제 kernel cost 측정 후 fusion 결정
- Heuristic threshold: large matmul 은 항상 cuBLAS (Inductor 현 구현에 있음)

현재 PT 2.0 은 conservative approach 취함 (matmul 은 cuBLAS 고정).  $\square$

</details>

---

<div align="center">

[◀ 이전](./03-aot-autograd.md) | [📚 README](../README.md) | [다음 ▶](./05-compile-distributed.md)

</div>
