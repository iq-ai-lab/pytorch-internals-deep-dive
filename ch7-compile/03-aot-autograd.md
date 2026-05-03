# 03. AOTAutograd — Forward + Backward Graph

## 🎯 핵심 질문

- **Ahead-Of-Time autograd** 의 의미는? Forward + backward 를 미리 graph 로 생성한다는 것이 어떻게 다른가?
- **Functionalization**: in-place mutation (`x.add_(1)`) 을 functional form (`x = x.add(1)`) 으로 변환하는 이유는?
- **Decomposition**: `aten::log_softmax` 가 `aten::sub + aten::log + aten::exp + aten::sum` 으로 축소되면 어떤 이점이 있는가?
- **Min-cut partition**: Forward 에서 어떤 intermediate 를 save 하고 backward 에서 어떤 intermediate 를 recompute 할 지 결정하는 방법은?
- Eager autograd 의 tape-based approach 와 본질적 차이는?

---

## 🔍 왜 AOT autograd 가 필수인가

**Eager mode autograd** (PT 1.x):

```python
# Forward
y = x @ w
# Backward (나중)
# tape 에서 (x, w) 를 꺼내 gradients 계산
```

문제점:
1. Forward 중에 intermediate (x, w) 를 메모리에 저장 필요
2. Backward 도 Python 에서 호출되므로dispatcher overhead
3. **Forward + backward 간의 fusion 불가능** (두 phase 가 분리됨)

**AOTAutograd** (PT 2.0):

```
Forward graph:   x, w → y
                 ↓
AOTAutograd: both graphs 동시 생성
                 ↓
Backward graph:  grad_y → grad_x, grad_w
```

이점:
1. **Forward + backward 를 하나의 graph 로 표현** → Inductor 가 fusion
2. **Intermediate save/recompute 최적화** (memory-speed trade-off)
3. **In-place operation 을 functional 로 변환** → 메모리 alias 문제 회피

---

## 📐 수학적 선행 조건

- **Autograd Deep Dive** (Ch2): VJP, backward graph, computational graph
- **Chain rule**: $\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}$
- **FX graph** (Ch7-01): node, edge, tracing
- **VJP 형식화**: $\text{vjp}(y, v_y) = (v_x, v_w)$ 의 linearization
- **Memory hierarchy** (Ch4): HBM/SRAM trade-off

---

## 📖 직관적 이해

### Eager vs AOTAutograd

```
EAGER MODE (PT 1.x)
┌──────────────────┐
│  Forward (eager) │  y = x @ w + b
└────────┬─────────┘
         │ Save: x, w, b to tape
         ↓
    ┌─────────────┐
    │   Tape[     │ ← Keep in GPU memory until backward
    │   x, w, b   │
    │   ]         │
    └─────────────┘
         │
         ↓ (later, when loss.backward() called)
┌──────────────────┐
│ Backward (eager) │ grad_y → grad_x, grad_w
└──────────────────┘

Problem: Forward & Backward disjoint phases
  → Fusion 불가능
  → Dispatcher overhead 두 번
  → Intermediate storage 예측 불가능

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AOTAUTOGRAD MODE (PT 2.0)
┌──────────────────────────────────────────┐
│ Forward graph (symbolic)                  │
│  x, w, b → y, (intermediate_saved)       │
└────────────┬─────────────────────────────┘
             │
        ╔════════════════════════════════╗
        ║ AOTAutograd engine              ║
        ║ (symbolic chain rule)           ║
        ╚════════════┬═══════════════════╝
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
┌─────────────────────┐  ┌──────────────────────┐
│ Backward graph      │  │ Partition strategy   │
│ grad_y,             │  │ (save or recompute   │
│ (intermediates)     │  │  each intermediate?) │
│ → grad_x, grad_w    │  │                      │
└─────────────────────┘  └──────────────────────┘

Benefit: Forward + Backward in ONE graph
  → Fusion 가능 (single kernel)
  → Intermediates fully controlled
  → Memory-speed trade-off optimal
```

### Functionalization 예시

```python
# Input code
x = torch.tensor([1.0, 2.0, 3.0])
x.add_(5.0)  # in-place operation

# AOTAutograd converts to
x = torch.tensor([1.0, 2.0, 3.0])
x = x.add(5.0)  # functional form

# Why?
# - in-place ops create aliasing: x_original, x_modified share storage
# - graph 에서 aliasing 은 optimization 을 복잡하게 함
# - functional form: data flow 명확, fusion 가능
```

### Decomposition 예시

```python
# High-level op
y = torch.log_softmax(x, dim=-1)

# AOTAutograd decomposes into primitives
x_max = x.max(dim=-1, keepdim=True)[0]  # numerical stability
x_exp = (x - x_max).exp()
x_sum = x_exp.sum(dim=-1, keepdim=True)
y = x.sub(x_max).sub(x_sum.log()).add(0)
# (simplified, actual order optimized)

# Why?
# - log_softmax is "composed" op (numerically stable softmax then log)
# - Primitive ops: sub, exp, sum, log → Inductor 가 더 자유롭게 fuse
# - Separate kernel 없이 single fused kernel 에서 처리
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Ahead-Of-Time Autograd

Forward graph $G_{\text{fwd}}: (x, w) \mapsto y$ 가 주어졌을 때, AOTAutograd 는:

**1. Forward graph 로부터 VJP 를 symbolic 하게 유도:**

$$
G_{\text{bwd}} = \text{vjp\_derive}(G_{\text{fwd}})
$$

**2. Backward graph 를 정의:**

$$
G_{\text{bwd}}: (\bar{y}, \text{saved\_intermediates}) \mapsto (\bar{x}, \bar{w})
$$

**3. 두 graph 를 **merged** 하여 single computation graph:**

$$
G_{\text{merged}} = G_{\text{fwd}} \sqcup G_{\text{bwd}}
$$

이를 Inductor 에 전달 → 단일 kernel 로 compile 가능.

### 정의 3.2 — Functionalization

In-place operation $\text{add\_}$ 를 functional form 으로 변환:

$$
\text{functional}(x.op\_(args)) \Rightarrow x = x.op(args)
$$

Formally:
- Input: Mutable tensors $\{x_i\}$ 와 in-place ops $\{\text{op}\_i\}$
- Output: Functional composition $y = \text{op}_n(...\text{op}_1(x_0)...)$
- Guarantee: Semantics 동일, aliasing 제거

### 정의 3.3 — Decomposition

High-level operation $\text{op}$ 를 primitive operations 로 분해:

$$
\text{op}(x, \text{args}) = G_{\text{primitives}}(x, \text{args})
$$

여기서 $G_{\text{primitives}}$ 는 basic ops (add, mul, exp, sum, etc.) 의 DAG.

예:
$$
\text{log\_softmax}(x, d) = \log \left( \frac{e^{x - \max(x, d)}}{\sum_d e^{x - \max(x, d)}} \right) \Rightarrow \{\text{max}, \text{sub}, \text{exp}, \text{sum}, \text{log}\}
$$

---

## 🔬 정리와 증명

### 정리 3.1 — AOTAutograd 의 정확성 (Horace He, 2022)

Forward graph $G_{\text{fwd}}$ 와 backward graph $G_{\text{bwd}} = \text{vjp\_derive}(G_{\text{fwd}})$ 에 대해:

$$
\boxed{\text{sgn} : \text{vjp}_{\text{eager}}(y, v_y) = \text{vjp}_{\text{aot}}(y, v_y, \text{saved})}
$$

즉, AOTAutograd 에서 계산한 gradient 와 eager autograd 의 gradient 가 numerically 동일.

**증명 sketch**:

AOTAutograd 는 forward 의 computational graph 를 node-by-node 분석하여 각 node $n$ 에 대해:

1. Forward VJP: $\text{vjp}_n(v_{\text{out}}) = v_{\text{in}}$ (chain rule 적용)
2. Backward node 추가: $G_{\text{bwd}}$ 에 추가
3. Saved intermediates: $S_n$ 을 forward 의 output 으로 표기

최종 gradient 는 backward graph 의 topological sort 로 계산. Chain rule 의 associativity 에 의해 eager 과 동일. $\square$

### 정리 3.2 — Functionalization 의 Semantic Preservation

In-place operation $x.op\_(args)$ 를 functional $x = x.op(args)$ 로 변환하면, 다음이 성립:

$$
\text{semantics}(x.op\_(args)) = \text{semantics}(x := x.op(args))
$$

**증명**:

In-place operation 은 storage 를 modify 하지만, 수학적으로는 $x \gets \text{op}(x, \text{args})$ 의 rebinding:

- Input: $x_{\text{old}}$, storage address $\alpha$
- After $x.op\_(args)$: $x_{\text{new}}$ at address $\alpha$ (same address)
- Functional: $x_{\text{new}} = \text{op}(x_{\text{old}}, \text{args})$ (new address $\alpha'$)

Computational graph 의 data dependency 관점에서는 동일 (두 경우 모두 $x_{\text{old}} \to x_{\text{new}}$ 의 edge). 따라서 gradient flow 도 동일. $\square$

### 정리 3.3 — Min-Cut Partition 최적화

Forward graph 의 모든 intermediate $I = \{i_1, \ldots, i_m\}$ 에 대해, backward 에서 save 할 subset $S \subseteq I$ 를 선택 문제:

**Minimize**: memory cost + recomputation cost

$$
\text{cost}(S) = |S| \cdot \text{mem\_cost} + |I \setminus S| \cdot \text{recomp\_cost}
$$

**Constraint**: Backward graph 가 valid 해야 함 (required gradients compute 가능).

이는 **min-cut problem** in DAG: save 와 recompute 의 경계를 최적화.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — AOTAutograd 결과 추출 및 확인

```python
import torch
import torch.nn as nn
from torch._functorch.partitioners import default_partition
from torch._functorch.compile import compiled_function
from torch._inductor import config as inductor_config

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20, bias=False)
        self.fc2 = nn.Linear(20, 1, bias=False)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel().cuda()
x = torch.randn(8, 10, device='cuda', requires_grad=True)

# AOTAutograd + Inductor trace
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

compiled = torch.compile(model)

print("=== Forward Pass (triggers compilation) ===")
y = compiled(x)

# Console output shows:
# [Inductor] Compiled forward+backward graph
# [Triton] Generated single fused kernel

print(f"\nForward output shape: {y.shape}")
```

### 실험 2 — In-place vs Functional 의 차이

```python
import torch

def with_inplace(x):
    x = x.clone()  # Copy for this test
    x.add_(1.0)    # In-place
    x.mul_(2.0)    # In-place
    return x

def functional(x):
    x = x.add(1.0)  # Functional
    x = x.mul(2.0)  # Functional
    return x

x = torch.randn(1000000, device='cuda')

# Compare: with_inplace and functional are mathematically same
result_inplace = with_inplace(x)
result_func = functional(x)

print(f"=== In-place vs Functional ===")
print(f"Max difference: {(result_inplace - result_func).abs().max():.2e}")
print(f"Results numerically identical: {torch.allclose(result_inplace, result_func)}")

# Now compile both
compiled_inplace = torch.compile(with_inplace)
compiled_func = torch.compile(functional)

import time
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    _ = compiled_inplace(x)
torch.cuda.synchronize()
t_inplace = time.time() - t0

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    _ = compiled_func(x)
torch.cuda.synchronize()
t_func = time.time() - t0

print(f"\nCompiled in-place: {t_inplace*1000:.2f}ms")
print(f"Compiled functional: {t_func*1000:.2f}ms")
print(f"Speedup ratio: {t_inplace/t_func:.2f}×")
# Functional likely faster due to better fusion
```

### 실험 3 — Decomposition 의 효과

```python
import torch
import torch.nn.functional as F

x = torch.randn(256, 10, device='cuda', requires_grad=True)

# High-level op
def with_log_softmax(x):
    return F.log_softmax(x, dim=-1)

# Decomposed version (manually)
def decomposed_log_softmax(x):
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_shifted = x - x_max
    x_exp = x_shifted.exp()
    x_sum = x_exp.sum(dim=-1, keepdim=True)
    return x_shifted - x_sum.log()

# Verify
y1 = with_log_softmax(x)
y2 = decomposed_log_softmax(x)
print(f"=== Decomposition Correctness ===")
print(f"Max difference: {(y1 - y2).abs().max():.2e}")

# Compile both
compiled_log_softmax = torch.compile(with_log_softmax)
compiled_decomposed = torch.compile(decomposed_log_softmax)

# Time them (with backward)
def benchmark(fn, x_):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        y = fn(x_)
        loss = y.sum()
        loss.backward()
        x_.grad.zero_()
    torch.cuda.synchronize()
    return time.time() - t0

t1 = benchmark(compiled_log_softmax, x.clone().requires_grad_(True))
t2 = benchmark(compiled_decomposed, x.clone().requires_grad_(True))

print(f"\nCompiled log_softmax: {t1*1000:.2f}ms")
print(f"Compiled decomposed: {t2*1000:.2f}ms")

# AOTAutograd may decompose internally,
# so both should have similar performance
```

### 실험 4 — Backward graph 의 correctness 검증

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
).cuda()

x = torch.randn(8, 10, device='cuda', requires_grad=True)

# Eager backward
y_eager = model(x)
loss_eager = y_eager.sum()
loss_eager.backward()
grad_eager = x.grad.clone()

# Clear gradients
x.grad.zero_()

# Compiled forward + backward
compiled_model = torch.compile(model)
y_compiled = compiled_model(x)
loss_compiled = y_compiled.sum()
loss_compiled.backward()
grad_compiled = x.grad.clone()

print("=== AOTAutograd Backward Correctness ===")
print(f"Forward outputs match: {torch.allclose(y_eager, y_compiled, rtol=1e-4, atol=1e-5)}")
print(f"Gradients match: {torch.allclose(grad_eager, grad_compiled, rtol=1e-4, atol=1e-5)}")
print(f"Max gradient difference: {(grad_eager - grad_compiled).abs().max():.2e}")
```

---

## 🔗 실전 활용

### 1. Graph rewrite 로 fusion 기회 증가

```python
# Avoid unnecessary intermediate materializations
# ❌ Bad: intermediate 가 multiple consumers
x = ...
a = x.sigmoid()
b = a * 2        # Uses a
c = a + 1        # Also uses a
z = b + c

# ✅ Better: single consumer pattern
x = ...
z = (x.sigmoid() * 2) + (x.sigmoid() + 1)
# (note: x.sigmoid() recomputed, but enables fusion)

# Or: use graph rewrite to fuse manually
from torch._dynamo import config
config.optimize_for_inference = False  # Use inference mode fusion
```

### 2. Memory vs Speed trade-off 제어

```python
# Default: AOTAutograd decides save/recompute
model = torch.compile(model)

# Force maximum memory saving (recompute everything)
torch._inductor.config.recompute_reduces = True

# Force memory maximization (save everything)
torch._inductor.config.memory_efficient_level = 0
```

### 3. Custom backward 의 AOTAutograd 호환성

```python
import torch
from torch.autograd import Function

class CustomOp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 2
    
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2

def model_with_custom_op(x):
    return CustomOp.apply(x) + 1

# Compile with custom op (PT 2.1+ supported)
compiled = torch.compile(model_with_custom_op)
x = torch.randn(8, 10, requires_grad=True)
y = compiled(x)
loss = y.sum()
loss.backward()  # Should work if CustomOp is traced correctly
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| All ops have defined VJP | Custom ops without backward: define backward |
| No mutable state in forward | Stateful layers (BatchNorm): running_mean/var update handled specially |
| Deterministic forward | Random ops: seed must be fixed ahead-of-time |
| Fixed graph structure | Dynamic control flow: split into segments (graph breaks) |
| Composition of traceable ops | Non-PyTorch code (numpy, C++): fallback to eager |

---

## 📌 핵심 정리

$$\boxed{\text{AOTAutograd} = \text{Forward graph} + \text{symbolic chain rule} + \text{Backward graph}}$$

| 개념 | 역할 | 이점 |
|------|------|------|
| **Forward graph** | Tensor operations | Static structure |
| **VJP derive** | Symbolic backward | Fusion 가능 |
| **Functionalization** | In-place → functional | Aliasing 제거 |
| **Decomposition** | High-level → primitives | Kernel fusion |
| **Min-cut partition** | Save vs recompute | Memory-speed trade-off |
| **Merged graph** | Forward + backward | Single kernel |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Ahead-Of-Time autograd 에서 "ahead-of-time" 의 의미는? Backward graph 가 forward 실행 전에 생성되면 어떤 이점이 있는가?

<details>
<summary>해설</summary>

**AOT**: Forward execution 전에 backward graph 를 compile time 에 생성.

**이점**:
1. Backward 를 forward 와 함께 fusion 가능 (single kernel)
2. Intermediate save/recompute 최적화 가능 (min-cut)
3. Memory allocation 을 ahead-of-time 에 계획 (no runtime tape overhead)

**Eager 비교**: Backward 가 runtime 에만 알려지므로 fusion 불가능, dispatcher 오버헤드 두 번.  $\square$

</details>

**문제 2** (심화): Functionalization 이 `x.add_(1)` 을 `x = x.add(1)` 으로 변환할 때, 왜 "aliasing 제거" 가 optimization 을 가능하게 하는가? 구체적 예시를 들어 설명하라.

<details>
<summary>해설</summary>

In-place operation 은 storage 를 modify:

```python
x = torch.tensor([1, 2, 3])  # address α
x.add_(1)                     # x now [2, 3, 4] at α
```

Graph 에서 aliasing:
- x_old (address α 의 값 [1,2,3])
- x_new (address α 의 값 [2,3,4], 같은 storage)
- 이 두 "logical" tensor 가 same storage 를 share

Optimizer 입장에서:
- x_old 와 x_new 가 구별되어야 함 (추후 users 결정)
- fusion 할 때 "어느 intermediate 를 reuse 할 수 있는가?" 판단 어려움
- Load-store pattern 불명확

Functional form:
```python
x_new = x_old.add(1)  # x_new 는 new address
```
- Data flow 명확: x_old → add op → x_new
- Intermediate 를 안전하게 fuse 가능
- Memory allocation pattern 예측 가능

따라서 **aliasing 제거 = optimization 공간 확대**.  $\square$

</details>

**问题 3** (논문 비평): He et al. (2022) AOTAutograd 논문에서 제시한 min-cut partition 문제의 최적성은? Greedy heuristic vs exact algorithm 의 trade-off는?

<details>
<summary>해설</summary>

**Min-cut partition problem**:

$$
\min_{S \subseteq I} \left( |S| \cdot \text{mem\_cost} + |I \setminus S| \cdot \text{recomp\_cost} \right)
$$

**Exact algorithm**: Min-cut in graph 는 polynomial-time (max-flow = min-cut, Ford-Fulkerson)

**PT 2.0 구현**: Greedy heuristic (cost-benefit analysis):
- 각 intermediate 에 대해 save cost vs recompute cost 계산
- cost-benefit ratio 로 정렬, threshold 이상이면 save

**Trade-off**:
- Exact: 최적 해, but O(n³) 이상 (n = intermediate 개수)
- Greedy: 근사 (보통 90% 이상), O(n log n)

실무에서 대부분 greedy 로 충분 (제한된 intermediate 개수). 논문은 problem formulation 제시, 구현은 heuristic 사용.  $\square$

</details>

---

<div align="center">

[◀ 이전](./02-torchdynamo.md) | [📚 README](../README.md) | [다음 ▶](./04-torchinductor.md)

</div>
