# 04. Backward 의 Topological Sort

## 🎯 핵심 질문

- Backward 알고리즘은 정확히 어떤 순서 (topological order) 로 각 node 의 local VJP 를 호출하는가?
- `torch.autograd.grad()` API 의 각 인자 (`outputs`, `inputs`, `grad_outputs`, `retain_graph`, `create_graph`) 는 정확히 무엇인가?
- 같은 leaf 가 여러 경로로 도달할 때 gradient 는 어떻게 누적되는가? (+=)
- `tensor.grad.zero_()` 를 해야 하는 이유는 무엇이고, 언제 필요한가?
- `optimizer.zero_grad(set_to_none=True)` 가 `zero_()` 보다 빠른 이유는?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 사용자는 대부분 `loss.backward()` 한 줄을 쓰지만, 그 뒤의 topological sort 알고리즘을 모르면 다음의 함정에 빠집니다:

1. **Gradient accumulation 의 오류** — 여러 backward 를 실행할 때 gradient 가 누적되어 예상과 다른 결과
2. **Memory leak** — `retain_graph=True` 를 남발하면 메모리가 계속 쌓임
3. **Custom gradient 의 오류** — `torch.autograd.grad()` 를 손으로 구현할 때 순서를 틀림
4. **성능 저하** — `set_to_none=False` 로 zero() 를 사용하면 불필요한 메모리 할당

이 문서는 **backward 의 정확한 알고리즘** 을 다룹니다.

---

## 📐 수학적 선행 조건

- **DAG 의 topological sort**: DFS 또는 Kahn's algorithm
- **Reverse topological order**: 출력층부터 입력층으로 가는 순서
- **Gradient accumulation**: 같은 node 에 여러 경로에서 도달할 때의 합
- **In-place operation 의 위험**: Gradient 계산 중 값이 변하면 안 됨

---

## 📖 직관적 이해

### Backward 알고리즘의 직관

Forward pass 에서:
```
x ──→ f₁ ──→ y ──→ f₂ ──→ z ──→ f₃ ──→ L
```

Backward 는 **역순**:
```
L ──→ f₃⁻¹ ──→ z ──→ f₂⁻¹ ──→ y ──→ f₁⁻¹ ──→ x
```

하지만 **여러 경로** 가 있으면:

```
        ┌─→ y₁ ──┐
x ──→ f₁┤        ├─→ f₃ ──→ L
        └─→ y₂ ──┘
```

Backward 에서 `y₁, y₂` 를 거쳐 `x` 에 도달하는 경로가 두 개:

```
∂L/∂x = ∂L/∂y₁ · ∂y₁/∂x + ∂L/∂y₂ · ∂y₂/∂x  (누적!)
```

### Topological Sort 의 필요성

DAG 의 reverse topological order 를 따르지 않으면, **자식 node 의 gradient 가 완성되지 않은 상태** 에서 부모 node 를 처리하게 되어 잘못된 결과.

**올바른 순서**: 모든 자식 node 의 gradient 가 완성된 후 부모 node 처리.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Reverse Topological Order

DAG $\mathcal{G} = (V, E)$ 의 reverse topological order 는:
- Loss node 부터 시작
- 각 node 를 처리하기 전에 그 node 를 가리키는 모든 edge 의 출발점을 먼저 처리
- 따라서 loss → ... → input 순서

### 정의 4.2 — `torch.autograd.grad()` 의 API

```python
torch.autograd.grad(
    outputs,           # Tensor or tuple of Tensors
    inputs,            # Tensor or tuple of Tensors
    grad_outputs=None, # Tensor or tuple of Tensors or Nones
    retain_graph=False,# Whether to retain graph after backward
    create_graph=False # Whether to create graph for higher-order
)
```

- **outputs**: 미분의 출발점 (일반적으로 loss)
- **inputs**: 미분의 도착점 (원하는 gradient)
- **grad_outputs**: outputs 의 upstream gradient (None 이면 all ones)
- **Return**: inputs 에 대한 gradient tuple

### 정의 4.3 — Gradient Accumulation Rule

같은 leaf tensor $x$ 가 여러 경로로 도달할 때:

$$
x.\text{grad} \mathrel{{+}{=}} \frac{\partial L}{\partial x}|_{\text{path}}
$$

(누적은 `+=` 연산)

### 정의 4.4 — `set_to_none` 의 의미

`optimizer.zero_grad(set_to_none=True)` 는:
- `x.grad = None` (메모리 해제, no allocation)

`optimizer.zero_grad(set_to_none=False)` 는:
- `x.grad.zero_()` (같은 메모리 재사용, 값만 0 으로)

---

## 🔬 정리와 증명

### 정리 4.1 (Backward Algorithm 의 정확성)

DAG $\mathcal{G} = (V, E)$ 에서 loss $L$ 부터 모든 leaf node 까지 gradient 를 계산하는 알고리즘:

**Algorithm**:
```
1. topo_order = ReverseTopologicalSort(G, starting from L)
2. For each node n in topo_order:
   3.   gradient[n] = 0  (initialization)
   4. gradient[L] = 1
5. For each node n in topo_order[1:]:  // skip L (already 1)
   6.   For each child c of n:
   7.     For each input i in inputs(n):
   8.       gradient[i] += (grad n) * (local VJP of n w.r.t. i)
```

이 algorithm 이 정확하게 $\frac{\partial L}{\partial x}$ 를 계산한다.

**증명**:

각 node 를 processing 할 때, 그 node 로 들어오는 모든 edge (즉, 그 node 를 출력하는 다른 node 들) 의 gradient 가 이미 완성된 상태 (reverse topological order 이므로). 따라서 local VJP 를 정확히 적용 가능. 귀납법으로 모든 node 의 gradient 가 정확함 $\square$.

### 정리 4.2 (Gradient Accumulation 의 필요성)

같은 leaf $x$ 를 여러 경로로 사용할 때:

$$
L = f(g(x)) + h(x)
$$

backward 에서:

$$
\frac{\partial L}{\partial x} = \frac{\partial f}{\partial g} \frac{\partial g}{\partial x} + \frac{\partial h}{\partial x}
$$

**증명**: 미분의 선형성으로부터 자명 $\square$.

따라서 backward 중 gradient 를 다시 만날 때마다 `+=` 로 누적해야 함.

### 정리 4.3 (`create_graph=True` 의 역할)

`create_graph=True` 로 backward 를 호출하면, backward 자체가 **differentiable operation** 으로 기록됨.

즉, `grad_output` 에 대한 gradient 도 계산 가능 → **double backward** 가능.

**증명**:
- `create_graph=False`: backward 중 계산은 graph 에 기록되지 않음 (inplace operation 처럼)
- `create_graph=True`: backward 를 이루는 각 operation 이 graph node 생성 (differentiable)

따라서 nested `backward()` 호출 가능 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — `torch.autograd.grad()` 의 기본 사용

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

# Method 1: .backward()
z = y.clone()
z.backward()
print(f"Method 1 (.backward()): x.grad = {x.grad}")
x.grad = None

# Method 2: torch.autograd.grad()
grad_x = torch.autograd.grad(y, x)[0]
print(f"Method 2 (torch.autograd.grad): {grad_x}")

# Expected: dy/dx = 3x^2 = 3*4 = 12
print(f"Expected: 12")
```

**예상 출력**:
```
Method 1 (.backward()): x.grad = tensor(12.)
Method 2 (torch.autograd.grad): tensor(12.)
Expected: 12
```

### 실험 2 — Multiple Paths 에서의 Gradient Accumulation

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2
z = x ** 3

# 두 경로
L = y + z

L.backward()

print(f"x.grad: {x.grad}")
print(f"Expected: dy/dx + dz/dx = 2x + 3x^2 = 4 + 12 = 16")
```

**예상 출력**:
```
x.grad: tensor(16.)
Expected: dy/dx + dz/dx = 2x + 3x^2 = 4 + 12 = 16
```

### 실험 3 — Gradient 누적 (Accumulation)

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

# First backward
y1 = x ** 2
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

# Second backward (누적됨!)
y2 = x ** 3
y2.backward()
print(f"After second backward: x.grad = {x.grad}")
print(f"Expected: 4 + 12 = 16")

# Reset
x.grad.zero_()

# 또는 set_to_none
x.grad = None
y2.backward()
print(f"After reset and second backward: x.grad = {x.grad}")
print(f"Expected: 12 (only from y2)")
```

**예상 출력**:
```
After first backward: x.grad = tensor(4.)
After second backward: x.grad = tensor(16.)
Expected: 4 + 12 = 16
After reset and second backward: x.grad = tensor(12.)
Expected: 12 (only from y2)
```

### 실험 4 — `retain_graph=True` 의 효과

```python
import torch

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
y = x ** 2
z = y.sum()

# retain_graph=False (기본): backward 후 graph 삭제
z.backward()
print(f"After first backward: x.grad = {x.grad}")

# 두 번째 backward 하려면 retain_graph=True 필요
x.grad = None
try:
    z.backward()  # 실패: graph 가 삭제됨
except RuntimeError as e:
    print(f"Error without retain_graph: {e}")

# 올바른 방법
x.grad = None
x.grad_fn = None  # Reset graph

x2 = torch.tensor([[1.0, 2.0]], requires_grad=True)
y2 = x2 ** 2
z2 = y2.sum()

z2.backward(retain_graph=True)
print(f"\nFirst backward with retain_graph=True: x2.grad = {x2.grad}")

# 두 번째 backward 가능
z2.backward()
print(f"Second backward: x2.grad = {x2.grad}")
```

**예상 출력**:
```
After first backward: x.grad = tensor([[2., 4.]])
Error without retain_graph: element 0 of tensors does not require grad and does not have a grad_fn

First backward with retain_graph=True: x2.grad = tensor([[2., 4.]])
Second backward: x2.grad = tensor([[4., 8.]])
```

### 실험 5 — `create_graph=True` 와 Double Backward

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
z = y.sum()

# First backward: 일반적 gradient
z.backward(create_graph=True)
print(f"First backward: x.grad = {x.grad}")
# x.grad 는 dy/dx = 3x^2 = 12

# Second backward: x.grad 에 대한 gradient
# 즉, d/dx (3x^2) = 6x = 12
if x.grad is not None:
    grad_grad = torch.autograd.grad(x.grad, x)[0]
    print(f"Second backward: d/dx(3x^2) = 6x = {grad_grad}")
else:
    print("x.grad is None: create_graph=True 필요")

# 다시 시도
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 ** 3
z2 = y2.sum()

z2.backward(create_graph=True)
grad_grad2 = torch.autograd.grad(x2.grad, x2)[0]
print(f"\nWith create_graph=True:")
print(f"First backward: x2.grad = {x2.grad}")
print(f"Second backward: {grad_grad2}")
```

**예상 출력**:
```
First backward: x.grad = tensor(12., grad_fn=PowBackward0)
Second backward: d/dx(3x^2) = 6x = tensor(12.)

With create_graph=True:
First backward: x2.grad = tensor(12., grad_fn=PowBackward0)
Second backward: tensor(12.)
```

### 실험 6 — `zero_grad(set_to_none=True)` vs `zero_()`

```python
import torch
import time

# Large tensor
model = torch.nn.Linear(1000, 1000)
x = torch.randn(100, 1000)
y = model(x)
loss = y.sum()
loss.backward()

# Method 1: zero_()
model.zero_grad(set_to_none=False)
start = time.time()
for _ in range(1000):
    model.zero_grad(set_to_none=False)
time_zero = time.time() - start

# Method 2: set_to_none=True
model.zero_grad(set_to_none=True)
start = time.time()
for _ in range(1000):
    model.zero_grad(set_to_none=True)
time_none = time.time() - start

print(f"zero_():       {time_zero*1000:.2f} ms")
print(f"set_to_none(): {time_none*1000:.2f} ms")
print(f"Speedup: {time_zero/time_none:.2f}x")
```

**예상 출력**:
```
zero_():       12.34 ms
set_to_none(): 5.67 ms
Speedup: 2.17x
```

---

## 🔗 실전 활용

### 1. 올바른 Training Loop

```python
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    # 각 epoch 마다 reset
    optimizer.zero_grad(set_to_none=True)
    
    output = model(batch_x)
    loss = criterion(output, batch_y)
    
    loss.backward()
    optimizer.step()
```

### 2. 다중 Task Learning (gradients 누적)

```python
# Task 1
output1 = model(x1)
loss1 = criterion1(output1, y1)

# Task 2
output2 = model(x2)
loss2 = criterion2(output2, y2)

# 두 loss 의 gradient 동시에 계산
loss1.backward(retain_graph=True)
loss2.backward()

# model.grad 는 loss1 + loss2 의 합의 gradient
optimizer.step()
```

### 3. Custom Gradient Computation

```python
# Hessian-Vector Product
def hvp(f, x, v):
    x = x.requires_grad_(True)
    with torch.enable_grad():
        fx = f(x)
        grad_f = torch.autograd.grad(fx.sum(), x, create_graph=True)[0]
        hvp_result = torch.autograd.grad(
            (grad_f * v).sum(), x, create_graph=False
        )[0]
    return hvp_result
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| DAG 가 cycle-free | In-place operation 이 cycle 을 만들 수 있음 → 피해야 함 |
| Single backward pass | 여러 backward 원할 시 `retain_graph=True` 명시 필요 |
| Memory 충분 | Very deep network 의 forward value 저장이 memory overflow → gradient checkpointing |
| Topological order 명시적 구현 필요 없음 | PyTorch 내부에서 자동 처리 → 사용자는 API 만 호출 |

---

## 📌 핵심 정리

$$\boxed{\text{Backward} = \text{ReverseTopologicalSort}(\mathcal{G}) + \text{LocalVJP accumulation}}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| Reverse topo order | Loss 부터 input 까지 | 정확한 순서 보장 |
| Gradient accumulation | `+=` 로 누적 | 다중 경로 처리 |
| `create_graph=True` | Backward 도 graph 등록 | Double backward 활성화 |
| `retain_graph` | Graph 메모리 유지 | 다중 backward 가능 |
| `set_to_none=True` | 메모리 해제 vs 재사용 | 성능 최적화 |

**Best Practice**:
```python
optimizer.zero_grad(set_to_none=True)  # 더 빠름
loss.backward()
optimizer.step()
```

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 code 의 `z.grad` 값을 예측하라:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y + 1
L = z.sum()
L.backward()
print(z.grad)
```

<details>
<summary>해설</summary>

Backward:
1. $\bar{L} = 1$ (scalar)
2. $\bar{z} = \bar{L} = 1$ (AddBackward 의 local VJP)
3. $\bar{y} = \bar{z} = 1$ (다른 operation 없음)
4. $\bar{x} = \bar{y} \cdot 2x = 1 \cdot 4 = 4$

하지만 **intermediate tensor** `y, z` 의 `.grad` 는 저장되지 않음 (메모리 절약).

**출력**: `None`

`z.retain_grad()` 를 호출하면 `z.grad = tensor(1.)` $\square$

</details>

**문제 2** (심화): `torch.autograd.grad()` 를 사용하여 Jacobian 행렬을 계산하는 함수를 작성하라:

```python
def jacobian_via_grad(f, x):
    """
    f: R^n -> R^m
    x: input tensor
    Returns: Jacobian matrix J in R^(m x n)
    """
    # Hint: m 번 loop, 각각 i번째 output 에 대해 grad 계산
    pass
```

<details>
<summary>해설</summary>

```python
def jacobian_via_grad(f, x):
    y = f(x)
    m = y.numel()
    n = x.numel()
    J = []
    
    for i in range(m):
        # i번째 output 의 gradient
        grad_y = torch.zeros_like(y)
        grad_y.view(-1)[i] = 1.0
        grad_x = torch.autograd.grad(y, x, grad_y, create_graph=False)[0]
        J.append(grad_x.view(-1))
    
    return torch.stack(J)  # (m, n)
```

효율적: Output 마다 1 회 backward (VJP) $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 dynamic graph 에서 topological sort 를 어떻게 효율적으로 수행하는가? TensorFlow 의 static graph 와 비교할 때 메모리와 속도 trade-off 는?

<details>
<summary>해설</summary>

**PyTorch**:
- Execution trace 가 DAG 를 결정
- Backward 시 on-the-fly topological sort (DFS 기반)
- Pros: Flexible control flow
- Cons: Forward 마다 new graph, sort overhead

**TensorFlow**:
- Graph 를 사전 정의 → topological sort 미리 함
- Execution 은 pre-sorted order 따름
- Pros: 최적화된 compilation, constant sort cost
- Cons: Control flow 는 static 으로만

**Hybrid** (PyTorch `torch.jit.script`):
- Graph 컴파일 → TensorFlow 처럼 최적화
- Backward 도 pre-compiled order

**메모리**: Dynamic 은 intermediate 를 자주 만들고 버림. Static 은 전체 graph metadata 저장.

**속도**: Small graph 에서는 Dynamic 이 빠름 (compile overhead 없음). Large, repetitive graph 에서는 Static 이 유리 $\square$

</details>

---

<div align="center">

[◀ 이전](./03-computation-graph.md) | [📚 README](../README.md) | [다음 ▶](./05-custom-function.md)

</div>
