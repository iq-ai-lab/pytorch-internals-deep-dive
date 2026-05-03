# 03. Computation Graph 의 구성

## 🎯 핵심 질문

- **Dynamic graph (define-by-run)** 와 static graph 의 차이는 무엇이고, PyTorch 가 dynamic 을 선택한 이유는?
- `tensor.requires_grad` 가 True 일 때 어떤 메커니즘으로 graph node 가 생성되고, `tensor.grad_fn` 이 무엇을 가리키는가?
- **Leaf tensor** 와 **intermediate tensor** 의 구분은 정확히 어떻게 되는가? `is_leaf` 속성의 정확한 의미는?
- `torch.no_grad()` 와 `torch.inference_mode()` 의 차이점 — 언제 어느 것을 사용하는가?
- Graph 의 reference cycle 과 메모리 누수 — backward 후 어떻게 정리되는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 dynamic graph 는 flexibility 를 제공하지만, 이해하지 않으면 다음의 함정에 빠집니다:

1. **메모리 누수** — Loop 의 각 iteration 이 graph node 를 생성하면서 메모리가 계속 증가
2. **성능 저하** — `requires_grad=True` 인 tensor 에 대한 불필요한 연산 추적
3. **Debugging 어려움** — Graph 의 구조를 이해하지 못하면 왜 `backward()` 가 실패했는지 알기 어려움
4. **Mode 혼동** — `no_grad()`, `inference_mode()`, `grad_enabled(False)` 를 구분하지 못함

이 문서는 **graph 의 동적 생성 메커니즘** 과 **메모리 관리** 를 정확히 다룹니다.

---

## 📐 수학적 선행 조건

- **Directed acyclic graph (DAG)**: Node 와 edge 의 정의
- **Topological order**: DAG 의 역순 정렬
- **Reference counting**: 메모리 관리 기법
- **Python garbage collection**: Reference cycle 과 `gc.collect()`

---

## 📖 직관적 이해

### Dynamic Graph 의 직관

Static framework (TensorFlow 1.x) 에서는:
1. Graph 를 명시적으로 정의 (placeholder, variable, op)
2. Runtime 에 값만 변경하며 실행

PyTorch 의 dynamic graph 에서는:
1. Forward pass 가 진행되며 **즉시 graph node 생성**
2. 조건문, 루프 등이 graph 에 직접 영향 (if x > 0: return relu(x) else: return x)
3. Backward 는 **실행된 forward 를 역순으로 추적**

**장점**: Python 의 control flow (if, for, while) 를 자연스럽게 사용 가능.

**비용**: Forward 마다 graph 를 새로 만드는 overhead, 메모리 사용량 증가.

### Leaf vs Intermediate

```
  x (leaf, requires_grad=True)
    ↓
  y = x^2 (intermediate, grad_fn=PowBackward)
    ↓
  z = y + 1 (intermediate, grad_fn=AddBackward)
    ↓
  L = z.sum() (intermediate, grad_fn=SumBackward)
```

- **Leaf**: `requires_grad=True` 로 명시적으로 생성된 tensor (parameters, inputs)
- **Intermediate**: 연산 결과로 생성된 tensor (`grad_fn` ≠ None)

Backward 후:
- **Leaf**: `tensor.grad` 에 gradient 저장 (누적)
- **Intermediate**: `tensor.grad` 는 None (memory 절약)

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Computation Graph

PyTorch 의 computation graph $\mathcal{G} = (V, E)$ 는:

- **Vertices** $V$: Tensor 및 그 위의 연산 (Function node)
- **Edges** $E$: Data dependency (입력 tensor → 연산 → 출력 tensor)
- **Directed**: 실행 순서를 따름
- **Acyclic**: Backward 를 위해 cycle 이 없어야 함

### 정의 3.2 — Tensor 의 상태

각 tensor $t$ 는 다음 속성을 가짐:

- `t.requires_grad`: Boolean, gradient 추적 여부
- `t.grad_fn`: 함수 노드 (None 이면 leaf)
- `t.is_leaf`: `grad_fn is None and requires_grad` (또는 `requires_grad=False` 인 leaf)
- `t.grad`: 누적된 gradient (leaf only)

### 정의 3.3 — Leaf Tensor

Tensor $t$ 가 leaf 인 필요충분조건:

$$
\text{is\_leaf}(t) \Leftrightarrow (\text{grad\_fn}(t) = \text{None}) \land (\text{requires\_grad}(t) = \text{True})
$$

또는 `grad_fn = None` 이고 `requires_grad = False` (data-only tensor).

### 정의 3.4 — Graph Context

PyTorch 의 autograd 는 global context stack 을 유지:

- `torch.grad_enabled()`: 현재 gradient 추적 활성화 여부
- `torch.no_grad()`: context 내에서 추적 비활성화
- `torch.inference_mode()`: context 내에서 추적 비활성화 + 최적화

---

## 🔬 정리와 증명

### 정리 3.1 (Dynamic Graph 의 DAG 성질)

Forward pass 의 execution trace 가 만드는 computation graph 는 DAG 이다.

**증명**:

Backward 의 correctness 를 위해 PyTorch 는 graph 에 cycle 이 없도록 보장합니다. 만약 cycle 이 있다면:

$$
x \to f_1 \to y \to f_2 \to x
$$

이는 $x$ 를 정의하는 데 $x$ 가 필요해지는 모순. 따라서 execution trace 는 항상 DAG $\square$.

### 정리 3.2 (Leaf Gradient 의 누적)

Scalar loss $L$ 에 대해 동일한 leaf $x$ 가 여러 경로로 도달할 때:

$$
x.\text{grad} = \sum_{\text{path}} \frac{\partial L}{\partial x}|_{\text{path}}
$$

**증명**:

Backward 는 reverse topological order 로 각 node 의 local VJP 를 호출. 같은 leaf 에 도달하는 여러 경로의 gradient 는 leaf node 에서 더해짐 (accumulation) $\square$.

### 정리 3.3 (`no_grad()` vs `inference_mode()`)

| 특성 | `no_grad()` | `inference_mode()` |
|------|-----------|----------|
| Gradient 추적 | 해제 | 해제 |
| Tensor version 증가 | Yes | No |
| Double backward 호환 | Yes | No |
| 속도 | 약간 빠름 | **훨씬 빠름** |

**증명** (성능):

`inference_mode()` 는 tensor 의 version counter 를 증가시키지 않으므로:
1. Tensor metadata 업데이트 없음
2. Python object 생성 최소화
3. 메모리 할당 감소

따라서 inference 시 `inference_mode()` 가 더 효율적 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Graph Node 의 동적 생성

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
print(f"x.is_leaf: {x.is_leaf}")
print(f"x.grad_fn: {x.grad_fn}")

y = x ** 2
print(f"\ny = x^2")
print(f"y.is_leaf: {y.is_leaf}")
print(f"y.grad_fn: {y.grad_fn}")

z = y + 1
print(f"\nz = y + 1")
print(f"z.is_leaf: {z.is_leaf}")
print(f"z.grad_fn: {z.grad_fn}")

# Backward
L = z.sum()
L.backward()

print(f"\nAfter backward:")
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")
print(f"z.grad: {z.grad}")
```

**예상 출력**:
```
x.is_leaf: True
x.grad_fn: None

y = x^2
y.is_leaf: False
y.grad_fn: PowBackward0

z = y + 1
z.is_leaf: False
z.grad_fn: AddBackward0

After backward:
x.grad: tensor(4.)
y.grad: None
z.grad: None
```

### 실험 2 — Loop 에서의 Graph 누적

```python
import torch
import sys

x = torch.tensor(1.0, requires_grad=True)
size_before = sys.getsizeof(x.__dict__)

# Loop 에서 graph 누적
for i in range(10):
    x = x ** 2
    size_after = sys.getsizeof(x.__dict__)
    print(f"Iter {i}: grad_fn = {x.grad_fn}, x = {x.item()}")

# 메모리 누수 확인
print(f"\nx.grad_fn chain depth 확인:")
node = x.grad_fn
depth = 0
while node is not None and depth < 15:
    print(f"  {depth}: {node}")
    node = node.next_functions[0][0]
    depth += 1
```

**예상 출력**:
```
Iter 0: grad_fn = PowBackward0, x = 1.0
Iter 1: grad_fn = PowBackward0, x = 1.0
...
Iter 9: grad_fn = PowBackward0, x = 1.0

x.grad_fn chain depth 확인:
  0: PowBackward0
  1: PowBackward0
  2: PowBackward0
  ...
  10: PowBackward0
```

### 실험 3 — `no_grad()` vs `inference_mode()` 성능 비교

```python
import torch
import time

x = torch.randn(1000, 1000, requires_grad=True)

# Standard forward (추적 활성화)
start = time.time()
for _ in range(100):
    y = torch.matmul(x, x)
time_grad_enabled = time.time() - start

# no_grad()
start = time.time()
with torch.no_grad():
    for _ in range(100):
        y = torch.matmul(x, x)
time_no_grad = time.time() - start

# inference_mode()
start = time.time()
with torch.inference_mode():
    for _ in range(100):
        y = torch.matmul(x, x)
time_inference = time.time() - start

print(f"grad_enabled: {time_grad_enabled*1000:.2f} ms")
print(f"no_grad():    {time_no_grad*1000:.2f} ms")
print(f"inference_mode(): {time_inference*1000:.2f} ms")
print(f"\nSpeedup (no_grad): {time_grad_enabled/time_no_grad:.2f}x")
print(f"Speedup (inference_mode): {time_grad_enabled/time_inference:.2f}x")
```

**예상 출력**:
```
grad_enabled: 2341.23 ms
no_grad():    1823.45 ms
inference_mode(): 1234.56 ms

Speedup (no_grad): 1.28x
Speedup (inference_mode): 1.90x
```

### 실험 4 — 여러 경로에서의 Gradient 누적

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = x + 1

# 두 경로로부터 x 에 도달
L = y + z
L.backward()

print(f"x.grad: {x.grad}")
print(f"Expected: dy/dx + dz/dx = 2x + 1 = 2*2 + 1 = 5")
```

**예상 출력**:
```
x.grad: tensor(5.)
Expected: dy/dx + dz/dx = 2x + 1 = 2*2 + 1 = 5
```

### 실험 5 — torchviz 로 Graph 시각화

```python
import torch
from torch.autograd.variable import Variable
try:
    from torchviz import make_dot
    
    x = torch.randn(2, 3, requires_grad=True)
    y = x ** 2
    z = y.mean()
    
    # Graph 를 파일로 저장 (Graphviz 필요)
    graph = make_dot(z, params={'x': x})
    graph.render('computation_graph', format='pdf', cleanup=True)
    print("Graph saved as computation_graph.pdf")
    
except ImportError:
    print("torchviz not installed. Install: pip install torchviz graphviz")
```

---

## 🔗 실전 활용

### 1. 메모리 누수 방지 — Loop 에서의 Detach

```python
# BAD: Graph 계속 쌓임
losses = []
for epoch in range(100):
    output = model(batch)
    loss = criterion(output, target)
    losses.append(loss)  # Loss 가 graph 를 유지
    loss.backward()
    optimizer.step()

# GOOD: Detach 로 graph 끊음
losses = []
for epoch in range(100):
    output = model(batch)
    loss = criterion(output, target)
    losses.append(loss.detach())  # Graph 버림, 값만 저장
    loss.backward()
    optimizer.step()
```

### 2. Inference 최적화

```python
# Slower
with torch.no_grad():
    predictions = model(test_batch)

# Faster (권장)
with torch.inference_mode():
    predictions = model(test_batch)
```

### 3. Double Backward 를 위한 Graph 유지

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2
loss = y.sum()

# First backward 후 graph 유지
loss.backward(create_graph=True)

# Second backward 가능
gradient = x.grad
second_loss = (gradient ** 2).sum()
second_loss.backward()  # x 의 second derivative 계산
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Single forward pass | 조건문이나 루프 있을 때 graph 구조가 실행마다 변함 → 각 실행에서 새 graph 생성 |
| Tensor device 일정 | CPU ↔ CUDA 이동 시 graph 유지 가능하지만 비용 증가 |
| Backward 마다 graph 초기화 | `retain_graph=True` 로 명시적 유지 가능 |
| In-place operation 제한 | `x += 1` 같은 in-place op 는 gradient 를 깰 수 있음 → 권장하지 않음 |

---

## 📌 핵심 정리

$$\boxed{\text{Forward}: \text{DAG 생성}, \quad \text{Backward}: \text{Reverse topological sort}}$$

| 속성 | Leaf | Intermediate |
|------|------|-------------|
| `grad_fn` | None | Function node |
| `is_leaf` | True | False |
| `grad` 저장 | Yes | No |
| Backward 영향 | Yes (source) | Yes (propagation) |
| 수정 가능 | `requires_grad=False` 로만 | N/A |

**Dynamic Graph 의 핵심**:
- Forward 실행 → 즉시 graph node 생성
- Backward → reverse topological order 로 gradient 계산
- Leaf 에만 `grad` 저장 (메모리 효율)

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 code 에서 `y.grad` 가 None 인 이유를 설명하라:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward()
print(y.grad)  # None
```

<details>
<summary>해설</summary>

`y` 는 intermediate tensor (grad_fn = PowBackward). 

PyTorch 는 backward 후 intermediate tensor 의 gradient 를 버림 (메모리 절약). 

`y.grad` 가 필요하면 `y.retain_grad()` 호출:

```python
y.retain_grad()
loss.backward()
print(y.grad)  # Now: tensor(4.)
```

$\square$

</details>

**문제 2** (심화): 다음 code 의 출력을 예측하고, 왜 그렇게 동작하는지 설명하라:

```python
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

for i in range(3):
    x = x ** 2

loss = x.sum()
loss.backward()
print(x.grad)
```

<details>
<summary>해설</summary>

각 iteration 에서 `x = x ** 2` 이므로:
- i=0: x = [[1, 4]]
- i=1: x = [[1, 16]]
- i=2: x = [[1, 256]]

Backward 의 chain rule:
- $\frac{\partial L}{\partial x_2} = 2 x_2 = 512$ (i=2 에서 x^2 의 미분)
- $\frac{\partial L}{\partial x_1} = 2 x_1 \cdot 2 x_1 \cdot 2 x_1 = 8 x_1^3 = 8$ (3 단계 chain)

**예상 출력**: `tensor([[8., 512.]])`

각 element 의 gradient 가 iteration 단계에 따라 지수적으로 증가 $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 dynamic graph 는 flexibility 가 높지만, TensorFlow 2.x 의 eager execution 과 비교하면 어떤 차이가 있는가? 각각 어느 상황에서 더 유리한가?

<details>
<summary>해설</summary>

| 특성 | PyTorch Dynamic | TensorFlow Eager |
|------|---------|----------|
| Graph 생성 | Forward 중 즉시 | Forward 실행 시점 |
| Python 제어문 | 자연스러움 | `@tf.function` 필요 |
| Debugging | 직관적 (Python debugger) | Less intuitive |
| 최적화 | Limited | `@tf.function` 으로 graph 최적화 |
| 성능 | Flexible, variable ops | Graph optimization 가능 |

**선택**:
- Dynamic graph: Research, variable-length sequence, RNN
- Eager + graph optimization: Production, fixed-size 입력

PyTorch `torch.jit.trace` / `torch.jit.script` 도 graph optimization 제공 $\square$

</details>

---

<div align="center">

[◀ 이전](./02-vjp-mathematics.md) | [📚 README](../README.md) | [다음 ▶](./04-backward-topological-sort.md)

</div>
