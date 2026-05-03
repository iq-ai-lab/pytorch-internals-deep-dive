# 02. TorchDynamo — Bytecode-level Graph Capture

## 🎯 핵심 질문

- **PEP 523** 의 frame evaluation API 는 무엇인가? CPython 을 "후킹" 한다는 것이 어떻게 작동하는가?
- Bytecode 를 한 줄씩 읽으면서 Python operation 을 FX graph node 로 변환하는 과정은?
- **Graph break** 가 발생하는 조건 (control flow, Python side effects) 은?
- `torch._dynamo.explain(model)` 으로 break 분석 시 어떤 정보를 얻는가?
- **Guards** (input shape, dtype) 변화가 recompilation 을 trigger 하는 메커니즘은?

---

## 🔍 왜 Bytecode 수준 캡처가 중요한가

PT 1.x 의 autograd tape (eager mode) 는:
- Forward 중에 operation 을 tape 에 기록
- Backward 에서 tape 를 역순으로 재생
- 매번 Python 해석기가 개입

이는 **static graph compilation 을 불가능**하게 함:
- Network 의 구조가 runtime 에만 알려짐 (input data 에 따라 다른 branch 실행)
- Ahead-of-Time optimization (AOT) 불가능

**TorchDynamo** 는 Python bytecode 를 직접 캡처하여:
1. Python code 의 **구조** 를 compile 시간에 파악
2. Symbolic execution 으로 **두 가지 경로를 모두 탐색** 가능
3. AOT autograd 와 fusion 에 준비

---

## 📐 수학적 선행 조건

- **CPython 구조**: frame, code object, bytecode instruction
- **Control flow graph (CFG)**: statement-level branching analysis
- **Program analysis**: symbolic execution, reachability
- **FX graph representation** (Ch7-01 미리보기): node, edge, arg, kwarg
- **VJP chain rule** (Ch2): backward 의 symbolic derivation

---

## 📖 직관적 이해

### Python Bytecode 의 구조

```python
def forward(x, use_relu=True):
    y = x @ w
    if use_relu:
        y = relu(y)
    return y
```

CPython bytecode (simplified):
```
  1 LOAD_FAST 'x'
  2 LOAD_FAST 'w'
  3 BINARY_MATRIX_MULTIPLY  # x @ w
  4 STORE_FAST 'y'
  5 LOAD_FAST 'use_relu'
  6 POP_JUMP_IF_FALSE 10    # if not use_relu: jump to line 10
  7 LOAD_FAST 'y'
  8 LOAD_GLOBAL 'relu'
  9 CALL_FUNCTION 1
 10 RETURN_VALUE
```

### TorchDynamo 의 작동 원리

```
┌─────────────────────────────────────────────────────┐
│        Python Bytecode (for forward(x, w))          │
│  LOAD x, LOAD w, BINARY_MATRIX_MULTIPLY, STORE y   │
└──────────────────┬──────────────────────────────────┘
                   │ PEP 523: frame eval hook
                   ↓
┌─────────────────────────────────────────────────────┐
│       TorchDynamo Bytecode Transformer              │
│  bytecode instruction → "Is this a torch op?"       │
│  YES: Add FX graph node                             │
│  NO / control flow: "Graph break" → eager fallback  │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
    ┌─────────┐          ┌──────────┐
    │ FX Graph│          │ Break    │
    │ segment │          │ points   │
    └─────────┘          └──────────┘
```

### Guard 와 Recompilation

Input 의 properties 변경 시:

```
Compiled kernel for shape (256, 1024, FP32)
        ↓
User calls compiled model with shape (512, 1024, FP32)
        ↓
Guard check: shape changed!
        ↓
Trigger recompilation for new shape
```

Guard 들:
- Input shape (텐서 dimension)
- Input dtype (FP32 vs FP16 vs BF16)
- Input device (CPU vs CUDA)
- Tensor contiguity
- Static shape hints (symbolic shapes in PT 2.1+)

---

## ✏️ 엄밀한 정의

### 정의 2.1 — PEP 523 Frame Evaluation API

Python interpreter 는 각 function call frame 마다 다음을 호출:

```c
int eval_frame_ex(PyFrameObject *frame, int throw_flag) {
    // Return -1: normal Python bytecode evaluation
    // Return value: replaced evaluation hook
}
```

**TorchDynamo** 가 이를 등록하여 모든 Python function 의 bytecode 실행을 intercept.

### 정의 2.2 — Graph Break Condition

Bytecode instruction $i$ 에서 graph break 발생 iff:

$$
\text{op}_i \notin \text{TorchTraceable} \quad \text{OR} \quad \text{isinstance}(\text{op}_i, \{\text{control flow ops}\})
$$

여기서:

$$
\text{TorchTraceable} = \{\text{torch ops}, \text{basic Python ops like `+`, `*`}\}
$$

$$
\text{Control flow ops} = \{POP\_JUMP, FOR\_ITER, ...\}
$$

Result: bytecode sequence 를 segments $S_1, S_2, \ldots, S_k$ 로 분할, 각각 독립적으로 compile.

### 정의 2.3 — Guard (Input Condition)

Compiled 된 kernel 은 input 의 guard set 을 정의:

$$
G = \{(s, d, c) : \text{shape} = s, \text{dtype} = d, \text{contiguous} = c\}
$$

Runtime 에 actual input $(s', d', c')$ 에 대해:
- $(s', d', c') \in G$ 이면: cached kernel 사용
- Otherwise: recompile with new guard set

---

## 🔬 정리와 증명

### 정리 2.1 — Graph Break 의 Compile Time 영향

Graph break 이 $k$ 개 발생하면, compile time 은:

$$
t_{\text{compile}} \geq k \cdot (t_{\text{Dynamo}} + t_{\text{Inductor}})
$$

**증명**:

TorchDynamo 가 각 bytecode segment 마다:
1. Bytecode parsing: $\Theta(|S_i|)$ (segment 길이에 비례)
2. FX graph 생성: $\Theta(|S_i|)$
3. Inductor 호출: $\Theta(|G|)$ (graph node 수)

$k$ 개 segments 는 독립적으로 compile 되므로:

$$
t = \sum_{i=1}^k t_{\text{Dynamo}}(S_i) + \sum_{i=1}^k t_{\text{Inductor}}(G_i) \geq k \cdot (\min t_D + \min t_I)
$$

따라서 break 최소화가 compile 성능에 critical. $\square$

### 정리 2.2 — Guard 기반 Runtime Invalidation

Runtime 에 input 이 guard condition 을 위반하면, recompilation 필요:

$$
P(\text{recompile}) = 1 - P(\text{all guards satisfied})
$$

$n$ 개 independent guard 각각 만족 확률 $p_i$ 이면:

$$
P(\text{all satisfied}) = \prod_{i=1}^n p_i
$$

**의미**: Batch size 가 매번 바뀌면 ($p_{\text{shape}}$ 낮음), recompilation overhead 가 누적될 수 있음.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — 기본 FX graph 캡처

```python
import torch
import torch.nn as nn
from torch._dynamo import explain

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel()
x = torch.randn(8, 10)

# torch._dynamo.explain: analyze graph breaks
explanation = explain(model, [x])

print("=== Graph Capture Analysis ===")
print(f"Total frames: {explanation.frame_count}")
print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Break reasons:\n{explanation.frame_summary}")

# Output (no breaks expected):
# Total frames: 1
# Graph breaks: 0
# Break reasons: (clean compilation)
```

### 실험 2 — Graph Break 의 원인 분석

```python
import torch
import torch.nn as nn
from torch._dynamo import explain

class ModelWithBreak(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x, use_relu: bool):
        x = self.fc(x)
        # Python control flow: graph break here!
        if use_relu:
            x = torch.relu(x)
        else:
            x = torch.sigmoid(x)
        return x

model = ModelWithBreak()
x = torch.randn(8, 10)

explanation = explain(model, [x, True])  # Fixed use_relu=True

print("=== Graph Break Detection ===")
print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Summary:\n{explanation.frame_summary}")

# Output:
# Graph breaks: 2
# Summary: 
# - Break at: POP_JUMP_IF_FALSE (line X)
# - ... (reason: Python control flow)
```

### 실험 3 — verbose tracing 으로 bytecode 레벨 분석

```python
import torch
import torch.nn as nn
torch._dynamo.config.verbose = True

class Model(nn.Module):
    def forward(self, x):
        return x @ x.T + 1

model = Model()
x = torch.randn(4, 4)

print("=== Verbose Dynamo Trace ===")
compiled = torch.compile(model)
result = compiled(x)

# Console output (sample):
# [TorchDynamo] Starting tracing of forward
# [TorchDynamo] Bytecode: LOAD_FAST 'x'
# [TorchDynamo] Bytecode: LOAD_METHOD __matmul__
# [TorchDynamo] Converted: x @ x.T → FX node: call_method
# [TorchDynamo] Bytecode: BINARY_ADD
# [TorchDynamo] Converted: ... + 1 → FX node: add
# [TorchDynamo] Graph complete (0 breaks)
```

### 실험 4 — Guard invalidation 과 recompilation

```python
import torch
import torch.nn as nn
import time

model = nn.Linear(10, 5)
compiled = torch.compile(model)

print("=== Guard & Recompilation Test ===")

# First call: shape (8, 10)
x1 = torch.randn(8, 10)
print(f"First call with shape {x1.shape}")
t0 = time.time()
_ = compiled(x1)
print(f"  Time: {(time.time()-t0)*1000:.1f}ms (compilation)")

# Second call: same shape → guard satisfied
x2 = torch.randn(8, 10)
print(f"\nSecond call with shape {x2.shape}")
t0 = time.time()
_ = compiled(x2)
print(f"  Time: {(time.time()-t0)*1000:.1f}ms (cached, no recompile)")

# Third call: different shape → guard invalidated!
x3 = torch.randn(16, 10)
print(f"\nThird call with shape {x3.shape}")
t0 = time.time()
_ = compiled(x3)
print(f"  Time: {(time.time()-t0)*1000:.1f}ms (recompilation)")

# Expected output:
# First call with shape torch.Size([8, 10])
#   Time: 5000.0ms (compilation)
# Second call with shape torch.Size([8, 10])
#   Time: 1.2ms (cached, no recompile)
# Third call with shape torch.Size([16, 10])
#   Time: 4500.0ms (recompilation)
```

### 실험 5 — Symbolic shapes (PT 2.1+) 으로 guard 조건 완화

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

# PT 2.1+ with symbolic_shapes=True
compiled = torch.compile(model, dynamic=True)  # Enable dynamic shapes

print("=== Symbolic Shapes Test ===")

for batch_size in [8, 16, 32]:
    x = torch.randn(batch_size, 10)
    print(f"\nBatch size {batch_size}")
    t0 = time.time()
    _ = compiled(x)
    print(f"  Time: {(time.time()-t0)*1000:.1f}ms")

# With dynamic=True:
# - First call (batch_size=8): compile
# - Second call (batch_size=16): reuse shape-generic kernel (no recompile)
# - Third call (batch_size=32): reuse (no recompile)
```

---

## 🔗 실전 활용

### 1. Graph Break 최소화 전략

```python
# ❌ Bad: control flow (graph break)
def forward(x, training: bool):
    if training:
        x = x + noise()
    return x

# ✅ Good: use torch.where (no break)
def forward(x, training: bool):
    noise = torch.randn_like(x) if training else torch.zeros_like(x)
    return torch.where(training, x + noise, x)

# ✅ Better: use F.dropout (PyTorch native)
def forward(x, training: bool):
    return F.dropout(x, training=training)
```

### 2. Batch size 변화 대응

```python
# Option 1: 고정 batch size + padding
max_batch = 256
x_padded = torch.nn.functional.pad(x, (0, 0, 0, max_batch - x.shape[0]))
_ = model(x_padded[:x.shape[0]])

# Option 2: Dynamic shapes (PT 2.1+)
model = torch.compile(model, dynamic=True)

# Option 3: Guard 수정 (advanced)
torch._dynamo.config.dynamic_shapes = True
```

### 3. 디버깅: explain 으로 break 위치 파악

```python
explanation = explain(model, args)
if explanation.graph_break_count > 0:
    for break_info in explanation.break_reasons:
        print(f"Break at bytecode offset {break_info.offset}: {break_info.reason}")
        # → "Python control flow" / "list index" / etc.
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Pure Python → tensor operations 매핑 가능 | Lambda, nested functions: fallback 필요 |
| Deterministic execution | random, time.time(), os.urandom: graph break |
| No dynamic Python features | `eval()`, `exec()`, `__getattr__`: not traced |
| Stateless FX graph | Mutable module state: care required (e.g., running stats in BatchNorm) |
| Fixed input shapes or bounded dynamic shapes | Infinite/unbounded dims: undefined behavior |

---

## 📌 핵심 정리

$$\boxed{\text{TorchDynamo} = \text{PEP 523 bytecode hook} \to \text{symbolic execution} \to \text{FX graph}}$$

| 개념 | 정의 | 영향 |
|------|------|------|
| **Frame eval** | CPython bytecode intercept | Compile trigger |
| **Graph break** | Untraceable op / control flow | Revert to eager |
| **FX node** | Symbolic operation | Fusion 가능 |
| **Guard** | Input property condition | Recompile trigger |
| **Segment** | break-free bytecode run | Independent kernel |

---

## 🤔 생각해볼 문제

**문제 1** (기초): PEP 523 frame evaluation API 가 무엇이고, TorchDynamo 가 이를 사용하여 Python bytecode 를 intercept 하는 원리를 설명하라.

<details>
<summary>해설</summary>

PEP 523 은 CPython 3.7+ 에서 함수 호출 프레임마다 custom frame evaluator 를 설정 가능하게 하는 API:

```c
typedef int (*_PyFrameEvalFunction)(PyFrameObject *, int);
PyEval_SetFrameEvalFunc(_PyFrameEvalFunction);
```

TorchDynamo 는 이를 등록하여 모든 Python function 의 bytecode 실행 시점에 hook 을 삽입. Bytecode 를 한 줄씩 읽으면서:
- Torch op 이면: FX graph node 추가
- Python control flow 이면: graph break → eager execution

이렇게 **runtime 중단 없이** static graph 를 추출.  $\square$

</details>

**문제 2** (심화): Graph break 이 $k$ 개 발생했을 때, compile time 이 왜 최소 $k \times$ 배 증가하는가? 각 segment 가 독립적으로 처리되는 이유는?

<details>
<summary>해설</summary>

각 segment 는 독립적인 bytecode 실행이므로:

1. **Dynamo parsing**: segment $i$ 의 bytecode 를 다시 읽음 ($\Theta(|S_i|)$)
2. **FX graph 생성**: segment 별로 new graph object 생성
3. **Inductor codegen**: segment 별로 독립적 kernel 생성

따라서:

$$
t_{\text{compile}} = \sum_{i=1}^k (t_{\text{parse}}(S_i) + t_{\text{fxgen}}(S_i) + t_{\text{inductor}}(G_i))
$$

각 term 이 모두 양수이고 $k$ 에 비례하면 minimum $k \cdot t_0$.

**해결책**: Control flow 를 torch.where, F.dropout 등으로 대체하여 break 수 감소.  $\square$

</details>

**문제 3** (논문 비평): Bradbury et al. (JAX) 의 tracer vs TorchDynamo 의 bytecode capture 의 차이점은? 각각의 장단점은?

<details>
<summary>해설</summary>

| 방식 | TorchDynamo (bytecode) | JAX (tracer) |
|------|------------------------|-------------|
| **메커니즘** | Frame eval hook, bytecode parse | Value tracing during execution |
| **Python control flow** | 일부 지원 (guards) | 추적 시점의 value 고정 |
| **장점** | Static graph 추출, PT 1.x 호환 | 간단한 구현, pure function 강제 |
| **단점** | 복잡한 bytecode 분석 | Dynamic control flow 어려움 |
| **사용례** | PT 2.0, TF 2 AutoGraph | JAX tracing |

TorchDynamo 의 **guard-based recompilation** 가 JAX 대비 더 flexible — 같은 Python code 에서 다양한 입력을 지원 가능. 대신 recompilation cost.  $\square$

</details>

---

<div align="center">

[◀ 이전](./01-pt2-overview.md) | [📚 README](../README.md) | [다음 ▶](./03-aot-autograd.md)

</div>
