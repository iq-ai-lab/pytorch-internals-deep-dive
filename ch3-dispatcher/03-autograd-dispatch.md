# 03. Autograd Key 와 Backward Dispatch

## 🎯 핵심 질문

- Forward pass 와 backward pass 는 dispatcher 의 다른 DispatchKey 를 통해 어떻게 분리 등록되는가?
- `Autograd[CUDA]` key 에 등록된 kernel 이 하는 일은? (metadata setup → redispatch)
- `torch::autograd::Function` C++ API 는 Python 의 `torch.autograd.Function` 과 어떻게 연관되는가?
- `torch.no_grad()` context 에서 Autograd key 가 disabled 되는 메커니즘은?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 "magic" — `loss.backward()` 한 줄이 자동으로 chain rule 을 계산하는 것 — 은 dispatcher 의 `Autograd` key 에서 비롯됩니다:

1. **Forward**: `Autograd[CUDA]` kernel → metadata 저장 (saved tensors, op 정보) → redispatch to CUDA kernel → 실제 계산
2. **Backward**: computation graph 역순 → 각 노드의 backward fn → local VJP 적용 → upstream gradient 누적

이 분리가 없으면, 모든 op 를 manual 로 구현하거나 (JAX-style), 아니면 op 마다 forward/backward 를 강제로 같이 정의해야 합니다. Custom `autograd.Function` 을 만들 때도, 이 메커니즘을 이해하면 backward 구현의 실수를 줄일 수 있습니다.

---

## 📐 수학적 선행 조건

- **Chapter 2 Autograd 핵심**: VJP (Vector-Jacobian Product), computation graph, chain rule
- **Dynamic dispatch**: DispatchKeySet, fallthrough mechanism (Chapter 3-01)
- **Variable tracking**: requires_grad, grad_fn, 자동 미분 전파
- (선택) Context manager protocol: `__enter__`, `__exit__`

---

## 📖 직관적 이해

### Forward + Backward 의 분리 dispatch

```
torch.add(x, y)  (x.requires_grad=True)
        │
        ↓
   Dispatcher
        │
        ├─ Check DispatchKeySet: {Autograd[CUDA]}
        │  ↓
        ├─ Call Autograd[CUDA] kernel
        │  ├─ 1. Save inputs/outputs for backward
        │  ├─ 2. Create AddBackward0 node
        │  ├─ 3. Set result.grad_fn = AddBackward0
        │  └─ 4. Redispatch with Autograd key removed
        │
        ├─ Check DispatchKeySet: {CUDA, FP32, ...}  (without Autograd)
        │  ↓
        ├─ Call CUDA kernel (actual computation)
        │  └─ return result (just Tensor, no grad info)
        │
        └─ Return result + grad_fn
```

**핵심**: Autograd key 는 "metadata layer" — 실제 계산은 하지 않고, backward 를 준비만 함.

### `torch.no_grad()` 의 메커니즘

```python
# with torch.no_grad():
#   Dispatcher: Autograd key 를 DispatchKeySet 에서 제거
#   → CUDA kernel 직접 호출
#   → result.requires_grad = False
```

C++ 내부:

```cpp
// torch/csrc/autograd/grad_mode.cpp
class GradModeGuard {
  bool prev_grad_enabled = GradMode::is_enabled();
  GradMode::set_enabled(false);  // Autograd key 제거
  // ... (destructor 에서 restore)
};
```

---

## ✏️ 엄밀한 정의

### 정의 3.7 — Autograd Kernel (Generic Forward)

For operator $f$, the Autograd kernel $f_\text{autograd}$ performs:

$$f_\text{autograd}(\mathbf{x}, \Phi) := \begin{cases}
1. & \text{Save} \: \mathbf{x}, \Phi, \text{ and op metadata to tape}\\
2. & \text{Create node } n = \text{BackwardFn}(f)\\
3. & \text{Redispatch } f(\mathbf{x}, \Phi) \text{ without Autograd key}\\
4. & \text{Attach } n \text{ to result as } \text{grad\_fn}\\
5. & \text{Return} \: (\mathbf{y}, n)
\end{cases}$$

여기서:
- $\mathbf{x}$ = input tensors
- $\Phi$ = non-tensor arguments (scalars, options)
- $f(\mathbf{x}, \Phi)$ = actual kernel (CPU/CUDA)
- $\text{BackwardFn}$ = C++ class implementing VJP

### 정의 3.8 — Backward Node (Autograd::Function)

```cpp
class AddBackward0 : public torch::autograd::Node {
  void release_variables() override {
    saved_self_.reset();  // clean up saved data
    saved_other_.reset();
  }
  
  variable_list apply(variable_list grads) override {
    // grads[0] = upstream gradient (dL/dy)
    // return: [dL/dx, dL/dy'] for next layer
    auto grad_output = grads[0];
    
    // VJP of add: ∇f^T v = (v, v) — both inputs get same grad
    return {grad_output, grad_output * alpha};  
  }
  
  torch::autograd::SavedVariable saved_self_, saved_other_;
  Scalar alpha_;
};
```

### 정의 3.9 — Dispatch Key 의 presence/absence

$$\text{DispatchKeySet}_{t} = \begin{cases}
\{\text{Autograd[CUDA], CUDA, FP32, ...}\} & \text{if} \: \text{requires\_grad} = \text{True}\\
\{\text{CUDA, FP32, ...}\} & \text{if} \: \text{requires\_grad} = \text{False} \: \text{or} \: \text{torch.no\_grad()}
\end{cases}$$

Dispatcher 는 "Autograd" 의 presence 여부로 backward graph 구축 여부 결정.

---

## 🔬 정리와 증명

### 정리 3.4 — Autograd Kernel 의 correctness

**주장**: Autograd kernel 이 input 을 저장하고 backward node 를 attach 한 후, 원래의 forward kernel 을 redispatch 하면, backward pass 에서 chain rule 이 정확히 적용된다.

**증명**:

Forward pass:
$$\mathbf{y} = f(\mathbf{x}), \quad \text{grad\_fn} = \text{BackwardFn}_f$$

Backward pass (역순 traverse):
$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{v} \cdot J_f$$

여기서 $J_f$ 는 forward kernel 에서 saved 된 inputs 를 이용해 계산 가능.

**핵심**: saved inputs 와 backward fn 의 분리 → lazy evaluation (필요한 순간까지 backward 계산 안 함) $\square$.

### 정리 3.5 — no_grad() 와 Autograd Key Removal

**주장**: `torch.no_grad()` context 에서 모든 ops 는 Autograd key 없이 dispatch 되므로, backward graph 가 생성되지 않는다.

**증명**:

1. `GradModeGuard(false)` enter → `GradMode::set_enabled(false)`
2. Forward op 호출 → `compute_dispatch_keys()` 함수 실행:
   ```cpp
   DispatchKeySet keys = {...};  // normally includes Autograd
   if (!GradMode::is_enabled()) {
     keys.remove(DispatchKey::Autograd);  // key 제거!
   }
   return keys;
   ```
3. Dispatcher: Autograd kernel lookup 안 함 → CUDA/CPU kernel 직접 호출
4. Result 의 `requires_grad = False` (no grad_fn 붙음)

따라서 backward graph 가 끊김 $\square$.

### 따름 정리 3.6 — requires_grad Propagation

$$\text{requires\_grad}(\mathbf{y}) = \text{requires\_grad}(\mathbf{x}_1) \lor \text{requires\_grad}(\mathbf{x}_2) \lor ...$$

(OR 연산 — input 중 하나라도 requires_grad=True 면, output 도 True)

이는 dispatcher 가 input 들의 DispatchKeySet 을 보고 Autograd 포함 여부 결정하기 때문.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Autograd key 의 활성 여부 확인

```python
import torch

# requires_grad=True → Autograd key 활성
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=False)
result = torch.add(x, y)

print(f"result.requires_grad: {result.requires_grad}")  # True
print(f"result.grad_fn: {result.grad_fn}")  # AddBackward0

# requires_grad=False → Autograd key 비활성
x2 = torch.tensor([1.0, 2.0], requires_grad=False)
result2 = torch.add(x2, y)

print(f"result2.requires_grad: {result2.requires_grad}")  # False
print(f"result2.grad_fn: {result2.grad_fn}")  # None
```

### 실험 2 — torch.no_grad() 의 영향

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# no_grad 밖 (Autograd key 활성)
with torch.enable_grad():
    y1 = x * 2
    print(f"y1.requires_grad: {y1.requires_grad}")  # True
    print(f"y1.grad_fn: {y1.grad_fn}")  # MulBackward0

# no_grad 안 (Autograd key 제거)
with torch.no_grad():
    y2 = x * 2
    print(f"y2.requires_grad: {y2.requires_grad}")  # False
    print(f"y2.grad_fn: {y2.grad_fn}")  # None

# Computation graph 비교
print(f"y1.requires_grad: {y1.requires_grad}")  # True (여전히)
print(f"y2.requires_grad: {y2.requires_grad}")  # False
```

### 실험 3 — Saved variables 의 크기 확인

```python
import torch

# Large tensor 를 input 으로
x = torch.randn(1000, 1000, requires_grad=True)
y = torch.randn(1000, 1000, requires_grad=True)
z = torch.add(x, y)

# z.grad_fn 은 AddBackward0
# 내부적으로 x 와 y 의 reference 를 saved_tensors 에 저장
# (또는 일부 경우, shape/dtype/stride 정보만 저장 후 필요 시 recompute)

# backward 호출
loss = z.sum()
loss.backward()

print(f"x.grad shape: {x.grad.shape}")  # [1000, 1000]
print(f"y.grad shape: {y.grad.shape}")  # [1000, 1000]

# Backward 이후, grad_fn 의 saved tensors 는 자동 cleanup
# (일부는 memory 절약을 위해 gradient checkpointing 로 recompute)
```

### 실험 4 — Custom autograd.Function 과 dispatcher

```python
import torch
from torch.autograd import Function

class MyAdd(Function):
    @staticmethod
    def forward(ctx, x, y):
        # Dispatcher 관점: Autograd[CUDA] kernel
        # 1. Save for backward
        ctx.save_for_backward(x, y)
        # 2. Actual computation (no Autograd key)
        return x + y
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward node 의 apply() 에 해당
        x, y = ctx.saved_tensors
        # VJP: both get same grad
        return grad_output, grad_output

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)

# MyAdd 를 통한 forward (Dispatcher 를 거치지 않고, 직접 custom fn 호출)
result = MyAdd.apply(x, y)
print(f"result.grad_fn: {result.grad_fn}")  # MyAddBackward

loss = result.sum()
loss.backward()
print(f"x.grad: {x.grad}")  # [1., 1.]
print(f"y.grad: {y.grad}")  # [1., 1.]
```

---

## 🔗 실전 활용

### 1. Custom Operation 의 backward 구현

`torch.autograd.Function` 을 상속하면, dispatcher 를 거치지 않고 forward/backward 를 pair 로 정의:

```python
class CustomOp(Function):
    @staticmethod
    def forward(ctx, x):
        # Custom forward logic
        y = x ** 2
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # Custom backward logic
        x, = ctx.saved_tensors
        return grad_output * 2 * x
```

### 2. Efficient Backward via Recomputation (Gradient Checkpointing)

Large forward pass 에서 intermediate tensors 를 저장하면 memory 폭증. 대신:

```python
# Low memory
def forward_recompute(x):
    return x ** 2

def forward_with_checkpoint(x):
    # Autograd key 제거 상태에서 forward → intermediate 저장 안 함
    # Backward 시 필요하면 forward recompute
    with torch.no_grad():
        y = forward_recompute(x)
    return y
```

### 3. Profiling Backward Graph

```python
import torch

x = torch.randn(10, requires_grad=True)
y = x ** 2
z = y.sum()

# backward graph 구조 확인
print(z.grad_fn)  # SumBackward0
print(z.grad_fn.next_functions)  # (PowBackward0, ...)
```

### 4. torch.compile 과 Autograd dispatch

TorchDynamo 는 graph 추출 시 Autograd key 상태 보존 → AOTAutograd 가 forward/backward 자동 분리.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 모든 op 가 Autograd key 등록 | 일부 낮은 수준 ops (metadata-only) 는 미등록 → manual Function 필요 |
| requires_grad propagation 이 자동 | Inplace ops (`.add_()`) 는 주의 — backward 가 깨질 수 있음 |
| Saved tensors 가 정확 | Gradient checkpointing 이나 memory pressure 시 approximation 가능 |
| Single-thread execution | Parallel/distributed setting 에서 gradient synchronization 필요 (DDP, FSDP) |

---

## 📌 핵심 정리

$$\boxed{\text{Forward: Autograd key} \xrightarrow{\text{save metadata}} \text{Redispatch} \to \text{Backward: VJP chain}}$$

| 개념 | 역할 |
|------|------|
| **Autograd key** | Dispatcher 의 tag — backward node 생성 여부 결정 |
| **Saved variables** | Forward input 들이 grad_fn 에 저장, backward 에서 Jacobian 계산용 |
| **requires_grad propagation** | Output 의 requires_grad = input 중 하나라도 True 면 True |
| **torch.no_grad()** | Autograd key 를 dispatch 에서 제거 → 계산 그래프 끝냄 |

**핵심**: Dispatcher 가 동적으로 Autograd key 를 on/off 하면서, 같은 kernel 이 forward 와 backward 정보를 모두 처리할 수 있게 해줍니다.

---

## 🤔 생각해볼 문제

**문제 1** (기초): `torch.no_grad()` 와 `x.detach()` 의 차이점을 명확히 설명하라. 둘 다 backward graph 를 끊지만, 내부 메커니즘이 어떻게 다른가?

<details><summary>해설</summary>

**torch.no_grad()**:
- Context manager — 범위 내 모든 ops 의 Autograd key 제거
- Dispatcher 차원에서 처리
- 범위 나가면 Autograd key 복구
- 효율적 (dispatcher 만 조작)

**x.detach()**:
- Tensor 메서드 — 새 tensor 반환, requires_grad=False
- 같은 storage 공유 (0-copy), grad_fn 만 제거
- Permanent (나중에 requires_grad=True 로 변경 불가)

```python
x = torch.tensor([1.0], requires_grad=True)

# no_grad
with torch.no_grad():
    y1 = x + 1  # y1.requires_grad = False
y2 = x + 1  # y2.requires_grad = True (graph 연결됨)

# detach
y3 = x.detach()  # y3.requires_grad = False
y4 = (x.detach()) + 1  # graph broken, y4.grad_fn = None
```

**메커니즘**:
- no_grad: Dispatcher 의 DispatchKeySet 조작
- detach: Tensor 의 requires_grad 플래그 조작 + grad_fn 제거

$\square$

</details>

**문제 2** (심화): Inplace operation (예: `x.add_(y)`) 에서 backward 가 어떻게 작동하고, 왜 위험한가? Dispatcher 의 관점에서 설명하라.

<details><summary>해설</summary>

**Forward**:
```python
x.add_(y)  # x 를 제자리에서 수정
# Dispatcher: Autograd key → backward node (AddBackward) 생성
# 그러나 x 의 storage 자체가 변함!
```

**Backward**:
```python
# Autograd node 가 saved_x (x 의 original value) 를 expect
# 하지만 x.add_(y) 이후 x 가 변했으므로...
# saved_x ≠ current x
# → Jacobian 계산 오류!
```

**예**:
```python
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0])
z = x.clone()  # z = 1.0
z.add_(x)  # z = 2.0, but backward fn expects z = 1.0
loss = z.sum()
loss.backward()  # ← gradient 부정확할 수 있음
```

**해결**:
- Inplace op 피하기 (PyTorch 권장)
- 또는 requires_grad=False 인 tensor 에만 inplace 사용
- Dispatcher 는 경고 발행하지만, 막지는 못함 (사용자 책임)

$\square$

</details>

**문제 3** (논문 비평): Modern PyTorch (2.0+) 의 AOTAutograd 는 dispatcher 의 Autograd key 를 어떻게 재정의/활용하고, 이것이 `torch.compile` 의 성능 향상과 어떻게 연결되는가? (Jia et al. "AOTAutograd" 논문 참고)

<details><summary>해설</summary>

**Traditional Autograd**:
- Dispatcher 의 Autograd key → forward kernel 호출 → backward node 생성
- Backward 는 dynamic (runtime 에 graph traverse)

**AOTAutograd** (PyTorch 2.0+):
- Graph capture (TorchDynamo) 에서 forward graph 추출
- `torch.func.grad()` 를 자동 적용 → backward graph 생성 (static)
- Forward + backward 를 모두 컴파일 가능한 형태로 변환
- Dispatcher 의 Autograd key 를 "functionalization" 단계에서 제거
- 대신 explicit backward kernel 을 생성 → 최적화 여지 증대

**성능 향상**:
1. Dynamic dispatch overhead 제거 (compile-time dispatch)
2. Kernel fusion (forward + backward 동시 최적화)
3. Memory efficiency (intermediate tensor 공유 가능)

**Dispatcher 와의 관계**:
- Autograd key 는 여전히 use 하지만, "static" 하게
- TorchDynamo 가 op 를 capture 할 때 Autograd key state 를 record
- Inductor 가 이 정보로 kernel generation 최적화

논문: Paszke, "Automatic Differentiation in PyTorch" (2017) 과 그 후속인 "AOTAutograd: An Efficient and Composable Autograd for JAX" (2021) 참고. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-aten-c10-layers.md) | [📚 README](../README.md) | [다음 ▶](./04-torch-library-custom-op.md)

</div>
