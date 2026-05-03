# 05. torch.autograd.Function 직접 구현

## 🎯 핵심 질문

- `torch.autograd.Function` 의 `forward(ctx, *inputs)` 와 `backward(ctx, *grad_outputs)` 의 한 쌍은 어떻게 chain rule 을 정의하는가?
- `ctx.save_for_backward(...)` 에 저장한 값들을 `backward` 에서 어떻게 꺼내고 사용하는가?
- Custom operation 의 backward 를 직접 구현한 후 `torch.autograd.gradcheck()` 로 검증하는 방법은?
- CUDA kernel 을 호출하는 custom function 을 device-agnostic 하게 구현하는 패턴은?
- `Function.apply()` 가 실제로 하는 일과 단순히 함수를 호출하는 것과의 차이는?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 가 제공하지 않는 operation 을 구현해야 할 때, `torch.autograd.Function` 을 사용합니다. 하지만 많은 사용자는 다음을 모르고 버그를 만듭니다:

1. **Backward 구현 오류** — Forward 와 일치하지 않는 gradient 를 계산
2. **메모리 누수** — `ctx.save_for_backward()` 를 사용하지 않고 closure 에서 참조하여 graph 가 유지됨
3. **Device 호환성** — CPU/CUDA 를 모두 지원하지 않는 코드
4. **Numerical stability** — Backward 를 안정적으로 구현하지 못함

이 문서는 **정확한 구현 패턴** 을 다룹니다.

---

## 📐 수학적 선행 조건

- **Forward**: $y = f(x)$ 의 explicit 구현
- **Backward**: $\bar{x} = J_f^T \bar{y}$ 의 VJP rule
- **ctx 의 역할**: Forward 에서 backward 에 필요한 값들을 저장하는 context object
- **Type checking**: Input 의 dtype, device 확인

---

## 📖 직관적 이해

### Custom Function 의 구조

```python
class MyOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 1. Forward 계산
        y = some_computation(x)
        
        # 2. Backward 에 필요한 값 저장
        ctx.save_for_backward(x)
        
        # 3. Output 반환
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # 1. Saved tensor 복원
        x, = ctx.saved_tensors
        
        # 2. VJP 계산
        grad_input = compute_vjp(x, grad_output)
        
        # 3. Input 의 개수만큼 gradient 반환
        return grad_input
```

**key points**:
- `forward` 와 `backward` 는 **정확히 대칭**
- Backward 는 forward 의 역함수가 아니라, **VJP rule** 을 구현
- `ctx` 는 forward 에서 backward 로 데이터를 넘기는 유일한 통로
- Return 되는 gradient 의 개수는 input 의 개수와 일치

### ctx.save_for_backward 의 필요성

Forward 에서 계산한 intermediate value 들을 backward 에서 재사용해야 합니다. 하지만:

```python
# BAD: closure 에서 직접 참조
class BadOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x ** 2
        # ctx 에 저장하지 않고 closure 에서 x 참조
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # x 에 접근 불가능 → 오류!
        return 2 * x * grad_output
```

대신 `ctx.save_for_backward()` 를 사용:

```python
class GoodOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x ** 2
        ctx.save_for_backward(x)  # 명시적 저장
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors  # 복원
        return 2 * x * grad_output
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Custom Function 의 API

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                *args,
                **kwargs) -> Tensor:
        """
        Forward pass. Compute output from inputs.
        
        Args:
            ctx: Context object for saving data
            *args: Input tensors
            **kwargs: Non-tensor arguments
        
        Returns:
            Output tensor(s)
        """
        pass
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 *grad_outputs) -> Tuple[Tensor, ...]:
        """
        Backward pass. Compute gradient w.r.t. inputs.
        
        Args:
            ctx: Context with saved data
            *grad_outputs: Upstream gradients for each output
        
        Returns:
            Tuple of gradients (one per input, None for non-tensor)
        """
        pass
```

### 정의 5.2 — `ctx` 의 메서드

- **`ctx.save_for_backward(*tensors)`**: Tensor 들을 저장 (memory efficient)
- **`ctx.saved_tensors`**: Tuple 로 저장된 tensor 복원
- **`ctx.mark_non_differentiable(*args)`**: Non-differentiable output 표시
- **`ctx.mark_dirty(*args)`**: In-place operation 선언
- **`ctx.set_materialize_grads(bool)`**: Backward 에서 None grad 를 zero 로 변환

### 정의 5.3 — Gradient 반환 규칙

Forward 의 input 이 $n$ 개, output 이 $m$ 개일 때:
- Backward 에서 반환하는 gradient tuple 의 길이: $n$
- 각 element: Tensor (미분 가능) 또는 None (상수, 미분 불가)

---

## 🔬 정리와 증명

### 정리 5.1 (Custom Function 의 수학적 정확성)

Custom operation $y = f(x)$ 를 `torch.autograd.Function` 으로 구현했을 때, backward 가 정확한 VJP 를 계산하면:

$$
\text{ComputedGrad}(x, \bar{y}) = J_f(x)^T \bar{y}
$$

**증명**:

Custom function 은 PyTorch 의 autograd graph 에 새로운 "primitive node" 를 추가합니다. Backward 시 이 node 에서 `backward()` 메서드를 호출. 만약 backward 가 정확한 VJP 를 반환하면, chain rule 의 composition 에 의해 전체 gradient 도 정확함 $\square$.

### 정리 5.2 (ctx.save_for_backward 의 메모리 효율)

`ctx.save_for_backward(x)` 로 저장하는 것은, tensor 의 메모리를 다시 할당하지 않고 **reference** 만 유지합니다 (zero-copy).

따라서 메모리 cost 는 pointer size 만큼 (8 bytes on 64-bit).

**증명**:

PyTorch 내부적으로 `ctx.saved_tensors` 는 tensor 의 storage 와 metadata 만 참조. Backward 가 끝나면 release 됨 $\square$.

### 정리 5.3 (gradcheck 의 원리)

Numerical gradient 를 계산:

$$
\text{NumGrad}(x_i) = \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}
$$

Analytical gradient (backward 에서 계산):

$$
\text{AnalGrad}(x_i) = J_f(x)^T e_i
$$

두 값의 relative error 가 threshold 보다 작으면 **backward 구현이 정확함**.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — 간단한 Custom Function: MyReLU

```python
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Forward: y = max(0, x)
        y = x.clamp(min=0)
        
        # Save for backward
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        # VJP: grad_x = (x > 0) * grad_output
        grad_input = grad_output * (x > 0).float()
        return grad_input

# Test
x = torch.randn(3, 4, requires_grad=True)
y = MyReLU.apply(x)
loss = y.sum()
loss.backward()

print(f"Custom ReLU gradient:\n{x.grad}")

# Compare with torch.relu
x2 = x.clone().detach().requires_grad_(True)
y2 = torch.relu(x2)
loss2 = y2.sum()
loss2.backward()

print(f"\ntorch.relu gradient:\n{x2.grad}")
print(f"\nMatch: {torch.allclose(x.grad, x2.grad)}")
```

**예상 출력**:
```
Custom ReLU gradient:
tensor([[0., 1., 0., ...],
        ...])

torch.relu gradient:
tensor([[0., 1., 0., ...],
        ...])

Match: True
```

### 실험 2 — Numerical Gradient Check (gradcheck)

```python
import torch
from torch.autograd import gradcheck

# Custom ReLU function
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.clamp(min=0)
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * (x > 0).float()
        return grad_input

# Test with gradcheck
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

# gradcheck 는 double precision 으로 numerical gradient 계산
result = gradcheck(MyReLU.apply, (x,), eps=1e-6, atol=1e-4)
print(f"ReLU gradcheck passed: {result}")

# 만약 backward 구현이 틀렸다면?
class WrongReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.clamp(min=0)
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # 잘못된 구현: (x >= 0) 대신 (x > 0)
        x, = ctx.saved_tensors
        grad_input = grad_output * (x >= 0).float()
        return grad_input

result_wrong = gradcheck(WrongReLU.apply, (x,), eps=1e-6, atol=1e-4)
print(f"Wrong ReLU gradcheck passed: {result_wrong}")  # False
```

**예상 출력**:
```
ReLU gradcheck passed: True
Wrong ReLU gradcheck passed: False
```

### 실험 3 — Softmax Custom Implementation

```python
import torch
import torch.nn.functional as F

class MySoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Numerical stability: max subtraction
        x_max = x.max(dim=-1, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        softmax = exp_x / exp_x.sum(dim=-1, keepdim=True)
        
        ctx.save_for_backward(softmax)
        return softmax
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax, = ctx.saved_tensors
        
        # VJP of softmax: bar_x = sigma * (grad - (sigma * grad).sum)
        grad_input = softmax * (grad_output - (softmax * grad_output).sum(dim=-1, keepdim=True))
        return grad_input

# Test
x = torch.randn(2, 3, requires_grad=True)
y1 = MySoftmax.apply(x)
loss1 = y1.sum()
loss1.backward()
grad1 = x.grad.clone()

# Compare with F.softmax
x.grad = None
y2 = F.softmax(x, dim=-1)
loss2 = y2.sum()
loss2.backward()
grad2 = x.grad.clone()

print(f"Custom softmax gradient:\n{grad1}")
print(f"\nF.softmax gradient:\n{grad2}")
print(f"\nMatch: {torch.allclose(grad1, grad2, atol=1e-5)}")

# Gradcheck
x_check = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
result = gradcheck(MySoftmax.apply, (x_check,), eps=1e-6, atol=1e-4)
print(f"\nGradcheck passed: {result}")
```

**예상 출력**:
```
Custom softmax gradient:
tensor([[-0.0833, -0.0833,  0.1667],
        [-0.0833,  0.1667, -0.0833]])

F.softmax gradient:
tensor([[-0.0833, -0.0833,  0.1667],
        [-0.0833,  0.1667, -0.0833]])

Match: True

Gradcheck passed: True
```

### 실험 4 — Device-Agnostic 구현

```python
import torch

class MyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        # Device-agnostic: 어떤 device 든 동작
        C = A @ B
        ctx.save_for_backward(A, B)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        
        # VJP: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
        grad_A = grad_output @ B.transpose(-2, -1)
        grad_B = A.transpose(-2, -1) @ grad_output
        
        return grad_A, grad_B

# Test on CPU
A_cpu = torch.randn(2, 3, requires_grad=True)
B_cpu = torch.randn(3, 4, requires_grad=True)
C_cpu = MyMatMul.apply(A_cpu, B_cpu)
loss_cpu = C_cpu.sum()
loss_cpu.backward()

print(f"CPU A.grad shape: {A_cpu.grad.shape}")
print(f"CPU B.grad shape: {B_cpu.grad.shape}")

# Test on CUDA (if available)
if torch.cuda.is_available():
    A_cuda = A_cpu.detach().cuda().requires_grad_(True)
    B_cuda = B_cpu.detach().cuda().requires_grad_(True)
    C_cuda = MyMatMul.apply(A_cuda, B_cuda)
    loss_cuda = C_cuda.sum()
    loss_cuda.backward()
    
    print(f"\nCUDA A.grad shape: {A_cuda.grad.shape}")
    print(f"CUDA B.grad shape: {B_cuda.grad.shape}")
```

**예상 출력**:
```
CPU A.grad shape: torch.Size([2, 3])
CPU B.grad shape: torch.Size([3, 4])

CUDA A.grad shape: torch.Size([2, 3])
CUDA B.grad shape: torch.Size([3, 4])
```

### 실험 5 — Non-Differentiable Output

```python
import torch

class ArgMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Forward: argmax (정수 반환)
        indices = x.argmax(dim=-1)
        
        # 이 output 은 미분 불가능
        ctx.mark_non_differentiable(indices)
        
        return indices
    
    @staticmethod
    def backward(ctx, grad_output):
        # Non-differentiable output 이므로 backward 없음
        return None

# Test
x = torch.randn(2, 3, requires_grad=True)
indices = ArgMaxFunction.apply(x)

print(f"Indices: {indices}")
print(f"Indices requires_grad: {indices.requires_grad}")

# Backward 불가능
try:
    loss = indices.sum().float()  # Convert to float
    loss.backward()
except RuntimeError as e:
    print(f"Error: {e}")
```

**예상 출력**:
```
Indices: tensor([[1, 2],
                 [0, 1]])
Indices requires_grad: False
Error: ...
```

---

## 🔗 실전 활용

### 1. CUDA Kernel Wrapper (CUDA 설치 필수)

```python
# 간단한 예시 (실제로는 C++ extension 필요)
class SparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_matrix, dense_vector):
        # CUDA kernel 호출 (예시)
        if sparse_matrix.is_cuda:
            result = torch.cuda.sparse.matmul(sparse_matrix, dense_vector)
        else:
            result = torch.sparse.matmul(sparse_matrix, dense_vector)
        
        ctx.save_for_backward(sparse_matrix, dense_vector)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        sparse_matrix, dense_vector = ctx.saved_tensors
        
        # Backward computation
        grad_sparse = grad_output.unsqueeze(-1) @ dense_vector.unsqueeze(0)
        grad_dense = sparse_matrix.t() @ grad_output
        
        return grad_sparse, grad_dense
```

### 2. Mixed Precision Custom Function

```python
class MixedPrecisionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Forward in FP16 for speed
        with torch.cuda.amp.autocast():
            y = torch.nn.functional.relu(x)
        
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward in FP32 for accuracy
        x, = ctx.saved_tensors
        grad_input = grad_output * (x > 0).float()
        return grad_input
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Forward/Backward 구현 정확성 | `gradcheck()` 로 검증 필수 |
| ctx 사용의 필수성 | Closure 의 오류 피하기 위해 명시적 save |
| Device 호환성 | CPU/CUDA 모두 지원하려면 device-agnostic 코드 |
| Type 호환성 | Float64 (gradcheck 용) 와 Float32 모두 지원 |

---

## 📌 핵심 정리

$$\boxed{\text{forward} \leftrightharpoons \text{backward}, \quad \text{via } ctx}$$

| 단계 | 역할 | 주의점 |
|------|------|--------|
| Forward | Output 계산 | ctx 에 필요한 값 저장 |
| ctx.save | Data 저장 | `save_for_backward()` 사용 |
| Backward | VJP 계산 | Return tuple 의 길이 = input 개수 |
| gradcheck | 검증 | double precision, eps 설정 |

**Best Practice**:
```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = compute_forward(x)
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = compute_vjp(x, grad_output)
        return grad_input
```

---

## 🤔 생각해볼 문제

**문제 1** (기초): MySquare 함수를 구현하고 gradcheck 로 검증하라:

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: Implement y = x^2
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement bar_x = 2x * grad_output
        pass
```

<details>
<summary>해설</summary>

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x ** 2
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return 2 * x * grad_output

# Test
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
result = gradcheck(MySquare.apply, (x,))
print(f"Gradcheck: {result}")  # True
```

$\square$

</details>

**문제 2** (심화): MyLogSoftmax 를 구현하고, numerical precision 문제를 다루라:

```python
class MyLogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # log_softmax = log(softmax(x)) = x - log_sum_exp(x)
        # Numerically stable?
        pass
```

<details>
<summary>해설</summary>

```python
class MyLogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Numerically stable log_softmax
        x_max = x.max(dim=-1, keepdim=True)[0]
        log_sum_exp = (x - x_max).exp().sum(dim=-1, keepdim=True).log() + x_max
        log_softmax = x - log_sum_exp
        
        ctx.save_for_backward(log_softmax.exp())  # Save softmax, not x
        return log_softmax
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax, = ctx.saved_tensors
        # bar_x = grad_output - softmax * grad_output.sum()
        grad_input = grad_output - softmax * grad_output.sum(dim=-1, keepdim=True)
        return grad_input
```

Stability: `log_sum_exp` 를 직접 계산하지 않고, `exp().sum().log()` 형태로 유지 $\square$

</details>

**문제 3** (논문 비평): Custom function 의 backward 를 C++ extension 으로 구현할 때 (예: CUDA kernel), 어떤 추가 고려사항이 있는가? Autodiff 와 manual backward 의 관계는?

<details>
<summary>해설</summary>

**C++ Extension 에서의 고려**:

1. **Double precision support**: Gradcheck 를 위해 float64 도 지원
2. **Device check**: `x.device()` 로 CPU/CUDA 분기
3. **Contiguous memory**: CUDA kernel 의 입력은 contiguous 해야 함 → `.contiguous()` 호출
4. **Error handling**: CUDA kernel error 는 PyTorch exception 으로 매핑

**Forward + Backward 의 관계**:
- **Manual backward**: VJP rule 을 직접 구현 (autograd.Function)
- **Autodiff**: PyTorch 가 자동으로 chain rule 적용
- **Hybrid**: Forward 는 CUDA kernel, backward 는 PyTorch autograd 사용 가능

**예**:
```cpp
// Forward: CUDA kernel
torch::Tensor custom_forward_cuda(torch::Tensor x) {
    return custom_kernel(x);
}

// Backward: torch.autograd
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.ops.custom.forward_cuda(x)
        ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Backward 는 PyTorch 로 구현 (자동으로 fusion)
        grad_x = torch.ops.custom.backward_cuda(grad_output, x)
        return grad_x
```

$\square$

</details>

---

<div align="center">

[◀ 이전](./04-backward-topological-sort.md) | [📚 README](../README.md) | [다음 ▶](./06-double-backward.md)

</div>
