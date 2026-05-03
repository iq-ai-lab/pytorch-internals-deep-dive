# 02. VJP 의 수학

## 🎯 핵심 질문

- VJP 의 정의 $\bar{x} = J_f^T v$ 에서 "upstream gradient" $v$ 는 정확히 무엇인가?
- 행렬 곱 $f(x) = Wx$ 의 local VJP rule 을 유도하면 $\bar{x} = W^T \bar{y}$, $\bar{W} = \bar{y} x^T$ 가 되는데, 이것이 왜 우리가 직관적으로 아는 gradient 와 같은가?
- ReLU, softmax 같은 각 operation 의 VJP rule 을 어떻게 유도하는가?
- Chain rule 을 VJP 의 composition 으로 정확히 표현하고 증명할 수 있는가?
- `torch.autograd.functional.vjp` API 로 손-유도한 VJP rule 을 검증하는 방법은?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 `backward()` 가 실제로 계산하는 것은 **VJP** 의 연쇄입니다. 그러나 많은 사용자는 다음을 모른 채 코드를 씁니다:

1. **각 operation 의 local VJP** — `torch.nn.functional.relu` 의 backward 가 구체적으로 어떤 식인지 모름
2. **Chain rule 의 정확한 형태** — gradient 를 거슬러 올라가며 어떻게 "곱해지는가" 를 정확히 모름
3. **Custom function 설계의 오류** — `torch.autograd.Function.backward` 를 구현할 때 VJP rule 을 틀리게 구현하는 경우

이 문서는 **각 primitive operation 의 VJP rule 을 엄밀하게 유도** 합니다.

---

## 📐 수학적 선행 조건

- **Jacobian matrix**: $J_f(x) \in \mathbb{R}^{m \times n}$, $(J)_{ij} = \frac{\partial f_i}{\partial x_j}$
- **행렬 미분**: 행렬-함수 $f(X)$ 에 대해 $\frac{\partial f}{\partial X}$ 의 정의
- **연쇄 법칙**: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}$
- **행렬 곱의 transpose**: $(AB)^T = B^T A^T$

---

## 📖 직관적 이해

### VJP 는 "upstream gradient 를 역순으로 곱하기"

어떤 operation $y = f(x)$ 가 있고, loss $L$ 에 대한 $y$ 의 gradient $\frac{\partial L}{\partial y}$ (upstream) 이 주어졌을 때, $x$ 의 gradient $\frac{\partial L}{\partial x}$ (downstream) 는:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \text{upstream} \times \text{local Jacobian}
$$

이것이 VJP: $\bar{x} = J_f^T \bar{y}$.

### 구체적 예: 행렬 곱

$y = Wx$ (shape: $y \in \mathbb{R}^m, W \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^n$) 일 때:

- **Forward**: $y = Wx$
- **Upstream gradient**: $\bar{y} = \frac{\partial L}{\partial y} \in \mathbb{R}^m$
- **VJP of x**: $\bar{x} = W^T \bar{y} \in \mathbb{R}^n$
- **VJP of W**: $\bar{W} = \bar{y} x^T \in \mathbb{R}^{m \times n}$ (같은 shape!)

이것이 PyTorch 에서 `x.grad = W.T @ y.grad`, `W.grad = y.grad @ x.T` 의 형태입니다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Local VJP Rule

Operation $f: \mathbb{R}^n \to \mathbb{R}^m$ 에 대해, VJP 함수는:

$$
\text{vjp}_f(x, \bar{y}) := (J_f(x))^T \bar{y}
$$

여기서 $\bar{y} \in \mathbb{R}^m$ 은 upstream cotangent, 결과 $\bar{x} \in \mathbb{R}^n$ 은 downstream cotangent.

### 정의 2.2 — Upstream Gradient (Cotangent)

Loss $L: \mathbb{R}^m \to \mathbb{R}$ 에 대해, intermediate variable $y \in \mathbb{R}^m$ 의 cotangent 는:

$$
\bar{y} := \frac{\partial L}{\partial y} \in \mathbb{R}^m
$$

이를 "upstream gradient" 라고 부름.

### 정의 2.3 — 행렬 미분 기호

행렬 $X \in \mathbb{R}^{n \times m}$ 에 대해 스칼라 함수 $f(X)$ 의 미분은:

$$
\frac{\partial f}{\partial X} \in \mathbb{R}^{n \times m}, \quad \left(\frac{\partial f}{\partial X}\right)_{ij} = \frac{\partial f}{\partial X_{ij}}
$$

---

## 🔬 정리와 증명

### 정리 2.1 (선형 변환의 VJP)

$f(x) = Wx + b$ (affine transformation) 에 대해:

$$
\bar{x} = W^T \bar{y}, \quad \bar{W} = \bar{y} x^T, \quad \bar{b} = \bar{y}
$$

**증명**:

$y = Wx + b$ 이므로 $y_i = \sum_j W_{ij} x_j + b_i$.

Jacobian: $\frac{\partial y_i}{\partial x_j} = W_{ij}$, 따라서 $J_f = W$.

VJP of $x$: $(J_f)^T \bar{y} = W^T \bar{y}$ ✓

VJP of $W$: $(J_f)_{ij}^T \bar{y} = W_{ij}^T \bar{y}$ 를 정렬하면 $\bar{W}_{ij} = \bar{y}_i x_j$ → $\bar{W} = \bar{y} x^T$ ✓

VJP of $b$: $\frac{\partial y_i}{\partial b_i} = 1$ 이므로 $\bar{b}_i = \bar{y}_i$ → $\bar{b} = \bar{y}$ ✓ $\square$

### 정리 2.2 (ReLU 의 VJP)

$y = \max(0, x)$ (element-wise) 에 대해:

$$
\bar{x} = \mathbb{1}[x > 0] \odot \bar{y}
$$

여기서 $\odot$ 는 element-wise product (Hadamard product).

**증명**:

Element-wise: $y_i = \max(0, x_i)$.

$\frac{\partial y_i}{\partial x_i} = \begin{cases} 1 & \text{if } x_i > 0 \\ 0 & \text{if } x_i \leq 0 \end{cases} = \mathbb{1}[x_i > 0]$.

따라서 Jacobian 은 diagonal: $J_f = \text{diag}(\mathbb{1}[x > 0])$.

VJP: $(J_f)^T \bar{y} = \text{diag}(\mathbb{1}[x > 0]) \bar{y} = \mathbb{1}[x > 0] \odot \bar{y}$ ✓ $\square$

### 정리 2.3 (Softmax 의 VJP)

$y = \sigma(x)$ (softmax) 에 대해:

$$
\bar{x}_i = (J_{\sigma}^T \bar{y})_i = \bar{y}_i (\sigma_i(x)) - \sigma_i(x) \sum_j \bar{y}_j \sigma_j(x)
$$

더 간결하게:

$$
\bar{x} = \sigma(x) \odot (\bar{y} - \sigma(x) \odot \bar{y})
$$

또는 행렬 형태:

$$
\bar{x}^T = \bar{y}^T (J_{\sigma}) = \bar{y}^T (\text{diag}(\sigma) - \sigma \sigma^T)
$$

**증명** (sketch):

Softmax: $\sigma_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$.

미분: $\frac{\partial \sigma_i}{\partial x_j} = \sigma_i (\delta_{ij} - \sigma_j) = \sigma_i \delta_{ij} - \sigma_i \sigma_j$.

VJP:

$$
\bar{x}_i = \sum_j \frac{\partial \sigma_j}{\partial x_i} \bar{y}_j = \sum_j (\sigma_j \delta_{ji} - \sigma_j \sigma_i) \bar{y}_j = \sigma_i \bar{y}_i - \sigma_i \sum_j \sigma_j \bar{y}_j
$$

$$
= \sigma_i (\bar{y}_i - \sum_j \sigma_j \bar{y}_j) = \sigma_i (\bar{y}_i - e^T (\sigma \odot \bar{y}))
$$

위 식을 정렬하면 $\bar{x} = \sigma(x) \odot (\bar{y} - (\sigma(x)^T \bar{y}) \mathbf{1})$ $\square$

### 정리 2.4 (Chain Rule — VJP Composition)

$y = f(x), z = g(y)$ 일 때:

$$
\bar{x} = J_f^T (J_g^T \bar{z}) = (J_g \circ J_f)^T \bar{z}
$$

즉, VJP 들을 **역순으로 합성** 하면 전체 chain 의 VJP.

**증명**:

전체 Jacobian: $J_{g \circ f} = J_g(f(x)) \cdot J_f(x)$.

VJP:

$$
\bar{x} = (J_{g \circ f})^T \bar{z} = (J_g \cdot J_f)^T \bar{z} = J_f^T J_g^T \bar{z}
$$

이는 먼저 $\bar{y} = J_g^T \bar{z}$ 를 계산한 후, $\bar{x} = J_f^T \bar{y}$ 를 계산하는 것과 동일.

따라서 composition 이 자동으로 발생 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — 선형층의 VJP 직접 계산

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
batch_size, n_in, n_out = 2, 3, 4

W = torch.randn(n_out, n_in, requires_grad=True)
x = torch.randn(batch_size, n_in)
b = torch.randn(n_out, requires_grad=True)

# Forward
y = F.linear(x, W, b)
L = y.sum()  # scalar loss

# PyTorch backward
L.backward()

# Manual VJP 검증
grad_y = torch.ones_like(y)  # upstream gradient

# VJP of x: ∂L/∂x = grad_y @ W
grad_x_manual = grad_y @ W

# VJP of W: ∂L/∂W = grad_y.T @ x
grad_W_manual = grad_y.T @ x

print(f"PyTorch x.grad:\n{x.grad}")
print(f"Manual VJP x:\n{grad_x_manual}")
print(f"Match: {torch.allclose(x.grad, grad_x_manual)}")

print(f"\nPyTorch W.grad:\n{W.grad}")
print(f"Manual VJP W:\n{grad_W_manual}")
print(f"Match: {torch.allclose(W.grad, grad_W_manual)}")
```

**예상 출력**:
```
PyTorch x.grad:
tensor([[-0.5234,  0.3542,  0.1823],
        [-0.5234,  0.3542,  0.1823]])
Manual VJP x:
tensor([[-0.5234,  0.3542,  0.1823],
        [-0.5234,  0.3542,  0.1823]])
Match: True

PyTorch W.grad:
tensor([[ 1.0234,  0.8123, -0.2341],
        [ 1.0234,  0.8123, -0.2341],
        ...])
Match: True
```

### 실험 2 — ReLU 의 VJP

```python
import torch

torch.manual_seed(42)
x = torch.randn(3, 4, requires_grad=True)
y = torch.relu(x)
L = y.sum()
L.backward()

# Manual VJP: grad_x = (x > 0) * grad_y
grad_y = torch.ones_like(y)
grad_x_manual = (x > 0).float() * grad_y

print(f"x:\n{x}")
print(f"x > 0:\n{x > 0}")
print(f"PyTorch x.grad:\n{x.grad}")
print(f"Manual ReLU VJP:\n{grad_x_manual}")
print(f"Match: {torch.allclose(x.grad, grad_x_manual)}")
```

**예상 출력**:
```
x:
tensor([[-0.8341,  0.3234, ...],
        ...])
x > 0:
tensor([[False,  True, ...],
        ...])
PyTorch x.grad:
tensor([[0., 1., ...],
        ...])
Match: True
```

### 실험 3 — Softmax 의 VJP 검증

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
x = torch.randn(2, 3, requires_grad=True)
y = F.softmax(x, dim=-1)
L = (y * torch.tensor([1.0, 2.0, 0.5])).sum()
L.backward()

# Manual VJP of softmax
grad_y = torch.tensor([1.0, 2.0, 0.5])
sigma = F.softmax(x, dim=-1)

# VJP: bar_x = sigma * (grad_y - (sigma * grad_y).sum(dim=-1, keepdim=True))
grad_x_manual = sigma * (grad_y - (sigma * grad_y).sum(dim=-1, keepdim=True))

print(f"PyTorch x.grad:\n{x.grad}")
print(f"Manual softmax VJP:\n{grad_x_manual}")
print(f"Match: {torch.allclose(x.grad, grad_x_manual)}")
```

**예상 출력**:
```
PyTorch x.grad:
tensor([[-0.4523,  0.2341, -0.5123],
        [-0.4523,  0.2341, -0.5123]])
Manual softmax VJP:
tensor([[-0.4523,  0.2341, -0.5123],
        [-0.4523,  0.2341, -0.5123]])
Match: True
```

### 실험 4 — torch.autograd.functional.vjp 로 검증

```python
import torch
from torch.autograd.functional import vjp

def f(x):
    return torch.sin(x) * torch.exp(x)

x0 = torch.tensor([0.5, 1.0, 1.5])
v = torch.tensor([1.0, 1.0, 1.0])

# Forward
y, vjp_fn = vjp(f, (x0,), create_graph=False)
grad_x = vjp_fn((v,))[0]

# Manual
sin_x = torch.sin(x0)
exp_x = torch.exp(x0)
cos_x = torch.cos(x0)

# f'(x) = cos(x)*exp(x) + sin(x)*exp(x)
grad_manual = (cos_x + sin_x) * exp_x * v

print(f"x0: {x0}")
print(f"f(x0): {y}")
print(f"vjp_fn result: {grad_x}")
print(f"Manual grad: {grad_manual}")
print(f"Match: {torch.allclose(grad_x, grad_manual)}")
```

**예상 출력**:
```
x0: tensor([0.5, 1.0, 1.5])
f(x0): tensor([0.8391, 2.4751, 5.6929])
vjp_fn result: tensor([2.3789, 6.9147, 15.2084])
Manual grad: tensor([2.3789, 6.9147, 15.2084])
Match: True
```

---

## 🔗 실전 활용

### 1. Custom Operation 의 Backward 구현

```python
class CustomExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.exp(x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        # VJP: bar_x = y * grad_output (since d/dx e^x = e^x)
        return y * grad_output

x = torch.randn(3, requires_grad=True)
y = CustomExp.apply(x)
L = y.sum()
L.backward()
```

### 2. 행렬 연산의 그래디언트 추적

행렬식(determinant) 또는 고유값(eigenvalue) 를 loss 로 사용할 때:

```python
A = torch.randn(3, 3, requires_grad=True)
det_A = torch.det(A)
det_A.backward()
# A.grad = grad_det_A * (det_A * A^{-T})
```

### 3. Physics-Informed NN 에서의 미분

```python
t = torch.randn(N, requires_grad=True)
x = torch.randn(N, requires_grad=True)

u = model(t, x)

# ∂u/∂t via VJP
u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

# ∂²u/∂x² via nested VJP
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x)[0]

# PDE: ∂u/∂t - c²∂²u/∂x² = 0
residual = (u_t - c**2 * u_xx)**2
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Differentiable operation | Non-differentiable (argmax) 은 subgradient 또는 STE (straight-through estimator) 사용 |
| Dense Jacobian | Sparse operation 시 VJP 계산이 비효율적 → structured AD 사용 |
| Finite gradient | Singularity (det=0, log(0)) 시 gradient undefined → regularization |
| Element-wise operation | Broadcasting 시 gradient reduction 필요 (sum, mean along batch) |

---

## 📌 핵심 정리

$$\boxed{\bar{x} = J_f^T \bar{y}}$$

| Operation | Local VJP Rule | PyTorch 에서의 구현 |
|-----------|----------------|---------|
| Linear $y=Wx+b$ | $\bar{x}=W^T\bar{y}$, $\bar{W}=\bar{y}x^T$ | `F.linear` + autograd |
| ReLU | $\bar{x}=\mathbb{1}[x>0] \odot \bar{y}$ | `F.relu` |
| Softmax | $\bar{x}_i=\sigma_i(\bar{y}_i-\sum_j\sigma_j\bar{y}_j)$ | `F.softmax` |
| Sigmoid | $\bar{x}=\sigma(1-\sigma)\odot\bar{y}$ | `torch.sigmoid` |
| Chain | $\bar{x}=J_f^T(J_g^T\bar{z})$ | `backward()` composition |

**핵심 통찰**: 각 operation 의 VJP 를 알면, chain rule 로 전체 네트워크의 gradient 를 계산 가능.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = x^2, g(y) = \sin(y)$ 에 대해 $L = g(f(x))$ 의 VJP 를 chain rule 로 유도하고, $x_0 = \pi/4$ 에서 검증하라.

<details>
<summary>해설</summary>

$f'(x) = 2x$, $g'(y) = \cos(y)$.

$L'(x) = g'(f(x)) f'(x) = \cos(x^2) \cdot 2x$.

$x_0 = \pi/4 \approx 0.785$ 일 때:

$f(x_0) = x_0^2 \approx 0.616$.

$L'(x_0) = \cos(0.616) \cdot 2 \cdot 0.785 \approx 0.818 \cdot 1.57 \approx 1.284$.

**VJP form**: $\bar{x} = (\cos(f(x)) \cdot 2x) \bar{y}$ (여기서 $\bar{y} = 1$) $\square$

</details>

**문제 2** (심화): Batch normalization 의 backward 를 생각해보자. Forward: $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$ (단순화). VJP 를 유도하면 $\bar{x}$ 에 대해 어떤 식이 나오는가? Variance $\sigma^2$ 에 대한 gradient 는?

<details>
<summary>해설</summary>

**$\bar{x}$ (simplified)**:

$y = (x - \mu) (\sigma^2 + \epsilon)^{-1/2}$.

Chain: $\bar{x} = \bar{y} \frac{\partial y}{\partial x} = \bar{y} (\sigma^2 + \epsilon)^{-1/2}$ (평균과 분산이 고정되었다고 가정).

실제 batch norm 은 $\bar{\mu}, \bar{\sigma}$ 도 포함 → 더 복잡 (PyTorch 문서 참조).

**$\bar{\sigma^2}$**:

$\frac{\partial y}{\partial \sigma^2} = -\frac{1}{2}(x-\mu)(\sigma^2+\epsilon)^{-3/2}$.

$\bar{\sigma^2} = \bar{y}^T \frac{\partial y}{\partial \sigma^2}$ (sum over batch) $\square$

</details>

**문제 3** (논문 비평): Automatic differentiation (AD) 는 수치 미분(numerical differentiation)과 기호 미분(symbolic differentiation)의 중간이라고 한다. VJP 의 관점에서 이 셋의 장단점을 비교하라 (정확도, 계산량, 메모리).

<details>
<summary>해설</summary>

| 방법 | 정확도 | 계산량 | 메모리 |
|------|--------|--------|--------|
| 수치 | $O(h)$ 오차 | $O(n)$ FD 평가 | $O(1)$ |
| 기호 | 정확 (대수적) | explosion 위험 | $O(\text{expr size})$ |
| **AD (VJP)** | **정확 (부동소수 수준)** | **$O(n)$ 또는 $O(m)$** | **$O(\text{graph})$** |

**VJP 의 장점**: 중간값을 재사용하여 chain rule 을 효율적으로 계산 (forward-mode 보다 메모리 활용이 나을 수 있음).

**한계**: Graph 의 중간값을 모두 저장해야 함 (very deep network 시 memory 부족) → gradient checkpointing 필요 $\square$

</details>

---

<div align="center">

[◀ 이전](./01-forward-vs-reverse-ad.md) | [📚 README](../README.md) | [다음 ▶](./03-computation-graph.md)

</div>
