# 06. Higher-Order Gradients — Double Backward

## 🎯 핵심 질문

- `create_graph=True` 를 `backward()` 에 전달하면 backward 자체가 어떻게 differentiable operation 이 되는가?
- **Hessian-Vector Product (HVP)** $Hv = \nabla(\nabla L \cdot v)$ 를 구하는 것과, Explicit Hessian matrix 를 구하는 것의 차이는?
- **Pearlmutter trick (1994)** 이 무엇이고, 왜 explicit Hessian 을 피하는가?
- Physics-Informed NN (PINN) 에서 PDE residual 의 미분을 어떻게 계산하는가?
- 2차 optimizer (K-FAC, Newton method) 가 HVP 를 왜 필요로 하는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

대부분의 PyTorch 사용자는 1차 미분 (gradient) 만 사용하지만, 다음의 고급 응용에서는 2차 미분이 필수입니다:

1. **Physics-Informed NN** — PDE 제약 조건의 2차 미분 필요
2. **2차 최적화** — Newton method, Gauss-Newton 의 Hessian-Vector Product
3. **Adversarial robustness** — Adversarial training 의 Hessian 정보
4. **Meta-learning** — MAML 의 higher-order gradient

하지만 explicit Hessian matrix 를 구하면 $(p \times p)$ 메모리를 사용하므로, **Hessian-Vector Product** 로 효율적으로 계산해야 합니다.

이 문서는 **double backward 의 정확한 메커니즘** 을 다룹니다.

---

## 📐 수학적 선행 조건

- **Hessian matrix**: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$
- **Vector-Jacobian product (VJP)**: Ch2-02 의 VJP 개념
- **Chain rule 의 반복 적용**: $\nabla^2 L = \nabla(\nabla L)$
- **메모리-계산 trade-off**: Explicit vs implicit Hessian

---

## 📖 직관적 이해

### 1차 미분: Forward → Backward

```
Forward:  x ──[f]──> y ──[g]──> L
Backward: x <──[∂g]──< y <──[∂f]── L
          ↓
       x.grad = ∂L/∂x
```

### 2차 미분: Double Backward

첫 번째 backward 에서 `create_graph=True` 이면, backward 자체가 graph node 가 됨:

```
Forward:  x ──[f]──> y ──[g]──> L
                                 ↓
First Backward (create_graph=True):
          (x.grad 자체도 graph node!)
                      ↓
Second Backward:
          (x.grad의 gradient) = ∂²L/∂x²
```

### Hessian-Vector Product vs Explicit Hessian

**Explicit Hessian** (memory intensive):
```
H = [∂²L/∂x₁², ∂²L/∂x₁∂x₂, ...]  ← (p × p) matrix
    [∂²L/∂x₂∂x₁, ∂²L/∂x₂², ...]
    ...
```

Memory: $O(p^2)$, 매우 큼.

**Hessian-Vector Product** (Pearlmutter trick):
```
Hv = ∇(∇L · v)
   = ∇[∑ᵢ (∂L/∂xᵢ) · vᵢ]
```

Memory: $O(p)$, efficient!

---

## ✏️ 엄밀한 정의

### 정의 6.1 — 2차 미분 (Second Derivative)

Loss $L: \mathbb{R}^n \to \mathbb{R}$ 에 대해:

$$
\nabla L = \begin{bmatrix} \frac{\partial L}{\partial x_1} \\ \vdots \\ \frac{\partial L}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n
$$

2차 미분 (Hessian):

$$
H = \nabla^2 L = \begin{bmatrix} \frac{\partial^2 L}{\partial x_1^2} & \cdots & \frac{\partial^2 L}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 L}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 L}{\partial x_n^2} \end{bmatrix} \in \mathbb{R}^{n \times n}$$

### 정의 6.2 — Hessian-Vector Product

Vector $v \in \mathbb{R}^n$ 에 대해:

$$
\text{HVP}(v) := H v = \nabla(\nabla L \cdot v) = \nabla\left(\sum_i \frac{\partial L}{\partial x_i} v_i\right)
$$

결과: $\mathbb{R}^n$ vector.

### 정의 6.3 — Pearlmutter Trick (1994)

Direct Hessian 계산 대신, HVP 를 이용한 간접 계산:

$$
\text{For each } i = 1, \ldots, n:
\quad (Hv)_i = \sum_j \frac{\partial^2 L}{\partial x_i \partial x_j} v_j
$$

**연산**: $n$ 번의 backward (각각 forward 의 1 회 cost)

**Memory**: Forward intermediate 만 유지 (Hessian 행렬 저장 X)

### 정의 6.4 — `create_graph=True` 의 의미

```python
loss = model(x).sum()
loss.backward(create_graph=True)  # x.grad 도 graph node
```

Result: `x.grad` 가 backward graph 의 output 으로 등록되어, 다시 미분 가능.

---

## 🔬 정리와 증명

### 정리 6.1 (Double Backward 의 정확성)

`create_graph=True` 로 backward 를 호출한 후, `x.grad` 를 다시 미분하면:

$$
\frac{\partial (\nabla L)}{\partial x} = H = \nabla^2 L
$$

(2차 미분을 정확히 계산)

**증명**:

첫 번째 backward 에서 `create_graph=True` 이므로:
- 각 operation 의 local VJP 가 graph node 로 기록됨
- `x.grad` 의 출력이 **intermediate tensor** 로 처리됨

두 번째 backward 에서:
- `x.grad` 를 input 으로 하는 backward graph 를 traverse
- Chain rule 의 composition 에 의해 Hessian 계산 $\square$.

### 정리 6.2 (Hessian-Vector Product 의 메모리 효율)

Explicit Hessian $H \in \mathbb{R}^{n \times n}$ 의 계산:

- **Direct**: $n$ 번의 backward (각각 memory: forward intermediates)
- **HVP via Pearlmutter**: 1 번의 forward + 1 번의 backward (memory: forward intermediates only)
- **메모리 절약**: $(n-1) \times (\text{forward intermediate memory})$

**증명**:

Pearlmutter 는 중간값을 재사용:

$$
Hv = \nabla(\sum_i g_i v_i), \quad g_i := \frac{\partial L}{\partial x_i}
$$

Scalar loss 로 reduce → 1 회 backward 로 충분 $\square$.

### 정리 6.3 (Physics-Informed NN 의 2차 미분)

PDE: $\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2}$ 에서:

residual = $\left(\frac{\partial u}{\partial t} - \nu \frac{\partial^2 u}{\partial x^2}\right)^2$

1차 미분: $\frac{\partial u}{\partial t}$ (1 회 backward)

2차 미분: $\frac{\partial^2 u}{\partial x^2}$ (nested backward, `create_graph=True`)

**증명**:

```python
u = model(t, x)
u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x)[0]
residual = (u_t - nu * u_xx)**2
```

각 `create_graph=True` 가 graph 를 유지 → nested backward 가능 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Double Backward 의 기본

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

# Forward
y = x ** 3
loss = y

# First backward: with create_graph=True
loss.backward(create_graph=True)
print(f"First backward (x.grad): {x.grad}")
print(f"x.grad.requires_grad: {x.grad.requires_grad}")
print(f"Expected: dy/dx = 3x^2 = 12")

# Second backward
if x.grad.requires_grad:
    grad_grad = torch.autograd.grad(x.grad, x)[0]
    print(f"\nSecond backward (d²y/dx²): {grad_grad}")
    print(f"Expected: d/dx(3x^2) = 6x = 12")
```

**예상 출력**:
```
First backward (x.grad): tensor(12., grad_fn=PowBackward0)
x.grad.requires_grad: True
Expected: dy/dx = 3x^2 = 12

Second backward (d²y/dx²): tensor(12.)
Expected: d/dx(3x^2) = 6x = 12
```

### 실험 2 — Hessian-Vector Product

```python
import torch

def hvp(f, x, v):
    """
    Compute Hessian-Vector Product: Hv = ∇(∇f(x) · v)
    
    Args:
        f: Scalar function
        x: Input tensor
        v: Vector for HVP
    
    Returns:
        Hessian-Vector Product
    """
    x.requires_grad_(True)
    
    # First gradient
    fx = f(x)
    grad_fx = torch.autograd.grad(fx, x, create_graph=True)[0]
    
    # Second gradient: d/dx (grad_f · v)
    hvp_result = torch.autograd.grad(
        (grad_fx * v).sum(), x, create_graph=False
    )[0]
    
    return hvp_result

# Test function: f(x) = x^T A x / 2 (quadratic)
# Expected Hessian: H = A
A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
x = torch.tensor([1.0, 2.0])
v = torch.tensor([1.0, 0.0])

def f(x):
    return 0.5 * (x @ A @ x)

hvp_result = hvp(f, x, v)
expected = A @ v

print(f"HVP result: {hvp_result}")
print(f"Expected (A @ v): {expected}")
print(f"Match: {torch.allclose(hvp_result, expected)}")
```

**예상 출력**:
```
HVP result: tensor([2., 1.])
Expected (A @ v): tensor([2., 1.])
Match: True
```

### 실험 3 — Explicit Hessian Matrix 계산

```python
import torch

def hessian_matrix(f, x):
    """
    Compute full Hessian matrix H ∈ R^(n×n)
    
    Args:
        f: Scalar function
        x: Input tensor
    
    Returns:
        Hessian matrix
    """
    n = x.numel()
    x.requires_grad_(True)
    
    # First gradient
    fx = f(x)
    grad_f = torch.autograd.grad(fx, x, create_graph=True)[0]
    
    # For each component of gradient
    H = []
    for i in range(n):
        # Compute ∂/∂x_j (∂f/∂x_i)
        grad_f[i].backward(retain_graph=(i < n-1))
        H.append(x.grad.clone())
        x.grad.zero_()
    
    return torch.stack(H)

# Test: f(x) = x1^2 + 2*x1*x2 + 3*x2^2
# H = [[2, 2], [2, 6]]
x = torch.tensor([1.0, 2.0], requires_grad=True)

def f(x):
    return x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2

H = hessian_matrix(f, x.detach().clone())

print(f"Hessian matrix:\n{H}")
print(f"Expected: [[2, 2], [2, 6]]")
```

**예상 출력**:
```
Hessian matrix:
tensor([[2., 2.],
        [2., 6.]])
Expected: [[2, 2], [2, 6]]
```

### 실험 4 — Physics-Informed NN (간단한 예시)

```python
import torch
import torch.nn as nn

# Simple PINN for u_t = u_xx
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))

t = torch.randn(10, 1, requires_grad=True)
x = torch.randn(10, 1, requires_grad=True)
tx = torch.cat([t, x], dim=1)

# Forward
u = model(tx)

# First derivatives
u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

# Second derivative u_xx
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=False)[0]

# PDE residual: u_t - u_xx = 0
pde_loss = ((u_t - u_xx) ** 2).mean()

print(f"u shape: {u.shape}")
print(f"u_t shape: {u_t.shape}")
print(f"u_xx shape: {u_xx.shape}")
print(f"PDE loss: {pde_loss.item()}")

# Can we backward on pde_loss?
pde_loss.backward()
print(f"Model parameters updated: {model[0].weight.grad is not None}")
```

**예상 출력**:
```
u shape: torch.Size([10, 1])
u_t shape: torch.Size([10, 1])
u_xx shape: torch.Size([10, 1])
PDE loss: 0.567
Model parameters updated: True
```

### 실험 5 — torch.func.hessian (JAX-style)

```python
try:
    from torch.func import hessian, jacobian
    import torch
    
    def f(x):
        return (x ** 3).sum()
    
    x = torch.tensor([1.0, 2.0, 3.0])
    
    # Compute Hessian using torch.func
    H = hessian(f)(x)
    
    print(f"Hessian via torch.func:\n{H}")
    print(f"Expected: [[6*1, 0, 0], [0, 6*2, 0], [0, 0, 6*3]]")
    
    # Verify: diagonal elements should be 6*x_i
    for i in range(3):
        print(f"H[{i},{i}] = {H[i,i]}, expected: {6*x[i]}")
        
except ImportError:
    print("torch.func not available in this PyTorch version")
```

**예상 출력**:
```
Hessian via torch.func:
tensor([[ 6.,  0.,  0.],
        [ 0., 12.,  0.],
        [ 0.,  0., 18.]])
Expected: [[6*1, 0, 0], [0, 6*2, 0], [0, 0, 6*3]]
H[0,0] = 6.0, expected: 6.0
H[1,1] = 12.0, expected: 12.0
H[2,2] = 18.0, expected: 18.0
```

---

## 🔗 실전 활용

### 1. Newton Method 의 Hessian 근사

```python
import torch
import torch.nn as nn

def newton_step(loss_fn, params, lr=0.01, damping=1e-4):
    """
    Single Newton step using HVP
    """
    # Compute gradient
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = torch.cat([g.view(-1) for g in grads])
    
    # Compute Hessian-Vector Product for small step
    hvps = []
    for i, grad in enumerate(grads):
        if grad is not None:
            hvp = torch.autograd.grad(
                grad.sum(), params[i], retain_graph=True
            )[0]
            hvps.append(hvp)
    
    # Update: θ = θ - H^{-1} ∇L
    # Approximated using damped Hessian: (H + λI)^{-1}
    # For simplicity, use Gauss-Newton or gradient descent
```

### 2. Adversarial Robustness (FGSM with Hessian)

```python
import torch

def adversarial_example_hessian(model, x, y, epsilon=0.1):
    """
    Generate adversarial using gradient and Hessian info
    """
    x.requires_grad_(True)
    
    # Forward and loss
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    
    # First gradient
    loss.backward(create_graph=True)
    grad = x.grad.clone()
    
    # Hessian information (if needed)
    # Can use HVP for curvature
    
    # Adversarial perturbation
    x_adv = x + epsilon * torch.sign(grad)
    return x_adv
```

### 3. Meta-Learning (MAML-like)

```python
import torch
import torch.nn as nn

def maml_inner_loop(model, task_batch, inner_lr=0.01, n_steps=5):
    """
    MAML inner loop: compute higher-order gradients
    """
    for step in range(n_steps):
        loss = model.compute_loss(task_batch)
        
        # Backward with create_graph=True for meta-update
        loss.backward(create_graph=True)
        
        # Inner loop update (simulated SGD)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= inner_lr * p.grad
                    p.grad.zero_()
    
    # Return loss for outer loop (which will compute meta-gradient)
    final_loss = model.compute_loss(task_batch)
    return final_loss
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Memory 충분 | Very deep network 의 double backward 는 memory intensive → gradient checkpointing |
| Smooth function | Kinks (ReLU) 에서 2차 미분이 불연속 → smooth activation 사용 |
| Single HVP 필요 | 전체 Hessian 행렬이 필요하면 explicit computation 피할 수 없음 |
| Numerical stability | 2차 미분은 1차보다 수치 오차 증가 → double precision 권장 |

---

## 📌 핵심 정리

$$\boxed{Hv = \nabla(\nabla L \cdot v) \quad \text{(Pearlmutter trick)}}$$

| 개념 | 정의 | Cost |
|------|------|------|
| 1차 미분 | $\nabla L = J^T \bar{1}$ | 1 backward |
| 2차 미분 (Hessian) | $H = \nabla^2 L \in \mathbb{R}^{p \times p}$ | $p$ backwards |
| **HVP** | $Hv = \nabla(\nabla L \cdot v)$ | **1 backward** |
| Explicit Hessian | Full matrix computation | $O(p^2)$ memory |
| **Double backward** | `create_graph=True` + nested backward | $O(p)$ memory |

**Best Practice**:
```python
# For HVP (efficient)
loss.backward(create_graph=True)
hvp = torch.autograd.grad((loss.grad * v).sum(), params)[0]

# For full Hessian (memory expensive, use sparingly)
H = hessian_matrix(loss_fn, x)
```

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = x^4$ 의 2차 미분 $f''(x)$ 를 PyTorch 로 계산하고, 손으로 계산한 값과 비교하라.

<details>
<summary>해설</summary>

$f(x) = x^4$ → $f'(x) = 4x^3$ → $f''(x) = 12x^2$.

PyTorch:
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 4
y.backward(create_graph=True)
grad2 = torch.autograd.grad(x.grad, x)[0]

print(f"f''(2) = {grad2}")  # 12 * 4 = 48
```

Expected: $12 \times 2^2 = 48$ ✓ $\square$

</details>

**問題 2** (심화): Hessian-Vector Product 의 계산 복잡도를 분석하라. Explicit Hessian 과 비교해서 몇 배 빠른가?

<details>
<summary>해설</summary>

**Explicit Hessian**:
- Forward: $O(N)$ (network size)
- Backward (first): $O(N)$
- Backward (for each column $i$): $O(N)$ × $p$ times
- **Total**: $O(p \times N)$ operations, $O(p \times M)$ memory (M = intermediate size)

**HVP (Pearlmutter)**:
- Forward: $O(N)$
- First backward: $O(N)$ (compute $\nabla L$)
- Second backward: $O(N)$ (compute $\nabla(\nabla L \cdot v)$)
- **Total**: $O(N)$ operations, $O(M)$ memory

**Speedup**: $(p \times N) / N = p$ times faster in computation, $p$ times less memory!

For $p = 1000$ parameters: **1000x 더 효율적** $\square$

</details>

**문제 3** (논문 비평): Physics-Informed NN 에서 높은 차수의 미분 (3차, 4차) 이 필요할 때 어떻게 구현할 것인가? Automatic differentiation 의 한계는?

<details>
<summary>해설</summary>

**구현 패턴**:

3차 미분까지 필요한 경우:
```python
u = model(x)
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
u_xxx = torch.autograd.grad(u_xx.sum(), x)[0]
```

각 단계에서 `create_graph=True` 유지.

**한계**:
1. **Memory explosion**: 각 단계마다 intermediate 저장 → exponential memory cost
2. **Numerical stability**: 미분이 여러 번 반복되면서 오차 누적 (3차 이상에서 심각)
3. **Speed**: 매 step 마다 backward → $n$ 차 미분은 $O(n \times N)$ cost
4. **Automatic vs Manual**: 손으로 짜는 것이 더 효율적일 수 있음 (e.g., finite difference, symbolic differentiation)

**대안**:
- **Symbolic differentiation** (SymPy): PDE 정의를 미리 미분
- **Finite difference**: Numerical approximation
- **Custom backward**: 고차 미분을 명시적으로 구현

$\square$

</details>

---

<div align="center">

[◀ 이전](./05-custom-function.md) | [📚 README](../README.md) | [다음 ▶](../ch3-dispatcher/01-dispatcher-architecture.md)

</div>
