# 01. Forward (JVP) vs Reverse (VJP) AD

## 🎯 핵심 질문

- Forward-mode AD 에서 $\dot{y} = J \dot{x}$ 를 계산하는 것과 reverse-mode AD 에서 $\bar{x} = J^T \bar{y}$ 를 계산하는 것은 어떻게 **같은 chain rule** 의 두 구현인가?
- 왜 ML 에서 scalar loss ($m=1$) 일 때 **reverse-mode 1 회로 모든 parameter gradient** 를 구할 수 있는가?
- Forward 모드는 $O(n)$ 의 비용이 input dimension 마다 드는데, reverse 는 왜 output dimension 마다 드는가?
- JAX 의 `jvp` vs `grad` API 의 비교를 통해 두 모드의 차이를 어떻게 실전에서 선택하는가?
- Dual number 를 사용한 forward-mode 의 직관적 구현이 어떻게 동작하는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 `loss.backward()` 는 reverse-mode AD 를 사용합니다. 하지만 많은 사용자가 "자동 미분" 을 하나의 검은 상자로 보며, 다음을 정확히 알지 못한 채 코드를 씁니다:

1. **왜 하필 reverse-mode 인가** — Forward-mode 가 항상 나쁜가? 아니다. ML 의 $m \ll n$ 구조에서 reverse 가 효율적일 뿐.
2. **비용 분석의 부재** — `backward()` 한 줄의 계산 복잡도가 실제로 어떻게 input/output dimension 에 비례하는지 모름.
3. **Custom autograd 설계의 실패** — `torch.autograd.Function` 의 backward 를 설계할 때 reverse-mode 를 가정하지 않으면 성능 함정에 빠짐.

이 문서는 JVP vs VJP 의 **수학적 정확한 비교** 와 **PyTorch/JAX 의 실제 선택** 을 다룹니다.

---

## 📐 수학적 선행 조건

- **미적분**: 다변수 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$, Jacobian $J_f \in \mathbb{R}^{m \times n}$
- **행렬 대수**: 행렬-벡터 곱 $Jv$, 행-벡터-행렬 곱 $u^T J$
- **연쇄 법칙 (Chain Rule)**: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}$
- **계산 복잡도**: 시간 복잡도를 operation count 로 정의

---

## 📖 직관적 이해

### Forward-Mode AD 의 직관

$f: \mathbb{R}^n \to \mathbb{R}^m$ 의 미분을 "tangent propagation" 이라고 부릅니다. Input 의 작은 변화 $\dot{x}$ 를 따라가며 output 이 어떻게 변하는지 추적:

$$
\dot{y} = J_f \dot{x}
$$

**장점**: 정확한 1차 미분 (중간 계산에 오차 누적이 없음), single pass.

**단점**: Input dimension 이 많을수록 **여러 번 반복** 해야 함 — 모든 $e_1, e_2, \ldots, e_n$ (standard basis) 에 대해 $Jv$ 를 각각 계산.

### Reverse-Mode AD 의 직관

반대로 output 의 변화 $\bar{y}$ 에서 시작하여 input 까지 거슬러 올라가는 "cotangent propagation":

$$
\bar{x} = J_f^T \bar{y}
$$

**장점**: Output dimension 이 적을수록 **한 번에 모든 input** 의 gradient 를 구함. Scalar loss ($m=1$) 일 때 $\bar{y} = 1 \in \mathbb{R}$ 이므로 1 회 계산.

**단점**: Backward 를 위해 forward 의 중간값들을 저장해야 함 (memory overhead).

### ML 에서의 비용 분석

Neural network training 에서:
- **Inputs (features)**: $n \gg 10^6$ (이미지의 pixel, 임베딩 차원)
- **Outputs (loss)**: $m = 1$ (scalar loss)
- **Parameters**: $p \approx n$ (fully connected 기준)

**Forward-mode**: Gradient 를 구하려면 $J^T v$ 를 모든 input basis 에 대해 $n$ 번 계산 → $O(nm)$ = $O(n)$ (since $m=1$).

**Reverse-mode**: Loss 에서 거슬러올라가며 1 회 만에 모든 parameter gradient 획득 → $O(m \cdot \text{nnz(graph)}) = O(\text{# connections})$ (dense network 에서도 $\approx O(n)$).

```
Forward:  loss → para_grad  (need n iterations)
           ↑      ↑
          input dimension 만큼

Reverse:  loss → para_grad  (1 iteration)
           ↑      ↑
          output dimension = 1
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Jacobian-Vector Product (JVP)

함수 $f: \mathbb{R}^n \to \mathbb{R}^m$ 와 tangent vector $\dot{x} \in \mathbb{R}^n$ 에 대해:

$$
\text{JVP}(f, x, \dot{x}) := J_f(x) \dot{x} = \left[ \frac{\partial f_i}{\partial x_j}(x) \dot{x}_j \right]_{i,j}
$$

결과는 $\dot{y} \in \mathbb{R}^m$.

### 정의 1.2 — Vector-Jacobian Product (VJP)

함수 $f: \mathbb{R}^n \to \mathbb{R}^m$ 와 cotangent vector $\bar{y} \in \mathbb{R}^m$ 에 대해:

$$
\text{VJP}(f, x, \bar{y}) := J_f(x)^T \bar{y} = \left[ \frac{\partial f_i}{\partial x_j}(x) \bar{y}_i \right]_{i,j}
$$

결과는 $\bar{x} \in \mathbb{R}^n$.

### 정의 1.3 — Forward-Mode AD 의 비용

$f$ 를 $N$ 개의 elementary operations 의 composition 이라 할 때, JVP 를 구하는 cost:

$$
\text{Cost}_{\text{forward}}(f, \text{all gradients}) = n \cdot O(N)
$$

(Input dimension 마다 한 번씩)

### 정의 1.4 — Reverse-Mode AD 의 비용

$$
\text{Cost}_{\text{reverse}}(f, \text{all gradients}) = m \cdot O(N)
$$

(Output dimension 마다 한 번씩)

---

## 🔬 정리와 증명

### 정리 1.1 (Chain Rule — Forward 와 Reverse 의 동등성)

두 함수 $g: \mathbb{R}^n \to \mathbb{R}^p$, $f: \mathbb{R}^p \to \mathbb{R}^m$ 이 합성된 $L = f \circ g$ 에 대해:

$$
J_L(x) = J_f(g(x)) \cdot J_g(x)
$$

Forward-mode 로 계산 (left-to-right):

$$
\dot{y} = J_f(g(x)) \cdot [J_g(x) \cdot \dot{x}]
$$

Reverse-mode 로 계산 (right-to-left):

$$
\bar{x} = [J_g(x)^T] \cdot [J_f(g(x))^T \cdot \bar{y}]
$$

둘 다 **같은 Jacobian** 을 나타냄.

**증명**: 행렬 곱셈의 결합성에 의해 자명 $\square$.

### 정리 1.2 (ML 에서 Reverse-Mode 의 최적성)

Scalar loss $L: \mathbb{R}^n \to \mathbb{R}$ (즉 $m=1$) 에 대해, 모든 parameter 의 gradient $\nabla L \in \mathbb{R}^n$ 을 구하는 비용:

$$
\text{Cost}_{\text{forward}} = n \cdot O(N), \quad \text{Cost}_{\text{reverse}} = 1 \cdot O(N) = O(N)
$$

따라서 **Reverse-mode 는 forward-mode 보다 $n$ 배 효율적**.

**증명**: JVP 정의에서 모든 input 의 gradient 를 각각 구하려면 $e_1, \ldots, e_n$ 에 대해 $n$ 번 호출. VJP 정의에서 $\bar{y} = 1$ 이므로 1 회 호출로 충분 $\square$.

### 따름 정리 1.3 (Forward-Mode 가 나은 경우)

Output dimension 이 input 보다 훨씬 많은 경우, 예를 들어 generating model 에서 $m \gg n$:

$$
\text{Cost}_{\text{reverse}} = m \cdot O(N) \gg n \cdot O(N) = \text{Cost}_{\text{forward}}
$$

**사례**: StyleGAN 의 generator $z \in \mathbb{R}^{512} \to \text{image} \in \mathbb{R}^{3 \times 1024 \times 1024}$ 에 대해 forward-mode 가 더 효율적.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Dual Number 를 사용한 Forward-Mode AD

```python
import torch
import numpy as np

class Dual:
    """Dual number: a + b*ε (ε² = 0)."""
    def __init__(self, val, deriv=0.0):
        self.val = val
        self.deriv = deriv
    
    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val,
                       self.val * other.deriv + self.deriv * other.val)
        else:
            return Dual(self.val * other, self.deriv * other)
    
    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.deriv + other.deriv)
        else:
            return Dual(self.val + other, self.deriv)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)

# f(x) = x^3 + 2x^2
def f(x):
    return x * x * x + 2 * x * x

# Forward-mode: 직접 계산
x_val = 2.0
x_dual = Dual(x_val, 1.0)  # seed with ė_x = 1
result_dual = f(x_dual)

print(f"f({x_val}) = {result_dual.val}")
print(f"f'({x_val}) = {result_dual.deriv}")
print(f"Expected f'(2) = 3*2^2 + 4*2 = 20")

# Verify with PyTorch
x_torch = torch.tensor(2.0, requires_grad=True)
y_torch = x_torch ** 3 + 2 * x_torch ** 2
y_torch.backward()
print(f"PyTorch: f'(2) = {x_torch.grad.item()}")
```

**예상 출력**:
```
f(2.0) = 12.0
f'(2.0) = 20.0
Expected f'(2) = 3*2^2 + 4*2 = 20
PyTorch: f'(2) = 20.0
```

### 실험 2 — Forward vs Reverse Cost 비교

```python
import torch
import time

# Network: n inputs → m outputs
n_input = 100
m_output = 1  # scalar loss

# Random network
W1 = torch.randn(50, n_input, requires_grad=True)
W2 = torch.randn(m_output, 50, requires_grad=True)

def forward(x):
    h = torch.relu(x @ W1.T)
    return (h @ W2.T).sum()

# Reverse-mode (PyTorch default)
x = torch.randn(1, n_input)
y = forward(x)
start = time.time()
y.backward()
time_reverse = time.time() - start

print(f"Scalar loss → Reverse-mode: {time_reverse*1000:.3f} ms")
print(f"  Computed: all {W1.grad.numel() + W2.grad.numel()} gradients in 1 backward pass")
```

**예상 출력**:
```
Scalar loss → Reverse-mode: 0.234 ms
  Computed: all 5150 gradients in 1 backward pass
```

### 실험 3 — JAX: jvp vs grad

```python
try:
    import jax
    import jax.numpy as jnp
    from jax import jacobian, jvp, vjp, grad
    
    def f(x):
        return jnp.sin(x) * jnp.exp(x)
    
    x0 = jnp.array(1.0)
    
    # Forward-mode JVP
    tangent = jnp.array(1.0)
    y_val, y_dot = jvp(f, (x0,), (tangent,))
    
    # Reverse-mode VJP
    y_val2, vjp_fn = vjp(f, x0)
    cotangent = jnp.array(1.0)
    grad_x = vjp_fn(cotangent)[0]
    
    # Using grad (convenience wrapper for scalar output)
    grad_fn = grad(f)
    grad_x2 = grad_fn(x0)
    
    print(f"f({x0}) = {y_val}")
    print(f"JVP: f'({x0}) = {y_dot}")
    print(f"VJP: f'({x0}) = {grad_x}")
    print(f"grad: f'({x0}) = {grad_x2}")
    print(f"Numerical: {float((jnp.sin(x0+1e-5)*jnp.exp(x0+1e-5) - jnp.sin(x0)*jnp.exp(x0))/1e-5)}")
except ImportError:
    print("JAX not installed. Install with: pip install jax jaxlib")
```

**예상 출력**:
```
f(1.0) = 1.8184
JVP: f'(1.0) = 3.5434
VJP: f'(1.0) = 3.5434
grad: f'(1.0) = 3.5434
Numerical: 3.5434
```

---

## 🔗 실전 활용

### 1. Custom Autograd Function 설계

VJP 를 명시적으로 구현할 때 reverse-mode 를 전제:

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):  # grad_output = ∂L/∂output
        x, = ctx.saved_tensors
        return 2 * x * grad_output  # ∂L/∂x = ∂output/∂x * ∂L/∂output (VJP!)
```

### 2. Jacobian 행렬 계산이 필요한 경우

JAX `jacfwd` (forward-mode) vs `jacrev` (reverse-mode):

```python
# jacfwd: 효율적 if output dim < input dim
J_fwd = jax.jacfwd(f)(x)  # O(m * cost(f))

# jacrev: 효율적 if input dim < output dim  
J_rev = jax.jacrev(f)(x)  # O(n * cost(f))
```

### 3. Physics-Informed NN (PINN)

PDE: $\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2}$ 를 PyTorch 로 구현할 때 공간 미분을 `backward` 를 이용해 계산:

```python
x = torch.randn(128, requires_grad=True)
u = model(x)
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x)[0]
residual = (u_t - nu * u_xx).pow(2).mean()
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Deterministic forward pass | Stochastic (dropout, batch norm) 시 VJP 가 부정확할 수 있음 → inference_mode 사용 |
| Smooth, differentiable function | ReLU 등 kink 가 있는 함수는 subgradient 사용 |
| Scalar loss | Vector loss 일 때 reverse-mode 도 여러 번 필요 → jacobian_matrix 사용 |
| Dense graph | Sparse graph 일 때 reverse-mode overhead 증가 가능 |
| Limited memory | Very deep network 시 reverse-mode 의 memory footprint 문제 → gradient checkpointing |

---

## 📌 핵심 정리

$$\boxed{\text{Forward}: \dot{y} = J \dot{x} \quad \text{Reverse}: \bar{x} = J^T \bar{y}}$$

| 모드 | 비용 (모든 gradient) | 효율적 상황 | PyTorch/JAX |
|------|-----|----------|---------|
| Forward (JVP) | $O(n \cdot N)$ | $m \gg n$ (generator) | `jax.jacfwd` |
| **Reverse (VJP)** | $O(m \cdot N)$ | **$m \ll n$ (ML)** | `loss.backward()`, `grad` |

**핵심 결론**: Neural network training 에서 $m=1$ (scalar loss) 이므로 **reverse-mode 가 자연스러운 선택** (Baydin et al. 2018).

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = \sin(x), g(y) = e^y$ 에 대해 $L = g(f(x))$ 의 forward-mode JVP 와 reverse-mode VJP 를 손으로 계산하고, $x_0 = \pi/4, \dot{x} = 0.1, \bar{L} = 1$ 일 때 결과를 비교하라.

<details>
<summary>해설</summary>

**Forward-mode**: $\dot{y} = \cos(x_0) \dot{x} = \cos(\pi/4) \cdot 0.1 = \frac{\sqrt{2}}{2} \cdot 0.1 \approx 0.0707$.

$\dot{L} = e^{f(x_0)} \dot{y} = e^{\sin(\pi/4)} \cdot 0.0707 \approx e^{0.707} \cdot 0.0707 \approx 0.150$.

**Reverse-mode**: $\bar{L} = 1$ (scalar).

$\bar{y} = e^{f(x_0)} \bar{L} = e^{0.707} \approx 2.029$.

$\bar{x} = \cos(x_0) \bar{y} = \cos(\pi/4) \cdot 2.029 \approx 0.707 \cdot 2.029 \approx 1.434$.

**검증**: $\frac{d}{dx} g(f(x)) = e^{\sin(x)} \cos(x)$ 이므로 $x_0 = \pi/4$ 에서 $1.434$ ✓ $\square$

</details>

**문제 2** (심화): Neural network 에서 gradient 계산의 비용을 정확히 분석하라. Parameter 개수 $p$, layer 수 $\ell$, batch size $b$, input/output dimension 을 명시했을 때 reverse-mode 의 시간 복잡도는?

<details>
<summary>해설</summary>

각 layer forward 의 cost $\approx$ 행렬-행렬 곱 = $O(ab \cdot c)$ ($a$ input, $b$ output, $c$ = 배치).

전체 $\ell$ layer → forward $O(p)$ (총 parameter 수만큼).

Reverse-mode: backward 도 각 layer 에서 local VJP → forward 와 동일한 cost $O(p)$.

**총 cost**: forward + backward = $O(2p)$ = $O(p)$.

**비교**: Forward-mode 로 각 input feature 에 대해 JVP → $O(n \cdot p)$ (훨씬 느림).

실제 PyTorch: forward $\approx 1.0 \times$ cost, backward $\approx 2.0 \times$ cost (intermediate 저장 + computation) → total $3.0 \times$ cost of forward-only $\square$

</details>

**문제 3** (논문 비평): Baydin et al. (2018) "Automatic differentiation in machine learning: a survey" 를 읽고, forward-mode 가 언제 역으로 더 효율적일 수 있는지 논하라. Hessian 계산, Jacobian 행렬 모두 필요한 경우는?

<details>
<summary>해설</summary>

**Hessian**: $H = \nabla^2 f = J(\nabla f)$. 

$\nabla f$ 를 reverse-mode 로 구한 후, 각 component 에 대해 다시 forward-mode JVP → Hessian-Vector Product (HVP) 계산 가능 (Ch6 참조).

Explicit Hessian matrix $H \in \mathbb{R}^{p \times p}$ 를 구하려면 forward-mode $p$ 번 또는 reverse-mode $p$ 번 필요 — 동등.

**Jacobian 행렬 $J \in \mathbb{R}^{m \times n}$**: 
- Forward-mode: $n$ 번 (input 마다) → cost $O(n \cdot \text{ops}(f))$
- Reverse-mode: $m$ 번 (output 마다) → cost $O(m \cdot \text{ops}(f))$

선택: $m < n$ 이면 reverse, $m > n$ 이면 forward.

**JAX 의 해법**: `jacfwd ∘ jacrev` 로 최적 순서 결정 (각 방향이 $\sqrt{mn}$ cost) — recursive 구조의 이점 $\square$

</details>

---

<div align="center">

[◀ 이전](../ch1-tensor/05-device-and-cuda-context.md) | [📚 README](../README.md) | [다음 ▶](./02-vjp-mathematics.md)

</div>
