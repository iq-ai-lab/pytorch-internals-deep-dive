# 05. functorch — vmap, grad, jacrev/jacfwd

## 🎯 핵심 질문

- `torch.func.grad(f)` 가 pure function 에서 어떻게 gradient 를 계산하는가?
- `torch.func.vmap(f)` 의 "batch dimension 제거" 가 어떻게 작동하며, 실제 loop 를 제거하는 메커니즘은?
- `jacrev` (역방향) vs `jacfwd` (정방향) 의 Jacobian 계산 복잡도는?
- `vmap(grad(f))` 로 per-sample gradient 를 어떻게 계산하고, DP-SGD 에서 왜 중요한가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

functorch (functional transforms library) 는 **JAX 의 철학을 PyTorch 에 도입한 것**입니다. JAX (Bradbury et al. 2018) 의 핵심 아이디어:

> **Pure function** 에 대해, grad/vmap/jit 같은 transformation 을 **composable** 하게 적용 가능.

PyTorch 의 traditional autograd 는:
- `loss.backward()` — imperative, dynamic graph
- 장점: 유연함, 조건문/루프 쉬움
- 단점: batch 처리 수동, per-sample gradient 복잡함

functorch 는:
- `torch.func.grad(loss_fn)` — declarative, transformable
- 장점: functional composition, 자동 per-sample, batch 처리
- 단점: pure function 요구 (side effects 없어야 함)

**실전**:
- Jacobian/Hessian 계산: `jacrev`, `jacfwd`, `hessian`
- Per-sample gradients (DP-SGD): `vmap(grad(loss_fn))`
- Model ensemble: `vmap` over multiple parameter sets

---

## 📐 수학적 선행 조건

- **Autograd 기초** (Chapter 2): VJP, Jacobian, chain rule
- **JAX paper** (Bradbury et al. 2018): transformation 의 개념, composability
- **Linear algebra**: Jacobian matrix (m×n) 의 structure
- **Numerical differentiation**: forward-mode vs reverse-mode AD 의 비용

---

## 📖 직관적 이해

### grad: Pure function → Gradient function

**Traditional PyTorch**:
```python
def loss_fn(params, x, y):
    pred = model(params, x)
    return ((pred - y) ** 2).mean()

# Imperative: graph 내부에 backward 구현
loss_fn(params, x, y).backward()
```

**functorch (Pure)**:
```python
def loss_fn(params, x, y):
    pred = model(params, x)
    return ((pred - y) ** 2).mean()

grad_loss_fn = torch.func.grad(loss_fn)
grads = grad_loss_fn(params, x, y)
# grads: params 와 같은 shape, ∇_params loss
```

**Difference**: loss_fn 은 side effects 없는 pure function — backward 를 명시 호출 안 함, 자동으로 derivative 취함.

### vmap: Explicit batch dimension 제거

**Traditional (manual loop)**:
```python
def process_batch(params, xs):
    results = []
    for x in xs:
        results.append(model(params, x))
    return torch.stack(results)  # [batch_size, output_dim]
```

**functorch (vmap)**:
```python
def process_single(params, x):
    return model(params, x)  # Single sample

process_batch = torch.func.vmap(process_single, in_dims=(None, 0))
# in_dims=(None, 0): params 는 batch 아님 (broadcast), x 는 dim 0 batched
results = process_batch(params, xs)  # xs: [batch_size, ...]
```

**내부**: vmap 이 loop 를 제거하고, vectorized operation 으로 변환. 예:

```
Manual: for i in range(batch_size): y[i] = model(params, x[i])
vmap:   y = vmap_model(params, xs)  # single vectorized op
```

### jacrev vs jacfwd

```
Jacobian J = ∂y / ∂x  (m × n matrix)

jacrev (reverse-mode, VJP 기반):
  - 각 output dimension 마다 VJP 계산
  - Cost: O(m) × O(n) = O(m·n)
  - 최적: m < n (output 작음)

jacfwd (forward-mode, JVP 기반):
  - 각 input dimension 마다 JVP 계산
  - Cost: O(n) × O(m) = O(n·m)
  - 최적: n < m (input 작음)
```

---

## ✏️ 엄밀한 정의

### 정의 3.13 — Pure Function Transformation

$$f_{\text{pure}} : (\mathbf{params}, \mathbf{x}) \to \mathcal{R}$$

where $f_{\text{pure}}$ has **no side effects** (no global state mutation, no I/O).

Transform $T$ 는:
$$T(f) : (\mathbf{params}, \mathbf{x}) \to \mathcal{R}'$$

예:
$$\text{grad}(f) : (\mathbf{params}, \mathbf{x}) \to \nabla_{\mathbf{params}} f$$
$$\text{vmap}(f, \text{in\_dims}) : \text{batched inputs} \to \text{batched outputs}$$

### 정의 3.14 — Jacobian (행렬 형태)

함수 $f : \mathbb{R}^n \to \mathbb{R}^m$ 에 대해:

$$J_f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n}\\
\vdots & \ddots & \vdots\\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

**jacrev**: reverse-mode AD 를 m 번 (각 output 마다)
$$J_f = \begin{bmatrix} \nabla y_1 \\ \vdots \\ \nabla y_m \end{bmatrix}$$

**jacfwd**: forward-mode AD 를 n 번 (각 input 마다)
$$J_f = \begin{bmatrix} \partial y / \partial x_1 & \cdots & \partial y / \partial x_n \end{bmatrix}$$

### 정의 3.15 — Vmap in Broadcasting Semantics

```python
torch.func.vmap(f, in_dims=mapping, out_dims=output_mapping)
```

- `in_dims=(None, 0, 1)`: 첫째 arg 는 batched 아님 (broadcast), 둘째는 dim 0, 셋째는 dim 1
- `out_dims=0`: output 의 batch dimension 은 0

내부:
$$\text{vmap}(f)(\mathbf{p}, \mathbf{X}) = \text{stack}([f(\mathbf{p}, \mathbf{x}_i) \text{ for } \mathbf{x}_i \in \mathbf{X}])$$

---

## 🔬 정리와 증명

### 정리 3.8 — grad 의 정확성 (Differentiable 함수에서)

**주장**: `torch.func.grad(f)` 가 반환하는 함수는 $\nabla_{\text{argnums}} f$ 를 정확히 계산한다.

**증명 (스케치)**:

`torch.func.grad` 는 내부적으로:
1. Input 을 추적 (tape recording)
2. Autograd engine 을 trigger
3. Backpropagation (reverse-mode AD)
4. Gradient 반환

이는 traditional `loss.backward()` 와 같은 VJP 계산이므로, 수학적으로 동일. $\square$

### 정리 3.9 — vmap 과 Loop Elimination

**주장**: `vmap(f)` 는 개념적으로 loop 과 동일하지만, 실제로는 vectorized operation 으로 최적화된다.

**증명**:

**Loop semantics** (정확):
$$\text{loop}(f, \mathbf{X}) = [\, f(x_i) \text{ for } i = 1 \cdots n \,]$$

**vmap semantics** (동일하지만 vectorized):
$$\text{vmap}(f)(\mathbf{X}) = [\, f(x_i) \text{ for } i = 1 \cdots n \,]$$

**최적화**:
- Elementwise ops: loop unrolling 또는 SIMD
- Matrix ops: 각 batch item 의 matmul 을 single batched matmul 로 변환
- 예: `loop: 1000 × (size[256, 512])` → `vmap: size[1000, 256, 512]`

따라서 **semantically same, computationally faster** $\square$.

### 정리 3.10 — jacrev vs jacfwd 의 복잡도

**주장**: 
- `jacrev`: O(m) reverse passes (output dimension)
- `jacfwd`: O(n) forward passes (input dimension)

**증명**:

jacrev (VJP 기반):
$$J = \begin{bmatrix} \nabla y_1 \\ \vdots \\ \nabla y_m \end{bmatrix}$$

각 $\nabla y_i$ 는 1 VJP pass → m passes total.

jacfwd (JVP 기반):
$$J = \begin{bmatrix} \partial y / \partial x_1 & \cdots & \partial y / \partial x_n \end{bmatrix}$$

각 column 은 1 JVP pass → n passes total.

**Best choice**:
- If $m \ll n$ (output small): jacrev
- If $n \ll m$ (input small): jacfwd
- If $m \approx n$ (balanced): either, but jacrev slightly faster (AD 최적화)

$\square$

### 정리 3.11 — Per-sample Gradient via vmap(grad(...))

**주장**: `vmap(grad(loss), in_dims=0)` 는 batch 내 각 sample 의 gradient 를 동시에 계산, loop 보다 빠르다.

**증명**:

Traditional (loop):
```python
grads = []
for x_i, y_i in zip(xs, ys):
    grad = grad_loss_fn(params, x_i, y_i)
    grads.append(grad)
# Time: O(batch_size × single_grad_time)
```

functorch (vmap):
```python
grads = vmap(grad_loss_fn, in_dims=(None, 0, 0))(params, xs, ys)
# Time: O(batch_size × single_grad_time) 이론상,
# 실제: vectorization overhead 감소 → faster
```

**메커니즘**:
1. `grad(loss)` 는 single sample 의 gradient function
2. `vmap` 이 batch 처리로 변환
3. Dispatcher 가 batched tensor operation 호출
4. GPU 의 SIMD/warp level parallelism 활용

따라서 **per-sample gradient 를 batch dimension 으로 vectorize** $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — grad 의 기본 사용

```python
import torch
import torch.func as F

def loss_fn(params, x, y):
    pred = params[0] * x + params[1]
    return ((pred - y) ** 2).mean()

params = [torch.tensor([2.0], requires_grad=True),
          torch.tensor([1.0], requires_grad=True)]
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([5.0, 7.0, 9.0])

# grad_fn: returns gradient w.r.t. params
grad_fn = F.grad(loss_fn)
grads = grad_fn(params, x, y)

print(f"Gradient w.r.t. params[0] (weight): {grads[0]}")
print(f"Gradient w.r.t. params[1] (bias): {grads[1]}")

# Verify with traditional autograd
params_trad = [p.clone().detach().requires_grad_(True) for p in params]
loss = loss_fn(params_trad, x, y)
loss.backward()
print(f"Traditional grad[0]: {params_trad[0].grad}")
# Should match
```

### 실험 2 — vmap 의 batch 처리

```python
import torch.func as F

def model(params, x):
    return params @ x  # Linear: [d_out, d_in] @ [d_in] -> [d_out]

params = torch.randn(3, 4)  # [output_dim=3, input_dim=4]
xs = torch.randn(10, 4)     # [batch_size=10, input_dim=4]

# Manual loop
results_loop = torch.stack([model(params, x) for x in xs])
# Results: [10, 3]

# vmap
model_batch = F.vmap(model, in_dims=(None, 0), out_dims=0)
results_vmap = model_batch(params, xs)

print(f"Manual shape: {results_loop.shape}")  # [10, 3]
print(f"Vmap shape: {results_vmap.shape}")     # [10, 3]
print(f"Results match: {torch.allclose(results_loop, results_vmap)}")  # True

# vmap 은 loop 을 제거하고 vectorized matmul 사용
# Time: vmap << manual loop (GPU 에서)
```

### 실험 3 — jacrev vs jacfwd

```python
import torch.func as F

def f(x):
    # Quadratic function, m=1, n=5
    return (x ** 2).sum()

x = torch.randn(5)

# Jacobian (정확히는 gradient 1×5 matrix)
# jacrev: 1 reverse pass
jac_rev = F.jacrev(f)(x)
print(f"jacrev result: {jac_rev}")  # [5] = grad

# jacfwd: 5 forward passes (각 input dimension 마다)
jac_fwd = F.jacfwd(f)(x)
print(f"jacfwd result: {jac_fwd}")  # [5]

print(f"Match: {torch.allclose(jac_rev, jac_fwd)}")  # True

# 더 흥미로운 예: f: R^n -> R^m, m과 n이 다를 때
def f_complex(x):
    # x: [n], output: [m]
    A = torch.randn(10, len(x))
    return A @ x  # [10] (m=10)

n = 5
x = torch.randn(n)

# jacrev (10 reverse passes)
J_rev = F.jacrev(f_complex)(x)
print(f"jacrev shape: {J_rev.shape}")  # [10, 5]

# jacfwd (5 forward passes)
J_fwd = F.jacfwd(f_complex)(x)
print(f"jacfwd shape: {J_fwd.shape}")  # [10, 5]

# 여기서 m=10 > n=5 → jacfwd 가 더 효율 (5 passes < 10 passes)
```

### 실험 4 — Per-sample gradients (DP-SGD)

```python
import torch
import torch.func as F

def loss_fn(params, x, y):
    pred = torch.nn.functional.linear(x, params[0], params[1])
    return torch.nn.functional.mse_loss(pred, y, reduction='none')  # [batch_size]

# Model parameters
d_in, d_out = 10, 5
params = [torch.randn(d_out, d_in), torch.randn(d_out)]

# Batch
batch_size = 32
xs = torch.randn(batch_size, d_in)
ys = torch.randn(batch_size, d_out)

# Per-sample gradient (DP-SGD 에서 필요)
grad_loss_fn = F.grad(loss_fn, argnums=0)  # grad w.r.t. params[0]

# vmap: batch 의 각 sample 에 대해 gradient
per_sample_grads = F.vmap(
    lambda x_i, y_i: grad_loss_fn(params, x_i, y_i),
    in_dims=(0, 0)  # xs[i], ys[i]
)(xs, ys)

print(f"per_sample_grads shape: {per_sample_grads.shape}")
# [batch_size, d_out, d_in] — 각 sample 의 gradient 가 batch dimension 으로 쌓임

# DP-SGD: clipping
max_norm = 1.0
grad_norms = torch.norm(per_sample_grads.reshape(batch_size, -1), dim=1)
clipped = per_sample_grads / torch.clamp(grad_norms / max_norm, min=1.0)[:, None, None]

# Aggregate (averaging)
avg_grad = clipped.mean(dim=0)
print(f"Averaged gradient shape: {avg_grad.shape}")  # [d_out, d_in]
```

---

## 🔗 실전 활용

### 1. Hessian 계산 (2차 미분)

```python
import torch.func as F

def loss_fn(params, x, y):
    pred = params @ x
    return ((pred - y) ** 2).mean()

params = torch.randn(5)
x = torch.randn(5)
y = torch.randn(1)

# Hessian = Jacobian of Jacobian
# H[i,j] = ∂²loss / ∂params[i] ∂params[j]

hessian_fn = F.hessian(loss_fn)
H = hessian_fn(params, x, y)

print(f"Hessian shape: {H.shape}")  # [5, 5]

# Or manually: vmap(grad(grad(...)))
grad_fn = F.grad(loss_fn)
hess_manual = F.jacobian(lambda p: grad_fn(p, x, y))(params)
print(f"Manual hessian shape: {hess_manual.shape}")
```

### 2. Model Ensemble via vmap

```python
import torch.func as F

def forward(params_dict, x):
    # Single model forward
    return params_dict['weight'] @ x + params_dict['bias']

# Multiple sets of parameters
params_ensemble = {
    'weight': torch.randn(10, 5, 3),  # [num_models=10, d_out=5, d_in=3]
    'bias': torch.randn(10, 5)
}

x = torch.randn(3)

# Ensemble forward (vmap over model dimension)
ensemble_forward = F.vmap(
    forward,
    in_dims=({'weight': 0, 'bias': 0}, None)
)(params_ensemble, x)

print(f"Ensemble output shape: {ensemble_forward.shape}")  # [10, 5]
# Each row = one model's output
```

### 3. Jacobian with respect to specific dimensions

```python
import torch.func as F

def f(x, y):
    return x ** 2 + x * y

x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])

# Jacobian w.r.t. x (argnums=0)
jac_x = F.jacrev(f, argnums=0)(x, y)
print(f"∂f/∂x: {jac_x}")  # [2, 2] (diagonal-ish)

# Jacobian w.r.t. y
jac_y = F.jacrev(f, argnums=1)(x, y)
print(f"∂f/∂y: {jac_y}")  # [2, 2]

# Combined: jacobian w.r.t. (x, y)
jac_all = F.jacrev(f, argnums=(0, 1))(x, y)
print(f"∂f/∂(x,y): {len(jac_all)} Jacobians")  # 2 jacobians
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Pure function (no side effects) | In-place ops, global state mutation 불가 → `torch.nn` layers 는 FunctionalModule 필요 |
| Differentiable path | Non-differentiable ops (argmax, quantize) 불가 → custom vjp_rule 정의 필요 |
| Static graph | Control flow (if/while) 는 traced 되어 static 됨 → dynamic control 은 어려움 |
| torch.compile 미완전 지원 | functorch + compile 조합 시 edge case 존재 → pure Python fallback 필요할 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\text{torch.func: } \text{grad} + \text{vmap} + \text{jacrev/jacfwd} = \text{JAX-like transforms in PyTorch}}$$

| Transform | Input | Output | 용도 |
|-----------|-------|--------|------|
| **grad** | f: (...) → scalar | ∇f: (...) → gradients | Single gradient |
| **vmap** | f: single → ... | vf: batched → ... | Batch processing, remove loop |
| **jacrev** | f: R^n → R^m | J: (...) → [m,n] | Jacobian (output-driven) |
| **jacfwd** | f: R^n → R^m | J: (...) → [m,n] | Jacobian (input-driven) |
| **compose** | grad(vmap(...)) | Vectorized gradients | Per-sample gradients (DP-SGD) |

**핵심**: Functional transform 의 composition 으로, 복잡한 미분·배치 연산을 선언적으로 표현.

---

## 🤔 생각해볼 문제

**문제 1** (기초): `torch.func.grad(f)` 에서 argnums 가 여러 개일 때 (예: argnums=(0, 2)), 반환값은 무엇인가? 어떻게 사용하는가?

<details><summary>해설</summary>

```python
def f(a, b, c):
    return a * b + c

grad_fn = torch.func.grad(f, argnums=(0, 2))
grads = grad_fn(a, b, c)  # Returns tuple (∂f/∂a, ∂f/∂c)

# grads[0] = ∂f/∂a = b
# grads[1] = ∂f/∂c = 1
```

**여러 argnums 시**:
- 반환값은 tuple of gradients (argnums 순서대로)
- 각 gradient 는 해당 arg 와 같은 shape

**활용**:
```python
grads_a, grads_c = grad_fn(a, b, c)  # Unpacking
```

This is useful for parameter + data gradient 분리 (예: domain adaptation 에서 domain-specific params 만 update). $\square$

</details>

**문제 2** (심화): `vmap(grad(...), out_dims=...)` 에서 out_dims 를 설정하는 이유는? 어떤 경우에 out_dims ≠ 0 일까?

<details><summary>해설</summary>

```python
def loss_fn(params, x):
    return (params @ x) ** 2

params = torch.randn(5, 10)  # vmap 할 대상 아님 (None)
xs = torch.randn(3, 10)      # [batch_size=3, input_dim=10]

# 기본 (out_dims=0)
grads_0 = torch.func.vmap(
    torch.func.grad(loss_fn, argnums=0),
    in_dims=(None, 0),
    out_dims=0  # output gradient 가 batch dim 0
)(params, xs)

# grads_0 shape: [3, 5, 10]
# grads_0[i] = ∂loss_i / ∂params

# out_dims=1 사용 사례
# 만약 output gradient 를 다른 차원에 쌓고 싶으면
grads_1 = torch.func.vmap(
    torch.func.grad(loss_fn, argnums=0),
    in_dims=(None, 0),
    out_dims=1  # output gradient 가 batch dim 1
)(params, xs)

# grads_1 shape: [5, 3, 10]
# grads_1[:, i, :] = ∂loss_i / ∂params
```

**언제 쓰나**:
- Output 의 natural dimension 이 0 이 아닐 때
- Downstream operation 이 특정 batch dim 을 기대할 때
- Memory layout 최적화 (cache-friendly dimension ordering)

실제로 대부분의 경우 out_dims=0 (기본값) 이면 충분. $\square$

</details>

**문제 3** (논문 비평): Bradbury et al. "JAX: Composable transformations of NumPy programs" (2018) 에서 주장하는 "transformation composability" 가 왜 새로운 아이디어였고, PyTorch functorch 가 이를 채택할 때 어떤 설계 결정을 했는가? (forward-mode AD 를 먼저 구현한 이유 등)

<details><summary>해설</summary>

**JAX 의 혁신** (Bradbury 2018):
- Pure function 에 grad/vmap/jit 을 **자유롭게 compose** 가능
- `grad(vmap(f))` = per-sample gradient
- `vmap(vmap(...))` = multi-dimensional batch
- 반면 TensorFlow/PyTorch (pre-2020) 는 batch/gradient 를 분리하거나 manual 로 처리

**설계 challenge**:
1. Pure function 요구 → in-place ops, mutable state 불가
2. Composability → grad(vmap(...)) 시 nested autograd 복잡도 증가
3. Performance → loop unrolling/vectorization 

**PyTorch functorch 의 선택**:
- **Forward-mode AD (JVP) 먼저** 구현: vmap(grad(...)) 에서 reverse-mode nested AD 의 overhead 감소
- Reverse-mode만으로 jacrev → m reverse passes (비쌈)
- Forward-mode jacfwd → n forward passes (input dimension driven, 가벼움)
- 따라서 jacfwd + vmap composition 이 빠름

**논문 참고**:
- Bradbury et al. "JAX: Composable..." (ICLR 2022) — JAX 의 공식 소개
- PyTorch 블로그 "Introducing functorch, an eager-mode autodiff library for PyTorch" (2021) — design decision

결론: composition 의 flexibility vs performance 의 trade-off 를 forward-mode AD 로 해결. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-torch-library-custom-op.md) | [📚 README](../README.md) | [다음 ▶](../ch4-cuda/01-gpu-architecture.md)

</div>
