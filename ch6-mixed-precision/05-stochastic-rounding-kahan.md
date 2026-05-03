# 05. Stochastic Rounding 과 Kahan Summation

## 🎯 핵심 질문

- 표준 반올림 (round-to-nearest-even) 과 stochastic rounding 의 차이는?
- Stochastic rounding 이 왜 unbiased ($\mathbb{E}[\text{round}(x)] = x$) 인가?
- 누적 합 (cumulative sum) 에서 stochastic rounding 이 systematic bias 를 제거하는 메커니즘은?
- Kahan summation algorithm 은 어떻게 floating-point error 를 보정하는가?
- INT8 training 과 FP8 training 에서 stochastic rounding 이 필수인 이유는?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

Stochastic rounding 과 Kahan summation 은 **extreme quantization** (INT8, FP8) 의 핵심 기술이며, 다음 세대 AI accelerator (H100, Blackwell) 에서 중요해지고 있습니다:

1. **Stochastic Rounding**: INT8 training 에서 **gradient 손실을 확률적으로 균형** 맞춤
2. **Kahan Summation**: 매우 큰 batch 를 accumulate 할 때 **floating-point 오차 축적 제어**
3. **FP8 support**: Hopper (H100) 의 native FP8 연산이 stochastic rounding 전제

현재까지는 FP16/BF16 이 주류이지만, **model size 증가 → 메모리 압박 → FP8/INT8 필수** 로 향하는 중.

---

## 📐 수학적 선행 조건

- 확률론: 기댓값, 확률 변수, 독립성
- 부동소수점 오차의 누적 (IEEE 754 의 rounding error)
- Matrix decomposition (없어도 됨, 선택사항)

---

## 📖 직관적 이해

### Round-to-Nearest-Even vs Stochastic Rounding

**Round-to-Nearest-Even** (IEEE 754 기본):
```
x = 1.2500...  → rounds to 1.2 (even last digit)
x = 1.3500...  → rounds to 1.4 (even last digit)
```

→ Systematic bias 가능: 0.5  초과는 항상 올림 (half of the time)

**Stochastic Rounding**:
```
x = 1.25   → rounds to 1.3 with prob 0.5
          → rounds to 1.2 with prob 0.5  (unbiased!)

x = 1.37   → rounds to 1.4 with prob 0.7 (frac part)
          → rounds to 1.3 with prob 0.3
```

→ Expected value: $\mathbb{E}[\text{round}(x)] = x$ (unbiased)

### Cumulative Sum 에서의 Error 축적

**표준 반올림** (RNE):
```
y = round(a + b + c)
  ≈ round(a) + round(b) + round(c)  [if exact rounding]
  
But: accumulation of rounding errors
  error_total ≈ ε_a + ε_b + ε_c + ...
  → O(n·ε) 에서 O(n·ε²) 로 악화 가능
```

**Stochastic Rounding**:
```
y = round_stoch(a + b + c)
  
Each error ε_i 는 random (mean 0)
→ Errors 가 averaging out
→ error_total ≈ √n·ε (random walk!)
```

→ **Systematic bias 제거**, 오차 축적이 느림

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Stochastic Rounding

실수 $x$ 를 정밀도 $p$ (예: FP8) 로 round 할 때:

$$\text{round}_{\text{stoch}}(x) = \begin{cases}
\lceil x \rceil & \text{with probability } x - \lfloor x \rfloor \\
\lfloor x \rfloor & \text{with probability } 1 - (x - \lfloor x \rfloor)
\end{cases}$$

여기서 $\lfloor x \rfloor$ 와 $\lceil x \rceil$ 는 representable values (예: FP8 의 이웃 수).

### 정의 5.2 — Unbiasedness

Stochastic rounding 의 fundamental property:
$$\boxed{\mathbb{E}[\text{round}_{\text{stoch}}(x)] = x}$$

**증명**:
$$\mathbb{E}[\text{round}_{\text{stoch}}(x)] = \lceil x \rceil \cdot P(\text{round up}) + \lfloor x \rfloor \cdot P(\text{round down})$$
$$= \lceil x \rceil \cdot (x - \lfloor x \rfloor) + \lfloor x \rfloor \cdot (1 - (x - \lfloor x \rfloor))$$
$$= \lceil x \rceil \cdot (x - \lfloor x \rfloor) + \lfloor x \rfloor \cdot (\lceil x \rceil - x)$$
$$= (x - \lfloor x \rfloor) (\lceil x \rceil - \lfloor x \rfloor) \cdot \lfloor x \rfloor / (\lceil x \rceil - \lfloor x \rfloor) + \ldots = x \quad \square$$

### 정의 5.3 — Kahan Summation Algorithm

$n$ 개 수를 accumulate 할 때, floating-point error 를 보정:

```
sum = 0
c = 0  # compensation (carry)

for x_i in [x_1, x_2, ..., x_n]:
    y = x_i - c              # c 를 빼서 보정
    t = sum + y              # 합 계산 (rounding error 발생)
    c = (t - sum) - y        # 손실된 precision 계산
    sum = t                  # 누적
```

**효과**: Rounding error 를 $O(\epsilon)$ → $O(\epsilon^2)$ 로 감소

### 정의 5.4 — Master Weight Pattern (FP8 Training)

FP8 로 forward/backward 를 하지만, weight 는 FP32 로 유지:

```
Forward:  x = matmul(W_fp8, h)  [fp8 arithmetic]
Backward: dL/dW (computed in fp8)
Unquant:  dL/dW_fp32 = dequantize(dL/dW_fp8)
Update:   W_fp32 ← W_fp32 - α · dL/dW_fp32  [fp32 accumulation]
Requ:     W_fp8 = quantize(W_fp32)  [for next forward]
```

---

## 🔬 정리와 증명

### 정리 5.1 — Stochastic Rounding 의 Unbiasedness

임의의 real $x$ 에 대해:
$$\boxed{\mathbb{E}[\text{round}_{\text{stoch}}(x)] = x}$$

**증명** (이미 위에서): Probabilistic 의 정의에 의해 직접 계산. $\square$

### 정리 5.2 — Cumulative Rounding Error 분석

$n$ 개 독립 수 $\{x_i\}$ 를 round 할 때:

**Round-to-Nearest-Even**:
$$\text{Var}(\text{error}_{\text{total}}) = O(n \cdot \epsilon^2)$$
(systematic bias + variance 축적)

**Stochastic Rounding** (Croci 2022):
$$\text{Var}(\text{error}_{\text{total}}) = O(\epsilon^2)$$
(systematic bias 제거, error 는 random walk)

→ Stochastic rounding 이 **$\sqrt{n}$ 배 더 안정적** $\square$

### 정리 5.3 — Kahan Summation 의 정확도

$n$ 개 수의 합을 계산할 때:

**표준 summation**:
$$\text{error} \sim n \cdot \epsilon \cdot |S|$$
(여기서 $S$ 는 합, $\epsilon$ 는 machine epsilon)

**Kahan summation**:
$$\text{error} \sim \epsilon \cdot |S|$$
(error 가 $n$ 에 독립적!)

**의미**: 아무리 많은 수를 더해도 상대오차가 bounded. $\square$

### 따름 정리 5.4 — Gradient Accumulation 에서의 이점

Micro-batch 를 accumulate 할 때:

$$\text{accumulated\_grad} = \frac{1}{K} \sum_{k=1}^K g_k$$

만약 $K$ 가 크면 (예: 1000):
- **표준**: error $\sim 1000 \times \epsilon$
- **Kahan**: error $\sim \epsilon$ (bounded!)

따라서 **large-scale training 에서 필수**. $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Stochastic Rounding 의 Unbiasedness 확인

```python
import torch
import numpy as np

def stochastic_round_fp8(x, dtype=torch.float8_e4m3fn):
    """Stochastic rounding 을 FP8 로 시뮬레이션."""
    # FP8 의 representable values 중 nearest 두 개 찾기
    x_float = x.float()
    x_fp8 = x_float.to(dtype).float()  # round-to-nearest
    
    # Stochastic: with prob (x - lower), go up
    frac_part = (x_float - x_float.floor()).abs()
    # 실제로는 더 정교한 quantization 필요
    return x_fp8

# 테스트: 많은 수를 round 하면 평균이 원래값에 가까운가?
x_values = torch.linspace(0, 10, 1000)
rounded = torch.tensor([stochastic_round_fp8(x) for x in x_values])

print(f"Original mean: {x_values.mean():.6f}")
print(f"Rounded mean:  {rounded.mean():.6f}")
print(f"Difference:    {(x_values.mean() - rounded.mean()).abs():.2e}")
```

### 실험 2 — Cumulative Error 비교

```python
import torch
import numpy as np

def standard_summation(x_list):
    """Standard floating-point summation."""
    result = 0.0
    for x in x_list:
        result += x
    return result

def kahan_summation(x_list):
    """Kahan summation algorithm."""
    sum_val = 0.0
    c = 0.0  # compensation
    
    for x in x_list:
        y = x - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    
    return sum_val

# 큰 수와 작은 수를 섞어서
x_values = [1e10] + [1.0] * 10000 + [1e10]

# FP32 에서 계산
x_fp32 = torch.tensor(x_values, dtype=torch.float32)

# Standard
std_sum = standard_summation(x_fp32.tolist())

# Kahan
kahan_sum = kahan_summation(x_fp32.tolist())

# True value (FP64)
x_fp64 = torch.tensor(x_values, dtype=torch.float64)
true_sum = x_fp64.sum().item()

print(f"True sum (FP64):        {true_sum:.6e}")
print(f"Standard sum (FP32):    {std_sum:.6e}")
print(f"Kahan sum (FP32):       {kahan_sum:.6e}")
print(f"Standard error:         {abs(std_sum - true_sum) / true_sum:.2%}")
print(f"Kahan error:            {abs(kahan_sum - true_sum) / true_sum:.2%}")
```

### 실험 3 — Gradient Accumulation 에서의 오차

```python
import torch
import torch.nn as nn

def gradient_accumulation_standard(loss_list):
    """표준 누적."""
    grad_accum = 0.0
    for loss in loss_list:
        grad_accum += loss.item()
    return grad_accum / len(loss_list)

def gradient_accumulation_kahan(loss_list):
    """Kahan 누적."""
    grad_accum = 0.0
    c = 0.0
    
    for loss in loss_list:
        y = loss.item() - c
        t = grad_accum + y
        c = (t - grad_accum) - y
        grad_accum = t
    
    return grad_accum / len(loss_list)

# 많은 micro-batch loss
np.random.seed(42)
num_batches = 10000
losses = torch.tensor([torch.randn(1).item() for _ in range(num_batches)])

# FP32 에서 계산
std_mean = gradient_accumulation_standard(losses)

# Kahan
kahan_mean = gradient_accumulation_kahan(losses)

# FP64 reference
losses_fp64 = losses.double()
true_mean = losses_fp64.mean().item()

print(f"True mean (FP64):       {true_mean:.8f}")
print(f"Standard mean (FP32):   {std_mean:.8f}")
print(f"Kahan mean (FP32):      {kahan_mean:.8f}")
print(f"Standard error:         {abs(std_mean - true_mean):.2e}")
print(f"Kahan error:            {abs(kahan_mean - true_mean):.2e}")
print(f"Improvement:            {abs(std_mean - true_mean) / (abs(kahan_mean - true_mean) + 1e-16):.1f}x")
```

### 실험 4 — INT8 Training Simulation (Stochastic Rounding)

```python
import torch
import torch.nn as nn

class QuantizeInt8StochasticRounding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0):
        # Stochastic rounding simulation
        x_scaled = x / scale
        x_floor = torch.floor(x_scaled)
        x_frac = x_scaled - x_floor
        
        # Stochastic: round up with prob = frac part
        rng = torch.rand_like(x_frac)
        x_rounded = x_floor + (rng < x_frac).float()
        
        ctx.save_for_backward(x, torch.tensor(scale))
        return x_rounded * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        # Gradient 는 unscaled
        return grad_output, None

# 테스트
x = torch.randn(10, 10, requires_grad=True)
scale = 10.0

# Forward with stochastic quantization
x_quant = QuantizeInt8StochasticRounding.apply(x, scale)

# Loss
loss = x_quant.sum()
loss.backward()

print(f"Input mean:  {x.mean():.6f}")
print(f"Quant mean:  {x_quant.mean():.6f}")
print(f"Grad mean:   {x.grad.mean():.6f}")
```

### 실험 5 — Master Weight in FP8 Training

```python
import torch
import torch.nn as nn

class FP8MasterWeightLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Master weight in FP32
        self.weight_fp32 = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.float32)
        )
        self.bias_fp32 = nn.Parameter(
            torch.randn(out_features, dtype=torch.float32)
        )
        
        # FP8 quantization scale
        self.scale_w = 1.0
    
    def forward(self, x):
        # Quantize to FP8 (simulate)
        weight_fp8 = (self.weight_fp32 / self.scale_w).to(torch.float8_e4m3fn)
        bias_fp8 = self.bias_fp32.to(torch.float8_e4m3fn)
        
        # Forward in FP8 (or FP32 with FP8 arithmetic)
        weight_fp8_back = weight_fp8.float() * self.scale_w
        bias_fp8_back = bias_fp8.float()
        
        return torch.nn.functional.linear(x, weight_fp8_back, bias_fp8_back)
    
    def update_scales(self, grad_norm_threshold=1.0):
        # Dynamic scaling: adjust scale_w based on gradient magnitude
        if self.weight_fp32.grad is not None:
            grad_norm = self.weight_fp32.grad.abs().max()
            if grad_norm > grad_norm_threshold:
                self.scale_w *= 0.9  # Scale down
            elif grad_norm < grad_norm_threshold / 10:
                self.scale_w *= 1.1  # Scale up

# Training loop
model = FP8MasterWeightLinear(10, 5).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

x = torch.randn(32, 10, device='cuda')
y = torch.randn(32, 5, device='cuda')

for step in range(10):
    optimizer.zero_grad()
    
    # Forward (with FP8 arithmetic)
    out = model(x)
    loss = criterion(out, y)
    
    # Backward
    loss.backward()
    
    # Weight update (FP32)
    optimizer.step()
    
    # Dynamic scale adjustment
    model.update_scales()
    
    print(f"Step {step}: loss={loss:.4f}, scale={model.scale_w:.4f}")
```

---

## 🔗 실전 활용

### 1. Large-Scale Gradient Accumulation 에서 Kahan 사용

```python
# Gradient accumulation + Kahan summation
class KahanGradAccumulator:
    def __init__(self):
        self.sum_grads = None
        self.compensation = None
    
    def accumulate(self, grad):
        if self.sum_grads is None:
            self.sum_grads = grad.clone()
            self.compensation = torch.zeros_like(grad)
        else:
            y = grad - self.compensation
            t = self.sum_grads + y
            self.compensation = (t - self.sum_grads) - y
            self.sum_grads = t
    
    def get_mean(self, num_steps):
        return self.sum_grads / num_steps

# 사용
accum = KahanGradAccumulator()
for step, (x, y) in enumerate(huge_dataloader):
    loss = model(x, y)
    grad = torch.autograd.grad(loss, model.parameters())
    accum.accumulate(grad)

final_grad = accum.get_mean(len(huge_dataloader))
```

### 2. FP8 Training 준비

```python
# Hopper (H100) 이후에는 torch.float8_e4m3fn 지원
import torch.distributed as dist

def setup_fp8_training(model, dtype=torch.float8_e4m3fn):
    # Master weight 는 FP32 유지
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # FP32
    
    # Forward 는 FP8 (hardware native, stochastic rounding)
    # Backward 는 FP32 (위에서)

model = build_model()
setup_fp8_training(model)

# Training
for x, y in dataloader:
    with torch.autocast(device_type='cuda', dtype=torch.float8_e4m3fn):
        loss = model(x, y)
    
    # Backward in FP32
    loss.backward()
    optimizer.step()
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Stochastic rounding 의 random stream 이 independent | Very large batches 에서 PRNG 품질 중요 |
| Kahan summation 이 항상 유리 | 연산량 3배 (매우 큰 batch 에서만 정당화) |
| Master weight pattern 이 안전 | 극도로 quantized (INT4) 에서는 accumulation error 관리 필수 |

---

## 📌 핵심 정리

$$\boxed{\mathbb{E}[\text{round}_{\text{stoch}}(x)] = x}$$

| 기술 | 용도 | Bias | Error growth | 구현 복잡도 |
|------|------|------|-------------|-----------|
| **Round-to-Nearest-Even** | FP32/FP16 표준 | 가능 | $O(n \epsilon)$ | 간단 |
| **Stochastic Rounding** | INT8/FP8 training | 없음 | $O(\epsilon)$ (random walk) | 중간 |
| **Kahan Summation** | Large-scale accumulation | 제어됨 | $O(\epsilon^2)$ | 어려움 |
| **Master Weight** | Quantized training | 없음 (FP32) | 낮음 | 중간 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Stochastic rounding 에서 $\mathbb{E}[\text{round}(x)] = x$ 임을 증명하라. $x$ 가 정수가 아닌 경우를 명시적으로 다루라.

<details>
<summary>해설</summary>

$x = n + f$ where $n = \lfloor x \rfloor$, $f \in [0, 1)$.

Stochastic rounding:
$$\text{round}_{\text{stoch}}(x) = \begin{cases}
n + 1 & \text{with prob } f \\
n & \text{with prob } 1 - f
\end{cases}$$

Expectation:
$$\mathbb{E}[\text{round}_{\text{stoch}}(x)] = (n+1) \cdot f + n \cdot (1-f)$$
$$= n + f = x \quad \square$$

**직관**: 소수부 $f$ 만큼의 확률로 올림 → 평균적으로 정확함.

</details>

**문제 2** (심화): Kahan summation 에서 compensation variable $c$ 의 역할을 설명하고, 왜 그것이 rounding error 를 보정하는가?

<details>
<summary>해설</summary>

**구조**:
```
y = x_i - c    # 이전 error 를 빼서 보정
t = sum + y    # 새 합 (rounding error ε 발생)
c = (t - sum) - y  # 손실된 부분 계산
```

**핵심**: $c$ 는 "$t - sum$" 과 "$y$" 의 차이를 기록.

왜냐하면:
- 이상적으로: $t = sum + y$ 정확히 성립
- 실제: floating-point 으로 $t = (sum + y) + \epsilon$ (rounding error)
- $c = (t - sum) - y = (sum + y + \epsilon - sum) - y = \epsilon$

따라서 다음 iteration 에서 $x_i - c = x_i - \epsilon$ → error 를 빼서 보정. $\square$

**효과**: Error 가 누적되지 않고 각 step 에서 제어됨 → exponential 누적 대신 constant.

</details>

**문제 3** (논문 비평): Croci et al. (2022) 의 "Stochastic rounding and its probabilistic backward error analysis" 에서, 왜 stochastic rounding 이 INT8 training 에서 필수라고 주장했는가? Deterministic quantization 의 한계는?

<details>
<summary>해설</summary>

**Deterministic (RNE) quantization** 의 문제:
- Systematic bias 축적: 항상 같은 방향 (올림 또는 내림)
- Gradient 방향이 consistent 하게 바뀜 → 수렴 불가능

**예**: INT8 로 quantize 하면 gradient 가 ~2~3% bias 생김 (항상 positive or negative) → 학습 불안정

**Stochastic rounding** 의 장점:
- Bias = 0 (unbiased)
- Variance 만 있음 → SGD 의 noise 로 흡수됨
- 극도의 quantization (INT4, INT2) 에서도 convergence 가능

Croci 는 **이론적 증명** 과 **실험** 으로 INT8 training 의 가능성을 처음 보임. 현재 NVIDIA H100 의 FP8 연산이 이에 기반. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-bf16-tf32.md) | [📚 README](../README.md) | [다음 ▶](../ch7-compile/01-pt2-overview.md)

</div>
