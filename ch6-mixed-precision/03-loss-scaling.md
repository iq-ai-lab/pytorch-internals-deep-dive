# 03. Loss Scaling 과 GradScaler

## 🎯 핵심 질문

- Loss scaling $L_{\text{scaled}} = L \cdot S$ 가 정확히 어떻게 gradient underflow 를 우회하는가?
- Backward pass 에서 chain rule 이 모든 gradient 를 $S$ 배로 scale 하는 메커니즘은?
- Optimizer step 직전 unscale $g \leftarrow g / S$ 가 왜 정확한 학습을 보장하는가?
- `GradScaler.scale() / unscale_() / step() / update()` 의 각 단계가 하는 일은?
- Dynamic scaling algorithm 이 NaN/Inf 를 감지해서 $S$ 를 조정하는 방식은?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

`torch.cuda.amp.GradScaler` 는 mixed precision training 의 **핵심 기술**이지만, 많은 사용자는 단지 "loss scaling 에 쓰는 거" 정도로 이해합니다. 정확히 알아야 할 것:

1. **Loss scaling 의 수학적 정당성** — gradient 를 $S$ 배 하면 underflow 를 피하지만, loss 자체는 왜 영향을 받지 않는가?
2. **Unscale 의 필요성** — weight 업데이트 전에 반드시 $g / S$ 를 하는 이유
3. **Dynamic adjustment** — NaN/Inf 가 나타나면 $S$ 를 어떻게 조정하고, clean step 이 몇 개여야 $S$ 를 올리는가?
4. **다른 optimizer 와의 상호작용** — momentum, weight decay 를 가진 optimizer 에서도 unscaling 이 안전한가?

이 문서는 loss scaling 의 수학적 증명부터 GradScaler 의 실제 구현, PyTorch native `torch.cuda.amp` 와 NVIDIA Apex 의 차이까지 다룹니다.

---

## 📐 수학적 선행 조건

- Chain rule 과 gradient 계산 (Ch2)
- FP16 의 range problem (Ch6-02)
- Optimizer 기초 (SGD, Adam — gradient update rule)

---

## 📖 직관적 이해

### Loss Scaling 의 아이디어

```
Without scaling:
  Forward:  y = f(x)      [FP16]
  Loss:     L = ℓ(y)      [FP32, scalar]
  Backward: dL/dx = ...   [FP16 calculations]
                ↑ many gradients < 6.1e-5 → 0 (underflow!)

With scaling:
  Forward:  y = f(x)          [FP16]
  Scaled:   L_s = L × S       [FP32 scalar, S ≈ 2^16]
  Backward: dL_s/dx = S × dL/dx  [chain rule → all grads ×S]
                ↑ S × gradient >> 6.1e-5 (safe!)
  
  Update:   W ← W - α × (dL_s/dx) / S = W - α × dL/dx  [correct!]
```

**핵심**: Loss scaling 은 backward 의 **chain rule** 을 이용해 모든 gradient 를 한 번에 scale.

### Unscale 의 필요성

```
After backward with scaling:
  g_scaled = dL_s/dx = S × dL/dx  [large enough for FP16]

Before optimizer step:
  g_unscaled = g_scaled / S = dL/dx  [correct gradient]

Weight update:
  W_new = W_old - α × g_unscaled = W_old - α × dL/dx  ✓
```

만약 unscale 을 빼먹으면:
```
W_new = W_old - α × S × dL/dx  [S배로 과도한 업데이트!]
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Loss Scaling

Scalar loss $L \in \mathbb{R}$ 에 대해, scaling factor $S > 0$ 를 사용한 scaled loss:
$$L_{\text{scaled}} = L \times S$$

Backward pass 에서 chain rule:
$$\frac{\partial L_{\text{scaled}}}{\partial W} = S \times \frac{\partial L}{\partial W}$$

### 정의 3.2 — Unscaling

Optimizer 적용 직전:
$$g_{\text{final}} = \frac{g_{\text{scaled}}}{S} = \frac{\partial L_{\text{scaled}}}{S \partial W} = \frac{\partial L}{\partial W}$$

### 정의 3.3 — Dynamic Scaling Algorithm

초기 scaling factor $S_0$ (보통 $2^{16} = 65536$) 에서 시작:

1. **NaN/Inf Detection**: $g_{\text{scaled}}$ 에서 NaN/Inf 감지 시
   $$S \leftarrow S / 2$$
   그 step skip (optimizer 적용 안 함), accumulation 초기화

2. **Growth Condition**: 연속 $n_{\text{growth}}$ steps (보통 2000) 동안 NaN/Inf 없으면
   $$S \leftarrow \min(S \times 2, S_{\max})$$

3. **Convergence**: $S_{\max} = 2^{24}$ (더 이상 올리지 않음)

---

## 🔬 정리와 증명

### 정리 3.1 — Loss Scaling 의 Correctness

Loss scaling $L_s = L \times S$ 를 사용한 backward 의 gradient 는:
$$g_s = \frac{\partial L_s}{\partial W} = S \times \frac{\partial L}{\partial W} = S \times g$$

Unscale 후 optimizer 업데이트:
$$W_{\text{new}} = W_{\text{old}} - \alpha \times \frac{g_s}{S} = W_{\text{old}} - \alpha \times g$$

따라서 **scaling 이 없는 경우와 정확히 동일한 weight 업데이트를 보장** $\square$

**증명**: Chain rule 의 선형성. $\frac{\partial (c \times L)}{\partial W} = c \times \frac{\partial L}{\partial W}$ 이므로 scaling 은 모든 gradient 에 같은 상수를 곱함. Optimizer 에서 unscale 하면 최종 업데이트는 동일.

### 정리 3.2 — FP16 Representability 보장

Scaling factor $S$ 를 충분히 크게 설정하면, FP16 underflow 를 피할 수 있다:
$$S \geq \frac{x_{\min}}{\bar{|g|}},$$

여기서 $x_{\min} = 6.1 \times 10^{-5}$ (FP16 min normal), $\bar{|g|}$ 는 typical gradient magnitude.

**예**: $\bar{|g|} \approx 10^{-5}$ 이면 $S \geq 6.1$ 이면 충분. 보통 $S = 2^{16}$ 은 과도하지만 안전.

**정성적 증명**: Scale up 하면 모든 gradient 가 더 크다 → FP16 representable range 에 들어감. $\square$

### 따름 정리 3.3 — Momentum 과의 호환성

Momentum optimizer (e.g., SGD with momentum):
$$v_t = \beta v_{t-1} + (1-\beta) g_t$$
$$W_t = W_{t-1} - \alpha v_t$$

Loss scaling 을 적용하면:
$$v_t = \beta v_{t-1} + (1-\beta) \times \frac{g_s}{S} = \beta v_{t-1} + (1-\beta) g$$

즉, **momentum 상태는 unscaled gradient 로만 누적** (올바름) $\square$

### 따름 정리 3.4 — Weight Decay 와의 호환성

AdamW style ($L_2$ regularization on weight):
$$g_{\text{wd}} = g + \lambda W$$

Loss scaling 시:
- Forward: $L_s = L \times S$ (scaled)
- Backward: $\frac{\partial L_s}{\partial W} = S \times \frac{\partial L}{\partial W}$ (scaled gradient)
- **Weight decay 는 unscaled gradient 에 추가**: $g_{\text{final}} = \frac{g_s}{S} + \lambda W$ (올바름)

PyTorch 의 native optimizer 가 이를 자동 처리 $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — GradScaler 의 기본 사용

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

model = nn.Linear(10, 1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = GradScaler()

x = torch.randn(32, 10, device='cuda')
y = torch.ones(32, 1, device='cuda')
criterion = nn.MSELoss()

for step in range(5):
    optimizer.zero_grad()
    
    # Forward + loss
    with autocast(dtype=torch.float16):
        pred = model(x)
        loss = criterion(pred, y)
    
    # Backward with scaling
    scaler.scale(loss).backward()
    
    # Unscale 후 optimizer step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    
    # Dynamic update
    scaler.update()
    
    print(f"Step {step}: loss={loss:.4f}, scale={scaler.get_scale():.0f}")
```

### 실험 2 — Scale Factor 의 영향

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).cuda()

x = torch.randn(256, 256, device='cuda')
y = torch.ones(256, 1, device='cuda')
criterion = nn.MSELoss()

# 여러 S 값 비교
for S in [1.0, 256.0, 65536.0]:
    model.zero_grad()
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        pred = model(x)
        loss = criterion(pred, y)
    
    # 수동 scaling
    (loss * S).backward()
    
    # Gradient 통계
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.extend(param.grad.flatten().detach().abs())
    grads = torch.tensor(grads)
    
    print(f"S={S:8.0f}: mean_grad={grads.mean():.2e}, "
          f"min={grads.min():.2e}, max={grads.max():.2e}")
```

### 실험 3 — Unscaling 의 필요성 검증

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(32, 10)
y = torch.ones(32, 1)
criterion = nn.MSELoss()

S = 1000.0

# Case 1: Correct (with unscaling)
model.zero_grad()
loss = criterion(model(x), y)
(loss * S).backward()
g_scaled = model.weight.grad.clone()
g_unscaled = g_scaled / S

print(f"With unscaling:")
print(f"  scaled grad:   {g_scaled[0, 0]:.6f}")
print(f"  unscaled grad: {g_unscaled[0, 0]:.6f}")

# Case 2: Wrong (without unscaling)
model.zero_grad()
loss = criterion(model(x), y)
loss.backward()
g_correct = model.weight.grad.clone()

print(f"\nCorrect gradient (no scaling):  {g_correct[0, 0]:.6f}")
print(f"Match after unscaling: {torch.allclose(g_unscaled, g_correct, atol=1e-5)}")
```

### 실험 4 — Dynamic Scaling 의 NaN/Inf 처리

```python
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

class OverflowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    
    def forward(self, x):
        # Intentional overflow: huge numbers
        return self.layer(x) * 1e10

model = OverflowNet().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = GradScaler(init_scale=1024.0)

x = torch.randn(32, 10, device='cuda')
y = torch.ones(32, 1, device='cuda')
criterion = nn.MSELoss()

print("Dynamic scaling with NaN/Inf handling:")
for step in range(10):
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        pred = model(x)
        loss = criterion(pred, y)
    
    scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    
    # NaN/Inf 감지
    has_nan = any(
        torch.isnan(p.grad).any() if p.grad is not None else False
        for p in model.parameters()
    )
    has_inf = any(
        torch.isinf(p.grad).any() if p.grad is not None else False
        for p in model.parameters()
    )
    
    if has_nan or has_inf:
        print(f"  Step {step}: NaN/Inf detected, skip step")
        scaler.step(optimizer)  # Skip 하지만 scale 조정
    else:
        print(f"  Step {step}: OK, scale={scaler.get_scale():.0f}")
        scaler.step(optimizer)
    
    scaler.update()
```

### 실험 5 — FP32 Master Weight Pattern

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# FP32 master weight
model = nn.Linear(10, 1)
for param in model.parameters():
    param.data = param.data.float()  # FP32

model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = GradScaler()

x = torch.randn(32, 10, device='cuda')
y = torch.ones(32, 1, device='cuda')
criterion = nn.MSELoss()

for step in range(5):
    optimizer.zero_grad()
    
    # Forward in FP16
    with autocast(dtype=torch.float16):
        pred = model(x)
        loss = criterion(pred, y)
    
    # Backward → gradient 계산 (intermediate FP16)
    scaler.scale(loss).backward()
    
    # Unscale → FP32 master weight 에 적용
    scaler.unscale_(optimizer)
    
    # Update (FP32)
    scaler.step(optimizer)
    scaler.update()
    
    # Weight 는 여전히 FP32
    print(f"Step {step}: weight dtype={model.weight.dtype}, "
          f"grad dtype={model.weight.grad.dtype if model.weight.grad is not None else 'None'}")
```

---

## 🔗 실전 활용

### 1. Mixed Precision Training 의 전형적인 패턴

```python
model = build_model().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()

for epoch in range(max_epochs):
    for batch in train_loader:
        x, y = batch
        
        optimizer.zero_grad()
        
        # 1. Forward in FP16
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)
        
        # 2. Backward with scaling
        scaler.scale(loss).backward()
        
        # 3. Unscale + gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 4. Optimizer step
        scaler.step(optimizer)
        
        # 5. Dynamic scaling update
        scaler.update()
```

### 2. Gradient Accumulation 과의 결합

```python
accumulation_steps = 4

for step, (x, y) in enumerate(train_loader):
    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = criterion(logits, y) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 3. Custom Training Loop 에서 주의점

```python
# WRONG: unscale 없이 gradient 접근
scaler.scale(loss).backward()
print(model.weight.grad)  # scaled! (S배 커짐)

# CORRECT: unscale 후 접근
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
print(model.weight.grad)  # unscaled
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| FP16 underflow 가 주요 문제 | Overflow 는 loss scaling 으로 동시에 처리 (NaN 감지) |
| Fixed batch size | Dynamic batch size 시 scaling 전략 재검토 필요 |
| Optimizer 가 gradient-based | 다른 optimization 방법 (evolution, population) 은 scaling 불필요 |
| PyTorch native amp 사용 | NVIDIA Apex (구식) 와 약간의 수동 코드 차이 있음 |

---

## 📌 핵심 정리

$$\boxed{L_{\text{scaled}} = L \times S, \quad g_{\text{final}} = \frac{g_{\text{scaled}}}{S} = \frac{\partial L}{\partial W}}$$

**GradScaler 의 단계**:
1. `scaler.scale(loss)` — $L_s = L \times S$ 계산
2. `.backward()` — chain rule: $\frac{\partial L_s}{\partial W} = S \times \frac{\partial L}{\partial W}$
3. `scaler.unscale_(optimizer)` — $g \leftarrow g / S$
4. `scaler.step(optimizer)` — weight 업데이트
5. `scaler.update()` — dynamic scaling 조정

---

## 🤔 생각해볼 문제

**문제 1** (기초): Loss scaling $L_s = L \times S$ 에서, gradient $\frac{\partial L_s}{\partial W}$ 가 정확히 왜 $S$ 배가 되는가? Chain rule 을 명시적으로 쓰라.

<details>
<summary>해설</summary>

Chain rule:
$$\frac{\partial L_s}{\partial W} = \frac{\partial (L \times S)}{\partial W} = S \times \frac{\partial L}{\partial W}$$

$S$ 는 상수이므로 (gradient 계산과 무관), 선형성에 의해 모든 gradient 가 $S$ 배.

Backward computation graph:
```
L_s = L * S
↓
dL_s/dW = S * dL/dW  [scale factor propagates]
```

$\square$

</details>

**문제 2** (심화): Unscale 후에도 weight update 가 정확한 이유를 momentum optimizer 의 경우로 증명하라.

<details>
<summary>해설</summary>

**Momentum SGD**:
$$v_t = \beta v_{t-1} + (1-\beta) g_t$$
$$W_t = W_{t-1} - \alpha v_t$$

Loss scaling:
- Backward: $g_s = S \times g$
- Unscale: $\hat{g} = g_s / S = g$

그 후 optimizer:
$$v_t = \beta v_{t-1} + (1-\beta) \hat{g} = \beta v_{t-1} + (1-\beta) g$$

즉, **momentum 상태는 unscaled gradient 로 누적** (scaling 의 영향 없음). Final update:
$$W_t = W_{t-1} - \alpha v_t = W_{t-1} - \alpha [\beta v_{t-1} + (1-\beta) g]$$

이것은 **scaling 이 없는 경우와 정확히 동일**. $\square$

</details>

**문제 3** (논문 비평): Micikevicius et al. (2018) 에서 제안한 dynamic scaling algorithm 이 왜 $S \leftarrow S/2$ (underflow 시) 와 $S \leftarrow 2S$ (2000 step clean 시) 를 선택했을까? 다른 값은 왜 안 되는가?

<details>
<summary>해설</summary>

**$S \leftarrow S/2$ (underflow/overflow 시)**:
- Exponent 공간에서 logarithmic adjustment
- 한 번에 반으로 줄임 → overcompensation 피함
- Too aggressive: $S \leftarrow 0.1 \times S$ 라면 너무 빠르게 내려감
- Too conservative: $S \leftarrow 0.9 \times S$ 라면 수렴 느림

**$S \leftarrow 2S$ (growth, 2000 steps)**:
- Factor of 2 는 binary scaling 과 대칭
- 2000 steps 는 batch size 256, learning rate epoch 기준으로 typical convergence time
- NVIDIA 의 empirical tuning (논문에서는 정확한 tuning 과정 미공개)

**최적성**: Binary exponent ($2^k$ 형태) 선택이 computational 편의와 수렴 간의 balance 제공. $\square$

Modern 구현 (PyTorch 2.0+) 은 `growth_interval`, `backoff_factor` 등을 tunable parameter 로 노출하기도 함.

</details>

---

<div align="center">

[◀ 이전](./02-fp16-range-problem.md) | [📚 README](../README.md) | [다음 ▶](./04-bf16-tf32.md)

</div>
