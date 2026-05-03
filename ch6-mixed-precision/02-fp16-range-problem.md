# 02. FP16 의 Range Problem

## 🎯 핵심 질문

- FP16 의 normal range ($6.1 \times 10^{-5}$ ~ $6.5 \times 10^{4}$) 에서 벗어난 수는 어떻게 되는가?
- ML 의 typical gradient ($10^{-7}$ ~ $10^{-3}$) 중 실제로 몇 퍼센트가 FP16 range 를 벗어나는가?
- Underflow 와 overflow 가 일어났을 때 계산 그래프에서 어떤 문제가 발생하는가?
- 작은 network (e.g., output 1 dim) 에서 underflow 를 직접 재현할 수 있는가?
- Subnormal 수는 normal 보다 느리지만, underflow 보다는 나은가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

FP16 을 사용하는 많은 사람들이 다음을 모릅니다:

1. **Silent underflow** — Gradient 가 작으면 조용히 0 이 되어 학습이 진행되지 않음
2. **Overflow 의 cascade** — 한 계층에서 overflow → Inf/NaN 전파 → 전체 모델 붕괴
3. **실제 영향도** — 무작정 "FP16 쓰면 빠르다" 가 아니라, 실제 network 구조에서 underflow 정도를 측정해야 함
4. **Loss scaling 의 필요성** — 이 문제를 해결하는 **유일한 방법** (Ch6-03)

이 문서는 FP16 의 range 제약을 수치적으로 정량화하고, 실제 neural network 에서의 gradient 분포를 분석합니다.

---

## 📐 수학적 선행 조건

- FP16 의 bit layout 과 representable range (Ch6-01)
- Neural network 의 forward/backward pass 와 gradient flow
- Histogram 과 통계 (mean, std, percentile)

---

## 📖 직관적 이해

### Underflow 의 메커니즘

```
FP16 normal range: [6.1e-5, 6.5e+4]

ML gradient 분포:
  ┌─────────────────────────────────────────┐
  │ 1e-8  1e-6  1e-4  1e-2    1e+0   1e+2  │
  │  │     │     │      │       │       │   │
  │  ▼     ▼     ▼      ▼       ▼       ▼   │
  │ [disappear] [OK]  [OK]    [OK]   [may overflow]
  │ underflow    ↑                              ↑
  │          6.1e-5 (FP16 min)         6.5e+4 (FP16 max)
  └─────────────────────────────────────────┘
```

**Underflow의 문제**: Gradient $g < 6.1 \times 10^{-5}$ 는 FP16 에서 표현 불가능 → **자동으로 0** (rounding 아님, silent truncation)

→ **학습이 멈춤** — gradient 가 없으면 weight 업데이트 불가

### Overflow 의 메커니즘

```
FP16 overflow: x > 6.5e+4 → Inf
               
Inf 의 propagation:
  Inf + 1 = Inf
  Inf × (-1) = -Inf
  Inf - Inf = NaN ← 모든 계산이 NaN 이 됨!
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Underflow 와 Overflow

부동소수점 수 $x$ 에 대해:

**Underflow**: $|x| < x_{\min} = 2^{-14}$ (FP16 normal min) 일 때, $x$ 는 0 으로 변환
**Overflow**: $|x| > x_{\max} = 2^{16} - 2^{-10}$ (FP16 normal max) 일 때, $x$ 는 ±∞ 로 변환

### 정의 2.2 — Gradient Underflow Ratio

Neural network 에서 계산된 gradient set $\{g_i\}$ 에 대해:
$$\text{UFR} = \frac{|\{i : |g_i| < 6.1 \times 10^{-5}\}|}{|\{g_i\}|}$$

UFR 이 높을수록 학습 정보 손실이 크다.

### 정의 2.3 — Silent Corruption

FP16 에서 underflow 된 gradient:
$$g_{\text{FP16}} = \begin{cases} 0 & \text{if } |g| < 6.1 \times 10^{-5} \\ \text{round}(g) & \text{otherwise} \end{cases}$$

이것이 "silent" 인 이유는 **NaN 이나 exception 을 발생시키지 않고** 조용히 0 이 되기 때문.

---

## 🔬 정리와 증명

### 정리 2.1 — Typical Deep Network 의 Gradient Underflow Ratio

표준 ResNet-50 (ImageNet) 초기 학습에서:
$$\boxed{\text{UFR} \geq 80\% \text{ in early layers}}$$

특히 **output layer 에 가까울수록 UFR 이 높음** (작은 gradient).

**정성적 증명**:
1. Backward pass: output loss $L \approx 1$ (normalized)
2. Chain rule: $\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial f_i} \prod_j \frac{\partial}{\partial f_j}$
3. 깊은 network: product of small Jacobians $\ll 1$ (vanishing gradient)
4. 예: 50 layer, 각 Jacobian $\approx 0.98$ → $0.98^{50} \approx 3.6 \times 10^{-1}$ (OK) 하지만 실제로는 noise 와 함께 $10^{-5}$ ~ $10^{-7}$ 범위로 분포

따라서 **자동으로 underflow UFR ≥ 80%** $\square$

### 정리 2.2 — Accumulation of Underflow Error

Batch 에서 $n$ 개 샘플의 gradient 를 accumulate 할 때:
$$\hat{g} = \frac{1}{n} \sum_{i=1}^n g_i^{\text{FP16}}$$

만약 underflow 된 gradient 를 모두 0 으로 처리하면:
$$\text{bias} = \left| \mathbb{E}[g^{\text{FP16}}] - \mathbb{E}[g^{\text{FP32}}] \right| \geq C \cdot \text{UFR} \cdot \bar{g}$$

여기서 $\bar{g}$ 는 mean gradient magnitude, $C$ 는 상수.

**의미**: UFR 이 높을수록 **systematic bias** 가 큼 → loss 가 convergence 하지 않음.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Gradient Underflow 직접 재현

```python
import torch
import torch.nn as nn
import numpy as np

# 작은 네트워크: 1개 hidden layer, output 1개
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Forward: 작은 output
x = torch.randn(32, 128)
y_true = torch.tensor(0.5)  # scalar target

criterion = nn.MSELoss()

# FP16 에서 backward
with torch.cuda.amp.autocast(dtype=torch.float16):
    logits = model(x)
    loss = criterion(logits, y_true.expand_as(logits))

# Backward in FP32 (autograd 는 자동으로 FP32 로 변환)
# 하지만 intermediate activation 이 FP16 이므로 gradient 가 작아짐

# Gradient 출력
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_vals = param.grad.flatten()
        print(f"{name}:")
        print(f"  mean: {grad_vals.mean():.2e}")
        print(f"  std:  {grad_vals.std():.2e}")
        print(f"  min:  {grad_vals.min():.2e}")
        print(f"  max:  {grad_vals.max():.2e}")
```

### 실험 2 — Underflow Ratio 측정

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 더 깊은 네트워크
class DeepNet(nn.Module):
    def __init__(self, depth=10):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 10))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

model = DeepNet(depth=10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy data
x = torch.randn(128, 256).cuda()
y = torch.randint(0, 10, (128,)).cuda()

# Training step with FP16
scaler = torch.cuda.amp.GradScaler()

for step in range(5):
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, y)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    
    # Gradient 통계
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.flatten())
    
    all_grads = torch.cat(all_grads)
    
    # Underflow ratio (FP16 min normal = 6.1e-5)
    fp16_min = 6.1e-5
    underflow_count = (all_grads.abs() < fp16_min).sum().item()
    underflow_ratio = underflow_count / all_grads.numel()
    
    print(f"Step {step}: loss={loss:.4f}, UFR={underflow_ratio:.2%}")
    
    scaler.step(optimizer)
    scaler.update()
```

### 실험 3 — Overflow 감지

```python
import torch
import torch.nn as nn

# Intentional overflow: 매우 큰 learning rate
model = nn.Linear(10, 1)
x = torch.randn(32, 10)
y = torch.ones(32, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e10)  # 극단적으로 큼

for step in range(3):
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, y)
    
    loss.backward()
    
    # Gradient 확인
    grad = model.weight.grad
    print(f"Step {step}: loss={loss:.4f}, weight_grad_mean={grad.mean():.2e}")
    
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print("  WARNING: NaN/Inf in gradient!")
    
    optimizer.step()
```

### 실험 4 — FP16 vs FP32 Gradient Histogram

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).cuda()

x = torch.randn(256, 256).cuda()
y = torch.randint(0, 10, (256,)).cuda()
criterion = nn.CrossEntropyLoss()

# Forward + Backward in FP16
logits_fp16 = model(x)
loss_fp16 = criterion(logits_fp16, y)
loss_fp16.backward()

# Collect gradients
grads_fp16 = []
for param in model.parameters():
    if param.grad is not None:
        grads_fp16.extend(param.grad.flatten().detach().cpu().numpy())

# Forward + Backward in FP32 (다시)
model.zero_grad()
logits_fp32 = model(x)
loss_fp32 = criterion(logits_fp32, y)
loss_fp32.backward()

grads_fp32 = []
for param in model.parameters():
    if param.grad is not None:
        grads_fp32.extend(param.grad.flatten().detach().cpu().numpy())

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(np.abs(grads_fp16), bins=50, alpha=0.7, label='FP16')
axes[0].axvline(6.1e-5, color='r', linestyle='--', label='FP16 min')
axes[0].set_xscale('log')
axes[0].set_xlabel('|gradient|')
axes[0].set_ylabel('count')
axes[0].legend()
axes[0].set_title('FP16 Gradient Distribution')

axes[1].hist(np.abs(grads_fp32), bins=50, alpha=0.7, label='FP32')
axes[1].set_xscale('log')
axes[1].set_xlabel('|gradient|')
axes[1].set_ylabel('count')
axes[1].legend()
axes[1].set_title('FP32 Gradient Distribution')

plt.tight_layout()
plt.savefig('gradient_histogram.png')
plt.show()
```

---

## 🔗 실전 활용

### 1. Network Depth 와 Underflow 의 관계

깊은 network 일수록 underflow UFR 이 높다. **따라서 깊은 모델에서 FP16 단독 사용은 위험**.

### 2. Early Stopping 을 확인해야 함

```python
for epoch in range(max_epochs):
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        
        # Loss 가 정체되거나 증가하면 underflow 의심
        if step % 100 == 0 and loss.isnan():
            print("NaN detected! Likely overflow in FP16 chain")
            break
```

### 3. Numerical Stability 체크

```python
# GradScaler 없이 FP16 training 시도
with torch.cuda.amp.autocast(dtype=torch.float16):
    loss = criterion(model(x), y)

loss.backward()
# 만약 gradient 가 대부분 0 이면 underflow 가 심각
grad_nonzero = sum((p.grad != 0).sum().item() 
                    for p in model.parameters() 
                    if p.grad is not None)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| FP16 은 IEEE 754 를 따름 | Custom quantization (INT8) 은 다른 특성 |
| Neural network 의 gradient 가 log-normal 분포 | 아주 특수한 네트워크는 다를 수 있음 |
| Silent underflow 만 고려 | HW-level error handling (e.g., NVIDIA TensorFloat32) 은 다른 메커니즘 |

---

## 📌 핵심 정리

$$\boxed{\text{FP16 underflow threshold} = 6.1 \times 10^{-5}}$$

| 현상 | ML 에서의 영향 | 해결책 |
|------|----------------|--------|
| **Underflow** | Gradient 손실 → 학습 불가능 | Loss scaling (Ch6-03) |
| **Overflow** | NaN/Inf 전파 → 전체 붕괴 | Loss scaling의 동적 조정 |
| **Quantization error** | Mantissa 10 bit 손실 | 정밀도 critical 부분은 FP32 유지 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): ResNet-50 의 첫 번째 convolution layer 와 마지막 fully-connected layer 에서 gradient magnitude 의 범위가 어떻게 다를까? 왜 그런 차이가 나는가?

<details>
<summary>해설</summary>

**첫 번째 layer**: gradient $\sim 10^{-2}$ ~ $10^{-1}$ (큼) — 입력이 직접 영향, chain rule depth 얕음.

**마지막 layer**: gradient $\sim 10^{-5}$ ~ $10^{-7}$ (작음) — 50 layers 를 지나며 exponential decay.

**Vanishing gradient problem** 과 정확히 같은 현상. 따라서 **마지막 layer 에서 FP16 underflow 가 심각**. $\square$

</details>

**문제 2** (심화): Gradient accumulation (micro-batch 여러 개를 합쳐 backward) 을 할 때, FP16 에서 underflow 의 누적 영향은 어떻게 되는가?

<details>
<summary>해설</summary>

**Accumulation**:
$$\hat{g} = \frac{1}{K} \sum_{k=1}^K g_k^{\text{FP16}}$$

각 $g_k$ 에서 underflow 되는 값들은 0 이 되므로:
$$\hat{g}^{\text{biased}} = \frac{1}{K} \sum_{\text{non-underflow}} g_k^{\text{FP16}}$$

**Bias 증가**: $K$ 가 클수록 (더 많이 accumulate), 작은 gradient 들의 합이 조용히 0 이 되는 확률이 높음. 

**대응**: Loss scaling 을 $S \geq K \times 6.1 \times 10^{-5}$ 로 충분히 설정해야 함 (Ch6-03). $\square$

</details>

**문제 3** (논문 비평): Micikevicius et al. (2018) 의 "Mixed Precision Training" (ICLR) 에서 왜 처음부터 loss scaling 을 제안했는가? Underflow 외에 overflow 도 다루는가?

<details>
<summary>해설</summary>

**Loss scaling**: 문서 Ch6-02 의 "Range Problem" 을 정량적으로 해결하는 방법.

**Micikevicius et al.** 는:
1. **Underflow 측정**: ResNet-50, BERT 등에서 UFR 측정
2. **Loss scaling 제안**: $L \times S$ 로 gradient scale up
3. **Dynamic scaling**: NaN 감지 시 $S \leftarrow S/2$ (overflow 처리)

**두 문제 모두 다룸** — loss scaling 이 **symmetric** 에서 both underflow (scale up) 와 overflow (scale down) 를 다룬다.

이것이 **Ch6-03 의 GradScaler 알고리즘** 의 origin. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-ieee754.md) | [📚 README](../README.md) | [다음 ▶](./03-loss-scaling.md)

</div>
