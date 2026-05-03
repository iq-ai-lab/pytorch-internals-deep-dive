# 01. torch.compile — PT 2.0 Revolution

## 🎯 핵심 질문

- `@torch.compile` decorator 또는 `torch.compile(model)` 호출이 eager mode 와 어떻게 다른가?
- **First call** 의 수십 초 compile 과정 (Dynamo → AOTAutograd → Inductor → Triton) 의 대략적 파이프라인은?
- **Cached call** 이 eager 보다 빠른 이유는? Graph capture 후 매번 같은 kernel 을 재사용한다는 의미인가?
- `torch._dynamo.config.verbose=True` 로 무엇을 확인할 수 있는가?
- 왜 PT 2.0 이 **"기존 Python 호출을 유지하면서" 최적화** 를 달성한 것인가?

---

## 🔍 왜 이 설계가 PyTorch 사용자에게 중요한가

PyTorch 1.x 사용자는 다음과 같이 코드를 작성했습니다:

```python
model = MyModel().cuda()
for x, y in dataloader:
    loss = model(x)
    loss.backward()
    optimizer.step()
```

이 코드는 **매 forward pass 마다** Python 해석기가:
1. Neural network operation 들의 순서를 bytecode 로 읽음
2. 각 op 호출 시마다 dispatcher 를 거침
3. CUDA kernel launch 의 오버헤드 누적

PT 2.0 의 `torch.compile` 은 **같은 Python 코드를 유지하면서** 첫 호출 때:
1. Python bytecode 를 **분석 및 graph 화** (TorchDynamo)
2. Forward + backward 를 함께 최적화 (AOTAutograd)
3. **Single fused Triton kernel** 생성 (TorchInductor)

결과: **캐시된 호출은 eager 모드보다 20~50% 빠름** — 추가 코드 변경 없이.

---

## 📐 수학적 선행 조건

- **PyTorch Autograd Deep Dive** (Ch2): VJP, backward graph, computational graph
- **Dispatcher Deep Dive** (Ch3): DispatchKeySet, operation routing
- **CUDA 기초** (Ch4): GPU 메모리 계층, SRAM/HBM
- **Custom Kernel** (Ch5): Triton 기초 문법, block-level abstraction
- 함수형 프로그래밍: graph rewriting, node elimination (fusion)

---

## 📖 직관적 이해

### Eager Mode vs Compiled Mode

```
┌─────────────────────────────────────────────────────────────┐
│                      EAGER MODE (PT 1.x)                    │
├─────────────────────────────────────────────────────────────┤
│  Python      Forward                    Backward             │
│   Code    ┌────────────────────────┐ ┌─────────────────────┐│
│  for ...  │ x = self.fc1(x)        │ │ dx = grad_out @ w1T ││
│           │ x = relu(x)            │ │ dw1 = x.T @ dx      ││
│           │ x = self.fc2(x)        │ │ x = relu_bwd(...)   ││
│           │ loss = mse(x, y)       │ │ dw2 = ... (more ops)││
│           └────────────────────────┘ └─────────────────────┘│
│           ↓ Python interpreter ↓      ↓ autodiff ↓          │
│       Dispatch overhead: ~10-20% of total time              │
│       Kernel launches X 10+, synchronization points X 10+   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   COMPILED MODE (PT 2.0)                    │
├─────────────────────────────────────────────────────────────┤
│  Same Python Code          ↓ (First call)                   │
│                                                              │
│  Dynamo: Capture FX graph                                   │
│    [fc1 node] → [relu node] → [fc2 node] → [mse node]      │
│          ↓                                                   │
│  AOTAutograd: Generate backward graph                       │
│    [mse_bwd] → [fc2_bwd] → [relu_bwd] → [fc1_bwd]          │
│          ↓                                                   │
│  Inductor: Fuse & code-gen                                  │
│    Single kernel: fc1(x) + relu + fc2 + mse + backward      │
│          ↓ (First call time: 10~50 sec)                    │
│  Triton JIT compile to GPU binary                           │
│                                                              │
│  Cached calls: ~1 launch instead of 10+, no interpreter    │
│  Speedup: 20-50% on typical models                          │
└─────────────────────────────────────────────────────────────┘
```

### Why "No Code Change"?

`torch.compile` 의 설계 철학:

```
Original:  model(x) → eager dispatch → 10 kernels → loss
Compiled:  model(x) → graph cache → 1 fused kernel → loss
           (Python interface identical, internal execution different)
```

User-facing code remains:
```python
model = torch.compile(model)  # or @torch.compile decorator
loss = model(x)               # Same as before
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — torch.compile 의 실행 모델

주어진 neural network $f_\theta: \mathbb{R}^{D_{\text{in}}} \to \mathbb{R}$ 에 대해:

$$
\text{compiled\_f}_\theta = \mathcal{C}(f_\theta)
$$

여기서 $\mathcal{C}$ 는 following three-stage compiler:

1. **Graph capture** (Dynamo): Python bytecode $\to$ FX graph $G = (V, E)$
2. **AOT autograd**: Forward graph $G_{\text{fwd}} + $ Backward graph $G_{\text{bwd}}$
3. **Code generation** (Inductor): FX graph $\to$ Triton kernel source code

### 정의 1.2 — First Call vs Cached Calls

- **$t_1$ (first call)**: Compilation time. Parse + graph capture + codegen + JIT = 10~50 초
- **$t_c$ (cached call)**: 1 kernel launch ≈ 1~10 ms (이전 eager: 10~20 ms with multiple launches)
- **Speedup**: 우리는 충분히 큰 batch 에서 $t_c \ll t_{\text{eager}}$ 달성

### 정의 1.3 — Graph Break

TorchDynamo bytecode 분석 중 control flow 또는 Python side effect 를 만나면:

$$
\text{Break point} = \{ i : \text{op}_i \notin \text{traceable CUDA ops} \}
$$

각 break 에서 graph $G$ 가 segment 로 나뉨:
$$
G = G_1 \sqcup G_2 \sqcup \cdots \sqcup G_k
$$

$k-1$ 개의 eager fallback 발생 (성능 하락).

---

## 🔬 정리와 증명

### 정리 1.1 — Fusion 에 의한 속도 향상의 상한

**가정**: Kernel fused 로 인해 memory roundtrip 이 $n$ 배 감소, computation density 가 $\rho$ 배 증가.

**결론**: Compiled mode 의 throughput $\Gamma_c$ 는 eager $\Gamma_e$ 대비:

$$
\boxed{\Gamma_c \geq \Gamma_e \cdot \frac{\rho}{1 + (1-\rho)/n}}
$$

**증명**:

Roofline model (Ch4-02) 에서, compute intensity $I = \text{FLOPs} / \text{Bytes}$.

각 kernel launch 에 dispatch overhead $O_{\text{dispatch}}$ 가 있다면:

$$
t_{\text{eager}} = \sum_i \left( \frac{\text{FLOP}_i}{B_{\text{peak}}} + O_{\text{dispatch}} + \frac{\text{Bytes}_i}{BW_{\text{peak}}} \right)
$$

Fusion 후 single kernel:

$$
t_{\text{compiled}} = \frac{\sum_i \text{FLOP}_i}{B_{\text{peak}}} + O_{\text{dispatch}} + \frac{\sum_i \text{Bytes}_i}{BW_{\text{peak}}}
$$

DispatchKey routing 과 memory round-trip 감소로 최대 $n \times$ 향상. 하지만 compute saturation 효과 $\rho \in [0, 1]$ 고려하면 상한은 위 식.  $\square$

### 정리 1.2 — Graph Break 의 compile 성능 영향

Graph break 이 $k$ 개 발생하면, compile time 은 roughly:

$$
t_{\text{compile}} = k \cdot t_{\text{Dynamo}} + t_{\text{Inductor}} + k \cdot t_{\text{Triton JIT}}
$$

**증명**: TorchDynamo 가 각 segment 마다 separate compile pass 필요 (bytecode re-analysis). Inductor 도 segment 마다 독립적으로 code-gen. Triton JIT 도 각 segment (추정 중복 kernel 아님) 마다 invoke.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — 기본 torch.compile 속도 측정

```python
import torch
import torch.nn as nn
import time

torch.manual_seed(42)
torch.cuda.empty_cache()

# 간단한 MLP
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model_eager = SimpleNet().cuda()
model_compiled = torch.compile(SimpleNet().cuda(), mode='reduce-overhead')

x = torch.randn(256, 1024, device='cuda')
y_target = torch.randint(0, 10, (256,), device='cuda')
loss_fn = nn.CrossEntropyLoss()

# Warmup eager
for _ in range(5):
    logits = model_eager(x)
    loss = loss_fn(logits, y_target)

# Time eager mode
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    logits = model_eager(x)
    loss = loss_fn(logits, y_target)
torch.cuda.synchronize()
t_eager = time.time() - t0

print(f"=== torch.compile speedup measurement ===")
print(f"Eager mode: {t_eager*1000:.2f}ms for 100 iters")

# Warmup compiled (first call triggers compilation)
print("\nCompiling model... (first call, 10-50 sec expected)")
for _ in range(5):
    logits = model_compiled(x)
    loss = loss_fn(logits, y_target)

# Time compiled mode
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    logits = model_compiled(x)
    loss = loss_fn(logits, y_target)
torch.cuda.synchronize()
t_compiled = time.time() - t0

print(f"Compiled mode: {t_compiled*1000:.2f}ms for 100 iters")
print(f"Speedup: {t_eager/t_compiled:.2f}×")

# Expected output (A100 GPU):
# Eager mode: 1250.00ms for 100 iters
# Compiled mode: 950.00ms for 100 iters
# Speedup: 1.31×
```

**Expected output** (A100, 256 batch):
```
=== torch.compile speedup measurement ===
Eager mode: 1250.00ms for 100 iters
Compiling model... (first call, 10-50 sec expected)
Compiled mode: 950.00ms for 100 iters
Speedup: 1.31×
```

### 실험 2 — Verbose mode 로 compile 파이프라인 추적

```python
# torch.compile 의 verbose logging 활성화
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

model = SimpleNet().cuda()
compiled_model = torch.compile(model, mode='reduce-overhead')

x = torch.randn(32, 1024, device='cuda')

print("=== First forward pass (triggering compilation) ===")
with torch.no_grad():
    _ = compiled_model(x)

# Console output 예시:
# [TorchDynamo] COMPILING ...
# [Dynamo] Graph break at: (list index out of bounds)
# [Inductor] Compiling Triton kernel: output_code = ...
# [Triton] JIT compiling ...
```

### 실험 3 — Graph break 분석 with torch._dynamo.explain

```python
import torch
from torch._dynamo import explain

model = SimpleNet().cuda()
x = torch.randn(32, 1024, device='cuda')

# Analyze graph breaks
explanation = explain(model, [x])

print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Breakpoints:\n{explanation.frame_summary}")

# Output format:
# Graph breaks: 0
# Breakpoints:
# (none - clean compilation)
```

### 실험 4 — 다양한 compile mode 비교

```python
import time

modes = ['default', 'reduce-overhead', 'max-autotune']
results = {}

for mode in modes:
    model = SimpleNet().cuda()
    compiled = torch.compile(model, mode=mode)
    
    x = torch.randn(256, 1024, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = compiled(x)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = compiled(x)
    torch.cuda.synchronize()
    t = time.time() - t0
    
    results[mode] = t

print("\n=== Comparison of compile modes ===")
for mode, t in results.items():
    print(f"{mode:20s}: {t*1000:8.2f}ms")
```

---

## 🔗 실전 활용

### 1. 프로덕션에서 torch.compile 적용

```python
# Training 코드에서
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

# 이 한 줄만 추가!
model = torch.compile(model, mode='reduce-overhead')

for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = model(x)  # Forward, 첫 epoch: 10~50s 컴파일
        loss.backward()  # Backward, 이미 그래프에 포함됨
        optimizer.step()
```

### 2. 서로 다른 input shape 의 recompilation 비용

Input shape 변경 시 recompilation 발생:

```python
compiled = torch.compile(model)

# First batch: B=256
x1 = torch.randn(256, 1024).cuda()
_ = compiled(x1)  # Compiles for shape (256, 1024)

# Second batch: B=512 → RECOMPILE!
x2 = torch.randn(512, 1024).cuda()
_ = compiled(x2)  # Recompiles for shape (512, 1024)
```

해결책: Dynamic shapes 활용하거나 shape padding.

### 3. Graph break 최소화 전략

- Python control flow 최소화
- List comprehension 대신 vectorized ops
- Type 조건문 제거 (shape/dtype 은 guards 로 처리)

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| CUDA 이상의 GPU (CPU compile 도 가능하나 덜 효과적) | CPU training: 2~5% 향상만 (memory bandwidth less bottleneck) |
| Fixed input shapes (또는 pre-padding) | Dynamic shapes: shape guards 이용 또는 symbolic shapes (PT 2.1+) |
| Pure tensor operations | Custom loops, Python built-ins: graph break → eager fallback |
| Deterministic execution | tf.random 같은 random ops: compile 전에 seed 고정 필요 |
| No external I/O | File read/write, print: graph break |

---

## 📌 핵심 정리

$$\boxed{\text{PT 2.0} = \text{Dynamo (capture)} \to \text{AOTAutograd (graph)} \to \text{Inductor (codegen)}}$$

| 단계 | 역할 | 시간 |
|------|------|------|
| **Dynamo** | Python bytecode → FX graph | ~1초 |
| **AOTAutograd** | Forward + backward graph | ~0.1초 |
| **Inductor** | FX → Triton kernel codegen | ~1초 |
| **Triton JIT** | Kernel binary 컴파일 | ~5~50초 |
| **첫 cached call** | 생성된 kernel 실행 | ~1 ms |
| **Typical speedup** | Eager 대비 | 1.2~1.5× |

---

## 🤔 생각해볼 문제

**문제 1** (기초): `torch.compile` 의 첫 호출이 느린 이유를 Dynamo, AOTAutograd, Inductor 의 세 단계로 설명하라.

<details>
<summary>해설</summary>

1. **Dynamo**: Python bytecode 를 frame evaluation API (PEP 523) 으로 capture 하면서 매 op 마다 FX graph node 생성 — 수백 줄 코드라면 수백 개 node, 초단위 분석 필요
2. **AOTAutograd**: Forward graph 에서 backward 를 symbolic 하게 derive — chain rule 을 따라 모든 intermediate 를 graph 에 추가 (memory 증가)
3. **Inductor + Triton**: FX graph 를 Triton 코드 문자열로 생성한 후, LLVM → cubin 컴파일 — 이것이 가장 오래 걸림 (5~50s)

합쳐서: 첫 호출은 10~50 초, 이후 캐시된 호출은 1 ms 수준.  $\square$

</details>

**문제 2** (심화): 왜 `torch.compile` 이 "코드 변경 없이" 작동하는가? 즉, `model(x)` 를 호출하면 자동으로 compiled version 으로 라우팅되는 메커니즘은?

<details>
<summary>해설</summary>

`torch.compile(model)` 은 model 의 `forward` 메서드를 wrapping:

```python
class CompiledModule:
    def __init__(self, original_module):
        self.module = original_module
    
    def forward(self, x):
        # First call: trigger compilation
        if not self._compiled:
            self._compile()  # Dynamo + AOT + Inductor
        # Subsequent calls: use cached kernel
        return self._kernel(x)
```

따라서 `model(x)` 는 Python 레벨에서는 같아 보이지만, 내부적으로는 compiled kernel 을 호출. 이것이 "Python interface 유지하면서 내부 최적화" 의 핵심.  $\square$

</details>

**문제 3** (논문 비평): Ansel et al. (2024, PLDI) 의 PyTorch 2.0 논문에서, 왜 AOTAutograd 가 "head-of-time" 이라는 이름을 얻었는가? Eager-mode autograd 와 어떤 본질적 차이가 있는가?

<details>
<summary>해설</summary>

**Eager autograd**: Forward 실행 중에 intermediate value 를 tape 에 저장 → backward 에서 tape 를 역순 재생

**AOTAutograd**: Backward graph 를 **미리** (ahead-of-time) 컴파일 시점에 symbolic 하게 생성 → Inductor 가 forward + backward 를 함께 최적화 가능 (memory 재사용, fusion, etc.)

이로 인해:
- Forward-only kernel 보다 forward + backward 를 fuse 한 단일 kernel 생성 가능
- Triton 이 최적화 공간 확대
- 메모리 할당 예측 가능 (no runtime tape overhead)

논문의 key contribution.  $\square$

</details>

---

<div align="center">

[◀ 이전](../ch6-mixed-precision/05-stochastic-rounding-kahan.md) | [📚 README](../README.md) | [다음 ▶](./02-torchdynamo.md)

</div>
