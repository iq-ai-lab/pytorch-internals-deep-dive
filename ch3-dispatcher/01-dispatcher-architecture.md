# 01. Dispatcher Architecture

## 🎯 핵심 질문

- `aten::add(a, b)` 한 호출이 CPU, CUDA, 또는 autograd kernel 중 어느 것으로 실행되는가?
- DispatchKey (Autograd, CUDA, CPU, BackendSelect, Functionalize, AutocastCUDA 등 50+ 개) 는 무엇이고, 어떤 순서로 우선순위가 정해지는가?
- DispatchKeySet (bit mask) 은 어떻게 작동하고, "highest priority key 부터 순차 dispatch (fallthrough chain)" 의 메커니즘은?
- `torch._C._dispatch_dump_table('aten::add')` 로 등록 상태를 어떻게 확인하는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 사용자가 `y = torch.add(x, torch.tensor(2.0, requires_grad=True))` 를 부르면, 내부적으로:

1. **어떤 device 에서 실행할 것인가** (CPU / CUDA / TPU)
2. **어떤 dtype 의 kernel 을 부를 것인가** (FP32 / FP16 / BF16)
3. **backward graph 를 등록할 것인가** (autograd key 활성?)
4. **특수 변환을 적용할 것인가** (functionalize, autocast, ...)

이 모든 결정이 **Dispatcher** 라는 routing system 이 자동으로 내립니다. Custom op 를 만들거나 성능 최적화를 위해 특정 kernel 의 우선순위를 바꿀 때, 또는 `torch.compile` 의 graph decomposition 을 이해할 때 Dispatcher 를 모르면 불가능합니다.

---

## 📐 수학적 선행 조건

- **ATen / c10 기초**: Tensor 의 metadata (device, dtype, layout)
- **C++ template metaprogramming**: `TORCH_LIBRARY_IMPL`, macro 전개
- **bit mask 연산**: `|`, `&` 로 DispatchKeySet 조작
- (선택) 컴파일러 이론 — 동적 dispatch vs static dispatch 의 성능 trade-off

---

## 📖 직관적 이해

### Router 로서의 Dispatcher

Dispatcher 는 **restaurant kitchen 의 order routing system** 에 비유할 수 있습니다:

```
Customer: "aten::add(x_cuda_fp32, y_cuda_fp32)"
                        │
                        ↓
                  [ Dispatcher ]
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   BackendSelect  Autograd[CUDA]    CUDA kernel
     (choose)     (setup backward)   (compute)
        │               │               │
        └───────────────┴───────────────┘
                        │
                        ↓
           result + computation graph
```

각 DispatchKey 는 한 요리사(handler) — BackendSelect 는 device/dtype 을 선택, Autograd[CUDA] 는 backward 를 준비, CUDA 는 실제 연산을 수행.

### DispatchKeySet 의 bit mask

DispatchKey 는 enum (0 ~ 64 정도), DispatchKeySet 은 uint64 bit mask:

```
DispatchKeySet = { CUDA, Autograd, Strided, FP32, ... }
               = 0b0010_1001_0011...  (각 bit 가 key 를 대표)
```

**highest priority 부터 fallthrough**: bit 가 높을수록 우선순위 높음. Dispatcher 는 최상위 bit 부터 찾아 해당 kernel 을 실행.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — DispatchKey Enum

```cpp
enum class DispatchKey {
    // 0-15: "meta" 영역 — device/dtype 선택용
    Undefined = 0,
    CPU = 1,
    CUDA = 2,
    // ...
    
    // 16-31: "functionality" — transformation
    Autograd = 16,
    AutocastCPU = 17,
    AutocastCUDA = 18,
    Functionalize = 19,
    BackendSelect = 20,
    // ... 50+ keys total
};
```

### 정의 3.2 — DispatchKeySet

$$\text{DispatchKeySet} = \{ k \in \text{DispatchKey} \mid \text{bit}_k = 1 \}$$

예: $\text{CUDA} \cup \text{Autograd} \cup \text{FP32} = 0b...00101...$

### 정의 3.3 — Dispatch Table & Kernel Registry

For operator `op_name` and dispatch key combination $\mathcal{K}$:

$$\text{kernels}[\text{op_name}][\mathcal{K}] = f : \text{Tensor...} \to \text{Tensor...}$$

DispatchKey 조합이 여러 개 가능 (예: `(CUDA, FP32)`, `(CUDA, FP16)`, `(CPU, FP32)`).

---

## 🔬 정리와 증명

### 정리 3.1 — Fallthrough Chain 의 정확성

**주장**: Dispatcher 가 DispatchKeySet 의 최상위 bit 부터 kernel 을 탐색할 때, 모든 입력이 같은 DispatchKeySet 을 공유하면 단 하나의 kernel 이 선택된다.

**증명**:
1. DispatchKeySet $\mathcal{K} = \{k_1, k_2, ..., k_n\}$ 에서 $k_1 > k_2 > ... > k_n$ (priority order)
2. Dispatcher 는 $k_1$ 에 대한 handler 를 먼저 탐색
3. 만약 $k_1$ 에 대한 kernel 이 등록되어 있으면 → 실행, 반환
4. 아니면 $k_2$ 로 fallthrough, 반복
5. 모든 $k_i$ 가 unregistered 면 error

따라서 **priority order 에 따라 단 하나의 execution path** 가 결정됨 $\square$.

### 따름 정리 3.2 — Autograd Key 의 우선순위

Autograd key 는 BackendSelect 보다 높은 priority 를 가지므로, backward 등록이 필요한 op 의 경우:

$$\text{Autograd[device]} \to \text{device kernel} \to \text{result + grad_fn}$$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Dispatch Table 직접 확인

```python
import torch

# aten::add 의 dispatch table 출력
torch._C._dispatch_dump_table('aten::add')

# 예상 출력 (부분):
# DispatchTable for aten::add:
#   CPU (FP32): native::add_cpu_fp32
#   CUDA (FP32): at::cuda::add_cuda_fp32
#   Autograd[CUDA]: autograd::AddBackward0
#   BackendSelect: native::add  (router)
#   ...
```

### 실험 2 — Device 별 kernel 선택 확인

```python
import torch

# CPU tensor
x_cpu = torch.tensor([1.0, 2.0])
y_cpu = torch.tensor([3.0, 4.0])
result_cpu = torch.add(x_cpu, y_cpu)
print(f"CPU result device: {result_cpu.device}")  # device(type='cpu')

# CUDA tensor (GPU 있을 때)
if torch.cuda.is_available():
    x_cuda = torch.tensor([1.0, 2.0]).cuda()
    y_cuda = torch.tensor([3.0, 4.0]).cuda()
    result_cuda = torch.add(x_cuda, y_cuda)
    print(f"CUDA result device: {result_cuda.device}")  # device(type='cuda', index=0)
```

dispatch table 은 두 케이스에서 다른 kernel 을 선택.

### 실험 3 — Autograd key 활성 여부 확인

```python
import torch

# Autograd key 활성
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0])
result = torch.add(x, y)
print(f"result.grad_fn: {result.grad_fn}")  # AddBackward0
print(f"requires_grad: {result.requires_grad}")  # True

# Autograd key 비활성 (no_grad)
with torch.no_grad():
    result_no_grad = torch.add(x, y)
    print(f"result_no_grad.grad_fn: {result_no_grad.grad_fn}")  # None
    print(f"result_no_grad.requires_grad: {result_no_grad.requires_grad}")  # False

# Dispatcher 가 torch.no_grad context 감지 → Autograd key 제거
# → BackendSelect → device kernel (backward 미등록)
```

### 실험 4 — Custom kernel 등록과 우선순위

```python
import torch
from torch.library import TORCH_LIBRARY

# 새 op 정의
TORCH_LIBRARY(mylib, m) {
    m.def("myop(Tensor x) -> Tensor")
}

# CUDA kernel 등록
TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("myop", lambda x: x * 2)  # 임시
}

# CPU kernel 등록
TORCH_LIBRARY_IMPL(mylib, CPU, m) {
    m.impl("myop", lambda x: x * 3)  # 우선순위 낮음
}

# CUDA 에서 실행하면 CUDA kernel 우선
x = torch.randn(3).cuda()
result = torch.ops.mylib.myop(x)
# dispatcher 는 CUDA key 를 먼저 확인 → myop_cuda 실행
```

---

## 🔗 실전 활용

### 1. Custom Operator 의 Backend 확장

새로운 device (TPU, Neuron) 를 지원할 때, custom op 에 dispatcher key 를 추가하면 자동으로 routing 됩니다.

### 2. Mixed Precision (AMP) 의 dispatcher 통합

AutocastCUDA key 는 FP32 → FP16 conversion 을 자동 적용. Dispatcher 가 AutocastCUDA key 를 감지하면, `torch.autocast()` context 내의 ops 는 자동으로 낮은 precision 으로 dispatch.

### 3. Performance Profiling

`torch._C._dispatch_dump_table()` 로 어떤 kernel 이 호출되는지 확인, 예상과 다른 device 에서 실행되는 경우 디버그.

### 4. torch.compile 과 dispatcher

TorchDynamo 는 graph 를 추출할 때 dispatcher 의 routing 정보를 보존, Inductor 가 kernel 생성 시 참고.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 단일 tensor — 모든 input 이 같은 device/dtype | 혼합 device (CPU + CUDA) 시 implicit transfer 필요 → cudaHostRegister 로 pinned memory 활용 |
| DispatchKey 가 static (compile-time) | Dynamic dispatch 는 성능 저하 (cache miss) → prioritize compile-time decision 최적화 (TorchScript, `@torch.compile`) |
| Kernel 등록이 일관적 | Missing backend 시 runtime error → fallback kernel 또는 slow generic implementation 제공 |
| Linear fallthrough chain | 실제로는 priority bitmap 으로 O(1) lookup (bit scanning) |

---

## 📌 핵심 정리

$$\boxed{\text{op}(\text{inputs}) \xrightarrow{\text{DispatchKeySet}} \xrightarrow{\text{Priority}} \text{device·dtype·backward kernel}}$$

| 개념 | 역할 |
|------|------|
| **DispatchKey** | op 를 어떤 handler 로 routing 할지 지정하는 tag (50+ enum 값) |
| **DispatchKeySet** | 입력 tensor 들의 device, dtype, autograd 상태를 bit mask 로 표현 |
| **Fallthrough chain** | 최상위 priority key 부터 kernel 탐색, 미등록 시 다음 key 로 |
| **Dispatcher** | DispatchKeySet → 해당 kernel 을 O(1) 로 lookup 후 실행 |

**핵심**: PyTorch 의 모든 op 호출은 dispatcher 를 거치며, 이것이 dynamic dispatch 의 유연성 (같은 코드 CPU/CUDA 에서 실행)을 가능하게 합니다.

---

## 🤔 생각해볼 문제

**문제 1** (기초): DispatchKeySet 에서 CPU 와 CUDA key 가 동시에 set 되는 경우가 있을까? 왜 또는 왜 아닌가?

<details><summary>해설</summary>

아닙니다. Tensor 는 single device 에만 존재하므로, CPU 또는 CUDA 중 정확히 하나만 set 됩니다. 

실제로 PyTorch 는 혼합 device op (예: `torch.add(x_cpu, y_cuda)`) 를 만나면 명시적으로 한쪽을 transfer 하거나 error 를 냅니다 — dispatcher 차원에서 지원하지 않음.

**예외**: "meta device" (device 를 지정하지 않은 abstract tensor) 는 shape/dtype 만 추적, 실제 연산은 하지 않음. Symbolic tracing (TorchScript) 에서 사용. $\square$

</details>

**문제 2** (심화): Fallthrough chain 에서 특정 DispatchKey 에 대한 kernel 이 "intentionally" 등록되지 않을 때, 이것이 error 인가 아니면 next key 로 gracefully fallthrough 하는가? PyTorch 의 policy 를 설명하라.

<details><summary>해설</summary>

PyTorch 의 policy:

1. **Essential keys** (BackendSelect, CPU, CUDA) — 모든 op 는 최소 이들 중 하나를 등록해야 함. 미등록 → error
2. **Optional keys** (Autograd, Functionalize, AutocastCUDA) — 미등록 시 fallthrough 하며 그 기능 자체가 disabled 됨
   - Autograd 미등록 → `requires_grad` 가 False 로 처리
   - Functionalize 미등록 → mutation 있는 op 는 실패할 수 있음

**설계 의도**: backward 구현이 어려운 ops (예: `conv_transpose3d` 의 일부 variant) 는 정의 상 autograd 를 지원하지 않고, fallthrough 하면서 warning 발행.

Modern PyTorch (2.0+) 는 AOTAutograd 로 자동 backward generation, fallthrough 최소화. $\square$

</details>

**문제 3** (논문 비평): Edward Yang 의 "Let's Talk About the PyTorch Dispatcher" (2021) 에서 dispatcher 의 O(1) lookup 이 어떻게 구현되고, 이것이 dynamic dispatch 임에도 불구하고 성능에 미치는 영향이 minimal 한 이유는?

<details><summary>해설</summary>

**구현**: DispatchKeySet 을 `uint64` bit vector 로 표현, `__builtin_clzll()` (count leading zeros) 로 최상위 set bit 찾기 → O(1).

```cpp
// pseudocode
DispatchKeySet keys = compute_dispatch_keys(inputs);
DispatchKey highest = __builtin_clzll(keys);  // bit scanning
KernelFunction* kernel = dispatch_table[op_name][highest];
return kernel(inputs);
```

**성능 영향이 minimal 한 이유**:

1. **bit scanning 비용** << 실제 kernel 실행 비용 (수 microseconds vs ms)
2. **CPU cache**: DispatchKeySet 계산 (dtype, device 확인) 은 Tensor metadata (already in L1), bit vector lookup 은 small table (inline cache-friendly)
3. **Compiler optimization**: 자주 호출되는 op (add, mul) 는 JIT 로 specialization 또는 inline 됨

실제로 PyTorch 2.1+ 에서는 `torch.compile` 을 통해 dispatching 자체를 compile-time 에 제거 (graph specialization). $\square$

</details>

---

<div align="center">

[◀ 이전](../ch2-autograd/06-double-backward.md) | [📚 README](../README.md) | [다음 ▶](./02-aten-c10-layers.md)

</div>
