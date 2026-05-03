# 04. TORCH_LIBRARY 와 Custom Operator

## 🎯 핵심 질문

- `TORCH_LIBRARY(mylib, m) { m.def("myop(Tensor x) -> Tensor"); }` 의 정확한 역할은?
- `TORCH_LIBRARY_IMPL(mylib, CUDA, m)` 로 CUDA kernel 을 어떻게 등록하는가?
- Python 에서 `torch.ops.mylib.myop(x)` 로 호출된 op 는 내부적으로 어떻게 dispatcher 를 거치는가?
- AOTAutograd / Inductor 가 custom op 를 자동으로 통합할 때, 어떤 정보를 필요로 하는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 built-in ops (`torch.add`, `torch.matmul`) 는 ATen 에 정의되고, dispatcher 가 자동으로 CPU/CUDA/autograd 를 처리합니다. 하지만 custom op 를 만들려면?

**시나리오**:
1. 새로운 알고리즘 (예: custom attention kernel)
2. Domain-specific operation (예: 3D vision 의 특수 projection)
3. 외부 라이브러리 통합 (예: NCCL collective operation)

이 경우 `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL` 으로 dispatcher 에 등록하면, 자동으로:
- Python binding 생성
- backward 지원 (manual or auto)
- `torch.compile` 호환성 (metadata function 등록 시)

---

## 📐 수학적 선행 조건

- **Dispatcher architecture** (Chapter 3-01): DispatchKey, fallthrough
- **c10/ATen layering** (Chapter 3-02): schema, code generation
- **Custom autograd** (Chapter 2-05, `torch.autograd.Function`): backward 구현
- **C++ template basics**: `TORCH_LIBRARY`, macro usage

---

## 📖 직관적 이해

### TORCH_LIBRARY 의 역할

```
Step 1: Schema 정의 (TORCH_LIBRARY)
┌─────────────────────────────────────────┐
│ m.def("myop(Tensor x) -> Tensor")       │
│   Dispatcher 에게: "myop 이라는 op 있음" │
│   정보: input Tensor, output Tensor     │
└─────────────────────────────────────────┘
         │
         ↓
Step 2: Backend 별 kernel 등록 (TORCH_LIBRARY_IMPL)
┌──────────────────────────────────────────┐
│ TORCH_LIBRARY_IMPL(mylib, CUDA, m)      │
│   m.impl("myop", my_cuda_kernel)        │
│   Dispatcher: myop × CUDA → my_cuda_... │
└──────────────────────────────────────────┘
         │
         ↓
Step 3: Python binding 자동 생성 (pybind11)
┌──────────────────────────────────────────┐
│ torch.ops.mylib.myop(x)  ← Python API   │
└──────────────────────────────────────────┘
```

### Autograd 등록 패턴

Custom op 에 backward 를 추가:

```cpp
// Method 1: torch::autograd::Function (manual)
TORCH_LIBRARY_IMPL(mylib, Autograd, m) {
  m.impl("myop", torch::autograd::Function<MyOpAutograd>::apply);
};

// Method 2: torch.library.register_kernel (Python, 편함)
torch.library.register_kernel("mylib::myop", 
    implementation=my_backward_fn,
    dispatch_key="Autograd")
```

---

## ✏️ 엄밀한 정의

### 정의 3.10 — TORCH_LIBRARY Macro

```cpp
TORCH_LIBRARY(mylib, m) {
  // m: a module binding object
  
  // Schema declaration
  m.def("myop(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
  m.def("myop_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  // (a!) = mutates input
}
```

역할:
- Op 의 schema (signature) 를 dispatcher 에 등록
- Python binding 의 scaffold 생성
- Native function table 에 entry 추가

### 정의 3.11 — TORCH_LIBRARY_IMPL Macro

```cpp
TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
  // m: implementation binding object
  
  // Kernel registration
  m.impl("myop", &my_cuda_kernel);  // function pointer
  m.impl("myop_", &my_cuda_kernel_inplace);
}
```

역할:
- `mylib::myop` × `CUDA` → `my_cuda_kernel` 의 mapping 을 dispatcher table 에 추가

### 정의 3.12 — Custom Autograd Function (C++ API)

```cpp
class MyOpAutograd : public torch::autograd::Function<MyOpAutograd> {
  static tensor_list apply(AutogradContext* ctx, 
                          const Tensor& self, 
                          const Tensor& other) {
    // Forward: save for backward
    ctx->save_for_backward({self, other});
    
    // Redispatch without Autograd
    return {impl::myop_cuda(self, other)};
  }
  
  static tensor_list backward(AutogradContext* ctx, 
                             tensor_list grad_outputs) {
    // Backward: compute VJP
    auto [self, other] = ctx->get_saved_tensors();
    auto grad_self = ...; // VJP computation
    auto grad_other = ...;
    return {grad_self, grad_other};
  }
};

// Registration
TORCH_LIBRARY_IMPL(mylib, Autograd, m) {
  m.impl("myop", torch::autograd::Function<MyOpAutograd>::apply);
}
```

---

## 🔬 정리와 증명

### 정리 3.6 — Custom Op 의 Dispatcher Integration

**주장**: TORCH_LIBRARY + TORCH_LIBRARY_IMPL 로 등록된 custom op 는, built-in op 과 동일하게 dispatcher 의 fallthrough chain 을 따르며, 올바른 backend kernel 을 호출한다.

**증명**:

1. TORCH_LIBRARY 로 schema 등록 → dispatcher table 에 op entry 생성
2. TORCH_LIBRARY_IMPL(lib, CUDA) 로 CUDA kernel 등록 → dispatcher 의 `kernels[lib::op][CUDA]` 에 함수 포인터 저장
3. Python `torch.ops.mylib.myop(x)` 호출
4. Dispatcher 계산:
   ```cpp
   DispatchKeySet keys = compute_keys(x);  // {CUDA, ...}
   KernelFn kernel = dispatch_table["mylib::myop"][keys];
   return kernel(x);  // my_cuda_kernel
   ```
5. 따라서 registered kernel 이 호출 $\square$.

### 정리 3.7 — Autograd 자동 통합

**주장**: Custom op 가 `torch.autograd.Function` 으로 backward 를 정의하고 Autograd key 에 등록하면, forward/backward 가 dispatcher 를 통해 자동으로 분리 실행된다.

**증명**:

Forward (requires_grad=True):
1. Dispatcher: Autograd key 포함 → `MyOpAutograd::apply` 호출
2. apply 내부: `ctx->save_for_backward()` 실행
3. Redispatch (Autograd key 제거) → `my_cuda_kernel` 호출
4. 결과에 grad_fn=MyOpAutograd 붙임

Backward:
1. Reverse topological order → `MyOpAutograd::backward` 호출
2. Saved tensors 복원 → gradient 계산
3. Upstream gradient 와 곱하기 (chain rule)

따라서 forward/backward 분리가 정확 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — C++ Custom Op (cpp_extension)

**File: my_op.cpp**

```cpp
#include <torch/extension.h>

torch::Tensor my_add_cpu(torch::Tensor self, torch::Tensor other) {
  return self + other;
}

torch::Tensor my_add_cuda(torch::Tensor self, torch::Tensor other) {
  // (CUDA kernel implementation)
  return self + other;  // placeholder
}

TORCH_LIBRARY(mylib, m) {
  m.def("my_add(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(mylib, CPU, m) {
  m.impl("my_add", &my_add_cpu);
}

TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
  m.impl("my_add", &my_add_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // pybind11 binding (manual, optional if using TORCH_LIBRARY)
}
```

**Python 에서 사용**:

```python
import torch
from torch.utils.cpp_extension import load

# Compile
my_module = load(
    name="mylib",
    sources=["my_op.cpp"],
    is_python_module=False  # TORCH_LIBRARY 사용하면 False
)

# Use
x = torch.randn(3).cuda()
y = torch.randn(3).cuda()
result = torch.ops.mylib.my_add(x, y)  # CUDA kernel 호출
```

### 실험 2 — Python 에서 Custom Op 등록

```python
import torch
from torch.library import TORCH_LIBRARY, TORCH_LIBRARY_IMPL

# 1. Schema 정의
torch.library.define("mylib::my_sigmoid(Tensor x) -> Tensor")

# 2. CPU kernel 등록
torch.library.impl("mylib::my_sigmoid", 
    lambda x: torch.sigmoid(x), 
    dispatch_key="CPU")

# 3. CUDA kernel 등록
torch.library.impl("mylib::my_sigmoid",
    lambda x: torch.sigmoid(x),  # placeholder
    dispatch_key="CUDA")

# 4. 사용
x = torch.randn(3)
result = torch.ops.mylib.my_sigmoid(x)
print(result)
```

### 실험 3 — Backward 등록

```python
import torch
from torch.autograd import Function

class MySigmoidAutograd(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sigmoid(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig_x = torch.sigmoid(x)
        return grad_output * sig_x * (1 - sig_x)

# Autograd 등록
torch.library.impl("mylib::my_sigmoid",
    MySigmoidAutograd.apply,
    dispatch_key="Autograd")

# 이제 backward 가능
x = torch.randn(3, requires_grad=True)
result = torch.ops.mylib.my_sigmoid(x)
print(f"result.grad_fn: {result.grad_fn}")  # MySigmoidAutogradBackward

loss = result.sum()
loss.backward()
print(f"x.grad: {x.grad}")
```

### 실험 4 — torch.compile 과 custom op

```python
import torch

# Custom op 정의 (위와 동일)
torch.library.define("mylib::my_add(Tensor x, Tensor y) -> Tensor")
torch.library.impl("mylib::my_add",
    lambda x, y: x + y,
    dispatch_key="CPU")

# torch.compile
@torch.compile
def f(x, y):
    return torch.ops.mylib.my_add(x, y)

x = torch.randn(3)
y = torch.randn(3)
result = f(x, y)

# TorchDynamo 가 torch.ops.mylib.my_add 를 graph 에 포함
# Inductor 가 이를 인식 → kernel fusion 최적화 (가능하면)
```

---

## 🔗 실전 활용

### 1. External CUDA Kernel 통합

```cpp
// MyCustomKernel.cu
__global__ void my_kernel_cuda(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = x[idx] * 2;
}

// my_op_cuda.cpp
torch::Tensor my_op_cuda(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  my_kernel_cuda<<<blocks, threads>>>(
    x.data_ptr<float>(), 
    y.data_ptr<float>(), 
    n);
  return y;
}

TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
  m.impl("my_op", &my_op_cuda);
}
```

### 2. MetaTensor/FakeTensor 지원 (torch.compile 호환)

```cpp
// Meta kernel (shape/dtype 정보만, 실제 연산 X)
torch::Tensor my_op_meta(torch::Tensor x) {
  return torch::empty_like(x);  // dummy, 실제 계산 없음
}

TORCH_LIBRARY_IMPL(mylib, Meta, m) {
  m.impl("my_op", &my_op_meta);
}

// Dispatcher: torch.compile 의 symbolic execution 시
// Meta kernel 호출 → shape/dtype 만 propagate
```

### 3. Operator Fusion (AOTAutograd)

Custom op 가 composite 로 정의되면, AOTAutograd 가 자동으로 forward+backward graph 최적화:

```python
# Composition 으로 정의
torch.library.define("mylib::my_fused_act(Tensor x) -> Tensor")

# 구현: linear + ReLU
torch.library.impl("mylib::my_fused_act",
    lambda x: torch.relu(x),  # placeholder: 실제로는 fused kernel
    dispatch_key="CUDA")

# torch.compile 에서:
# - Forward: x → ReLU(x) 로 decompose
# - Backward: d(ReLU(x))/dx = (x > 0) 자동 생성
# - Inductor: 가능하면 single kernel 로 fuse
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Schema 가 정적 | Op signature 변경 버전 호환성 어려움 → 새 schema 정의 (mylib::my_op_v2) |
| 모든 backend 에 kernel 등록 | Missing backend 시 fallback 필요 → CompositeImplicitAutograd 로 other ops composition |
| Backward 자동 지원 | Manual backward 어려운 경우 많음 → torch::autograd::Function 명시 필요 |
| torch.compile 호환 | 특수 kernel (custom loop) 의 경우 Meta kernel 작성 어려울 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\text{TORCH_LIBRARY (schema)} \to \text{TORCH_LIBRARY_IMPL (backends)} \to \text{Dispatcher} \to \text{torch.ops...}}$$

| 단계 | 역할 |
|------|------|
| **TORCH_LIBRARY** | Op 의 schema 정의 → dispatcher table entry 생성 |
| **TORCH_LIBRARY_IMPL** | Backend 별 kernel 등록 → dispatcher 의 fallthrough chain 구성 |
| **Autograd 등록** | torch::autograd::Function 로 backward 정의 → forward/backward 분리 |
| **Python binding** | pybind11 자동 생성 → torch.ops.lib.op 로 호출 가능 |

**핵심**: Built-in op 과 동일한 dispatcher mechanism 으로 custom op 를 통합, CPU/CUDA/Autograd 를 자동 처리.

---

## 🤔 생각해볼 문제

**문제 1** (기초): TORCH_LIBRARY 에서 schema 정의 시, `Tensor(a!)` 의 의미는 무엇이고, 왜 inplace op 에서 중요한가?

<details><summary>해설</summary>

**`Tensor(a!)`**: "alias to input a, mutated"
- `a` = 첫 번째 input tensor
- `!` = inplace 수정 (aliased 됨)

```yaml
- func: add_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  # self 를 제자리 수정, output 도 self 를 가리킴
```

**중요성**:
1. **Dispatcher 에게 알림**: 이 op 는 inplace → backward 시 주의
2. **Autograd**: backward 에서 saved_for_backward 시 `self` 의 original value 를 저장해야 함 (나중 수정된 값과 다를 수 있음)
3. **torch.compile**: FakeTensor execution 에서도 aliasing 추적

inplace op 미선언 시, autograd 가 잘못된 gradient 계산할 수 있음. $\square$

</details>

**문제 2** (심화): Custom op 가 `CompositeImplicitAutograd` 로 정의될 때, AOTAutograd 와 torch.compile 이 어떻게 자동으로 backward 를 생성하는가? 어떤 조건이 필요한가?

<details><summary>해설</summary>

**CompositeImplicitAutograd**:
```python
torch.library.define("mylib::my_composite(Tensor x) -> Tensor")

# 구현을 다른 ops 의 composition 으로
torch.library.impl("mylib::my_composite",
    lambda x: torch.sin(x) * torch.cos(x),  # sin·cos 는 이미 backward 등록됨
    dispatch_key="CompositeImplicitAutograd")
```

**AOTAutograd 의 자동 backward**:
1. torch.func.grad() 를 forward function 에 적용
2. `torch.sin` 과 `torch.cos` 의 backward 는 이미 존재 → chain rule 자동 적용
3. Result: composite op 의 backward = `lambda g_out: g_out * (cos²x - sin²x)`

**필요조건**:
- Composed ops 들이 모두 backward 지원
- Python function 으로 정의 (C++ Composite 는 torch::func::grad 필요)
- No inplace 또는 explicit mutation handling

**장점**:
- Manual backward 코드 작성 안 함
- torch.compile 에서 decompose 되어 fusible
- Gradient checkpointing 자동 적용 가능

반면 low-level custom CUDA kernel 은 manual backward 필수. $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 TORCH_LIBRARY 설계와 JAX 의 primitive function registration, NumPy 의 universal functions (ufuncs) 의 개념적 유사성과 차이점을 비교하라. 각각이 extensibility 를 어떻게 지원하는가?

<details><summary>해설</summary>

| 접근 | Extensibility | 성능 | 유연성 |
|------|---------------|------|--------|
| **PyTorch TORCH_LIBRARY** | Schema 기반 dispatcher 등록 | Dispatcher overhead 최소 (static) | Backend 별 kernel 명시 필요 |
| **JAX primitives** | `core.primitive_register` → JAX tracer 에 rule 정의 | Custom tracer rules 비용 | 높음 (functional 철학) |
| **NumPy ufuncs** | C extension 또는 `__array_ufunc__` | Native C 속도 | 낮음 (fixed op shape) |

**설계 철학**:
- **PyTorch**: "Statically declarative" (schema 미리 정의) → compile-time 최적화, multi-backend dispatch
- **JAX**: "Dynamically composable" (pure functions) → automatic differentiation, functional purity
- **NumPy**: "C-level efficiency" (ufunc protocol) → vectorization, broadcasting

**Modern trend**:
- PyTorch 2.0+: schema + AOTAutograd = JAX 의 functional clarity + PyTorch 의 backend efficiency
- JAX: pure python 에서 schema-like 구조 도입 (e.g., jaxtyping)

실제로 둘 다 "namespace'd op + backend 등록" 구조로 수렴. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-autograd-dispatch.md) | [📚 README](../README.md) | [다음 ▶](./05-functorch-transforms.md)

</div>
