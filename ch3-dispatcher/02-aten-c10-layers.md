# 02. ATen 과 c10 — Layered 구조

## 🎯 핵심 질문

- c10, ATen, torch 의 의존성 계층은 정확히 어떤 모양인가?
- c10 은 "core types" (Tensor, Storage, Device, ScalarType, DispatchKey) 만 정의하고 ATen 에 의존하지 않는가?
- `native_functions.yaml` 에서 op 의 schema 를 정의하고, 이로부터 자동 생성되는 코드 (RegisterCPU.cpp, RegisterCUDA.cpp) 는 무엇인가?
- Edward Yang 의 PyTorch Internals blog 에서 "schema-driven" 설계의 핵심은 무엇인가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 의 핵심 설계는 **세 계층의 명확한 분리** 에 있습니다:

```
┌─────────────────────────────────────────┐
│  torch (Python binding, user-facing)    │  ← 사용자는 여기서 호출
└─────────────┬───────────────────────────┘
              │ pybind11
┌─────────────▼───────────────────────────┐
│  ATen (operations, kernels)             │  ← add, matmul, conv2d, ... 구현
│  - Schema definition                    │
│  - CPU/CUDA kernel registration         │
└─────────────┬───────────────────────────┘
              │ depends on
┌─────────────▼───────────────────────────┐
│  c10 (core types)                       │  ← Tensor, Storage, Device
│  - No dependencies on ATen              │
└─────────────────────────────────────────┘
```

이 구조가 없으면, C++ 코드를 수정할 때마다 manual kernel registration 을 해야 하고, 새로운 backend (TPU, Neuron) 를 지원할 때 모든 op 를 다시 구현해야 합니다. Schema-driven 접근법이 이를 자동화합니다.

---

## 📐 수학적 선행 조건

- **C++ templates & metaprogramming**: `TORCH_LIBRARY`, macro 전개
- **Code generation**: 컴파일 시간에 코드 생성의 장점 (type safety, dispatch overhead 감소)
- **Interface segregation principle**: 계층 간 의존성 최소화
- (선택) Compiler design — code generation pass

---

## 📖 직관적 이해

### 세 계층의 역할

**c10**: "Core Ten" (처음 10개 type + 기본 구조)
- Tensor 의 metadata: `(Storage, size, stride, offset, device, dtype, layout)`
- Device enum: CPU, CUDA, ...
- ScalarType enum: Float32, Float64, Int64, ...
- DispatchKey enum

**ATen**: "A Tensor library"
- "schema" 로 op 의 signature 정의 (입출력 type, 옵션)
- `native_functions.yaml` 에서 모든 op 나열
- 각 backend 에 대한 kernel 구현 (native_functions 폴더에서 자동 생성)

**torch**: Python binding
- pybind11 로 ATen C++ 함수를 Python method 로 노출
- `torch.add(x, y)` → `at::add` C++ 함수 호출

### "Schema-driven" 의 의미

```yaml
# native_functions.yaml excerpt
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  variants: function, method
  
  # 이 schema 로부터:
  # 1. add 의 C++ signature 자동 생성
  # 2. Python binding 자동 생성
  # 3. CPU kernel 등록 scaffold 생성
  # 4. CUDA kernel 등록 scaffold 생성
```

한 번의 schema 정의로 여러 backend 에 자동 확산.

---

## ✏️ 엄밀한 정의

### 정의 3.4 — c10 의 최소 API

```cpp
namespace c10 {
  // Device abstraction
  enum class DeviceType { CPU, CUDA, ... };
  struct Device { DeviceType type; int index; };
  
  // Type abstraction
  enum class ScalarType { Float, Double, Int64, ... };
  
  // Dispatch abstraction
  enum class DispatchKey { CPU, CUDA, Autograd, ... };
  using DispatchKeySet = uint64_t;  // bit mask
  
  // Storage abstraction
  class Storage { void* data; size_t nbytes; Device device; };
  
  // Tensor metadata (no computation)
  class TensorImpl {
    Storage storage_;
    std::vector<int64_t> sizes_;      // shape
    std::vector<int64_t> strides_;    // memory layout
    int64_t storage_offset_;
    ScalarType dtype_;
    Device device_;
    DispatchKeySet dispatch_keys_;
  };
}
```

**중요**: c10 은 어떤 op 도 정의하지 않음. Tensor 생성·조회만.

### 정의 3.5 — ATen 의 op Schema

```cpp
namespace at {
  // native_functions.yaml 에서 자동 생성
  Tensor add(const Tensor& self, const Tensor& other, 
             const Scalar& alpha = 1);  // Tensor 반환
  
  Tensor matmul(const Tensor& self, const Tensor& other);
  
  Tensor _unsafe_view(const Tensor& self, IntArrayRef size);
}
```

각 op 마다:
- **CPU kernel**: `native::add_cpu(...)` in `cpu/Add.cpp`
- **CUDA kernel**: `cuda::add_cuda(...)` in `cuda/Add.cu`
- **Dispatcher registration**: `TORCH_LIBRARY_IMPL` macro 로 backend 별 kernel 등록

### 정의 3.6 — Dispatcher Table (재정의)

ATen 의 모든 op 에 대해, dispatcher table:

$$\text{kernels}[\text{op\_name}][\text{DispatchKeySet}] : \text{Tensor...} \to \text{Tensor...}$$

예:

$$
\begin{align}
\text{kernels}[\text{"add"}][\{\text{CUDA, Autograd, FP32}\}] &= \text{autograd::AddBackward}\\
\text{kernels}[\text{"add"}][\{\text{CUDA, FP32}\}] &= \text{cuda::add\_cuda\_fp32}\\
\text{kernels}[\text{"add"}][\{\text{CPU, FP32}\}] &= \text{native::add\_cpu\_fp32}
\end{align}
$$

---

## 🔬 정리와 증명

### 정리 3.2 — Schema-driven Code Generation 의 type safety

**주장**: `native_functions.yaml` 에서 schema 정의 → 자동 생성된 C++ signature 는 type-safe 이며, 런타임 에러 (type mismatch) 의 대부분을 컴파일 타임에 잡는다.

**증명 스케치**:

1. YAML schema: `add(Tensor, Tensor) -> Tensor`
2. Code generator 가 읽음 → C++ 함수 생성:
   ```cpp
   Tensor add(const Tensor& self, const Tensor& other) { ... }
   ```
3. Dispatcher 가 이 signature 를 expect
4. Python binding 도 같은 signature 에서 생성
5. 따라서 Python → C++ → dispatcher → kernel 의 전 경로에서 **type mismatch 는 불가능** (static typing)

**따름정리**: 반면 op 를 manual 로 등록하면 signature mismatch 가능 (dispatcher crash) $\square$.

### 정리 3.3 — c10 독립성의 장점

**주장**: c10 이 ATen 에 의존하지 않으므로, 새로운 backend (TPU, Neuron) 를 추가할 때 c10 은 수정하지 않아도 된다.

**증명**:
- c10 은 Tensor 의 shape/stride/device 만 정의
- ATen 은 이를 기반으로 op 구현
- 새 backend 추가 = ATen 의 RegisterTPU.cpp 추가
- c10 전혀 touch 안 함

따라서 **core type 의 stability** 를 보장 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — native_functions.yaml 에서 schema 확인

```python
# PyTorch source code 에서:
# aten/src/ATen/native/native_functions.yaml 일부

# - func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
#   device_check: NoCheck
#   dispatch:
#     CompositeImplicitAutograd: native::add
#     CUDA: native::add_cuda
#     CPU: native::add_cpu
```

이 한 줄의 schema 로부터:
1. 함수 signature 자동 생성
2. 각 backend 별 kernel registration 생성
3. Python binding 생성

### 실험 2 — c10 의 Device 와 dtype 확인

```python
import torch

# c10::Device 를 Python 에서 확인
x = torch.randn(3, 4)
print(f"Device: {x.device}")  # device(type='cpu')
print(f"Dtype: {x.dtype}")  # torch.float32
print(f"Layout: {x.layout}")  # torch.strided

# 내부적으로 c10::TensorImpl 이 이들 정보 저장
# c10::Device, c10::ScalarType 이 dispatcher 에 의해 확인됨
```

### 실험 3 — ATen kernel registration 추적

```python
import torch

# torch.add 의 backend 별 kernel 등록 상태 확인
# (정교한 inspection 은 C++ 코드 필요, Python API 는 제한적)

# 간단한 확인: CPU vs CUDA kernel 호출 여부
x_cpu = torch.tensor([1.0, 2.0])
y_cpu = torch.tensor([3.0, 4.0])
result_cpu = torch.add(x_cpu, y_cpu)  # native::add_cpu 호출

if torch.cuda.is_available():
    x_cuda = x_cpu.cuda()
    y_cuda = y_cpu.cuda()
    result_cuda = torch.add(x_cuda, y_cuda)  # native::add_cuda 호출
    
    # 결과는 같지만 다른 kernel 실행
    print(torch.allclose(result_cpu, result_cuda.cpu()))  # True
```

### 실험 4 — Strided vs. Dense layout 의 dispatcher impact

```python
import torch

# strided tensor (dense)
x_dense = torch.randn(3, 4)
print(f"x_dense.is_contiguous(): {x_dense.is_contiguous()}")  # True

# transposed tensor (non-contiguous strided)
x_t = x_dense.T  # .transpose(0, 1)
print(f"x_t.is_contiguous(): {x_t.is_contiguous()}")  # False

# 하지만 둘 다 dispatcher 입장에서는 Strided layout
# → 같은 kernel 호출, stride 를 고려해 memory access pattern 조정
```

---

## 🔗 실전 활용

### 1. Custom Op 작성 시 native_functions.yaml 수정

새 op 를 만들면:

1. `aten/src/ATen/native/native_functions.yaml` 에 schema 추가
2. Code generator 실행 → C++ scaffold 생성
3. `aten/src/ATen/native/cpu/MyOp.cpp`, `aten/src/ATen/native/cuda/MyOp.cu` 구현
4. `TORCH_LIBRARY_IMPL` 로 각 backend 등록
5. 자동으로 Python binding 생성

### 2. Backend 별 kernel 최적화

같은 op 의 CPU 와 CUDA kernel 을 다르게 구현:
- CPU: OpenMP + BLAS
- CUDA: custom CUDA kernel 또는 cuBLAS

Dispatcher 가 device 자동 감지하고 올바른 kernel 호출.

### 3. Dtype promotion rules

ATen 은 자동으로 dtype promotion 처리:
- `torch.add(float32_tensor, int64_tensor)` → float32 로 promotion
- 이는 ATen 의 "promote_type" utility 에서 정의, 모든 op 가 일관되게 적용

### 4. torch.compile 과 c10/ATen

TorchDynamo 는 ATen 의 schema 정보 활용:
- op 의 type signature 검증
- AOTAutograd 에서 forward/backward graph differentiation 최적화
- Inductor 가 kernel fusion 시 dtype/device 정보 참고

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| c10 이 never change | 실제로는 backward compatibility 유지하며 천천히 진화 (e.g., DispatchKey 추가) |
| 모든 op 가 schema 에 정의됨 | 일부 실험적 op 는 수동 등록 (빠른 prototyping 위해) |
| Code generation 이 완벽 | edge case 에서 manual tweak 필요 (rare) |
| 모든 backend 에 kernel 등록 | 일부 op 는 특정 backend 에만 지원 (e.g., CUDA custom kernel) |

---

## 📌 핵심 정리

$$\boxed{\text{c10 (types)} \to \text{ATen (ops + schema)} \to \text{torch (Python)}}$$

| 계층 | 역할 | 의존성 |
|------|------|--------|
| **c10** | Tensor 의 metadata 정의, 0 의존성 | none |
| **ATen** | op 들, native_functions.yaml schema 기반 code gen | depends on c10 |
| **torch** | Python binding via pybind11, user-facing | depends on ATen |

**핵심**: Schema-driven design 으로 code generation 자동화 → 새 backend 추가/op 확장 비용 극소화.

---

## 🤔 생각해볼 문제

**문제 1** (기초): c10 이 ATen 에 의존하지 않는 것의 장점을 구체적 예시로 설명하라. 만약 c10 이 ATen 을 import 했다면 어떤 문제가 생기겠는가?

<details><summary>해설</summary>

**장점**:
- c10 은 "stable core" — ATen 에서 op 를 자주 추가/변경해도 c10 의 type definition 은 고정
- 새로운 backend (예: NEURON) 를 추가할 때, c10 은 수정 불필요

**c10 이 ATen 을 import 했다면**:
```
circular dependency risk:
c10 <- ATen <- c10 (bad!)
```
또는:
```
c10 become heavyweight:
c10 (originally 10KB) + ATen (1000KB) = unnecessary bloat
```

실제로 c10 은 TorchScript 의 기본 type system 으로도 사용되므로, lightweight 여야 함. $\square$

</details>

**문제 2** (심화): `native_functions.yaml` 에서 `dispatch: CompositeImplicitAutograd` vs explicit backend registration 의 차이점을 설명하고, 이것이 AOTAutograd 와 어떻게 연결되는가?

<details><summary>해설</summary>

**CompositeImplicitAutograd**:
```yaml
func: add.Tensor(...) -> Tensor
dispatch:
  CompositeImplicitAutograd: native::add  # 한 구현이 모든 backend 커버
```

의미: native::add 는 다른 ATen ops 의 composition 으로 정의됨:
```cpp
Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  return self + other * alpha;  // 내부적으로 mul, add 를 chained call
}
```

→ Dispatcher 는 "다른 ops 로 decompose" 한다고 알음 → AOTAutograd 가 automatic backward generation 가능

**Explicit backend registration**:
```yaml
func: add.Tensor(...) -> Tensor
dispatch:
  CPU: native::add_cpu
  CUDA: native::add_cuda
  Autograd: ...
```

의미: 각 backend 에 대해 직접 구현.

→ AOTAutograd 가 forward/backward 자동 분리 불가능 → manual backward 필수

**Modern approach** (PyTorch 2.0+): CompositeImplicitAutograd 로 정의 후 custom backward 는 torch::autograd::Function 으로 (optional). $\square$

</details>

**문제 3** (논문 비평): Edward Yang 의 "PyTorch Internals" blog post 에서 주장하는 "schema-driven" design 의 핵심 이점은 무엇이고, 이것이 JAX (functional, no schema) 와의 설계 차이점은?

<details><summary>해설</summary>

**PyTorch schema-driven**:
- 모든 op 가 declarative schema 로 정의
- Code generation 으로 type-safe binding, dispatcher registration 자동화
- 장점: compile-time 검증, 새 backend 추가 용이, 일관된 interface

**JAX functional**:
- op 들이 순수 Python 함수, schema 없음
- `jax.numpy.add = lambda x, y: ...` 스타일
- 장점: flexibility, rapid prototyping 용이
- 단점: manual dispatch 관리, type mismatch 런타임 발견

**차이**:
- PyTorch: "statically known" op 들의 빠른 실행, 많은 backend 지원
- JAX: "dynamically composable" ops, 함수형 프로그래밍 친화

실제로 JAX 도 최근 버전에서 op 의 정규화 시작 (규모 커지면서 schema-like 구조 도입). $\square$

</details>

---

<div align="center">

[◀ 이전](./01-dispatcher-architecture.md) | [📚 README](../README.md) | [다음 ▶](./03-autograd-dispatch.md)

</div>
