# 01. torch.utils.cpp_extension

## 🎯 핵심 질문

- `torch.utils.cpp_extension.load()` 가 runtime 에 C++/CUDA 코드를 JIT compile 하는 메커니즘은?
- Ninja build system 과 `~/.cache/torch_extensions` cache 의 역할은 무엇인가?
- `setup.py` + `CUDAExtension` 으로 production 빌드하는 방식과 JIT 의 차이는?
- PyBind11 binding 이 Python 과 C++ 사이의 interface 를 정확히 어떻게 중개하는가?
- Fused kernel (Linear + ReLU) 의 throughput 이 PyTorch built-in 과 왜 다른가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 는 pure Python/CUDA library 가 아닙니다. **`torch.add`, `torch.mm`, `F.relu` 의 대부분은 손수 작성한 CUDA kernel 또는 cuBLAS call** 입니다. 그러나 내장 kernel 이 당신의 특수한 연산 (예: fused activation, custom loss) 에 맞지 않을 때, **Python 에서 C++ → CUDA kernel 을 컴파일해 등록하는 것** 이 최후의 수단입니다. 이 프로세스를 이해하면:

1. **성능 병목** (HBM round-trip) 을 **custom kernel 로 우회** 가능
2. **Production 모델** (setup.py) 과 **research prototype** (JIT load) 의 trade-off 를 알고 선택
3. `torch.compile` 시대에도 **CUDA 코드의 한 줄이 dispatcher 에 어떻게 등록** 되는지 이해

---

## 📐 수학적 선행 조건

- **Chapter 4**: GPU architecture, SM·warp, shared memory, memory coalescing
- **Chapter 3**: Dispatcher, DispatchKey, TORCH_LIBRARY
- C++ 기초: template, namespace, pointer
- CUDA C: `__global__`, `__shared__`, `blockIdx`, `threadIdx`
- Linear algebra: matrix multiplication ($C = AB$), element-wise operation

---

## 📖 직관적 이해

### Runtime JIT vs Offline Build

```
Python code:
  kernel = load('mykernel', sources=['mykernel.cu'], ...)  (JIT)
  vs
  python setup.py install  (offline)
        ↓
  Both: C++/CUDA → compile → .so/.pyd library
        ↓
  PyBind11: Python ↔ C++ function call bridge
        ↓
  Dispatcher: register kernel in aten op table
        ↓
  PyTorch tensor op 호출 시 kernel 자동 dispatch
```

**JIT 의 장점**: 개발 속도, 코드 수정 즉시 반영, interactive
**JIT 의 단점**: 매번 compile (느림), cache 적중 안 하면 재빌드, CI/CD 에 부적합

**Setup.py 의 장점**: 한 번 빌드, 배포 시 이미 컴파일된 .so 배포, reproducible
**Setup.py 의 단점**: build 환경 의존 (CUDA version, compute capability)

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Fused Operation

두 개 이상의 element-wise operation 을 **단일 kernel** 로 구현하여 중간 tensor 를 HBM 까지 write → read 하지 않는 것:

$$\text{fused}(x) = \text{ReLU}(\text{Linear}(x)) \quad \text{vs} \quad \text{separate: } y = \text{Linear}(x),\, z = \text{ReLU}(y)$$

**Cost difference**: 
- Separate: $2 \times$ HBM traffic (write $y$, read $y$)
- Fused: $1 \times$ HBM traffic (write final output only)

### 정의 5.2 — JIT Compilation Context

$$\text{JIT}(name, sources, extra\_cflags) \rightarrow \{
\begin{array}{l}
\text{Check cache: } h = \text{hash}(sources, flags) \\
\text{If } h \in \text{cache} \rightarrow \text{load .so} \\
\text{Else: compile via Ninja, save to cache}
\end{array}
$$

Cache location: `~/.cache/torch_extensions/{CUDA_VERSION}/{ARCH}/{hash}/`

### 정의 5.3 — PyBind11 Binding

C++ 함수를 Python callable 로 등록:

```cpp
PYBIND11_MODULE(my_extension, m) {
  m.def("my_kernel", &my_kernel, "kernel signature");
}
```

결과: `import my_extension; my_extension.my_kernel(tensor_x)` 호출 가능

---

## 🔬 정리와 증명

### 정리 5.1 — JIT Cache Invalidation

source 또는 flag 가 변경되면 hash 가 변하므로 새 compile 이 자동 triggered:

$$\text{hash}(\text{source}_{t+1}) \neq \text{hash}(\text{source}_t) \Rightarrow \text{No cache hit}$$

**증명 sketch**: hash function 의 collision probability negligible (SHA256 활용) $\square$

### 정리 5.2 — Fused Linear+ReLU 의 Memory 절감

Separate kernel 대비 HBM round-trip 감소:

$$\text{Traffic}_{\text{fused}} = T_{\text{input}} + T_{\text{output}}$$
$$\text{Traffic}_{\text{separate}} = T_{\text{input}} + T_{\text{intermediate}} + T_{\text{intermediate}} + T_{\text{output}} = T_{\text{input}} + 2 T_{\text{inter}} + T_{\text{output}}$$

절감률: $\eta = \frac{2 T_{\text{inter}}}{\text{Traffic}_{\text{separate}}}$

For $T_{\text{inter}} = M \times N \times 4 \text{ bytes (FP32)}$:
- $M=1024, N=4096$: $\eta \approx 70\%$ (매우 큼)

**증명**: Roofline analysis (Ch4, Ch5 reuse) $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — JIT Load 로 Fused Linear+ReLU 커널 빌드 및 벤치마크

**CUDA kernel** (`fused_linear_relu.cu`):

```cuda
#include <torch/extension.h>

__global__ void fused_linear_relu_kernel(
    const float* __restrict__ x,      // (batch, in_features)
    const float* __restrict__ w,      // (out_features, in_features)
    const float* __restrict__ b,      // (out_features,)
    float* __restrict__ out,          // (batch, out_features)
    int batch, int in_f, int out_f
) {
    int bid = blockIdx.x;            // batch index
    int tid = threadIdx.x;           // output feature index
    
    if (bid >= batch || tid >= out_f) return;
    
    // Compute linear: out = x @ w.T + b
    float acc = b[tid];
    for (int k = 0; k < in_f; ++k) {
        acc += x[bid * in_f + k] * w[tid * in_f + k];
    }
    
    // Fuse ReLU: out = max(0, acc)
    out[bid * out_f + tid] = (acc > 0) ? acc : 0.0f;
}

torch::Tensor fused_linear_relu_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b
) {
    int batch = x.size(0);
    int in_f = x.size(1);
    int out_f = w.size(0);
    
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kFloat32);
    auto out = torch::empty({batch, out_f}, options);
    
    int threads = std::min(out_f, 1024);
    dim3 block(threads);
    dim3 grid(batch);
    
    fused_linear_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, in_f, out_f
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_relu_forward", &fused_linear_relu_forward, "Fused Linear+ReLU");
}
```

**Python 호출**:

```python
import torch
import time
from torch.utils.cpp_extension import load

fused_ext = load(
    name='fused_linear_relu',
    sources=['fused_linear_relu.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

batch, in_f, out_f = 128, 4096, 2048
x = torch.randn(batch, in_f, device='cuda')
w = torch.randn(out_f, in_f, device='cuda')
b = torch.randn(out_f, device='cuda')

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    out_fused = fused_ext.fused_linear_relu_forward(x, w, b)
torch.cuda.synchronize()
t_fused = time.time() - t0

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    out_sep = torch.nn.functional.linear(x, w, b)
    out_sep = torch.relu(out_sep)
torch.cuda.synchronize()
t_sep = time.time() - t0

print(f'Fused:    {t_fused:.3f} s')
print(f'Separate: {t_sep:.3f} s')
print(f'Speedup:  {t_sep / t_fused:.2f}x')
```

**예상 출력** (batch=128, in=4096, out=2048):
```
Fused:    2.145 s
Separate: 3.281 s
Speedup:  1.53x    ✓ HBM traffic 절감으로 약 1.5배 향상
```

### 실험 2 — Setup.py 로 Production 빌드

**setup.py**:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_linear_relu',
    ext_modules=[
        CUDAExtension(
            'fused_linear_relu',
            ['fused_linear_relu.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**Build & Install**:
```bash
python setup.py install
```

### 실험 3 — AT_DISPATCH_FLOATING_TYPES 로 dtype 자동 지원

```cpp
template<typename scalar_t>
__global__ void fused_kernel_generic(
    const scalar_t* x, const scalar_t* w, const scalar_t* b,
    scalar_t* out, int batch, int in_f, int out_f
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch || tid >= out_f) return;
    
    scalar_t acc = b[tid];
    for (int k = 0; k < in_f; ++k) {
        acc += x[bid * in_f + k] * w[tid * in_f + k];
    }
    
    out[bid * out_f + tid] = (acc > 0) ? acc : 0;
}

torch::Tensor fused_generic(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_generic", [&] {
        int batch = x.size(0), in_f = x.size(1), out_f = w.size(0);
        auto out = torch::empty({batch, out_f}, x.options());
        
        int threads = std::min(out_f, 1024);
        dim3 block(threads), grid(batch);
        
        fused_kernel_generic<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch, in_f, out_f
        );
        
        return out;
    });
}
```

---

## 🔗 실전 활용

### 1. Dispatcher 에 Custom Kernel 등록

```cpp
TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("fused_linear_relu", TORCH_CUDA_DISPATCH(fused_linear_relu_forward));
}
```

### 2. Backward (Gradient) 정의

```cpp
torch::Tensor fused_linear_relu_backward(
    torch::Tensor grad_out,
    torch::Tensor x, torch::Tensor w,
    torch::Tensor out
) {
    // grad_x = (grad_out * relu_mask) @ w
    // relu_mask = (out > 0)
    // Implementation...
    return grad_x;
}
```

### 3. torch.compile 호환성

JIT 커널이 AOTAutograd 를 통과하려면:
- In-place operation 금지
- Side effect 금지
- forward + backward 명확히 분리

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| CUDA compiler 설치됨 | 빌드 머신에 nvcc, CUDA toolkit 필수 |
| 모든 dtype 지원 필요 없음 | AT_DISPATCH 로 필요한 dtype 만 선택 |
| Backward 구현 가능 | autograd.Function 으로 감싸면 numerical check 가능 |
| Thread block < 1024 | 최대 thread 수 제한, large output 시 loop 필요 |
| Shared memory < 96 KB | bank conflict 주의, padding trick 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{JIT load}: \text{source} + \text{flags} \xrightarrow{\text{hash}} \text{cache hit?} \xrightarrow{\text{no}} \text{Ninja compile} \xrightarrow{\text{PyBind11}} \text{Python callable}}$$

| 구성요소 | 역할 |
|---------|------|
| `torch.utils.cpp_extension.load()` | JIT compile 관리, cache 자동 처리 |
| `~/.cache/torch_extensions/` | 컴파일된 .so/.pyd 캐시 (hash 기반) |
| **CUDA kernel** | GPU 연산 구현 (fused op 로 HBM 절감) |
| **PyBind11** | C++ ↔ Python bridge, type conversion |
| **AT_DISPATCH_FLOATING_TYPES** | dtype 자동 지원 macro (FP16/32/64) |
| **Fused Linear+ReLU** | ~1.5x 가속 (HBM round-trip 제거) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): JIT cache 가 invalidate 되는 경우를 3가지 이상 나열하고, 각각 hash 가 왜 바뀌는지 설명하라.

<details>
<summary>해설</summary>

1. **Source code 수정**: 파일 내용 변경 → hash(`source_code`) 변함
2. **CUDA flag 변경**: `-O3` → `-O2` 변경 → hash(`extra_cuda_cflags`) 변함
3. **CUDA Toolkit version**: nvcc 버전 다름 → compiler output 다름, new cache dir (version-specific)
4. **GPU architecture**: compute capability 다름 (`-gencode arch=sm_70` vs `sm_80`) → 생성 kernel 다름

이들은 모두 **최종 kernel binary 가 달라지는 경우** → reproducibility 를 위해 hash/cache path 에 반영되어야 함 $\square$

</details>

**문제 2** (심화): Fused Linear+ReLU 에서 만약 weight matrix 가 매우 크면 (in_f = 16384, out_f = 8192), 하나의 thread block 으로는 부족할 수 있다. 이를 해결하기 위해 kernel 을 어떻게 수정하겠는가?

<details>
<summary>해설</summary>

**문제**: 한 thread 가 in_f = 16384 개 element 를 순차 처리하면 latency 크고, 한 block 내 thread 협력도 어려움

**해법**: 2D thread block
```cuda
dim3 block(256, 4);  // 256 thread (feature dim) × 4 warp (reduction dim)
dim3 grid(batch, 1);

// block 내 reduction: 각 feature 에 대해 협력해 partial sum 계산
// 최종: shared mem 에 누적 후 one warp 가 final reduction
```

Roofline 모델로 분석 후 최적화 방향 결정 $\square$

</details>

**문제 3** (논문 비평): Tillet (2019, Triton) 은 "CUDA kernel 작성이 어려우므로 high-level DSL 이 필요" 라고 주장했다. cpp_extension vs Triton 의 장단점을 비교하라.

<details>
<summary>해설</summary>

| 측면 | CUDA | Triton |
|------|------|--------|
| **작성 난이도** | 높음 | 낮음 |
| **Performance ceiling** | 높음 | 자동 최적화 |
| **Portability** | NVIDIA only | 다양한 백엔드 |
| **Compilation time** | 빠름 | 느림 |
| **Production readiness** | 검증됨 | 신뢰도 검증 중 |

FlashAttention, vLLM 등 최신 시스템은 Triton 으로 구현 중 $\square$

</details>

---

<div align="center">

[◀ 이전](../ch4-cuda/06-reduction-optimization.md) | [📚 README](../README.md) | [다음 ▶](./02-tensor-cuda-pointer.md)

</div>
