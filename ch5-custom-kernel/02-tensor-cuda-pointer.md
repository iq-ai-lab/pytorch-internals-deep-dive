# 02. Tensor 와 CUDA Pointer

## 🎯 핵심 질문

- `tensor.data_ptr<float>()` 로 얻는 raw GPU pointer 는 무엇이고 안전성은?
- Stride 정보를 커널에 전달해야 하는 이유는? Index → offset 계산의 정확한 공식은?
- `AT_DISPATCH_FLOATING_TYPES` 매크로가 어떻게 template 을 자동 인스턴스화하는가?
- `tensor.is_contiguous()` 검사 후 `.contiguous()` fallback 은 언제 필수인가?
- `at::cuda::getCurrentCUDAStream()` 로 stream 을 명시하지 않는 것의 위험성은?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch tensor 는 high-level abstraction (size, dtype, device) 뒤에 **raw GPU memory pointer** 를 숨기고 있습니다. Custom kernel 을 작성하려면 이 **metadata → pointer 의 변환 과정** 을 정확히 이해해야 합니다:

1. **Memory safety**: `data_ptr()` 를 잘못 쓰면 buffer overflow, dangling pointer
2. **Non-contiguous tensor**: stride 가 default 가 아니면 index → offset 공식이 복잡해짐
3. **Type genericity**: 같은 kernel 을 FP16/32/64 에서 동시 지원하려면 `AT_DISPATCH` macro 필수
4. **Stream safety**: explicit stream 지정 없으면 default stream 으로 순차 실행 (성능 손실)

---

## 📐 수학적 선행 조건

- **Chapter 1**: Tensor metadata (size, stride, offset, dtype, device)
- **Chapter 4**: CUDA Stream, synchronization
- **Chapter 5-01**: PyBind11 binding, CUDA kernel basics
- 선형대수: row-major vs column-major indexing
- C++ templates: template specialization, instantiation

---

## 📖 직관적 이해

### Tensor → Raw Pointer 의 변환

```
PyTorch Tensor (High-level)
├── storage: raw GPU memory (1D array)
├── size: (d0, d1, ..., dn)
├── stride: (s0, s1, ..., sn)
└── offset: base

C++ API:
    data_ptr<T>()  →  (base + offset) as T*
    is_contiguous()  →  stride == default_stride?
    
CUDA kernel (Low-level):
    __global__ kernel(T* ptr, int *stride, int *size)
    {
        int idx[n] = {blockIdx.x, blockIdx.y, ..., threadIdx.x};
        int offset = base + sum(idx[k] * stride[k]);
        T value = ptr[offset];
    }
```

### Non-contiguous 와 Stride

```
Contiguous (C-order):
  Tensor shape (2, 3)
  stride = (3, 1)  ← natural row-major
  offset(i, j) = i*3 + j
  
  memory: [a00 a01 a02 | a10 a11 a12]

Transposed (non-contiguous):
  shape (3, 2) after transpose
  stride = (1, 3)  ← swapped!
  offset(i, j) = i*1 + j*3
  
  memory: [a00 | a10 | a20 | a01 | a11 | a21]  ← scattered
```

---

## ✏️ 엄밀한 정의

### 정의 5.4 — Raw Pointer 와 Offset 계산

Tensor $\mathbf{x}$ 의 element $(i_0, i_1, \ldots, i_{n-1})$ 의 메모리 주소:

$$\text{addr}(i_0, \ldots, i_{n-1}) = \text{base} + \text{offset} + \sum_{k=0}^{n-1} i_k \cdot s_k$$

where:
- `base` = `storage.data_ptr()` (untyped pointer)
- `offset` = `tensor.storage_offset()` (bytes)
- $s_k$ = `tensor.stride(k)` (element count, not bytes)
- $i_k \in [0, \text{size}(k))$

### 정의 5.5 — Contiguity Check

Tensor 가 **C-contiguous** iff:

$$s_k = \prod_{j > k} d_j \quad \forall k$$

where $d_j = \text{tensor.size}(j)$.

**F-contiguous** (Fortran-order): stride 가 column-major ordering 만족.

### 정의 5.6 — AT_DISPATCH Macro

```cpp
AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "kernel_name", [&] {
    // scalar_t 는 tensor의 dtype 에 자동 매칭
    // float / double / float16 중 하나
    kernel<scalar_t><<<grid, block>>>(ptr, ...);
})
```

Template specialization 을 런타임 dispatch 로 전환 (동적 타입 지원).

---

## 🔬 정리와 증명

### 정리 5.3 — Non-contiguous Tensor 의 Memory 접근 패턴

Transpose 후 kernel 이 naive linear access 하면 **L1/L2 cache miss** 급증:

$$\text{Miss rate}_{\text{noncontig}} = O(1), \quad \text{Miss rate}_{\text{contig}} = O(\epsilon)$$

where $\epsilon \approx 1/\text{cache line size}$.

**증명**: Cache line = 128 bytes → 32 floats. Contiguous 는 consecutive 4개 access 시 1회 line fetch. Transposed 는 매 접근마다 새 line fetch 필요 $\square$

### 정리 5.4 — Contiguity 의 필요충분조건

Tensor $\mathbf{x}$ 가 C-contiguous iff

$$\forall (i, j) : \text{offset}(i+1, j) - \text{offset}(i, j) = \text{size}(1)$$

(row 들이 메모리에 연속으로 배치)

**증명**: stride[0] = size[1] 정의로부터 직접 $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Contiguous vs Non-contiguous 포인터 비교

```cpp
#include <torch/extension.h>
#include <stdio.h>

torch::Tensor check_memory_layout(torch::Tensor x) {
    auto size_vec = x.sizes();
    auto stride_vec = x.strides();
    
    printf("Tensor size: ");
    for (auto s : size_vec) printf("%ld ", s);
    printf("\n");
    
    printf("Tensor stride: ");
    for (auto s : stride_vec) printf("%ld ", s);
    printf("\n");
    
    printf("Storage offset: %ld\n", x.storage_offset());
    printf("Data pointer: %p\n", x.data_ptr<float>());
    printf("Is contiguous (C): %d\n", x.is_contiguous());
    printf("Is contiguous (F): %d\n", x.is_contiguous(torch::MemoryFormat::ChannelsLast));
    
    return x;  // dummy return
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check_memory_layout", &check_memory_layout);
}
```

**Python test**:

```python
import torch
from torch.utils.cpp_extension import load

check_mem = load(
    name='check_memory_layout',
    sources=['check_memory_layout.cu'],
    verbose=False
)

# Contiguous
x_contig = torch.randn(4, 8)
print("=== Contiguous ===")
check_mem.check_memory_layout(x_contig)

# After transpose (non-contiguous)
x_trans = x_contig.t()
print("\n=== After transpose ===")
check_mem.check_memory_layout(x_trans)

# After contiguous()
x_rechunk = x_trans.contiguous()
print("\n=== After .contiguous() ===")
check_mem.check_memory_layout(x_rechunk)
```

**예상 출력**:
```
=== Contiguous ===
Tensor size: 4 8 
Tensor stride: 8 1 
Is contiguous (C): 1

=== After transpose ===
Tensor size: 8 4 
Tensor stride: 1 8 
Is contiguous (C): 0

=== After .contiguous() ===
Tensor size: 8 4 
Tensor stride: 4 1 
Is contiguous (C): 1
```

### 실험 2 — Stride 기반 Kernel 구현

```cuda
#include <torch/extension.h>

__global__ void strided_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* sizes,      // (dim0, dim1, ...)
    const int* strides_src, // (stride0, stride1, ...)
    const int* strides_dst,
    int ndim,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear idx to multi-dimensional indices
    int multi_idx[4] = {0, 0, 0, 0};
    int remain = idx;
    for (int d = ndim - 1; d >= 0; --d) {
        multi_idx[d] = remain % sizes[d];
        remain /= sizes[d];
    }
    
    // Calculate offsets
    int offset_src = 0, offset_dst = 0;
    for (int d = 0; d < ndim; ++d) {
        offset_src += multi_idx[d] * strides_src[d];
        offset_dst += multi_idx[d] * strides_dst[d];
    }
    
    dst[offset_dst] = src[offset_src];
}

torch::Tensor strided_copy(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.dim() <= 4, "Max 4D tensor");
    TORCH_CHECK(src.numel() == dst.numel(), "Size mismatch");
    
    // Prepare size/stride arrays
    std::vector<int> sizes_vec, strides_src_vec, strides_dst_vec;
    for (int d = 0; d < src.dim(); ++d) {
        sizes_vec.push_back(src.size(d));
        strides_src_vec.push_back(src.stride(d));
        strides_dst_vec.push_back(dst.stride(d));
    }
    
    // Copy to GPU
    auto sizes_gpu = torch::from_blob(
        sizes_vec.data(), {(int)sizes_vec.size()},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    ).to(src.device());
    auto strides_src_gpu = torch::from_blob(
        strides_src_vec.data(), {(int)strides_src_vec.size()},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    ).to(src.device());
    auto strides_dst_gpu = torch::from_blob(
        strides_dst_vec.data(), {(int)strides_dst_vec.size()},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    ).to(src.device());
    
    int threads = 256;
    int blocks = (src.numel() + threads - 1) / threads;
    
    strided_copy_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        sizes_gpu.data_ptr<int>(),
        strides_src_gpu.data_ptr<int>(),
        strides_dst_gpu.data_ptr<int>(),
        src.dim(),
        src.numel()
    );
    
    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_copy", &strided_copy);
}
```

**Test**:

```python
x = torch.randn(4, 8, device='cuda')
y_contig = torch.zeros_like(x)
y_trans = torch.zeros(8, 4, device='cuda')

# Copy to contiguous
strided_ext.strided_copy(x, y_contig)
assert torch.allclose(x, y_contig)

# Copy to transposed view
strided_ext.strided_copy(x, y_trans.t())
assert torch.allclose(x, y_trans.t())
print("✓ Strided copy works")
```

### 실험 3 — AT_DISPATCH 로 FP16/32/64 자동 지원

```cpp
template<typename scalar_t>
__global__ void reduction_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = x[idx] * (scalar_t)2.0;
}

torch::Tensor reduce_scalar(torch::Tensor x) {
    int n = x.numel();
    auto out = torch::empty_like(x);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reduce_scalar", [&] {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        reduction_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            n
        );
    });
    
    return out;
}
```

**Test**:

```python
for dtype in [torch.float16, torch.float32, torch.float64]:
    x = torch.randn(1000, dtype=dtype, device='cuda')
    y = reduce_scalar(x)
    assert torch.allclose(y, x * 2, rtol=1e-3)
    print(f"✓ {dtype} passed")
```

---

## 🔗 실전 활용

### 1. Safe Pointer Extraction

```cpp
torch::Tensor safe_kernel_call(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "Must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Must be float32");
    
    // Ensure contiguous for cache efficiency
    auto x_contig = x.contiguous();
    
    auto out = torch::empty_like(x_contig);
    my_kernel<<<grid, block>>>(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        ...
    );
    
    return out;
}
```

### 2. Stream-aware Execution

```cpp
torch::Tensor stream_aware_kernel(torch::Tensor x) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty_like(x);
    
    my_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        ...
    );
    
    return out;
}
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| `data_ptr()` 호출 후 tensor 유지됨 | dangling pointer risk, holder 필수 |
| Tensor lifetime 명확함 | shared_ptr 로 ownership track |
| Contiguous 체크 충분 | multiidx → offset 계산 여전히 복잡 |
| Stream 동기화 자동 | explicit sync 필요 시 `.cuda().synchronize()` |

---

## 📌 핵심 정리

$$\boxed{\text{offset}(i_0, \ldots, i_{n-1}) = \text{base} + \sum_k i_k \cdot s_k}$$

| 항목 | 정의 |
|------|------|
| `data_ptr<T>()` | raw GPU pointer (typed) |
| `storage_offset()` | 메모리 상 실제 시작점 (element) |
| `stride(k)` | 차원 k 에서 다음 element 까지 거리 |
| `is_contiguous()` | C-order 인지 확인 |
| `.contiguous()` | 필요 시 copy 로 stride normalize |
| `AT_DISPATCH_*` | dtype 자동 template dispatch |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 크기 (2, 3) 인 row-major tensor 에서 `stride = (3, 1)` 이고, transpose 후 `stride = (1, 3)` 이 되는 이유를 설명하라.

<details>
<summary>해설</summary>

Row-major (C-order):
- Row 들이 메모리에 연속 배치
- 한 행은 3개 원소 → row 점프 시 3칸 이동 (stride[0] = 3)
- 같은 행 내 column 은 인접 → 1칸 이동 (stride[1] = 1)

Transpose:
- 이제 column 들이 "행" 처럼 작동
- 원래 column 은 memory 에 3칸씩 떨어짐 → stride[0] = 1 (transpose 후 새로운 row)
- 원래 row 간격이 1칸 → stride[1] = 3 (transpose 후 새로운 column)

**중요**: stride 만 바뀌고 실제 storage 는 변하지 않음 (zero-copy) $\square$

</details>

**문제 2** (심화): `is_contiguous()` 검사 후에도 `.contiguous()` 를 호출해야 하는 경우가 있을까? 언제?

<details>
<summary>해설</summary>

**예시**: 복잡한 indexing 후 tensor 생성
```python
x = torch.randn(10, 10, device='cuda')
y = x[::2, 1::3]  # Strided slicing
print(y.is_contiguous())  # False
```

Non-contiguous 이지만 일부 kernel 은 stride 를 무시하고 linear access 가정:
- 누군가의 naive kernel 이 pointer 만 받고 직선 메모리 접근
- `.contiguous()` 호출 시에만 안전

Custom kernel 작성 시:
- stride-aware kernel 쓰면 `.contiguous()` 불필요 (Ch5-02 실험 2)
- stride 무시 kernel 쓰면 항상 check & convert $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 동적 shape 와 stride 설계는 NumPy 에서 영감을 받았다 (Harris, 2020). 이 설계가 어떻게 view operation (zero-copy) 을 가능하게 했고, 현재 `torch.compile` 의 functionalization 과정에서 어떤 도전과제를 만드는가?

<details>
<summary>해설</summary>

**장점**: stride-based layout 으로 view, transpose, slice 를 zero-copy 로 구현 가능
- 메모리 효율적 (배치 연산에서 매우 중요)
- dynamic shape 지원 (if-else 로 shape 결정 가능)

**도전과제** (torch.compile):
1. **Non-contiguous 의 컴파일**: stride 를 컴파일 시점에 알 수 없음 → Triton kernel 생성이 어려움
2. **Mutation 제거** (AOTAutograd): `y[x > 0] = val` 같은 scatter-like op 이 stride-aware 이어야 하는데, graph 최적화가 복잡해짐
3. **Inductor 의 fusion heuristic**: non-contiguous 에서 fusion 할 때 stride 관계를 추적해야 함

현재 TorchInductor 는 **contiguous 강제** 로 우회 (성능 trade-off) → future work: stride-aware code generation $\square$

</details>

---

<div align="center">

[◀ 이전](./01-cpp-extension.md) | [📚 README](../README.md) | [다음 ▶](./03-triton.md)

</div>
