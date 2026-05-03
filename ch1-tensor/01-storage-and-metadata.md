# 01. Tensor = Storage + Metadata

## 🎯 핵심 질문

- Tensor 객체는 메모리에 어떻게 저장되는가 — `(Storage, size, stride, offset, dtype, device)` 의 각 원소는 무엇인가?
- Index `(i_0, i_1, ..., i_{n-1})` 에서 메모리 offset 으로 변환하는 공식은?
- 같은 Storage 를 공유하면서 metadata 만 다른 두 tensor 가 존재할 수 있는가? (view 의 본질)
- `tensor.untyped_storage().data_ptr()` 로 raw 메모리 주소를 직접 확인할 수 있는가?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 사용자는 흔히 `Tensor` 를 "단순한 다차원 배열" 로 생각합니다. 그러나 **실제 구현에서 Tensor 는 6개의 분리된 원소의 묶음** 입니다: raw memory (Storage), 각 차원의 크기 (size), 각 차원의 step size (stride), 메모리의 시작점 (offset), 데이터 타입 (dtype), 그리고 device. 이 분리가 다음을 가능하게 합니다:

1. **무복사 view 연산**: `.view()`, `.transpose()`, `.narrow()` 가 Storage 를 건드리지 않고 **stride 와 shape 만 조작** → zero-copy
2. **cuDNN kernel 선택**: 같은 shape 의 tensor 가 stride 에 따라 NCHW vs NHWC 로 해석되고, 이것이 GPU kernel 의 성능을 수십 배 바꿈
3. **Automatic type promotion**: dtype 이 분리되어 있기 때문에 `0.5 + 1` 의 결과가 `float32` 가 되는 이유를 이해 가능
4. **Device abstraction**: 같은 코드가 CPU / CUDA 에서 동작하는 이유 (device metadata 만 다름)

이 구조를 모르면 `.contiguous()` 가 왜 필요한지, `.view()` 가 언제 실패하는지, stride 가 왜 중요한지 **설명 불가능** 합니다.

---

## 📐 수학적 선행 조건

- **Linear Algebra**: 다차원 배열 (tensor) 의 개념, row-major vs column-major 메모리 배치
- **메모리 이론**: Virtual memory, memory layout, pointer arithmetic
- (선택) C++ 메모리 모델 — `*ptr` 역참조, stride 계산

---

## 📖 직관적 이해

### Storage 와 Index Mapping

**Storage** 는 단순한 **1D byte array** 입니다. 예를 들어 FP32 (4 byte) 를 100개 저장하면 400 byte 의 raw memory.

**2D Tensor** `shape = (3, 4)` 를 생각해봅시다. 이 tensor 는 실제로:
- 12개의 FP32 값 (4 byte × 12 = 48 byte 의 Storage)
- `size = (3, 4)` — "이 storage 를 3×4 로 해석하라"
- `stride = (4, 1)` — Row-major layout: 첫 차원을 1칸 이동하면 storage 에서 4칸, 둘째 차원을 1칸 이동하면 1칸

### 물리적 메모리 그림

```
Storage (raw 1D memory)
[0][1][2][3][4][5][6][7][8][9][10][11]

Tensor (3, 4):
[0] [1] [2] [3]
[4] [5] [6] [7]
[8] [9][10][11]

Stride = (4, 1) 해석:
- tensor[0, 0] → offset = 0*4 + 0*1 = 0 → Storage[0]
- tensor[0, 1] → offset = 0*4 + 1*1 = 1 → Storage[1]
- tensor[1, 0] → offset = 1*4 + 0*1 = 4 → Storage[4]
- tensor[2, 3] → offset = 2*4 + 3*1 = 11 → Storage[11]
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Tensor 구조

PyTorch 의 `Tensor` 는 다음 6원소 튜플로 정의됩니다:

$$\text{Tensor} = (\text{Storage}, \text{size}, \text{stride}, \text{offset}, \text{dtype}, \text{device})$$

- **Storage**: contiguous raw memory (1D byte array), reference-counted
- **size** $\in \mathbb{N}^n$: 각 차원의 크기 $(d_0, d_1, \ldots, d_{n-1})$
- **stride** $\in \mathbb{Z}^n$: 각 차원에서 1칸 이동 시 Storage 내 byte offset $(s_0, s_1, \ldots, s_{n-1})$
- **offset** $\in \mathbb{Z}$: Storage 내 시작 위치 (bytes)
- **dtype** $\in \{\text{float32}, \text{float16}, \ldots\}$: 원소 크기 결정 (4 byte, 2 byte, ...)
- **device** $\in \{\text{cpu}, \text{cuda:0}, \ldots\}$: memory location

### 정의 1.2 — Index to Offset Mapping

Tensor 의 index $(i_0, i_1, \ldots, i_{n-1})$ 에 대응하는 Storage 내 **byte offset** 은:

$$\text{byte\_offset}(i_0, \ldots, i_{n-1}) = \text{offset} + \sum_{k=0}^{n-1} i_k \cdot s_k$$

원소의 실제 메모리 주소 (float pointer):

$$\text{ptr} = \text{data\_ptr} + \frac{\text{byte\_offset}}{\text{sizeof(dtype)}}$$

### 정의 1.3 — View (무복사 tensor)

두 tensor $T_1, T_2$ 가 같은 Storage 를 공유하고 $(offset_1, size_1, stride_1) \neq (offset_2, size_2, stride_2)$ 일 때:
$$T_2 \text{ is a view of } T_1 \iff \text{Storage}(T_2) = \text{Storage}(T_1)$$

이 경우 `data_ptr(T_1) == data_ptr(T_2)` (같은 메모리를 가리킴).

---

## 🔬 정리와 증명

### 정리 1.1 — View 는 Storage 공유

`.view()` 또는 `.transpose()` 같은 연산으로 생성된 tensor 는 Storage 를 공유합니다.

**증명**:
1. `.view()` 는 매개변수로 new_shape 만 받음. Stride 를 재계산해도 Storage 는 변경 없음.
2. `.transpose(dim0, dim1)` 은 `stride[dim0]` 와 `stride[dim1]` 을 교환하고 `size[dim0]` 와 `size[dim1]` 을 교환. Storage 변경 없음.
3. 따라서 `storage.data_ptr()` 가 동일 → 메모리 위치 동일 $\square$

### 정리 1.2 — Contiguous 조건

Tensor 가 **C-contiguous (row-major)** 일 필요충분조건:

$$s_k = \prod_{j > k} d_j \quad \forall k \in [0, n)$$

즉, stride 가 뒤쪽 차원들의 크기의 곱과 일치해야 함.

**증명**:
- 연속된 메모리 배치에서 뒤의 차원을 1칸 이동하면 1 byte 이동 (dtype 고려) → $s_{n-1} = 1$
- 그 앞 차원을 1칸 이동하면 $d_{n-1}$ byte 이동 → $s_{n-2} = d_{n-1}$
- 귀납: $s_k = \prod_{j=k+1}^{n-1} d_j$ $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Storage 공유 확인 (view)

```python
import torch

# 3×4 tensor 생성
x = torch.arange(12.0).reshape(3, 4)
print(f"x:\n{x}")
print(f"x.stride() = {x.stride()}")
print(f"x.data_ptr() = {x.data_ptr()}")

# View: 같은 storage, 다른 shape/stride
y = x.view(2, 6)
print(f"\ny (view):\n{y}")
print(f"y.stride() = {y.stride()}")
print(f"y.data_ptr() = {y.data_ptr()}")

# Storage 공유 확인
print(f"\nStorage 공유? {x.data_ptr() == y.data_ptr()}")

# Storage 객체 직접 비교
print(f"untyped_storage 동일? {x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()}")

# 메모리 수정 시 영향
y[0, 0] = 999
print(f"\ny[0,0] = 999 후 x[0,0] = {x[0, 0]}")  # 999 로 변경됨 (같은 메모리)
```

**예상 출력**:
```
x:
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
x.stride() = (4, 1)

y (view):
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
        [ 6.,  7.,  8.,  9., 10., 11.]])
y.stride() = (6, 1)

Storage 공유? True
untyped_storage 동일? True

y[0,0] = 999 후 x[0,0] = 999.0
```

### 실험 2 — Stride 계산 (row-major vs column-major)

```python
import torch

# Row-major (C-order, default)
x_c = torch.arange(24.0).reshape(2, 3, 4)
print(f"C-order shape={x_c.shape}, stride={x_c.stride()}")
print(f"Expected stride: (3*4={3*4}, 4, 1) = {(12, 4, 1)}")

# Column-major (F-order, Fortran)
x_f = torch.arange(24.0).reshape(4, 3, 2).permute(2, 1, 0)  # (2, 3, 4)
print(f"\nPermute 후 shape={x_f.shape}, stride={x_f.stride()}")
print(f"is_contiguous: {x_f.is_contiguous()}")

# Contiguous 로 만들면?
x_f_contig = x_f.contiguous()
print(f"Contiguous 후 stride={x_f_contig.stride()}")
```

**예상 출력**:
```
C-order shape=torch.Size([2, 3, 4]), stride=(12, 4, 1)
Expected stride: (3*4=12, 4, 1) = (12, 4, 1)

Permute 후 shape=torch.Size([2, 3, 4]), stride=(2, 8, 24)
is_contiguous: False

Contiguous 후 stride=(12, 4, 1)
```

### 실험 3 — Index to Offset 검증

```python
import torch

x = torch.arange(12.0).reshape(3, 4)
stride = x.stride()
dtype_size = 4  # FP32 = 4 bytes

# 수작업으로 offset 계산
def manual_offset(i0, i1, stride=stride):
    byte_offset = i0 * stride[0] + i1 * stride[1]
    return byte_offset // dtype_size  # float 단위

# 검증
for i in range(3):
    for j in range(4):
        manual_idx = manual_offset(i, j)
        tensor_val = x[i, j].item()
        print(f"x[{i},{j}] = {tensor_val}, computed offset = {manual_idx}")
```

**예상 출력**:
```
x[0,0] = 0.0, computed offset = 0
x[0,1] = 1.0, computed offset = 1
x[0,2] = 2.0, computed offset = 2
x[0,3] = 3.0, computed offset = 3
x[1,0] = 4.0, computed offset = 4
...
```

### 실험 4 — Transpose 의 zero-copy 검증

```python
import torch

x = torch.arange(6.0).reshape(2, 3)
print(f"x shape={x.shape}, stride={x.stride()}")
print(f"x.data_ptr() = {x.data_ptr()}")

y = x.t()
print(f"\ny (transpose):")
print(f"y shape={y.shape}, stride={y.stride()}")
print(f"y.data_ptr() = {y.data_ptr()}")

print(f"\ndata_ptr 동일 (zero-copy)? {x.data_ptr() == y.data_ptr()}")

# 수정 시 영향
y[0, 0] = 999
print(f"y[0,0] = 999 후 x[0,0] = {x[0, 0]}")
```

**예상 출력**:
```
x shape=torch.Size([2, 3]), stride=(3, 1)

y (transpose):
y shape=torch.Size([3, 2]), stride=(1, 3)

data_ptr 동일 (zero-copy)? True

y[0,0] = 999 후 x[0,0] = 999.0
```

---

## 🔗 실전 활용

### 1. Stride 기반 Optimization

cuDNN convolution 은 입력의 stride pattern 을 확인해 최적의 kernel 을 선택합니다. NCHW 와 NHWC stride 가 다르면 전혀 다른 GPU kernel 이 호출되어 성능이 10배 이상 차이날 수 있습니다.

### 2. 메모리 절약

`.view()` 는 Storage 를 공유하므로 메모리 할당이 없습니다. 따라서 대규모 배치 데이터를 reshape 할 때 `.view()` 를 사용하면 메모리 효율적입니다.

### 3. Automatic Broadcasting

`a.shape = (1, 3)`, `b.shape = (2, 1)` 일 때 `a + b` 는 내부적으로 stride = 0 인 broadcast 를 사용해 메모리 할당 없이 계산합니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Storage 는 contiguous 한 1D array | Non-contiguous tensor 는 implicit copy 필요 (Ch1-03) |
| Offset 은 항상 유효 | Strided access 시 memory layout 에 따라 cache efficiency 차이 |
| dtype 은 고정 | Type casting 은 새로운 Storage 할당 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{offset}(i_0, \ldots, i_{n-1}) = \text{base} + \sum_{k=0}^{n-1} i_k \cdot s_k}$$

| 양 | 정의 | 역할 |
|----|------|------|
| Storage | 1D byte array | raw 메모리 |
| size | $(d_0, d_1, \ldots)$ | 논리적 형태 |
| stride | $(s_0, s_1, \ldots)$ | index → offset 변환 |
| offset | base pointer | 시작점 |
| dtype | element type | 크기 결정 |
| device | cpu/cuda | 메모리 위치 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Shape `(3, 4)` 의 row-major tensor 에서 `x[2, 3]` 의 byte offset 을 stride `(4, 1)` 을 사용해 계산하라. 이는 Storage 에서 정확히 어느 byte 위치인가?

<details>
<summary>해설</summary>

Stride = (4, 1) 이면 각 원소는 FP32 (4 byte) 이므로:

$$\text{offset}(2, 3) = 2 \times 4 + 3 \times 1 = 11 \text{ (float 단위)}$$

Byte 단위로:
$$11 \times 4 = 44 \text{ bytes}$$

따라서 Storage 의 44번째 byte 부터 4 byte 를 읽으면 `x[2, 3]` 의 값 $\square$

</details>

**문제 2** (심화): Tensor `x` 가 shape `(2, 3, 4)` 이고 transpose `x.permute(2, 0, 1)` 후 새로운 shape `(4, 2, 3)` 을 갖는다. 원본 stride 가 `(12, 4, 1)` 이면 permute 후 stride 는?

<details>
<summary>해설</summary>

Permute(2, 0, 1) 는 dimension 을 재배열:
- 새 dimension 0 = 원래 dimension 2 → stride[2] = 1
- 새 dimension 1 = 원래 dimension 0 → stride[0] = 12
- 새 dimension 2 = 원래 dimension 1 → stride[1] = 4

따라서 새로운 stride = (1, 12, 4) $\square$

</details>

**문제 3** (논문 비평): Edward Yang 의 PyTorch Tensor 구조 분석에서 Storage + Metadata 분리가 어떻게 modern AD system 과 호환되는지 설명하라. `.view()` 후 `.backward()` 에서 gradient shape 이 어떻게 결정되는가?

<details>
<summary>해설</summary>

**Storage-Metadata 분리의 AD 호환성**:
1. `.view(shape)` 는 Storage 공유, shape/stride 만 변경
2. Autograd 에서 view 는 **parameter 가 shape** 인 연산으로 등록됨
3. Backward: upstream gradient 를 원본 shape 으로 reshape 만 하면 됨 (no computation)
4. 따라서 `.view()` 는 **computational cost 가 0** 인 미분 불가능 연산

예시:
```python
x = torch.randn(6, requires_grad=True)
y = x.view(2, 3)
loss = y.sum()
loss.backward()
# x.grad shape = (6,) — view 의 shape 은 backward 에 영향 없음
```

(Edward Yang, "Tensor Considered Harmful" 및 PyTorch autograd C++ 코드) $\square$

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-stride-and-layout.md)

</div>
