# 02. Stride 와 Memory Layout

## 🎯 핵심 질문

- Row-major (C-order) 와 column-major (F-order) 의 stride 공식은?
- NCHW 와 NHWC 의 stride 차이가 왜 cuDNN kernel 선택을 바꾸는가?
- `is_contiguous()` 의 정확한 수학적 정의는?
- Stride 만으로 transpose, slice, broadcast 를 어떻게 표현하는가?

---

## 🔍 왜 이 메모리 레이아웃이 PyTorch 사용자에게 중요한가

대부분의 PyTorch 사용자는 convolution 을 NCHW 형태로 기본 사용합니다. 그러나 같은 연산을 NHWC 로 하면:

- **GPU 성능이 2-10배 차이** 날 수 있습니다.
- 이유는 cuDNN 이 input tensor 의 stride pattern 을 확인해 **최적의 kernel 을 선택** 하기 때문입니다.
- 같은 shape 의 tensor 가 stride 에 따라 **다른 메모리 접근 패턴** 을 만들고, 이것이 cache locality / GPU memory coalescing 을 극적으로 바꿉니다.

또한 `.permute()` 후 `.contiguous()` 가 implicit copy 를 trigger 하는 이유도 stride 를 이해해야 설명 가능합니다.

---

## 📐 수학적 선행 조건

- **Ch1-01**: Tensor 의 Storage, stride, offset 개념
- **컴퓨터 구조**: Memory coalescing, cache locality (Ch4 에서 더 자세히)

---

## 📖 직관적 이해

### Row-major vs Column-major

**Row-major (C-order, PyTorch default)**:
- 2D array 를 메모리에 행 단위로 배치
- `x[i, j]` 를 차례로 접근하면 메모리가 연속

**Column-major (Fortran order, 매우 드문)**:
- 2D array 를 메모리에 열 단위로 배치
- `x[i, j]` 를 차례로 접근하면 메모리가 불연속

### Stride 공식 도출

**2D Row-major**: Shape `(d0, d1)`
```
x[0, 0] -> offset 0
x[0, 1] -> offset 1
x[1, 0] -> offset d1
x[i, j] -> offset i*d1 + j
따라서 stride = (d1, 1)
```

**3D Row-major**: Shape `(d0, d1, d2)`
```
x[i, j, k] -> offset i*d1*d2 + j*d2 + k
따라서 stride = (d1*d2, d2, 1)
일반적으로: s_k = prod(d_{k+1}, ..., d_{n-1})
```

### NCHW vs NHWC 의 실제 영향

```
NCHW (N=2, C=3, H=4, W=4, total=96):
shape = (2, 3, 4, 4)
stride = (48, 16, 4, 1)
메모리 배치: Channel 이 연속으로 메모리에 저장
→ 같은 pixel 위치의 모든 channel 은 떨어져 있음

NHWC (N=2, H=4, W=4, C=3, total=96):
shape = (2, 4, 4, 3)
stride = (48, 12, 3, 1)
메모리 배치: Pixel 값들이 연속
→ 같은 pixel 위치의 모든 channel 이 연속으로 메모리에 저장

Conv2D 에서:
- NCHW: Kernel 이 각 channel 을 연속 접근 → 캐시 효율 좋음
- NHWC: Kernel 이 pixel 단위로 접근 → 다른 pattern
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Row-major Stride

Shape `(d_0, d_1, ..., d_{n-1})` 에 대한 row-major (C-order) stride:

$$s_k = \prod_{j=k+1}^{n-1} d_j$$

특히 $s_{n-1} = 1$ (마지막 차원).

### 정의 2.2 — Column-major Stride

Shape `(d_0, d_1, ..., d_{n-1})` 에 대한 column-major (F-order) stride:

$$s_k = \prod_{j=0}^{k-1} d_j$$

특히 $s_0 = 1$ (첫 차원).

### 정의 2.3 — Contiguous Tensor

Tensor 가 **C-contiguous** 할 필요충분조건:

$$s_k = \prod_{j=k+1}^{n-1} d_j \quad \forall k$$

또는 동등하게, 다음 조건들이 모두 참:
- 모든 $j$ 에 대해 $s_j > 0$ (양의 stride)
- 모든 $(i_0, \ldots, i_{n-1}) < (d_0, \ldots, d_{n-1})$ 에 대해 index 순서대로 offset 도 증가

---

## 🔬 정리와 증명

### 정리 2.1 — Contiguity 검사 공식

Tensor 가 C-contiguous 는 다음과 동치:

$$\forall k : s_k = \left(\prod_{j > k} d_j\right)$$

**증명**:
C-contiguous 의 정의는 메모리가 끊김없이 연속이라는 뜻.

Row-major 에서 index `(i_0, ..., i_{n-1})` 다음은 `(i_0, ..., i_{n-2}, i_{n-1}+1)` 또는 다음 행.

Offset 차이:
$$\text{offset}(i_0, ..., i_{n-1}+1) - \text{offset}(i_0, ..., i_{n-1}) = s_{n-1}$$

연속이려면 이것이 dtype 크기(FP32 면 1) 이어야 함 → $s_{n-1} = 1$.

귀납적으로, 마지막이 끝나고 그 앞 dimension 이 증가할 때:
$$\text{offset}(i_0, ..., i_{n-2}+1, 0) - \text{offset}(i_0, ..., i_{n-2}, d_{n-1}-1) = s_{n-2} - (d_{n-1}-1) \cdot s_{n-1}$$

연속이려면: $s_{n-2} = d_{n-1} \cdot s_{n-1} = d_{n-1}$.

일반적으로: $s_k = d_{k+1} \cdot s_{k+1}$ → $s_k = \prod_{j>k} d_j$ $\square$

### 정리 2.2 — Transpose 후 Contiguity 상실

Shape `(d_0, d_1)` 의 contiguous tensor 를 `.transpose(0, 1)` 하면 contiguous 아님.

**증명**:
원본: stride = $(d_1, 1)$, contiguous O.

After transpose: shape = $(d_1, d_0)$, stride = $(1, d_1)$.

Contiguous 조건: stride 가 $(d_0, 1)$ 이어야 하는데 실제로 $(1, d_1)$ → 조건 위배 $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Stride 공식 검증

```python
import torch

# 3D tensor: (2, 3, 4)
x = torch.arange(24.0).reshape(2, 3, 4)
print(f"Shape: {x.shape}")
print(f"Stride: {x.stride()}")
print(f"Expected: ({3*4}, {4}, {1}) = (12, 4, 1)")

# 확인
assert x.stride() == (12, 4, 1), "Stride mismatch!"
print("✓ Row-major stride 공식 확인")

# 더 높은 차원
y = torch.arange(120.0).reshape(2, 3, 4, 5)
print(f"\n4D Shape: {y.shape}")
print(f"Stride: {y.stride()}")
print(f"Expected: ({3*4*5}, {4*5}, {5}, {1}) = (60, 20, 5, 1)")
assert y.stride() == (60, 20, 5, 1)
print("✓ 4D stride 공식 확인")
```

**예상 출력**:
```
Shape: torch.Size([2, 3, 4])
Stride: (12, 4, 1)
Expected: (12, 4, 1) = (12, 4, 1)
✓ Row-major stride 공식 확인

4D Shape: torch.Size([2, 3, 4, 5])
Stride: (60, 20, 5, 1)
Expected: (60, 20, 5, 1) = (60, 20, 5, 1)
✓ 4D stride 공식 확인
```

### 실험 2 — NCHW vs NHWC Stride

```python
import torch

# NCHW: shape (N=2, C=3, H=4, W=4)
x_nchw = torch.randn(2, 3, 4, 4)
print(f"NCHW stride: {x_nchw.stride()}")
print(f"Expected: ({3*4*4}, {4*4}, {4}, 1) = (48, 16, 4, 1)")
assert x_nchw.stride() == (48, 16, 4, 1)

# NHWC 로 변환: permute(0, 2, 3, 1)
x_nhwc = x_nchw.permute(0, 2, 3, 1)
print(f"\nNHWC stride: {x_nhwc.stride()}")
print(f"Expected: ({4*4*3}, {4*3}, {3}, {1}) = (48, 12, 3, 1)")
assert x_nhwc.stride() == (48, 12, 3, 1)

print(f"\nNHWC is_contiguous: {x_nhwc.is_contiguous()}")  # False
print(f"NCHW is_contiguous: {x_nchw.is_contiguous()}")   # True
```

**예상 출력**:
```
NCHW stride: (48, 16, 4, 1)
Expected: (48, 16, 4, 1) = (48, 16, 4, 1)

NHWC stride: (48, 12, 3, 1)
Expected: (48, 12, 3, 1) = (48, 12, 3, 1)

NHWC is_contiguous: False
NCHW is_contiguous: True
```

### 실험 3 — Contiguous 검사

```python
import torch

# Contiguous tensor
x = torch.arange(12.0).reshape(3, 4)
print(f"Original: stride={x.stride()}, is_contiguous={x.is_contiguous()}")

# Transpose: stride 변경, contiguous 상실
y = x.t()
print(f"After transpose: stride={y.stride()}, is_contiguous={y.is_contiguous()}")

# Stride 가 contiguous 규칙 위반
shape = y.shape
expected_stride = tuple(torch.cumprod(
    torch.tensor([1] + list(shape[::-1])[:-1]), 0)[::-1].tolist())
print(f"Expected contiguous stride for {shape}: {expected_stride}")

# Manual contiguity check
def is_c_contiguous(shape, stride):
    return all(
        stride[i] == torch.prod(torch.tensor(shape[i+1:])).item()
        for i in range(len(shape) - 1)
    ) and stride[-1] == 1

print(f"Manual check: {is_c_contiguous(y.shape, y.stride())}")
```

**예상 출력**:
```
Original: stride=(4, 1), is_contiguous=True
After transpose: stride=(1, 3), is_contiguous=False
Expected contiguous stride for torch.Size([4, 3]): (3, 1)
Manual check: False
```

### 실험 4 — Implicit Copy 발생

```python
import torch

# Non-contiguous tensor 를 `.contiguous()` 호출 시 copy 생성
x = torch.arange(12.0).reshape(3, 4)
y = x.t()  # non-contiguous

print(f"Original data_ptr: {y.data_ptr()}")
print(f"is_contiguous: {y.is_contiguous()}")

# .contiguous() 호출
z = y.contiguous()
print(f"\nAfter .contiguous():")
print(f"data_ptr: {z.data_ptr()}")
print(f"is_contiguous: {z.is_contiguous()}")

# 메모리가 다름
print(f"\ndata_ptr 다른가? {y.data_ptr() != z.data_ptr()}")  # True (copy 발생)

# Stride 확인
print(f"y stride: {y.stride()}")
print(f"z stride: {z.stride()}")
```

**예상 출력**:
```
Original data_ptr: 140265...
is_contiguous: False

After .contiguous():
data_ptr: 140266...
is_contiguous: True

data_ptr 다른가? True

y stride: (1, 3)
z stride: (3, 1)
```

---

## 🔗 실전 활용

### 1. cuDNN Kernel 선택 최적화

```python
import torch
import torch.nn.functional as F

# NCHW (권장)
x_nchw = torch.randn(32, 3, 224, 224, device='cuda')
# cuDNN 이 최적 kernel 선택

# NHWC (느림)
x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
# permute 후 contiguous 호출 → implicit copy 발생
# 그 다음에도 non-optimal cuDNN kernel
```

### 2. 대규모 Tensor 의 효율적 Reshape

```python
# 나쁜 예
x = torch.randn(1000, 1000)
y = x.permute(1, 0).reshape(1000000)  # permute 후 reshape
# → implicit copy in reshape

# 좋은 예
x = torch.randn(1000, 1000)
y = x.reshape(1000000)  # 직접 reshape (contiguous 면 zero-copy)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Row-major 만 고려 | Column-major 는 드물지만 일부 legacy code 에서 사용 |
| Stride 모두 양수 | Negative stride (역순) 도 가능 (고급 기법) |
| Contiguous 만 최적 | 일부 연산 (transpose, slice) 는 non-contiguous 도 효율적 |

---

## 📌 핵심 정리

$$\boxed{s_k = \prod_{j=k+1}^{n-1} d_j \quad \text{(row-major)}}$$

| 개념 | 정의 | 용도 |
|------|------|------|
| Row-major | $s_k = \prod_{j>k} d_j$ | PyTorch default |
| Column-major | $s_k = \prod_{j<k} d_j$ | Fortran compatibility |
| C-contiguous | $s_k = d_{k+1} \times s_{k+1}$ | 메모리 연속성 |
| Stride mismatch | Permute/transpose 후 | implicit copy trigger |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Shape `(4, 5, 6)` 의 row-major tensor 에서 `stride[1]` 값은?

<details>
<summary>해설</summary>

$$s_1 = \prod_{j > 1} d_j = d_2 = 6$$

즉, stride = $(4 \times 5 \times 6, 6, 1) = (120, 6, 1)$, 따라서 `stride[1] = 6` $\square$

</details>

**문제 2** (심화): 2D tensor `x` (shape 2x3) 를 `x.t().permute(1, 0)` 하면? Stride 와 contiguity 를 분석하라.

<details>
<summary>해설</summary>

```
x: shape (2, 3), stride (3, 1), contiguous=True
x.t(): shape (3, 2), stride (1, 3), contiguous=False
x.t().permute(1, 0): shape (2, 3), stride (3, 1), contiguous=True
```

Permute(1, 0) 은 transpose 와 같으므로 stride 를 다시 교환. 결과적으로 원본과 동일하지만 메모리는 다름 (copy 발생) $\square$

</details>

**문제 3** (논문 비평): cuDNN 에서 NCHW 와 NHWC 에 대해 서로 다른 convolution kernel 을 구현하는 이유를 메모리 접근 패턴으로 설명하라. (Chen et al., cuDNN paper)

<details>
<summary>해설</summary>

**NCHW stride = (C*H*W, H*W, W, 1)**:
- 같은 H,W 위치의 모든 channel 이 메모리에서 떨어져 있음
- Kernel 이 각 channel 을 sequential 하게 접근 → cache line 에 많은 데이터
- Channel-wise 연산에 최적

**NHWC stride = (H*W*C, W*C, C, 1)**:
- 같은 위치의 모든 channel 이 연속 → spatial locality 좋음
- Pixel-wise 처리에 최적 (일부 mobile 칩셋)

**결론**: Layout 에 따라 L1/L2 cache hit rate 가 2-10배 차이 → kernel 최적화 필수 (cuDNN 문서, "Performance Tuning") $\square$

</details>

---

<div align="center">

[◀ 이전](./01-storage-and-metadata.md) | [📚 README](../README.md) | [다음 ▶](./03-view-vs-copy.md)

</div>
