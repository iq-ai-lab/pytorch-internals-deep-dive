# 03. View vs Copy 의 구분

## 🎯 핵심 질문

- `.view()` 는 항상 zero-copy 인가? 언제 실패하는가?
- `.reshape()` 는 view 인가 copy 인가? 어떻게 자동으로 선택되는가?
- `.contiguous()` 는 무엇을 강제하고 왜 필요한가?
- `.permute()` 후 다음 연산이 implicit copy 를 trigger 하는 메커니즘은?
- `.expand()` (broadcast) 와 `.repeat()` (copy) 의 차이는?

---

## 🔍 왜 View vs Copy 구분이 중요한가

PyTorch 사용자들은 자주 묻습니다: "shape 을 바꾸려면 `.view()` 를 써야 하나 `.reshape()` 를 써야 하나?"

**정답**: 대부분의 경우 자동으로 선택됩니다. 그러나 **memory-critical 응용**에서는 구분이 중요합니다:

1. **메모리 효율성**: Large tensor (GB 단위) 에서 unnecessary copy 는 OOM 을 야기합니다.
2. **Performance**: Copy 는 memory bandwidth 에 bound → 미리 예측 불가능하면 최적화 불가능.
3. **Gradient 흐름**: Backward 에서 view 는 재계산이 없고, copy 는 데이터 참조가 달라져 일부 고급 기법 (gradient checkpointing) 과 상호작용.

이 차이를 모르면 `data_ptr()` 가 변경되어 gradient 흐름이 끊어지는 subtle bug 를 만들 수 있습니다.

---

## 📐 수학적 선행 조건

- **Ch1-01, Ch1-02**: Tensor 의 Storage, stride, offset, contiguity
- **Computational Graph**: Forward → backward 에서 tensor reference 의 역할 (Ch2 에서 자세히)

---

## 📖 직관적 이해

### View: Metadata 만 조작

```
x = torch.arange(12).reshape(3, 4)
y = x.view(2, 6)

메모리:
[0 1 2 3 4 5 6 7 8 9 10 11]

x 해석:
row 0: [0 1 2 3]
row 1: [4 5 6 7]
row 2: [8 9 10 11]

y 해석 (같은 메모리):
row 0: [0 1 2 3 4 5]
row 1: [6 7 8 9 10 11]

x.data_ptr() == y.data_ptr()  # True
```

### Copy: 새로운 Storage 할당

```
x = torch.arange(12).reshape(3, 4)
y = x.contiguous()  # 이미 contiguous 면 no-op, 아니면 copy

메모리:
x: [0 1 2 3 4 5 6 7 8 9 10 11] (원본)
y: [0 1 2 3 4 5 6 7 8 9 10 11] (복사본, 다른 주소)

x.data_ptr() != y.data_ptr()  # True (다른 메모리)
```

### Reshape: 자동 선택

```python
x = torch.arange(12).reshape(3, 4)  # contiguous

# Case 1: 가능한 경우 → view
y = x.reshape(2, 6)  # view (zero-copy)
y = x.reshape(4, 3)  # view (zero-copy)

# Case 2: 불가능한 경우 → copy
x = torch.arange(12).reshape(3, 4).t()  # non-contiguous
y = x.reshape(2, 6)  # copy (메모리 새로 할당)
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — View Operation

Tensor `y` 가 `x` 의 **view** 는:

$$\text{Storage}(y) = \text{Storage}(x) \quad \text{and} \quad \text{data\_ptr}(y) = \text{data\_ptr}(x)$$

메모리 재할당 없이 **metadata 만 변경** 된 연산.

### 정의 3.2 — Copy Operation

Tensor `y` 가 `x` 의 **copy** 는:

$$\text{Storage}(y) \neq \text{Storage}(x)$$

새로운 Storage 에 원소 값들이 복제된 연산. 메모리 할당 비용 포함.

### 정의 3.3 — Reshape 의 동작

PyTorch 의 `.reshape(new_shape)` 는:

$$\text{reshape}(x) = \begin{cases}
\text{view}(x, \text{new\_shape}) & \text{if possible} \\
\text{contiguous}(x).\text{view}(\text{new\_shape}) & \text{otherwise}
\end{cases}$$

"가능"은 contiguity 또는 broadcasted stride 로 표현 가능함을 의미.

### 정의 3.4 — Expand vs Repeat

**Expand** (broadcast, zero-copy):
$$\text{expand}(x, \text{new\_shape})$$

Stride 에 0을 삽입하여 새로운 view 생성 (메모리 재사용).

**Repeat** (copy):
$$\text{repeat}(x, *\text{sizes})$$

Tensor 를 메모리에 복제 후 연결.

---

## 🔬 정리와 증명

### 정리 3.1 — View 가능 조건

Contiguous tensor 를 새로운 shape 으로 view 가능 할 필요충분조건:

$$\prod_k d_k = \prod_k d'_k$$

(총 원소 개수 동일)

**증명**: View 는 stride 재계산만 하므로, 원소 개수가 같으면 항상 새로운 stride 를 계산 가능. 역으로, 원소 개수가 다르면 불가능 $\square$

### 정리 3.2 — Non-contiguous Reshape 는 Copy

Non-contiguous tensor `x` 를 `x.reshape(shape)` 하면:

1. `.contiguous()` 호출 (implicit copy 발생)
2. 그 후 view

**증명**: Non-contiguous tensor 의 stride 로 새로운 shape 의 offset 을 계산할 수 없음 (stride 가 규칙을 따르지 않음). 따라서 먼저 메모리를 재배치해야 함 $\square$

### 정리 3.3 — Permute 후 다음 Op 에서 Implicit Copy

`.permute()` → non-contiguous. 다음 op 가 contiguous 요구 시:

```python
x = torch.randn(3, 4, device='cuda')
y = x.permute(1, 0)  # non-contiguous, stride = (1, 3)
z = torch.matmul(y, torch.randn(3, 5))  # 내부에서 y.contiguous() 호출 → copy
```

일부 op (mm, conv) 는 입력을 contiguous 로 요구. PyTorch 는 자동으로 처리하지만, copy 비용이 숨겨짐.

**증명**: 해당 BLAS/cuDNN kernel 이 non-contiguous buffer 를 받지 못하므로, dispatcher 가 자동으로 contiguous 변환 추가 (Ch3 에서 자세히) $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — View vs Copy: data_ptr 비교

```python
import torch

# Case 1: view
x = torch.arange(12.0).reshape(3, 4)
y = x.view(2, 6)
print(f"view: x.data_ptr() == y.data_ptr() ? {x.data_ptr() == y.data_ptr()}")
assert x.data_ptr() == y.data_ptr(), "view 이면 같은 메모리"

# Case 2: copy
z = x.contiguous()  # 이미 contiguous 면 아무것도 안 함
print(f"contiguous (already contig): x.data_ptr() == z.data_ptr() ? {x.data_ptr() == z.data_ptr()}")

# Case 3: non-contiguous → contiguous
x_noncontig = x.t()  # non-contiguous
w = x_noncontig.contiguous()
print(f"contiguous (from non-contig): ptr 다른가? {x_noncontig.data_ptr() != w.data_ptr()}")
assert x_noncontig.data_ptr() != w.data_ptr(), "copy 이면 다른 메모리"
```

**예상 출력**:
```
view: x.data_ptr() == y.data_ptr() ? True
contiguous (already contig): x.data_ptr() == z.data_ptr() ? True
contiguous (from non-contig): ptr 다른가? True
```

### 실험 2 — Reshape 자동 선택

```python
import torch

# Contiguous tensor: reshape 는 view
x = torch.arange(24.0).reshape(2, 3, 4)
y = x.reshape(6, 4)
print(f"Contiguous → reshape: same memory? {x.data_ptr() == y.data_ptr()}")

# Non-contiguous tensor: reshape 는 copy
x_t = x.permute(2, 1, 0)  # non-contiguous
print(f"Before reshape: is_contiguous={x_t.is_contiguous()}")
z = x_t.reshape(4, 6)
print(f"Non-contiguous → reshape: same memory? {x_t.data_ptr() == z.data_ptr()}")

# Manual confirmation
x_contig = x_t.contiguous()
z_manual = x_contig.view(4, 6)
print(f"Manual (contiguous + view): same result? {torch.allclose(z, z_manual)}")
```

**예상 출력**:
```
Contiguous → reshape: same memory? True
Before reshape: is_contiguous=False
Non-contiguous → reshape: same memory? False
Manual (contiguous + view): same result? True
```

### 실험 3 — Permute 후 Conv 에서 Implicit Copy

```python
import torch
import torch.nn.functional as F

x = torch.randn(2, 3, 32, 32, device='cuda')
weight = torch.randn(64, 3, 3, 3, device='cuda')

# Case 1: contiguous input
y_contig = F.conv2d(x, weight, padding=1)
print(f"Contiguous conv OK: {y_contig.shape}")

# Case 2: non-contiguous input (permute)
x_perm = x.permute(0, 2, 3, 1)  # (2, 32, 32, 3) - non-contiguous
print(f"After permute: is_contiguous={x_perm.is_contiguous()}")

# Conv2d 는 NCHW 를 기대, 우리는 NHWC 로 줌
# 내부적으로 contiguous 호출 → implicit copy
try:
    # 직접 호출하면 실패 또는 자동 변환
    y_perm = F.conv2d(x_perm, weight, padding=1)
    print(f"Permuted input conv: {y_perm.shape} (implicit contiguous 호출됨)")
except RuntimeError as e:
    print(f"Error: {e}")

# Memory allocation trace (Nsight 등으로 확인 가능)
# y_contig: x 로부터 zero-copy conv
# y_perm: x_perm 로부터 hidden contiguous() call → copy
```

**예상 출력**:
```
Contiguous conv OK: torch.Size([2, 64, 32, 32])
After permute: is_contiguous=False
Permuted input conv: torch.Size([2, 64, 32, 32]) (implicit contiguous 호출됨)
```

### 실험 4 — Expand vs Repeat

```python
import torch

# Expand: stride = 0 (zero-copy)
x = torch.randn(1, 3)
y = x.expand(32, 3)
print(f"expand: same data_ptr? {x.data_ptr() == y.data_ptr()}")
print(f"expand: y stride = {y.stride()}")  # (0, 1) - stride[0] = 0

# Repeat: 실제 복제 (copy)
z = x.repeat(32, 1)
print(f"\nrepeat: same data_ptr? {x.data_ptr() == z.data_ptr()}")
print(f"repeat: z stride = {z.stride()}")  # (3, 1) - 정상 stride

# 메모리 비교
print(f"\nMemory usage:")
print(f"expand: logical size {y.numel()}, storage size {y.storage().size()}")
print(f"repeat: logical size {z.numel()}, storage size {z.storage().size()}")
```

**예상 출력**:
```
expand: same data_ptr? True
expand: y stride = (0, 1)

repeat: same data_ptr? False
repeat: z stride = (3, 1)

Memory usage:
expand: logical size 96, storage size 3
repeat: logical size 96, storage size 96
```

---

## 🔗 실전 활용

### 1. Large Tensor 의 메모리 효율적 재배치

```python
import torch

# 나쁜 예: 불필요한 copy
x = torch.randn(10000, 10000, device='cuda')
y = x.permute(1, 0)  # non-contiguous
z = y.reshape(100000000)  # implicit copy!
# → peak memory = 2x

# 좋은 예
x = torch.randn(10000, 10000, device='cuda')
z = x.reshape(100000000)  # contiguous 이면 view
# → peak memory = 1x
```

### 2. Broadcast 최적화

```python
import torch

a = torch.randn(1, 1000)
b = torch.randn(32, 1)

# expand 사용 (zero-copy)
a_expanded = a.expand(32, 1000)
result = a_expanded + b.expand(32, 1000)  # broadcast, no copy

# repeat 대신 expand 권장
# a_repeated = a.repeat(32, 1)  # ← copy 발생
```

### 3. Gradient Checkpointing 과의 호환성

```python
import torch
from torch.utils.checkpoint import checkpoint

def forward(x):
    x = x.permute(...)  # non-contiguous
    # ... 많은 연산
    return x.sum()

x = torch.randn(1000, 1000, requires_grad=True)
# checkpoint 는 forward 를 다시 실행해 메모리 절약
# 그러나 non-contiguous tensor 는 implicit copy 가 숨어있을 수 있음
y = checkpoint(forward, x)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| View 는 항상 zero-copy | 매우 드문 경우 (advanced stride) 에는 copy 필요할 수 있음 |
| Reshape 자동 선택은 최적 | 명시적 `.view()` / `.contiguous()` 가 더 명확할 수 있음 |
| data_ptr 동일 = 같은 메모리 | View 이후 연쇄 연산에서 ptr 이 바뀔 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\text{reshape}(x) = \begin{cases}
\text{view} & \text{if C-contiguous} \\
\text{copy + view} & \text{otherwise}
\end{cases}}$$

| 연산 | 메모리 할당 | Stride 변경 | 용도 |
|------|-----------|-----------|------|
| `.view()` | No | Yes | contiguous 확실할 때 명시적 |
| `.reshape()` | Conditional | Yes | 항상 안전 (자동 선택) |
| `.contiguous()` | Conditional | Yes | non-contig → contig 강제 |
| `.expand()` | No | Yes (stride=0) | Broadcast |
| `.repeat()` | Yes | Yes | 실제 복제 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 코드에서 `y.data_ptr() == x.data_ptr()` 인가?
```python
x = torch.arange(12.0).reshape(3, 4)
y = x.view(4, 3)
```

<details>
<summary>해설</summary>

**예**: `view()` 는 항상 zero-copy. `x.is_contiguous() = True` 이므로 `y` 도 같은 메모리.

다만, `y.stride() = (3, 1)` 인데 shape (4, 3) 의 contiguous stride 는 `(3, 1)` 이므로 우연히 `y.is_contiguous() = True` 도 성립 $\square$

</details>

**문제 2** (심화): 다음 코드를 실행하면 몇 번의 copy 가 발생하는가?
```python
x = torch.randn(1000, 1000, device='cuda')
y = x.permute(1, 0)
z = y.reshape(1000000)
w = torch.nn.functional.linear(z.view(1000, 1000), 
                               torch.randn(256, 1000, device='cuda'))
```

<details>
<summary>해설</summary>

1. `y = x.permute(1, 0)`: view, 0 copy
2. `z = y.reshape(1000000)`: y 가 non-contiguous → 1 copy (y.contiguous() 호출)
3. `z.view(1000, 1000)`: view (z 는 이미 contiguous), 0 copy
4. `linear()`: contiguous input 가정, 최소 0 copy

**총: 1 copy** (reshape 에서)

만약 `z` 를 명시적으로 contiguous 하지 않았으면 reshape 에서 자동 처리 $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 eager execution 에서 view vs copy 의 선택이 왜 tensor dependency 가 아니라 **metadata 만**으로 결정되는가? Autograd 관점에서 이것이 어떤 이점을 주는가? (PyTorch internals documentation)

<details>
<summary>해설</summary>

**Why metadata-only decision**:
- View 의 핵심: Storage 공유 → backward 에서 gradient 계산이 자동으로 전파됨
- Copy 가 필요하면 그것도 op 로 등록 → backward 그래프에 노드 추가

**Autograd 이점**:
1. View 는 computational graph 에 노드 추가 안 함 → backward cost 없음
2. Copy 는 kernel 로 등록 → backward 가 transpose (또는 gather) 로 자동 생성

예:
```python
x = torch.randn(6, requires_grad=True)
y = x.view(2, 3)  # graph node 없음
loss = y.sum()
loss.backward()  # view 는 gradient shape 만 조정, computation 없음

z = x.permute(...)
w = z.reshape(...)  # reshape 안에 contiguous() → matmul 같은 op
# backward: reshape 는 view 의 역, contiguous 는 permute 의 역
```

(Edward Yang 의 "Tensor Considered Harmful" 및 PyTorch autograd graph 분석) $\square$

</details>

---

<div align="center">

[◀ 이전](./02-stride-and-layout.md) | [📚 README](../README.md) | [다음 ▶](./04-dtype-and-promotion.md)

</div>
