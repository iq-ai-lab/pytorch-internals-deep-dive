# 04. Dtype 계층과 Type Promotion

## 🎯 핵심 질문

- PyTorch 의 dtype 들은 어떤 계층 구조를 가지는가? (lattice)
- Type promotion 의 정확한 규칙은? (scalar vs tensor, complex vs real)
- `torch.result_type()` 은 무엇을 계산하는가?
- `torch.tensor(0.5) + 1` 에서 왜 결과가 float32 인가?
- FP16 vs BF16 vs TF32 의 대표 범위는? (Ch6 미리보기)

---

## 🔍 왜 Type Promotion 이 PyTorch 사용자에게 중요한가

Type promotion 은 종종 눈에 띄지 않는 자동 형변환으로, 대부분은 "예상대로" 동작합니다. 그러나:

1. **Silent precision loss**: `0.5 + 1` 이 float32 가 되어 float64 로 변환되지 않음
2. **Autograd 호환성**: Type mismatch 로 backward 가 실패할 수 있음 (tensor + scalar 에서)
3. **CUDA efficiency**: Float16/BF16 mixing 에서 자동 upcast 비용 발생
4. **Bug-prone**: `torch.ones(10, dtype=torch.int64) + 0.5` 의 결과 dtype 이 직관적이지 않음

이 규칙을 모르면 "왜 갑자기 dtype 이 달라져?" 하는 버그를 추적 불가능합니다.

---

## 📐 수학적 선행 조건

- **Number Theory**: Integer, rational, complex number 의 포함 관계
- **IEEE 754** (Ch6): Floating-point representation
- **Type System**: Lattice 이론 (partially ordered set)

---

## 📖 직관적 이해

### Dtype 계층 (Partial Order)

```
                complex128
                    |
    float64 ------ complex64
      |
    float32 ------- (float16, bfloat16)
      |
    int64 ------- int32 ------- int16 ------- int8
      |
    uint64 ------- uint32 ------- uint16 ------- uint8
      |
    bool
```

**규칙들**:
- **integer vs floating**: float > integer (항상 upcast to float)
- **complex vs real**: complex > real (항상 upcast to complex)
- **signed vs unsigned**: signed > unsigned (같은 크기면)
- **bit width**: 더 큰 bit width 가 승리 (보통)

### Scalar vs Tensor Promotion

```python
# scalar + tensor
0.5 (Python float = FP64) + torch.tensor([1, 2], dtype=torch.float32)
→ result dtype = float32  (tensor 의 dtype 을 따름)

# 역
torch.tensor(0.5, dtype=torch.float32) + 1 (Python int)
→ result dtype = float32  (tensor 의 dtype 을 따름)
```

**핵심**: Tensor 가 있으면 **tensor 의 dtype 을 따름** (보통).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Dtype 계층

PyTorch 의 dtype 계층을 부분 순서 (partial order) $\leq$ 로 정의:

$$\text{dtype}_1 \leq \text{dtype}_2 \iff \text{dtype}_2 \text{ 가 } \text{dtype}_1 \text{ 의 모든 값을 표현 가능}$$

예:
- `int8 ≤ int16 ≤ int32 ≤ int64`
- `float32 ≤ float64`
- `float32 ≤ complex64`
- `int32 < float32` (integer 의 모든 표현 가능)

### 정의 4.2 — Promotion Rule

두 dtype $d_1, d_2$ 의 promotion 결과 $d_1 \sqcup d_2$ (join, least upper bound):

**규칙**:
1. Scalar 가 없으면: 더 큰 dtype (bit width 가 크면) 또는 complex > real
2. Scalar 가 포함되면:
   - Tensor dtype 이 결정 (거의 항상)
   - Except: Python int/float → tensor 가 narrow 하면 upcast

### 정의 4.3 — Type Promotion Table

| 왼쪽 \ 오른쪽 | bool | int8 | int32 | int64 | float32 | float64 | complex64 | complex128 |
|---|---|---|---|---|---|---|---|---|
| **bool** | bool | int8 | int32 | int64 | float32 | float64 | complex64 | complex128 |
| **int8** | int8 | int8 | int32 | int64 | float32 | float64 | complex64 | complex128 |
| **int32** | int32 | int32 | int32 | int64 | float32 | float64 | complex64 | complex128 |
| **int64** | int64 | int64 | int64 | int64 | float32 | float64 | complex64 | complex128 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float64 | complex64 | complex128 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | complex128 | complex128 |

---

## 🔬 정리와 증명

### 정리 4.1 — Type Promotion 은 Associative

$(d_1 \sqcup d_2) \sqcup d_3 = d_1 \sqcup (d_2 \sqcup d_3)$

**증명**:
Type hierarchy 가 lattice (well-defined join) 를 형성하므로 associativity 자동 성립.

따라서 여러 operand 의 결과 dtype 은 순서에 무관 $\square$

### 정리 4.2 — Scalar + Tensor Promotion

Scalar $s$ (Python int/float) 와 tensor `x` ($\text{dtype} = d_x$) 의 연산:

$$\text{result\_dtype}(s, x) = d_x$$

즉, tensor dtype 이 항상 승리.

**증명**:
PyTorch dispatcher 는 tensor operand 를 먼저 확인. Scalar 는 tensor 의 dtype 으로 자동 변환됨 (implicit casting) $\square$

### 정리 4.3 — Transitivity of Promotion

$d_1 \leq d_2 \leq d_3$ 이면 $d_1 \leq d_3$.

**증명**: Definition 의 표현 가능성이 transitive 이므로 자명 $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — torch.result_type() 사용

```python
import torch

# 두 dtype 의 promotion 결과
print(f"int32 + float32 = {torch.result_type(torch.int32, torch.float32)}")
print(f"int64 + float64 = {torch.result_type(torch.int64, torch.float64)}")
print(f"float32 + complex64 = {torch.result_type(torch.float32, torch.complex64)}")
print(f"int8 + int32 = {torch.result_type(torch.int8, torch.int32)}")

# 예상
assert torch.result_type(torch.int32, torch.float32) == torch.float32
assert torch.result_type(torch.float32, torch.complex64) == torch.complex64
assert torch.result_type(torch.int8, torch.int32) == torch.int32
print("✓ torch.result_type 확인")
```

**예상 출력**:
```
int32 + float32 = torch.float32
int64 + float64 = torch.float64
float32 + complex64 = torch.complex64
int8 + int32 = torch.int32
✓ torch.result_type 확인
```

### 실험 2 — Scalar + Tensor Promotion

```python
import torch

# Python int/float + tensor
x_int = torch.tensor([1, 2, 3], dtype=torch.int32)
x_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
x_float64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

# Scalar + tensor 는 tensor dtype 을 따름
result1 = 0.5 + x_int  # 0.5 (float64 scalar) + int32 tensor
print(f"0.5 + int32 tensor: dtype = {result1.dtype}")
# → float32? float64? (PyTorch 1.x 에서는 float32 가 기본)

result2 = 1 + x_float32  # 1 (int scalar) + float32 tensor
print(f"1 + float32 tensor: dtype = {result2.dtype}")
# → float32 (tensor dtype 을 따름)

result3 = 1 + x_float64  # 1 + float64 tensor
print(f"1 + float64 tensor: dtype = {result3.dtype}")
# → float64 (tensor dtype 을 따름)

# 일관성 확인
print(f"\nTensor dtype 이 결과를 결정: {result3.dtype == x_float64.dtype}")
```

**예상 출력**:
```
0.5 + int32 tensor: dtype = torch.float32
1 + float32 tensor: dtype = torch.float32
1 + float64 tensor: dtype = torch.float64

Tensor dtype 이 결과를 결정: True
```

### 실험 3 — Tensor + Tensor Promotion

```python
import torch

x_int32 = torch.tensor([1, 2], dtype=torch.int32)
x_float32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
x_float64 = torch.tensor([1.0, 2.0], dtype=torch.float64)

# int32 + float32 → float32
result1 = x_int32 + x_float32
print(f"int32 + float32: dtype = {result1.dtype}")
assert result1.dtype == torch.float32

# float32 + float64 → float64
result2 = x_float32 + x_float64
print(f"float32 + float64: dtype = {result2.dtype}")
assert result2.dtype == torch.float64

# int32 + float64 → float64
result3 = x_int32 + x_float64
print(f"int32 + float64: dtype = {result3.dtype}")
assert result3.dtype == torch.float64

print("✓ Tensor-tensor promotion 규칙 확인")
```

**예상 출력**:
```
int32 + float32: dtype = torch.float32
float32 + float64: dtype = torch.float64
int32 + float64: dtype = torch.float64
✓ Tensor-tensor promotion 규칙 확인
```

### 실험 4 — Type Mismatch 에러 (Backward)

```python
import torch

# Backward 에서 dtype mismatch 시 에러 가능
x = torch.randn(5, dtype=torch.float32, requires_grad=True)
y = torch.randn(5, dtype=torch.float64, requires_grad=False)

# Forward: result dtype = float64 (x 를 upcast)
z = x + y

# Backward: x.grad 의 dtype 은?
loss = z.sum()
loss.backward()

print(f"x.grad dtype: {x.grad.dtype}")
print(f"Expected: {x.dtype} (original x dtype)")

# 일반적으로 x 의 dtype 이 유지됨
assert x.grad.dtype == x.dtype, "Gradient dtype 은 parameter dtype 을 따름"
print("✓ Gradient dtype 일관성")
```

**예상 출력**:
```
x.grad dtype: torch.float32
Expected: torch.float32 (original x dtype)
✓ Gradient dtype 일관성
```

### 실험 5 — torch.tensor(0.5) vs 0.5

```python
import torch

# 중요: torch.tensor() vs Python scalar 의 차이
py_float = 0.5  # Python float (내부적으로 float64)
torch_float = torch.tensor(0.5)  # torch tensor, dtype = float32 (default)

print(f"Python 0.5 type: {type(py_float)}")
print(f"torch.tensor(0.5) dtype: {torch_float.dtype}")

# 따라서
x = torch.ones(5, dtype=torch.float32)

# Case 1
result1 = x + 0.5  # Python float (float64 conceptually)
print(f"x (float32) + 0.5 (Python): dtype = {result1.dtype}")
# → float32 (tensor dtype 우위)

# Case 2
result2 = x + torch.tensor(0.5)  # torch tensor (float32)
print(f"x (float32) + torch.tensor(0.5) (float32): dtype = {result2.dtype}")
# → float32

# Case 3 - 혼란의 원인
result3 = torch.tensor(0.5) + 1  # tensor(0.5) dtype=float32, int scalar
print(f"torch.tensor(0.5) (float32) + 1 (int): dtype = {result3.dtype}")
# → float32 (tensor dtype)
```

**예상 출력**:
```
Python 0.5 type: <class 'float'>
torch.tensor(0.5) dtype: torch.float32
x (float32) + 0.5 (Python): dtype = torch.float32
x (float32) + torch.tensor(0.5) (float32): dtype = torch.float32
torch.tensor(0.5) (float32) + 1 (int): dtype = torch.float32
```

---

## 🔗 실전 활용

### 1. 정확성 보장 (float64 유지)

```python
import torch

# 수치적으로 민감한 연산은 float64 유지
x = torch.randn(100, dtype=torch.float64, requires_grad=True)
y = torch.randn(100, dtype=torch.float64)

# 작은 값 추가 (underflow 피하려면)
epsilon = 1e-10
result = torch.log(x + epsilon * y)  # epsilon 이 자동 float64 로 upcast
```

### 2. 혼합 정밀도 피하기

```python
import torch

# 나쁜 예: 자동 upcast 비용
x = torch.randn(1000, 1000, dtype=torch.float32, device='cuda')
w = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')

result = torch.matmul(x, w)  # x 는 float16 으로 downcast, 계산, 다시 float32?
# → implicit casting 비용 포함

# 좋은 예: 일관된 dtype
x = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')
w = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')
result = torch.matmul(x, w)  # 일관된 float16, 최적화 가능
```

### 3. Autograd 호환성

```python
import torch

# Type mismatch 는 backward 에서 문제 가능
x = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
y = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=False)

z = (x + y).sum()  # z dtype = float64
z.backward()  # x.grad 는 float32 유지 (automatic)

# 명시적 dtype 맞춤이 안전
x_casted = x.to(torch.float64)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Tensor dtype 항상 우위 | 특수 연산 (indexing) 에서는 다를 수 있음 |
| Promotion 은 항상 정의됨 | bool 과 complex 의 조합은 명확하지 않음 |
| Bit width 가 성능 지표 | float16 vs bfloat16 은 bit width 외에 정밀도 다름 (Ch6) |

---

## 📌 핵심 정리

$$\boxed{d_1 \sqcup d_2 = \text{join of } d_1, d_2 \text{ in dtype lattice}}$$

| Dtype | Range (대표) | Precision | 용도 |
|------|---------|-----------|------|
| float64 | $\sim 10^{-308}$ ~ $10^{308}$ | 15-17 decimal | 수치 분석 |
| float32 | $\sim 10^{-38}$ ~ $10^{38}$ | 6-7 decimal | **PyTorch default** |
| float16 | $6 \times 10^{-5}$ ~ $6.5 \times 10^4$ | 3-4 decimal | AMP (Ch6) |
| bfloat16 | $\sim 10^{-38}$ ~ $10^{38}$ | 2-3 decimal | AMP (modern) |
| int64, int32, int8 | $[-2^{n-1}, 2^{n-1}-1]$ | 정확 | 인덱싱, boolean |

---

## 🤔 생각해볼 문제

**문제 1** (기초): `torch.result_type(torch.int32, torch.float64)` 의 결과는?

<details>
<summary>해설</summary>

**float64**: int32 < float64 in the lattice (float > int). 

Promotion table 의 (int32, float64) 교점 = float64 $\square$

</details>

**문제 2** (심화): 다음 코드의 `x.grad.dtype` 은?
```python
x = torch.randn(10, dtype=torch.float32, requires_grad=True)
y = torch.randn(10, dtype=torch.float64, requires_grad=False)
z = (x + y).sum()
z.backward()
```

<details>
<summary>해설</summary>

1. `x + y`: float32 + float64 → float64
2. `z`: float64, requires_grad=True (upstream)
3. Backward: z 의 gradient 는 float64
4. **But `x.grad` 는?**: `x` 가 float32, parameter → grad 도 float32 유지

PyTorch 는 parameter dtype 을 유지. 따라서 **`x.grad.dtype = torch.float32`** $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 type promotion 규칙이 NumPy 와 어떻게 다른가? (NEP 에서 "type 계층" 논의) 왜 PyTorch 는 tensor dtype 을 더 강하게 우위에 두는가?

<details>
<summary>해설</summary>

**NumPy "type coercion" (traditional)**:
- Python scalar 와 NumPy scalar 를 구분
- `int + np.float64` → float64 (scalar 규칙 따름)

**PyTorch "type promotion" (modern)**:
- Tensor 가 있으면 tensor dtype 우위
- `torch.tensor(0.5) + 1` → float32 (tensor dtype)

**이유**:
1. **CUDA efficiency**: Tensor 가 특정 dtype 에 할당되어 있으면, 그 dtype 으로 계산하는 것이 가장 효율적
2. **Autograd**: Parameter 의 dtype 을 유지해야 gradient 도 같은 정밀도

(PyTorch type promotion design, Numerical Python conference 2020+) $\square$

</details>

---

<div align="center">

[◀ 이전](./03-view-vs-copy.md) | [📚 README](../README.md) | [다음 ▶](./05-device-and-cuda-context.md)

</div>
