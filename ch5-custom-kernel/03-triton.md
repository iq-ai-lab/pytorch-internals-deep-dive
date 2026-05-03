# 03. Triton — Python-level GPU Programming

## 🎯 핵심 질문

- `@triton.jit` 로 Python 함수를 GPU kernel 로 변환하는 메커니즘은?
- Block-level abstraction (thread 아닌 block 단위) 이 CUDA 보다 무엇을 단순화하는가?
- Pointer arithmetic 이 numpy-like 하다는 것이 무엇을 의미하는가?
- `@triton.autotune(configs=[...])` 이 block size · num_warps 를 자동 탐색하는 원리는?
- FlashAttention (Dao 2022) 과 vLLM 이 Triton 으로 구현된 이유는?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

2019년 Tillet 이 Triton 을 제안한 이래, **"CUDA 를 잘못 쓰면 10배 느리지만, Triton 으로 쓰면 자동 최적화"** 가 현실이 되고 있습니다. 특히:

1. **FlashAttention (Dao 2022)**: 4번의 Triton kernel 로 자체 CUDA 구현 대비 성능 동일 + 코드 간결
2. **vLLM, text-generation-webui**: Triton 으로 decode-stage kernel 구현 → maintainability ↑
3. **PyTorch 2.0 (Inductor)**: Graph compiler 가 자동으로 Triton kernel 생성 (Ch7)

Triton 을 이해하면:
- custom kernel 의 유지보수 비용 ↓
- perf ceiling 과 implementation complexity 의 trade-off 를 수치적으로 이해
- `torch.compile` 이 생성한 kernel 을 읽고 디버그 가능

---

## 📐 수학적 선행 조건

- **Chapter 4**: GPU memory hierarchy, coalescing, bank conflict, warp divergence
- **Chapter 5-01**: kernel fusion 개념, HBM round-trip
- CUDA C 기초: grid, block, shared memory (선택)
- NumPy: broadcasting, strides, slicing
- Python decorator pattern

---

## 📖 직관적 이해

### CUDA vs Triton 의 추상화 레벨

```
CUDA (Low-level):
  ├── Thread (32개씩 warp 로 묶임)
  ├── Warp shuffle, shared memory bank conflict 직접 관리
  ├── Coalescing 을 위해 memory access pattern 설계
  └── 개발자가 최적화의 모든 단계 담당

Triton (High-level):
  ├── Block (tile) 단위로 생각
  ├── "Load this tile of data, process, store"
  ├── Thread block 내부의 scheduling 은 Triton compiler 가 자동 최적화
  └── 개발자는 "무엇을" 구현, "어떻게"는 자동화
```

### Vector Addition Example

**CUDA**:
```cuda
__global__ void add_kernel(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) z[idx] = x[idx] + y[idx];
}
// grid/block size 선택, coalescing 확인, 성능 측정 필수
```

**Triton**:
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)
    z = x + y
    tl.store(z_ptr + idx, z)
# BLOCK_SIZE 는 autotuning 으로 자동 결정
```

---

## ✏️ 엄밀한 정의

### 정의 5.7 — Triton Kernel

```python
@triton.jit
def kernel(ptr0, ptr1, ..., PARAM0: tl.constexpr, ...):
    # 1. Index 계산 (block-level)
    idx = tl.arange(0, BLOCK_SIZE)
    
    # 2. Load (coalesced automatically)
    x = tl.load(ptr0 + idx, mask=idx < N)
    
    # 3. Compute (element-wise, within tile)
    y = x * 2
    
    # 4. Store (coalesced automatically)
    tl.store(ptr0 + idx, y, mask=idx < N)
```

**핵심**: Thread-level 아닌 **block-level 추상화** — compiler 가 thread ↔ block mapping 자동으로

### 정의 5.8 — Autotuning

$$\text{best\_config} = \arg\max_{(BS, NW) \in \text{configs}} \text{throughput}(BS, NW)$$

where:
- `BS` = block size (tile size)
- `NW` = num_warps (parallelism within block)

Compiler 가 각 (BS, NW) 에 대해 kernel 컴파일 → grid search

---

## 🔬 정리와 증명

### 정리 5.5 — Triton Compiler 의 자동 coalescing

Triton kernel 이 `tl.load(ptr + idx)` 로 연속 주소를 접근하면, compiler 가 자동으로 **coalesced memory transaction** 을 생성:

$$\text{Transactions} = \lceil \frac{N \cdot 4 \text{ bytes}}{128 \text{ bytes/transaction}} \rceil$$

**증명 sketch**: Triton LLVM backend 가 pointer arithmetic 을 분석 → coalescing 패턴 인식 → 자동으로 cache-friendly instruction sequence 생성 $\square$

### 정리 5.6 — Softmax Fusion 의 성능 이득

Fused softmax (Dao 2022):

$$\text{HBM access} = 4N \quad \text{(memory-bound 상태)}$$

vs. Naive 3-pass (exp, sum, div):

$$\text{HBM access} = 12N \quad \text{(3배 traffic)}$$

**증명**: Fused kernel 이 intermediate (exp 결과) 를 shared mem 에 보존, 다음 pass 에서 reuse $\square$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — Vector Add (Triton)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)

def add_triton(x, y):
    z = torch.empty_like(x)
    n = x.numel()
    
    def grid(meta):
        return (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        x, y, z, N=n,
        BLOCK_SIZE=1024
    )
    return z

# Test
x = torch.randn(1000000, device='cuda')
y = torch.randn(1000000, device='cuda')
z_triton = add_triton(x, y)
z_torch = x + y

assert torch.allclose(z_triton, z_torch, rtol=1e-5)
print("✓ Vector add works")
```

### 실험 2 — Matrix Multiplication (Triton)

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Tile indices
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K
    for k in range(0, triton.cdiv(K, BLOCK_K)):
        k_idx = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A tile
        a_idx = m_idx[:, None] * stride_am + k_idx[None, :] * stride_ak
        a_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(a_ptr + a_idx, mask=a_mask, other=0.0)
        
        # Load B tile
        b_idx = k_idx[:, None] * stride_bk + n_idx[None, :] * stride_bn
        b_mask = (k_idx[:, None] < K) & (n_idx[None, :] < N)
        b = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
        
        # Compute (tile)
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Store C
    c_idx = m_idx[:, None] * stride_cm + n_idx[None, :] * stride_cn
    c_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    tl.store(c_ptr + c_idx, acc, mask=c_mask)

def matmul_triton(A, B):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N, meta['BLOCK_N']))
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    return C

# Test
A = torch.randn(512, 512, device='cuda')
B = torch.randn(512, 512, device='cuda')
C_triton = matmul_triton(A, B)
C_torch = torch.matmul(A, B)

print(f"Relative error: {(C_triton - C_torch).abs().max() / C_torch.abs().max():.6f}")
```

### 실험 3 — Autotuning Block Size 와 Num Warps

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["N"]
)
@triton.jit
def reduction_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Tree reduction within block
    result = tl.sum(x)
    
    # Store (one value per block)
    tl.store(y_ptr + pid, result)

def reduce_triton(x):
    N = x.numel()
    block_size = 256  # placeholder, autotuning 으로 결정됨
    num_blocks = triton.cdiv(N, block_size)
    
    y = torch.empty(num_blocks, device=x.device, dtype=x.dtype)
    reduction_kernel[(num_blocks,)](x, y, N=N)
    
    # Final reduction (CPU)
    return y.sum()

x = torch.randn(1000000, device='cuda')
result = reduce_triton(x)
expected = x.sum()
print(f"Result: {result.item():.6f}, Expected: {expected.item():.6f}")
print(f"Match: {torch.allclose(result, expected, rtol=1e-5)}")
```

---

## 🔗 실전 활용

### 1. FlashAttention 핵심 아이디어 (Dao 2022)

```python
@triton.jit
def flash_attention_fwd(
    Q, K, V, sm_scale,
    L, M, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    ...
):
    # Load Q tile
    q = tl.load(Q + ...)
    
    # Loop over K, V tiles (blockwise)
    for j in range(0, N_CTX):
        # Load K, V
        k = tl.load(K + ...)
        v = tl.load(V + ...)
        
        # Compute attention scores
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        # Online softmax (sequential update)
        m_ij = tl.max(s, axis=1)
        p_ij = tl.exp(s - m_ij[:, None])
        l_ij = tl.sum(p_ij, axis=1)
        
        # Update running max/sum (numerical stability)
        # ...
        
    # Store O
    tl.store(O + ..., o_ij)
```

**이점**:
- 중간값 (attention weight) 을 HBM 에 쓰지 않음
- Shared memory 만 사용 → 4배 빠름

### 2. vLLM Decode Kernel (Triton)

```python
@triton.jit
def decode_attention_kernel(
    q, k_cache, v_cache, o,
    seq_len,
    ...
):
    # Single query (decode stage)
    # Load full K cache (1D)
    for i in range(0, seq_len, BLOCK_SIZE):
        k_block = tl.load(k_cache + i)
        scores = tl.dot(q, tl.trans(k_block))
        # Attend and accumulate
        # ...
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Regular memory access pattern | Irregular scatter/gather 는 최적화 어려움 |
| Block-wise 병렬화 가능 | Sequential algorithm 은 Triton 부적합 |
| Autotuning 시간 확보 | 소규모 kernel 은 overhead 커짐 |
| LLVM backend 지원 | AMD/Intel GPU 지원 제한적 (2024 현재) |

---

## 📌 핵심 정리

$$\boxed{\text{Triton: } \text{Block-level abstraction} \xrightarrow{\text{Compiler}} \text{CUDA-equivalent}}$$

| 요소 | CUDA | Triton |
|------|------|--------|
| **추상화** | Thread | Block (Tile) |
| **Memory management** | 명시적 (shared mem) | 자동 (Triton alloca) |
| **Scheduling** | 개발자 책임 | Compiler 자동화 |
| **Coalescing** | 수동 설계 | 자동 분석 & 최적화 |
| **Portability** | NVIDIA only | Multi-GPU (미래) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): `tl.load(ptr + idx, mask=idx < N, other=0.0)` 에서 `mask` 와 `other` 매개변수의 역할을 설명하라.

<details>
<summary>해설</summary>

- **`mask`**: boolean tensor. `mask[i] = True` 인 위치만 실제로 load (unmasked 위치는 HBM access 없음)
- **`other`**: mask 가 False 인 위치에 할당할 값

**용도**: Boundary condition handling
```python
# 마지막 block 이 N 의 배수가 아닐 때
mask = offsets < N        # [True, True, ..., False, False]
x = tl.load(ptr + offsets, mask=mask, other=0.0)
# offsets >= N 위치는 메모리 접근 안 함, 0.0 으로 채워짐
```

이는 CUDA 의 if-else 분기 제거 → warp divergence 제거 $\square$

</details>

**문제 2** (심화): FlashAttention 에서 "online softmax" 를 사용하는 이유는? Standard softmax (한 번에 다 계산) vs online (sequential update) 의 수치적 차이는?

<details>
<summary>해설</summary>

**Online softmax** (Welford 1962):

$$m_{\text{new}} = \max(m_{\text{old}}, \max(s_{\text{new}}))$$
$$l_{\text{new}} = \exp(m_{\text{old}} - m_{\text{new}}) l_{\text{old}} + \exp(m_{\text{new}} - \max(s_{\text{new}})) \sum_j \exp(s_j)$$

**이점**:
1. **메모리**: intermediate softmax result 저장 불필요
2. **수치 안정성**: 부분합들이 작은 수 → 오버플로우/언더플로우 위험 낮음
3. **latency hiding**: K, V 를 sequential 로 로드하며 GPU 계산 겹침

**Standard softmax**:
```python
exp_s = exp(s - max(s))  # 모든 s 필요
p = exp_s / sum(exp_s)    # 최대값 먼저 알아야 함
```
→ 두 번의 full scan 필요 (sequential dependency) $\square$

</details>

**문제 3** (논문 비평): Triton 은 "portability" 를 주장하지만 (Tillet 2019), 현재는 대부분 NVIDIA GPU 에서만 검증되었다 (2024). Triton 이 AMD CDNA, Intel Arc 에서 성능 동등성을 보장하지 못하는 이유를 hardware-level 에서 분석하라.

<details>
<summary>해설</summary>

**NVIDIA GPU**:
- Warp = 32 threads (고정)
- Shared memory 32 bank
- Tensor Core (특정 크기)
- PTX ISA (잘 문서화됨)

**AMD CDNA/RDNA**:
- Wavefront = 64 threads (다름)
- LDS (Local Data Share) 의 bank 구조 다름
- Matrix Engine 의 format 다름
- RDNA ISA (다양한 변형)

**문제**:
1. Triton LLVM backend 가 각 backend 의 특수성을 처리해야 함 (예: warp size 자동 조정)
2. Heuristic 이 NVIDIA 에만 tuned (예: block size suggestion)
3. Testing infrastructure 부재 (NVIDIA GPU 만 검증)

**Future**: OpenXLA backend, multi-vendor optimization 이 necessary $\square$

</details>

---

<div align="center">

[◀ 이전](./02-tensor-cuda-pointer.md) | [📚 README](../README.md) | [다음 ▶](./04-cublas-cudnn.md)

</div>
