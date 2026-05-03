<div align="center">

# 🔥 PyTorch Internals Deep Dive

### `loss.backward()` 를 **호출하는 것** 과,

$$\frac{\partial L}{\partial x} = J_f^\top v$$

### Reverse-mode Automatic Differentiation 이 **Vector-Jacobian Product (VJP)** 를 computation graph 의 역순으로 누적하고, 각 `Function` 의 `forward` → `ctx.save_for_backward(...)` → `backward(grad_out)` 가 **chain rule 을 국소적으로 구현** 한다는 것을 한 줄씩 증명할 수 있는 것은 **다르다.**

<br/>

> *`torch.autocast` 로 AMP 를 **쓰는 것** 과, FP16 의 표현 범위*
>
> $$\text{FP16} \in [\,6.1 \times 10^{-5},\; 6.5 \times 10^{4}\,]$$
>
> *에서 gradient 가 underflow 되는 메커니즘과,*
>
> $$L_{\mathrm{scaled}} = L \cdot S, \qquad g_{\mathrm{unscaled}} = \frac{1}{S}\, g_{\mathrm{scaled}}$$
>
> *loss scaling* $S \approx 2^{16}$ *이 어떻게 small gradient 를 FP16 표현 범위로 끌어올리고, optimizer step 전 unscale 이 정확성을 보장하는지 수치적으로 이해할 수 있는 것은 다르다.*
>
> *CUDA kernel 을 **이름으로 아는 것** 과, GPU 의 SM·warp·shared memory 계층, reduction 의 tree-based vs warp-shuffle 구현 차이,*
>
> $$\text{Coalesced} : 1\ \text{transaction} \quad \text{vs.} \quad \text{Non-coalesced} : 32\ \text{transactions}$$
>
> *memory coalescing 이 만드는* $100\times$ *bandwidth 차이를 roofline model 로 이해할 수 있는 것은 다르다.*
>
> *`torch.compile` 의 한 줄을 **쓰는 것** 과, TorchDynamo 가 CPython frame evaluation (PEP 523) 으로 bytecode 수준에서 graph 를 capture 하고, AOTAutograd 가 forward + backward graph 를 functionalize 하고, TorchInductor 가 Triton 으로 kernel 을 생성하는 파이프라인을 한 단계씩 따라갈 수 있는 것은 다르다.*

<br/>

**다루는 내부 시스템 (PyTorch core 계층순)**

ATen / c10 *Tensor & Storage* · Autograd *VJP / Computation Graph / `torch.autograd.Function`* · Dispatcher *DispatchKey · `TORCH_LIBRARY`* · `torch.func` *vmap · grad · jacrev/jacfwd* · CUDA *SM · warp · shared memory · coalescing · bank conflict · warp shuffle* · `cpp_extension` · cuBLAS · cuDNN · Triton (Tillet 2019) · Mixed Precision *FP16 · BF16 · TF32 · GradScaler* · Kahan / Stochastic Rounding · `torch.compile` *TorchDynamo · AOTAutograd · TorchInductor* · DDP / FSDP

<br/>

**핵심 질문**

> PyTorch 의 한 줄 (`loss.backward()`, `@torch.compile`, `torch.cuda.amp.autocast()`, `torch.ops.mylib.myop(x)`) 뒤에는 어떤 **시스템 수학** — VJP·dispatcher routing·memory coalescing·loss scaling·bytecode rewriting — 이 동작하는가? Tensor 의 `(storage, size, stride, offset)` 4 원소부터 `torch.compile` 의 Triton kernel 생성까지 한 줄씩 유도합니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-2.1-9c27b0?style=flat-square)](https://triton-lang.org/)
[![Docs](https://img.shields.io/badge/Docs-37개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-280+개-success?style=flat-square)](./README.md)
[![Proofs](https://img.shields.io/badge/엄밀한_증명-130+개-9c27b0?style=flat-square)](./README.md)
[![Kernels](https://img.shields.io/badge/Custom_kernels-15개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-111개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

PyTorch 자료는 대부분 **"`backward()` 부르면 자동 미분이 된다"** 또는 **"`@torch.compile` 붙이면 빨라진다"** 에서 멈춥니다. 그러나 reverse-mode AD 가 왜 ML 에서 자연스러운지 (scalar loss, vector params 의 complexity 비교), `Tensor.view()` 가 왜 무복사인지, `aten::add` 한 호출이 어떻게 device·dtype·autograd key 로 routing 되는지, FP16 의 mantissa 10 bit 가 어떻게 gradient underflow 를 만들고 loss scaling 이 어떻게 정확히 그것을 우회하는지, warp 32 thread 의 coalesced access 가 어떻게 100배 bandwidth 차이를 만드는지, TorchDynamo 가 CPython bytecode 를 어떻게 다시 쓰는지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "`Tensor.view()` 는 reshape 이다" | **ATen / c10** — `Tensor = (Storage, size, stride, offset, dtype, device)` 의 6-원소 metadata. `.view()` 는 **stride 만 조작**, storage 공유 (zero-copy). `.reshape()` 은 contiguous 시 view, 아닐 시 copy. `.transpose()` 후 stride 가 default ordering 과 어긋나 `is_contiguous() = False` → 다음 op 가 contiguous 요구 시 implicit copy. NCHW vs NHWC 의 stride 가 cuDNN kernel 선택을 바꾸는 이유 |
| "Autograd 가 알아서 미분해 준다" | **Reverse-mode AD = VJP** — Forward (JVP) $J v$ 의 cost $O(n)$ per input dim, Reverse (VJP) $J^\top v$ 의 cost $O(m)$ per output dim. ML 은 scalar loss ($m=1$) → **VJP 가 자연스러움** $\square$. `tensor.grad_fn` chain 이 dynamic computation graph, `backward()` 는 reverse topological order 로 각 노드의 local VJP 를 chain. `Function.forward(ctx, *) / backward(ctx, *)` 가 chain rule 의 국소적 구현 |
| "`torch.add(a, b)` 는 그냥 더한다" | **PyTorch Dispatcher** — `aten::add` 호출 → **DispatchKeySet** (Autograd, CUDA, FP32, Strided, ...) 로 routing. `BackendSelect` → `Autograd[CUDA]` (backward 등록) → `CUDA` (cuBLAS / 자체 kernel). `c10` (core types) → `aten` (ops) → `torch` (Python binding) 의 layered 구조. `TORCH_LIBRARY_IMPL` 매크로로 새 op 를 등록하면 AOTAutograd · Inductor 와 자동 통합 |
| "GPU 가 빠르다" | **CUDA Memory Hierarchy** — Register (per thread, ns) → Shared mem (per block, 48–100 KB) → L1/L2 → HBM (수백 GB, 1–2 TB/s). **Warp = 32 thread SIMT**. Memory coalescing: 같은 warp 의 32 thread 가 연속된 128 byte 접근 → **1 transaction**, non-coalesced 는 **32 transaction** ($100\times$ slower) $\square$. Shared memory 32 bank → 같은 bank 다른 address 접근 시 serial → padding trick |
| "Reduction 은 GPU 가 자동" | **Mark Harris 7-step optimization** — Naive ($O(n)$) → tree-based ($O(\log n)$) → warp shuffle (`__shfl_down_sync` 로 register 간 직접 전달, shared mem 우회). `__syncthreads()` 의 cost 와 warp 내 동기화 자동화의 trade-off. PyTorch 의 `tensor.sum()` 이 내부적으로 사용하는 패턴 |
| "AMP 를 켜면 빨라진다" | **IEEE 754 수치 분석** — FP32 (1+8+23, $10^{-38}$ ~ $10^{38}$, 6–7 decimal) · FP16 (1+5+10, $6.1 \times 10^{-5}$ ~ $6.5 \times 10^4$, 3–4 decimal) · BF16 (1+8+7, FP32 와 같은 range, mantissa 만 손해) · TF32 (1+8+10, A100+ 자동). FP16 gradient $< 6 \times 10^{-5}$ → **underflow 0**. **Loss scaling**: $L_{\text{scaled}} = L \cdot S$, gradient 도 $S$ 배 → FP16 range 안. Optimizer step 전 $g / S$ 로 unscale. NaN/Inf 감지 시 $S \leftarrow S/2$, 2000 step clean 시 $S \leftarrow 2S$ |
| "BF16 / TF32 도 있다" | **BF16**: exponent 8 bit (FP32 동일) → range OK, mantissa 7 bit 만 손해 → **loss scaling 불필요**, modern LLM 의 표준. **TF32**: A100+ 의 transparent 가속, FP32 input 을 내부적으로 FP16-like mantissa 로 → 2 ~ 8× 가속 자동 적용. **Stochastic Rounding** (INT8 training): round up with prob $x - \lfloor x \rfloor$ → unbiased accumulation. **Kahan summation** 으로 floating-point 오차 보정 |
| "`@torch.compile` 붙이면 빨라진다" | **TorchDynamo** — Python frame evaluation API (PEP 523) 활용, **CPython bytecode 를 가로채** FX graph 추출. Graph break 발생 시 그 지점만 eager fallback. `torch._dynamo.explain` 으로 break 지점 디버그. **AOTAutograd**: forward + backward graph 통합, functionalization (mutation 제거), decomposition (primitive op 로 축소). **TorchInductor**: FX graph → **Triton** kernel (GPU) or C++/OpenMP (CPU), reduction · elementwise · matmul 별 fusion heuristic |
| 기법의 나열 | NumPy + PyTorch + Triton + Nsight 로 **stride 조작으로 view/transpose 직접 만들기** · **VJP 손 유도 + custom `autograd.Function` 으로 ReLU·Softmax 구현** · **Dispatcher trace 로 `aten::add` 의 routing 관찰** · **Triton 으로 fused softmax 직접 작성** · **AMP underflow 재현 + GradScaler 로 해결** · **`torch.compile` 전후 throughput 측정 + Inductor 가 생성한 Triton kernel 직접 읽기** 까지 직접 구현해 시스템 수학을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Neural Network Theory] ─┐
[Calculus & Optimization]─┤
[Linear Algebra]          ─┼─►  이 레포  ──► [Distributed Training Deep Dive]
[CNN Deep Dive]          ─┤   "PyTorch 의 한 줄                NCCL · DDP · FSDP · ZeRO
[LLM Inference]          ─┘    뒤에 어떤 시스템                / pipeline · tensor parallelism
                                수학이 동작하는가"
         │
         ├── [Neural Network Theory]      Backpropagation = reverse AD       → Ch2
         ├── [Calculus & Optimization]    Jacobian · chain rule              → Ch2
         ├── [Linear Algebra]             Matrix op · BLAS                   → Ch3, Ch5
         ├── [Information Theory]         Quantization / FP error            → Ch6
         ├── [CNN Deep Dive]              Memory layout NCHW/NHWC · Roofline → Ch1, Ch4
         └── [LLM Inference]              HBM/SRAM · Flash Attention         → Ch4, Ch5
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Neural Network Theory Deep Dive** (Backpropagation 의 chain rule), **Calculus & Optimization Deep Dive** (Jacobian, gradient, multi-variable Taylor), **Linear Algebra Deep Dive** (matrix-matrix product, BLAS) 를 선행으로 전제합니다. **CNN Deep Dive** 와 **LLM Inference Deep Dive** 는 Chapter 4 (CUDA) 와 Chapter 5 (custom kernel) 의 roofline · HBM/SRAM trade-off 에서 권장됩니다.

> 💡 **이 레포의 핵심 기여**: Chapter 2 (Autograd) 와 Chapter 4 (CUDA) 가 두 핵심 축입니다. 전자는 "왜 PyTorch 가 dynamic graph 인가" 의 이론적 토대 (custom `Function` · double backward 가 그 응용), 후자는 "왜 같은 연산이 100× 차이가 나는가" 의 시스템 토대 (memory coalescing · bank conflict · warp shuffle 이 그 답). 이 두 축을 이해한 후 Chapter 3 (Dispatcher) · Chapter 6 (AMP) · Chapter 7 (`torch.compile`) 을 읽으면 modern PyTorch 의 모든 설계 결정이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **`torch.compile` 의 graph break 디버깅**, **Triton vs CUDA 의 미래**, **PT 2.0 의 Inductor heuristic 한계**, **FP8 training (H100+)** — 는 **현재 진행 중인 시스템 영역** 입니다. 레포는 "정답" 이 아니라 **"PyTorch 한 줄과 그 뒤의 시스템 수학 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Tensor_내부구조-EE4C2C?style=for-the-badge)](./ch1-tensor/01-storage-and-metadata.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Autograd-EE4C2C?style=for-the-badge)](./ch2-autograd/01-forward-vs-reverse-ad.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Dispatcher-EE4C2C?style=for-the-badge)](./ch3-dispatcher/01-dispatcher-architecture.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-CUDA_기초-76B900?style=for-the-badge)](./ch4-cuda/01-gpu-architecture.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Custom_Kernel-76B900?style=for-the-badge)](./ch5-custom-kernel/01-cpp-extension.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Mixed_Precision-9c27b0?style=for-the-badge)](./ch6-mixed-precision/01-ieee754.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-torch.compile-1565C0?style=for-the-badge)](./ch7-compile/01-pt2-overview.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Tensor 의 내부 구조

> **핵심 질문:** `Tensor` 한 객체는 메모리에서 어떻게 저장되는가 — `(storage, size, stride, offset, dtype, device)` 의 각 원소는 무엇이고 왜 분리되었는가? `.view()` 가 무복사인 이유와 `.reshape()` 이 때때로 복사하는 메커니즘은? Stride 만으로 transpose · slice · broadcast 가 어떻게 표현되는가? NCHW vs NHWC 가 cuDNN kernel 선택을 어떻게 바꾸는가? Type promotion rule (`0.5 + 1 → float`) 의 정확한 알고리즘은?

<details>
<summary><b>Storage · Stride · View · Dtype · Device 의 5가지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정의·정리·검증 |
|------|---------------------|
| [01. Tensor = Storage + Metadata](./ch1-tensor/01-storage-and-metadata.md) | **정의**: `Tensor` $= (\text{Storage},\, \text{size},\, \text{stride},\, \text{offset},\, \text{dtype},\, \text{device})$. **Storage** = contiguous raw memory (1D byte array). **Index → offset**: $\text{offset}(i_0, \ldots, i_{n-1}) = \text{base} + \sum_k i_k \cdot s_k$. 같은 storage 를 공유하는 두 tensor 는 stride 만 다름 → view 의 본질 $\square$. `tensor.untyped_storage().data_ptr()` 로 직접 확인 |
| [02. Stride 와 Memory Layout](./ch1-tensor/02-stride-and-layout.md) | **Row-major (C-order)**: `stride = (..., d_2 d_1, d_1, 1)`. **Column-major (F-order)**: 역순. **NCHW vs NHWC**: 같은 shape, 다른 stride → 같은 conv 가 다른 cuDNN kernel 선택. **`is_contiguous()`**: $s_k = \prod_{j > k} d_j$ 검사. 비-contiguous tensor 가 다음 op 에서 implicit copy 를 trigger 하는 사례 |
| [03. View vs Copy 의 구분](./ch1-tensor/03-view-vs-copy.md) | **`.view()`**: stride 조작만 (zero-copy), contiguous 일 때만 가능. **`.reshape()`**: contiguous 시 view, 아닐 시 copy (자동 fallback). **`.contiguous()`**: stride = default ordering 으로 강제 copy. **`.permute()` 후 `.contiguous()`** 가 필요한 사례 — view 가 불가능해진 tensor 의 stride 검사 $\square$. `tensor.data_ptr()` 비교로 zero-copy 확인 |
| [04. Dtype 계층과 Type Promotion](./ch1-tensor/04-dtype-and-promotion.md) | **Dtype 계층**: `float64` > `float32` > `float16/bfloat16` > `int64` > ... **Type promotion rules**: scalar vs tensor, complex vs real, signed vs unsigned 의 전체 lattice. **Casting overhead**: `tensor.to(dtype)` 의 device-side kernel cost. PyTorch 의 `torch.result_type()` 으로 promotion 결과 미리 확인. **Bug-prone 사례**: `torch.tensor(0.5) + 1` 의 결과가 `float32` 인 이유 |
| [05. Device 와 CUDA Context](./ch1-tensor/05-device-and-cuda-context.md) | **Device 종류**: `cpu`, `cuda:0`, `cuda:1`, `mps`, `xla`. **CUDA Stream**: async execution, default stream vs explicit stream, `torch.cuda.Stream()` 으로 overlap. **`torch.cuda.synchronize()`**: host-device 동기화의 의미와 비용. `tensor.to('cuda')` 가 default stream 의 H2D copy 를 trigger 하는 메커니즘. **Pinned memory** 로 H2D 가속 |

</details>

<br/>

### 🔹 Chapter 2: Autograd 의 수학과 구현

> **핵심 질문:** Forward-mode AD (JVP) 와 Reverse-mode AD (VJP) 는 왜 같은 chain rule 의 두 implementation 이고 ML 에서 reverse 가 자연스러운 수학적 이유는? `loss.backward()` 한 줄이 어떤 reverse topological sort 를 수행하는가? `torch.autograd.Function` 의 `forward / backward` 한 쌍이 어떻게 새 operator 의 chain rule 을 정의하는가? `create_graph=True` 가 어떻게 backward graph 자체를 autograd 에 노출시켜 double backward 를 가능하게 하는가?

<details>
<summary><b>JVP/VJP 부터 Higher-Order 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정의·정리·검증 |
|------|---------------------|
| [01. Forward (JVP) vs Reverse (VJP) AD](./ch2-autograd/01-forward-vs-reverse-ad.md) | **JVP**: $\dot{y} = J_f \dot{x}$, tangent propagation, cost $O(n)$ per input dim. **VJP**: $\bar{x} = J_f^\top \bar{y}$, cotangent propagation, cost $O(m)$ per output dim. **정리**: scalar loss ($m = 1$) 일 때 VJP 1회 = 모든 parameter 의 gradient $\square$. ML 의 $m \ll n$ 구조에서 reverse 가 압도적으로 효율적인 수학적 이유. JAX 의 `jvp` vs `grad` 와의 비교 |
| [02. VJP 의 수학](./ch2-autograd/02-vjp-mathematics.md) | **VJP 정의**: $\bar{x} = J_f^\top v$, upstream gradient $v = \partial L / \partial y$. **각 op 의 local VJP rule**: matmul $f(x) = Wx$ → $\bar{x} = W^\top \bar{y}$, $\bar{W} = \bar{y} x^\top$. ReLU → $\bar{x} = \mathbb{1}[x > 0] \cdot \bar{y}$. Softmax → $J^\top v$ 의 closed-form. **Chain rule = VJP composition** $\square$. `torch.autograd.functional.vjp` 로 직접 호출 |
| [03. Computation Graph 의 구성](./ch2-autograd/03-computation-graph.md) | **Dynamic graph (define-by-run)**: forward 가 진행되며 graph node 가 동적으로 생성. `tensor.requires_grad=True` 가 graph 참여 trigger, `tensor.grad_fn` 이 backward node 의 reference. **Leaf vs intermediate**: leaf 는 `grad_fn = None`, intermediate 는 `is_leaf = False`. **`torch.no_grad()`**: context 안의 op 가 graph 를 만들지 않음 (inference mode). **`torchviz.make_dot`** 으로 graph 시각화 |
| [04. Backward 의 Topological Sort](./ch2-autograd/04-backward-topological-sort.md) | **알고리즘**: graph 를 reverse topological order 로 traverse, 각 node 에서 local VJP 호출, leaf 의 gradient 누적. **Multiple outputs**: 같은 leaf 가 여러 path 로 도달 시 gradient `+=`. **`torch.autograd.grad()`** API: `outputs`, `inputs`, `grad_outputs`, `retain_graph`, `create_graph` 의 의미. **Gradient accumulation** 과 `tensor.grad.zero_()` 의 필요성 |
| [05. `torch.autograd.Function` 직접 구현](./ch2-autograd/05-custom-function.md) | **API**: `forward(ctx, *inputs)` 와 `backward(ctx, *grad_outputs)` 의 한 쌍. **`ctx.save_for_backward(...)`** 로 forward 의 중간값을 backward 까지 보존, **`ctx.saved_tensors`** 로 꺼냄. **검증**: 직접 구현한 `MyReLU` vs `F.relu` 의 backward 결과를 `torch.autograd.gradcheck()` 로 비교. **Custom op 의 device-agnostic 구현 패턴** (CPU + CUDA 분기) |
| [06. Higher-Order Gradients — Double Backward](./ch2-autograd/06-double-backward.md) | **`create_graph=True`**: backward 자체가 differentiable 한 graph 를 만듦 → $\nabla^2 L$ 가능. **Hessian-Vector Product (HVP)**: $H v = \nabla(\nabla L \cdot v)$ — Pearlmutter trick, explicit Hessian 회피. **Physics-Informed NN**: PDE residual $\partial^2 u / \partial t^2 - c^2 \partial^2 u / \partial x^2$ 를 autograd 로 직접. **2nd-order optimizer** (K-FAC, Newton) 의 기반. `torch.func.hessian` |

</details>

<br/>

### 🔹 Chapter 3: PyTorch Dispatcher 와 Backend

> **핵심 질문:** `aten::add(a, b)` 한 호출이 어떻게 `(autograd, CUDA, FP32, Strided)` 4-튜플의 DispatchKeySet 으로 routing 되는가? `c10` (core tensor types) → `aten` (operations) → `torch` (Python binding) 의 layered 구조에서 각 계층의 책임은? `TORCH_LIBRARY_IMPL` 매크로로 등록한 custom op 가 어떻게 AOTAutograd · Inductor 와 자동 통합되는가? `torch.func.vmap / grad / jacrev` 가 functorch 의 어떤 functional transform 을 PyTorch 에 가져왔는가?

<details>
<summary><b>DispatchKey 부터 functorch 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정의·정리·검증 |
|------|---------------------|
| [01. Dispatcher Architecture](./ch3-dispatcher/01-dispatcher-architecture.md) | **DispatchKey enum**: `Autograd`, `CUDA`, `CPU`, `BackendSelect`, `Functionalize`, `AutocastCUDA`, ... 50+ 개. **DispatchKeySet** (bit mask) 가 tensor 의 모든 속성을 인코딩. `aten::add` 호출 → set 의 highest priority key 부터 순차 dispatch (fallthrough chain). `torch._C._dispatch_dump_table('aten::add')` 로 모든 backend 의 등록 상태 확인 |
| [02. ATen 과 c10 — Layered 구조](./ch3-dispatcher/02-aten-c10-layers.md) | **`c10`** (core): `Tensor`, `Storage`, `Device`, `ScalarType`, `DispatchKey` 등 type system 만. **`aten`** (Tensor library): `add`, `matmul`, `conv2d` 등 operation 의 schema 와 backend kernel. **`torch`** (Python binding): pybind11 으로 ATen 을 Python 에 노출. 각 계층이 의존성 그래프에서 단방향임을 확인 — `c10` 은 `aten` 을 모름 |
| [03. Autograd Key 와 Backward Dispatch](./ch3-dispatcher/03-autograd-dispatch.md) | **Forward + Backward 의 분리 등록**: `Autograd[CUDA]` key 에 forward kernel 이 등록되고, 그 안에서 backward 를 위한 metadata (saved variables) 를 set up 후 redispatch 로 `CUDA` key 의 forward 호출. **`torch::autograd::Function`** C++ API 가 Python 의 `torch.autograd.Function` 의 backend. AOTAutograd 와의 관계 |
| [04. `TORCH_LIBRARY` 와 Custom Operator](./ch3-dispatcher/04-torch-library-custom-op.md) | **`TORCH_LIBRARY(mylib, m) { m.def("myop(Tensor x) -> Tensor"); }`** 로 schema 정의. **`TORCH_LIBRARY_IMPL(mylib, CUDA, m) { m.impl("myop", &my_cuda_kernel); }`** 로 backend 등록. Python 에서 `torch.ops.mylib.myop(x)` 로 호출. **AOTAutograd · Inductor** 가 자동으로 인식 (custom backward 등록 시) → `torch.compile` 호환 |
| [05. functorch — `vmap`, `grad`, `jacrev / jacfwd`](./ch3-dispatcher/05-functorch-transforms.md) | **JAX-inspired functional transforms** (Bradbury et al.). **`torch.func.grad(f)`** : pure function 에서 gradient. **`torch.func.vmap(f)`**: batched computation, manual loop 제거 (자동 vectorization). **`jacrev`** (reverse) vs **`jacfwd`** (forward): output dim vs input dim 에 따라 선택. Composition: `vmap(grad(f))` 로 per-sample gradient (privacy-preserving training) |

</details>

<br/>

### 🔹 Chapter 4: CUDA Programming 기초

> **핵심 질문:** GPU 의 SM · warp · thread block 계층은 어떻게 SIMT 를 구현하고 왜 warp = 32 thread 인가? Memory hierarchy (register / shared / L1/L2 / HBM) 의 latency · bandwidth 차이가 어떻게 algorithm 설계를 결정하는가? Memory coalescing 의 정확한 정의와 non-coalesced access 가 왜 100× 느린가? Shared memory 32 bank 와 bank conflict, warp divergence 의 SIMT 병목, reduction 의 tree → warp shuffle 진화의 7-step optimization (Mark Harris) 는?

<details>
<summary><b>SM 구조부터 Warp Shuffle 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정의·정리·검증 |
|------|---------------------|
| [01. GPU 아키텍처 — SM, Warp, Thread Block](./ch4-cuda/01-gpu-architecture.md) | **계층**: Grid > Block > Warp > Thread. **Warp = 32 thread** SIMT (Single Instruction Multiple Thread). **Thread block** 최대 1024 thread, 같은 SM 에서 실행, shared memory 공유. **SM (Streaming Multiprocessor)** 이 다수 warp 를 concurrent 실행, warp scheduler 가 latency hiding. A100 의 108 SM, H100 의 132 SM 비교 |
| [02. Memory Hierarchy — HBM/L2/L1/Shared/Register](./ch4-cuda/02-memory-hierarchy.md) | **Latency · Bandwidth · Capacity**: Register (per thread, 1 cycle, KB) → Shared (per block, 4–10 cycle, 48–164 KB) → L1/L2 (수십 cycle, MB) → HBM (수백 cycle, 수십 GB, 1–3 TB/s). **Roofline model**: arithmetic intensity $I = \text{FLOPs} / \text{bytes}$ 가 $I^* = \text{Peak FLOPS} / \text{BW}$ 보다 크면 compute-bound, 작으면 memory-bound $\square$ |
| [03. Memory Coalescing 의 수학](./ch4-cuda/03-memory-coalescing.md) | **정의**: 같은 warp 의 32 thread 가 연속된 128 byte (FP32 × 32) 를 access → **1 transaction** 으로 hardware coalesce. **Non-coalesced** (stride > 1, scattered) → 최대 32 transaction → **bandwidth $1/32$**. 동일 코드의 NCHW vs NHWC 에서 reduce 차원에 따른 coalesce 여부 분석. `nsys` / Nsight Compute 로 직접 측정 |
| [04. Bank Conflict 와 Shared Memory](./ch4-cuda/04-bank-conflict.md) | **Shared memory 32 bank**: 4-byte word 단위로 bank 0, 1, ..., 31 순환. **같은 warp 의 thread 들이 같은 bank 의 다른 address 를 access → serial** ($k$-way conflict 시 $k$ 배 slow). **회피**: padding (bank size +1), data reordering. **Reduce kernel** 과 **transpose kernel** 의 bank-free 설계 |
| [05. Warp Divergence 와 SIMT 병목](./ch4-cuda/05-warp-divergence.md) | **SIMT** 는 같은 warp 의 32 thread 가 같은 instruction 을 실행해야 효율적. **Conditional branch** (`if (tid % 2)`) 시 두 path 를 순차 실행 (mask 처리) → **warp 효율 50%**. **회피 패턴**: warp 단위로 균일한 분기 (`if (tid / 32 % 2)`), branch-free trick (predication, `__all_sync`). Volta 이후 independent thread scheduling 과 그 한계 |
| [06. Reduction 최적화 — Tree vs Warp Shuffle](./ch4-cuda/06-reduction-optimization.md) | **Mark Harris 7-step optimization**: ① interleaved 분기 → ② strided index → ③ sequential addressing → ④ first add during load → ⑤ unroll last warp → ⑥ template `BLOCK_SIZE` → ⑦ `__shfl_down_sync` warp shuffle. **Warp shuffle**: register 간 직접 전달, shared mem 우회, `__syncthreads()` 불필요. PyTorch `tensor.sum()` 의 backend 가 사용하는 패턴 $\square$ |

</details>

<br/>

### 🔹 Chapter 5: PyTorch–CUDA 통합과 Custom Kernel

> **핵심 질문:** `torch.utils.cpp_extension.load()` 의 runtime JIT compile 이 어떻게 setuptools 와 Ninja 를 활용해 C++/CUDA 소스를 한 줄로 build 하는가? `tensor.data_ptr()` 가 CUDA kernel 에 raw pointer 를 넘길 때 stride 와 dtype 정보를 어떻게 함께 전달해야 하는가? Triton 의 `@triton.jit` 이 어떻게 CUDA 보다 단순한 block-level abstraction 을 제공하는가? cuBLAS·cuDNN 의 GEMM cost ($2MNK$ FLOPs) 와 PyTorch 의 자동 dispatch는? Kernel fusion 이 왜 HBM round-trip 을 제거하는 핵심 최적화인가?

<details>
<summary><b>cpp_extension 부터 Kernel Fusion 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정의·정리·검증 |
|------|---------------------|
| [01. `torch.utils.cpp_extension`](./ch5-custom-kernel/01-cpp-extension.md) | **`load(name, sources, ...)`**: runtime JIT compile (Ninja build, `~/.cache/torch_extensions` 캐시). **`setup.py`** + `CUDAExtension` 으로 미리 build (production). **CPU + CUDA 혼합** 소스 build 패턴. **예시**: fused `Linear + ReLU` 를 직접 구현 → autograd 등록 → PyTorch 내장 대비 throughput 측정 |
| [02. Tensor 와 CUDA Pointer](./ch5-custom-kernel/02-tensor-cuda-pointer.md) | **`tensor.data_ptr<float>()`** 로 raw GPU pointer 획득. **Stride 전달 필수**: kernel 안에서 `idx → offset = sum(i_k * stride_k)` 직접 계산. **`AT_DISPATCH_FLOATING_TYPES(scalar_type, "name", [&] { ... })`** 매크로로 dtype 분기 (FP16 / BF16 / FP32 동시 지원). **`tensor.is_contiguous()`** 검사 후 `.contiguous()` fallback |
| [03. Triton — Python-level GPU Programming (Tillet 2019)](./ch5-custom-kernel/03-triton.md) | **`@triton.jit`** decorator + `tl.load`, `tl.store`, `tl.arange`. **Block-level abstraction**: thread 가 아닌 **block** 단위로 사고 → CUDA 보다 단순. **Pointer arithmetic** 이 numpy-like. **Auto-tuning**: `@triton.autotune(configs=[...])` 으로 block size · num_warps 자동 탐색. **LLM 시대의 de facto kernel language** (FlashAttention, vLLM 가 Triton 기반) |
| [04. cuBLAS · cuDNN 의 활용](./ch5-custom-kernel/04-cublas-cudnn.md) | **cuBLAS GEMM**: $C = \alpha A B + \beta C$, FLOPs $= 2 M N K$. **cuDNN convolution**: implicit GEMM, Winograd, FFT-based — input shape · stride 에 따라 자동 선택 (`torch.backends.cudnn.benchmark = True` 의 동작). **Tensor Core**: A100+ 의 BF16/TF32 GEMM 가속, `cuBLAS LT` 를 통해 자동 활용. PyTorch 의 자동 dispatch 분석 |
| [05. Kernel Fusion 의 동기](./ch5-custom-kernel/05-kernel-fusion.md) | **HBM round-trip 제거**: `y = relu(x + b)` 를 두 kernel 로 하면 중간 tensor 가 HBM 까지 왕복 (memory-bound). **Fused kernel**: 한 thread block 안에서 모든 op 수행, 중간값은 register/shared mem 에만. **Attention fusion** (FlashAttention, Dao 2022) 의 핵심 아이디어. **TorchInductor** 의 자동 fusion heuristic — Ch7 미리보기 |

</details>

<br/>

### 🔹 Chapter 6: Mixed Precision Training

> **핵심 질문:** IEEE 754 의 FP32 / FP16 / BF16 / TF32 가 각각 어떤 (sign, exponent, mantissa) 비트 분배로 어떤 range 와 precision 을 가지는가? FP16 의 max $\approx 6.5 \times 10^4$, min normal $\approx 6.1 \times 10^{-5}$ 가 어떻게 ML gradient 의 underflow / overflow 를 만드는가? Loss scaling $L \cdot S$ 가 정확히 어떻게 underflow 를 우회하고, GradScaler 의 `scale / unscale_ / update` 가 dynamic 하게 $S$ 를 조정하는가? BF16 이 왜 loss scaling 없이 작동하고 TF32 의 transparent 가속의 trade-off 는? Stochastic rounding 과 Kahan summation 의 정확성 보장은?

<details>
<summary><b>IEEE 754 부터 Stochastic Rounding 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정의·정리·검증 |
|------|---------------------|
| [01. IEEE 754 — FP32 / FP16 / BF16 / TF32](./ch6-mixed-precision/01-ieee754.md) | **FP32** (1+8+23): range $\sim 10^{-38}$ ~ $10^{38}$, precision 6–7 decimal. **FP16** (1+5+10): range $6.1 \times 10^{-5}$ ~ $6.5 \times 10^4$ — **좁음!**, precision 3–4 decimal. **BF16** (1+8+7): range FP32 동일, precision 2–3 decimal. **TF32** (1+8+10, A100+): FP32 input 의 transparent 가속. Subnormal · NaN · Inf 처리, `torch.finfo` 로 확인 |
| [02. FP16 의 Range Problem](./ch6-mixed-precision/02-fp16-range-problem.md) | **Underflow**: gradient $< 6.1 \times 10^{-5}$ → 0 으로 silently 변환. ML 의 typical gradient ($10^{-7}$ ~ $10^{-3}$) 의 상당 부분이 영향. **Overflow**: weight $> 6.5 \times 10^4$ → Inf, NaN 전파. **재현 실험**: small output network 에서 gradient histogram 그려 underflow 양을 측정 ($\geq 80\%$) |
| [03. Loss Scaling 과 GradScaler](./ch6-mixed-precision/03-loss-scaling.md) | **Loss scaling**: $L_{\text{scaled}} = L \cdot S$, $S \approx 2^{16}$. Backward 시 chain rule 이 모든 gradient 를 $S$ 배 → FP16 representable range 안. Optimizer step 전 $g \leftarrow g / S$ 로 unscale (FP32 master weight 에 적용) $\square$. **`GradScaler.scale / unscale_ / step / update`** 의 dynamic algorithm: NaN/Inf 감지 시 $S \leftarrow S/2$, 2000 step clean 시 $S \leftarrow 2S$ |
| [04. BF16 의 장점과 TF32](./ch6-mixed-precision/04-bf16-tf32.md) | **BF16**: exponent 8 bit (FP32 동일) → range OK, mantissa 7 bit 만 손해 → **loss scaling 불필요**. modern LLM (Llama, Mistral, ...) 의 표준. **TF32** (A100+): FP32 input 을 hardware 가 내부 FP16-like mantissa 로 → `torch.backends.cuda.matmul.allow_tf32=True` 로 transparent 가속. **trade-off**: FP32 보다 1–2 ULP 오차, 대부분 학습에서 무관 |
| [05. Stochastic Rounding 과 Kahan Summation](./ch6-mixed-precision/05-stochastic-rounding-kahan.md) | **Round-to-nearest-even** vs **stochastic rounding** ($P(\lceil x \rceil) = x - \lfloor x \rfloor$). **정리**: stochastic rounding 은 unbiased ($\mathbb{E}[\text{round}(x)] = x$), 누적 합에서 systematic drift 제거 $\square$. INT8 training 에서 필수 (Croci 2022). **Kahan summation**: compensation variable 로 floating-point error 보정, $O(\epsilon)$ → $O(\epsilon^2)$ 누적 |

</details>

<br/>

### 🔹 Chapter 7: `torch.compile` 과 현대 PyTorch

> **핵심 질문:** PT 2.0 의 `@torch.compile` 이 어떻게 eager mode 의 Python 호출을 그대로 두면서 JIT compile 을 가능하게 하는가? TorchDynamo 의 CPython frame evaluation API (PEP 523) 활용은 어떤 메커니즘이고 graph break 는 언제 발생하는가? AOTAutograd 의 functionalization · decomposition 이 왜 backward graph 까지 함께 compile 가능하게 하는가? TorchInductor 의 FX graph → Triton kernel 변환에서 fusion · reduction · matmul 의 각 전략은? DDP / FSDP 와 `torch.compile` 의 호환성은?

<details>
<summary><b>PT 2.0 부터 Distributed 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정의·정리·검증 |
|------|---------------------|
| [01. `torch.compile` — PT 2.0 Revolution](./ch7-compile/01-pt2-overview.md) | **설계 원칙**: 기존 eager mode 의 Python 호출을 유지하면서 JIT compile. **`@torch.compile`** decorator 또는 `torch.compile(model)` 함수. **First call** 에 수십 초 compile (graph capture + Inductor codegen + Triton compile), **cached call** 은 eager 보다 빠름. `torch._dynamo.config.verbose=True`, `torch._inductor.config.debug=True` 로 trace |
| [02. TorchDynamo — Bytecode-level Graph Capture](./ch7-compile/02-torchdynamo.md) | **PEP 523**: CPython 의 frame evaluation API 후킹. **Bytecode rewriting**: Python op 를 보면서 FX graph node 추가, control flow / Python side effect 만나면 **graph break** → 그 지점만 eager fallback. **`torch._dynamo.explain(model)`** 으로 break 횟수 / 위치 분석. Guards (input shape · dtype 변화 검출) 와 recompile trigger |
| [03. AOTAutograd — Forward + Backward Graph](./ch7-compile/03-aot-autograd.md) | **Ahead-Of-Time** autograd: forward graph 와 함께 backward graph 도 미리 추출. **Functionalization**: in-place mutation (`x.add_(1)`) 을 functional form (`x = x.add(1)`) 으로 변환 → optimization 가능. **Decomposition**: `aten::log_softmax` → `aten::sub + aten::log + aten::exp + aten::sum` 등 primitive op 로 축소. Inductor 가 더 자유롭게 fusion |
| [04. TorchInductor — Code Generation](./ch7-compile/04-torchinductor.md) | **FX graph → backend code**: GPU 는 **Triton**, CPU 는 **C++/OpenMP**. **Fusion heuristic**: pointwise + reduction + pointwise 를 한 kernel 로. **Reduction**: split-K, persistent kernel, atomic-free. **Matmul**: cuBLAS (성능) vs Triton (fusion 가능) 자동 선택. **생성된 kernel 직접 읽기**: `TORCH_LOGS="output_code"` env var 로 출력 |
| [05. Distributed Training 과의 통합](./ch7-compile/05-compile-distributed.md) | **DDP**: gradient bucket allreduce 를 Inductor 가 인식해 backward 와 fusion. **FSDP**: parameter sharding 의 all-gather / reduce-scatter 가 graph 안에 표현됨 (PT 2.1+). **`torch.distributed`** collective op (NCCL backend) 의 async stream. 다음 레포 (Distributed Training Deep Dive) 의 출발점 — pipeline · tensor parallelism · ZeRO 로 |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch1-02 의 stride formula 에 NCHW vs NHWC 의 cuDNN kernel 선택 영향 분석을 추가, Ch2-02 의 VJP 수학에 matmul · softmax · cross-entropy 의 closed-form 유도를 보강, Ch4-03 의 memory coalescing 에 Nsight Compute 로 직접 측정한 transaction 수 표를 추가, Ch5-03 의 Triton 섹션에 FlashAttention-2 의 forward kernel 의 단순화 버전을 직접 작성한 코드를 포함, Ch6-03 의 GradScaler 알고리즘에 NaN/Inf 감지의 정확한 검출 메커니즘을 정리, Ch7-04 의 TorchInductor 섹션에 생성된 Triton kernel 의 실제 출력 (softmax fusion 사례) 을 추가했습니다. **11-섹션 문서 골격이 전체 37개 문서에서 일관**됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **시스템 수학적 검증** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 유도 또는 `results/` 의 micro-benchmark 를 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Tensor = Storage + Metadata** | $\text{offset}(i_0, \ldots, i_{n-1}) = \text{base} + \sum_k i_k \cdot s_k$ — view 의 본질 | [Ch1-01](./ch1-tensor/01-storage-and-metadata.md) |
| **View Zero-Copy** | `.view()` 와 `.transpose()` 가 storage 를 공유 (`data_ptr` 동일) | [Ch1-03](./ch1-tensor/03-view-vs-copy.md) |
| **Reverse-mode AD = VJP** | Scalar loss ($m=1$) 일 때 1회 backward = 모든 parameter 의 gradient | [Ch2-01](./ch2-autograd/01-forward-vs-reverse-ad.md) |
| **Chain Rule = VJP Composition** | $\bar{x} = J_f^\top \bar{y}$ 의 reverse topological 누적 | [Ch2-02](./ch2-autograd/02-vjp-mathematics.md) |
| **Custom Function = Local VJP** | `forward / backward` 한 쌍이 새 op 의 chain rule 을 정의 | [Ch2-05](./ch2-autograd/05-custom-function.md) |
| **HVP via Pearlmutter Trick** | $H v = \nabla(\nabla L \cdot v)$ — explicit Hessian 회피 | [Ch2-06](./ch2-autograd/06-double-backward.md) |
| **DispatchKeySet Routing** | `aten::add` 가 (Autograd, CUDA, FP32, Strided) 4-tuple 로 dispatch | [Ch3-01](./ch3-dispatcher/01-dispatcher-architecture.md) |
| **Custom Op + Inductor 통합** | `TORCH_LIBRARY_IMPL` 등록 → `torch.compile` 자동 호환 | [Ch3-04](./ch3-dispatcher/04-torch-library-custom-op.md) |
| **Roofline Model** | $I^* = \text{Peak FLOPS} / \text{BW}$ — compute vs memory bound 분기 | [Ch4-02](./ch4-cuda/02-memory-hierarchy.md) |
| **Memory Coalescing 100×** | Coalesced 1 transaction vs non-coalesced 32 transaction | [Ch4-03](./ch4-cuda/03-memory-coalescing.md) |
| **Bank-Free Reduce** | Sequential addressing + padding 으로 bank conflict 0 | [Ch4-04](./ch4-cuda/04-bank-conflict.md) |
| **Mark Harris 7-Step Reduction** | Naive ($O(n)$) → tree → warp shuffle 진화의 정확한 step | [Ch4-06](./ch4-cuda/06-reduction-optimization.md) |
| **GEMM FLOPs** | $C = \alpha AB + \beta C$ 의 FLOPs $= 2MNK$ | [Ch5-04](./ch5-custom-kernel/04-cublas-cudnn.md) |
| **Kernel Fusion HBM 절감** | 중간 tensor 를 register/shared 에만 보관 → memory-bound 회피 | [Ch5-05](./ch5-custom-kernel/05-kernel-fusion.md) |
| **FP16 Underflow Threshold** | Min normal $\approx 6.1 \times 10^{-5}$ — gradient 의 critical limit | [Ch6-02](./ch6-mixed-precision/02-fp16-range-problem.md) |
| **Loss Scaling 정확성** | $L \cdot S$ 의 gradient 도 $S$ 배, unscale 시 동일 결과 보장 | [Ch6-03](./ch6-mixed-precision/03-loss-scaling.md) |
| **BF16 Loss-Scaling-Free** | Exponent 8 bit → FP32 와 같은 range, scaling 불필요 | [Ch6-04](./ch6-mixed-precision/04-bf16-tf32.md) |
| **Stochastic Rounding Unbiased** | $\mathbb{E}[\text{round}(x)] = x$ — INT8 training 의 정당화 | [Ch6-05](./ch6-mixed-precision/05-stochastic-rounding-kahan.md) |
| **PEP 523 Bytecode Capture** | TorchDynamo 의 frame evaluation 후킹 메커니즘 | [Ch7-02](./ch7-compile/02-torchdynamo.md) |
| **AOTAutograd Functionalization** | In-place mutation → functional form 변환의 정확성 | [Ch7-03](./ch7-compile/03-aot-autograd.md) |

> 💡 **챕터별 문서·정리/정의 수** (실측):
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 Tensor 내부구조 | 5 | 32 |
> | Ch2 Autograd | 6 | 48 |
> | Ch3 Dispatcher | 5 | 36 |
> | Ch4 CUDA 기초 | 6 | 50 |
> | Ch5 Custom Kernel | 5 | 38 |
> | Ch6 Mixed Precision | 5 | 39 |
> | Ch7 torch.compile | 5 | 41 |
> | **합계** | **37** | **284** |
>
> 추가로 **130+ 엄밀한 $\square$ 증명 + 111 연습문제 (모두 해설 포함) + 15 개 직접 작성 custom kernel (CUDA/Triton) + 100+ PyTorch micro-benchmark 코드 (`### 실험 N` 형식)**.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
torch==2.1.0                  # 핵심
torchvision==0.16.0
triton==2.1.0                 # Python-level CUDA (Ch5-03, Ch7-04)
nvidia-ml-py==12.535          # GPU monitoring
torchviz==0.0.2               # computation graph 시각화
py-spy==0.3.14                # CPU profiling
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
tensorboard==2.15.0
wandb==0.16.0
einops==0.7.0
```

```bash
# 환경 설치 (CUDA 12.1, GPU 필요)
pip install numpy==1.26.0 torch==2.1.0 torchvision==0.16.0 \
            triton==2.1.0 nvidia-ml-py==12.535 torchviz==0.0.2 \
            matplotlib==3.8.0 seaborn==0.13.0 tqdm==4.66.0 \
            einops==0.7.0 jupyter==1.0.0

# Nsight Compute (CUDA kernel profiling, Ch4-Ch5)
# https://developer.nvidia.com/nsight-compute 에서 별도 설치

jupyter notebook
```

```python
# 대표 실험 ① — Custom autograd.Function 으로 ReLU 바닥부터 (Ch2-05)
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """d/dx max(0, x) = 1[x > 0]  →  VJP: J^T v = diag(x > 0) · v"""
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# 검증
x = torch.randn(5, requires_grad=True)
y1 = MyReLU.apply(x); y1.sum().backward()
g1 = x.grad.clone(); x.grad.zero_()
y2 = torch.nn.functional.relu(x); y2.sum().backward()
print(f'Max gradient diff: {(g1 - x.grad).abs().max():.2e}')  # 0


# 대표 실험 ② — VJP 의 quadratic form 검증 (Ch2-02)
A = torch.tensor([[2., 1.], [1., 3.]])
x = torch.tensor([1., 2.], requires_grad=True)
f = x @ A @ x
f.backward()
analytic = (A + A.T) @ x       # ∇(x^T A x) = (A + A^T) x
print(f'Analytic: {analytic.tolist()},  Autograd: {x.grad.tolist()}')


# 대표 실험 ③ — AMP Loss Scaling 으로 underflow 우회 (Ch6-03)
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

class TinyOutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x) * 1e-5      # 일부러 작은 출력

if torch.cuda.is_available():
    model = TinyOutNet().cuda()
    scaler = GradScaler()
    x = torch.randn(32, 10).cuda()
    y = torch.randn(32, 1).cuda() * 1e-3

    with autocast():
        loss = ((model(x) - y) ** 2).mean()

    g_no = torch.autograd.grad(loss, model.fc.weight, retain_graph=True)[0]
    g_sc = torch.autograd.grad(scaler.scale(loss), model.fc.weight, retain_graph=True)[0]
    print(f'No scaling : zeros = {(g_no == 0).sum().item()} / {g_no.numel()}  → underflow')
    print(f'With S=2^16: gradient norm = {(g_sc.float() / scaler.get_scale()).norm():.2e}')


# 대표 실험 ④ — Memory Coalescing 의 100× 차이 (Ch4-03)
import time
N = 1024 * 1024
x = torch.randn(N, 128, device='cuda')

torch.cuda.synchronize(); t0 = time.time()
for _ in range(100): _ = x.sum(dim=1)         # contiguous, coalesced
torch.cuda.synchronize(); t_good = time.time() - t0

x_t = x.t().contiguous()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(100): _ = x_t.sum(dim=0)       # 같은 데이터, 비-coalesced 접근
torch.cuda.synchronize(); t_bad = time.time() - t0

print(f'Coalesced  : {t_good*1000:.2f} ms')
print(f'Non-coalesced: {t_bad*1000:.2f} ms  ({t_bad/t_good:.1f}× slower)')


# 대표 실험 ⑤ — Triton 으로 vector add (Ch5-03)
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

if torch.cuda.is_available():
    x = torch.randn(1024 * 1024, device='cuda')
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 1024),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    print(f'Triton vs PyTorch max diff: {(out - (x + y)).abs().max():.2e}')


# 대표 실험 ⑥ — torch.compile speedup 측정 (Ch7-01)
model = nn.Sequential(
    nn.Linear(1024, 4096), nn.ReLU(),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Linear(4096, 1024),
).cuda()
compiled = torch.compile(model)
x = torch.randn(256, 1024, device='cuda')

for _ in range(10): _ = model(x); _ = compiled(x)        # warmup
torch.cuda.synchronize(); t0 = time.time()
for _ in range(100): _ = model(x)
torch.cuda.synchronize(); t_eager = time.time() - t0

torch.cuda.synchronize(); t0 = time.time()
for _ in range(100): _ = compiled(x)
torch.cuda.synchronize(); t_compiled = time.time() - t0
print(f'Eager: {t_eager*1000:.1f}ms,  Compiled: {t_compiled*1000:.1f}ms,  Speedup: {t_eager/t_compiled:.2f}×')
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격** 으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 내부 구조가 PyTorch 사용자에게 중요한가** | 해당 시스템 수학이 어떤 PyTorch 한 줄을 설명하는지 |
| 3 | 📐 **수학적 선행 조건** | NN Theory · LA · Calc 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Computation graph · stride · memory hierarchy · IEEE 754 등의 시각화 |
| 5 | ✏️ **엄밀한 정의** | Tensor metadata · VJP · DispatchKey · roofline · loss scaling 등 |
| 6 | 🔬 **정리와 증명** | VJP 유도 · coalescing 분석 · loss scaling 정확성 등 $\square$ 종결 |
| 7 | 💻 **PyTorch + CUDA 구현 검증** | 4개 실험 (`### 실험 1` ~ `### 실험 4`) — custom Function · CUDA kernel · AMP · Triton |
| 8 | 🔗 **실전 활용** | 성능 최적화 가이드, profiling tools (Nsight, `torch.profiler`) |
| 9 | ⚖️ **가정과 한계** | 자동화의 한계, 수동 튜닝이 필요한 사례 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ 핵심 수식 + 표) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 기초 / 심화 / 논문 비평 의 3 문제, `<details>` 펼침 해설 |

> 📚 **연습문제 총 111개** (37 문서 × 3 문제): **기초 / 심화 / 논문 비평** 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. Tensor stride 손 계산부터 VJP 의 closed-form 유도, custom CUDA kernel 작성, GradScaler 의 dynamic 알고리즘, Triton block-level abstraction, TorchDynamo 의 graph break 분석까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 500~600줄 (정의·증명·코드·연습문제 포함) 기준 **약 60분~1시간 30분**. 전체 37문서는 약 **40~50시간** 상당 (custom CUDA kernel 작성·Nsight profiling 재현 포함 시 70시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "PyTorch 를 쓰지만 내부가 궁금하다" — 입문 투어 (1주, 약 14~16시간)</b></summary>

<br/>

```
Day 1  Ch1-01  Tensor = Storage + Metadata
       Ch1-02  Stride 와 Memory Layout
Day 2  Ch1-03  View vs Copy
       Ch2-01  Forward vs Reverse AD
Day 3  Ch2-02  VJP 의 수학
       Ch2-03  Computation Graph
Day 4  Ch2-05  Custom autograd.Function
       Ch3-01  Dispatcher Architecture
Day 5  Ch4-01  GPU 아키텍처
       Ch4-03  Memory Coalescing
Day 6  Ch6-01  IEEE 754 Floating Point
       Ch6-03  Loss Scaling
Day 7  Ch7-01  torch.compile 개관
       Ch7-02  TorchDynamo
```

</details>

<details>
<summary><b>🟡 "Autograd 와 CUDA 를 완전 정복한다" — 시스템 집중 (2주, 약 26~30시간)</b></summary>

<br/>

```
1주차 — Tensor · Autograd · Dispatcher
  Day 1   Ch1-01~03   Storage · Stride · View
  Day 2   Ch1-04~05   Dtype · Device & CUDA Context
  Day 3   Ch2-01~02   JVP/VJP · VJP 수학
  Day 4   Ch2-03~04   Computation Graph · Backward
  Day 5   Ch2-05~06   Custom Function · Double Backward
  Day 6   Ch3-01~03   Dispatcher · ATen/c10 · Autograd Key
  Day 7   Ch3-04~05   TORCH_LIBRARY · functorch

2주차 — CUDA · Custom Kernel · AMP · Compile
  Day 1   Ch4-01~02   GPU 아키텍처 · Memory Hierarchy
  Day 2   Ch4-03~04   Coalescing · Bank Conflict
  Day 3   Ch4-05~06   Warp Divergence · Reduction 최적화
  Day 4   Ch5-01~03   cpp_extension · CUDA Pointer · Triton
  Day 5   Ch5-04~05   cuBLAS/cuDNN · Kernel Fusion
  Day 6   Ch6-01~05   Mixed Precision 전체
  Day 7   Ch7-01~05   torch.compile 전체
```

</details>

<details>
<summary><b>🔴 "PyTorch 의 시스템 수학을 완전 정복한다" — 전체 정복 (8주, 약 40~50시간 + custom kernel·profiling 15~20시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — Tensor 내부구조
         → stride 손 계산으로 view/transpose 직접 만들기
         → NCHW vs NHWC 의 cuDNN dispatch 비교
         → Type promotion 의 lattice 직접 그리기

2주차   Chapter 2 전체 — Autograd
         → JVP/VJP 의 cost 비교 ($O(n)$ vs $O(m)$) 측정
         → Custom Function 으로 ReLU·Softmax·CrossEntropy 작성
         → torchviz 로 graph 시각화 + double backward 로 Hessian-Vector Product

3주차   Chapter 3 전체 — Dispatcher
         → torch._C._dispatch_dump_table 로 routing 관찰
         → TORCH_LIBRARY 로 custom op + autograd 등록
         → torch.func.vmap(grad(f)) 로 per-sample gradient

4주차   Chapter 4 전체 — CUDA 기초
         → Roofline plot 으로 elementwise/reduce/GEMM 분류
         → Coalesced vs non-coalesced micro-benchmark
         → Mark Harris 7-step reduction 직접 구현

5주차   Chapter 5 전체 — Custom Kernel
         → cpp_extension 으로 fused Linear+ReLU
         → Triton 으로 fused softmax + autotune
         → cuBLAS GEMM vs Triton GEMM throughput

6주차   Chapter 6 전체 — Mixed Precision
         → FP16 underflow 재현 → loss scaling 으로 해결
         → BF16 vs FP16 vs TF32 의 accuracy 비교
         → Stochastic rounding · Kahan summation 효과 측정

7주차   Chapter 7 (1~3) — torch.compile / Dynamo / AOTAutograd
         → torch._dynamo.explain 으로 graph break 분석
         → AOTAutograd 의 functionalization 결과 추출
         → 같은 모델 의 eager / dynamo / aot / inductor 단계별 throughput

8주차   Chapter 7 (4~5) + 종합
         → Inductor 가 생성한 Triton kernel 직접 읽기 (TORCH_LOGS=output_code)
         → torch.compile + DDP 호환성 실험 (small cluster 또는 simulated)
         → 다음: Distributed Training Deep Dive 로
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | Backpropagation · chain rule · loss landscape | **Ch2 전체** (reverse AD) |
| [calculus-and-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-and-optimization-deep-dive) | Jacobian · multi-variable Taylor · gradient descent | **Ch2 전체** (VJP), **Ch6** (loss scaling 의 정확성) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | Matrix–matrix product · BLAS · LAPACK | **Ch3** (aten ops), **Ch5** (cuBLAS) |
| [information-theory-deep-dive](https://github.com/iq-ai-lab/information-theory-deep-dive) | Quantization · floating-point error · entropy | **Ch6** (FP16/BF16/INT8 수치 분석) |
| [cnn-deep-dive](https://github.com/iq-ai-lab/cnn-deep-dive) | NCHW/NHWC · Roofline · Winograd | **Ch1-02** (stride), **Ch4** (memory hierarchy) |
| [llm-inference-deep-dive](https://github.com/iq-ai-lab/llm-inference-deep-dive) | Flash Attention · KV cache · HBM/SRAM | **Ch4-05** (warp divergence), **Ch5-03** (Triton), **Ch5-05** (kernel fusion) |
| [distributed-training-deep-dive](https://github.com/iq-ai-lab/distributed-training-deep-dive) *(다음)* | NCCL · DDP · FSDP · ZeRO · pipeline · tensor parallelism | **Ch7-05** 이후 자연스러운 연결 |

> 💡 이 레포는 **"PyTorch 한 줄 (`backward`, `compile`, `autocast`, `torch.ops.mylib.myop`) 뒤의 시스템 수학"** 에 집중합니다. NN Theory 에서 chain rule 을, LA 에서 BLAS 를, Calc 에서 Jacobian 을 익힌 후 오면 Chapter 2 (VJP) 와 Chapter 4 (CUDA) 의 유도가 훨씬 자연스럽습니다. **CNN Deep Dive** 와 **LLM Inference Deep Dive** 와 함께 보면 Ch4 의 memory hierarchy 와 Ch5 의 kernel fusion 이 실제 모델 성능과 어떻게 연결되는지 선명해집니다.

---

## 📖 Reference

### 🏛️ Autograd · Automatic Differentiation
- **Automatic Differentiation in Machine Learning: a Survey** (Baydin et al., 2018) — JVP/VJP 정리적 비교
- **Fast Exact Multiplication by the Hessian** (Pearlmutter, 1994) — **HVP trick 효시**
- **Functional Programming for Machine Learning** (Wang et al., 2019) — functorch / JAX 의 영감

### 🏗️ PyTorch Internals
- **PyTorch Documentation** — pytorch.org/docs (공식)
- **PyTorch Internals** (Edward Yang, 2019) — 내부 구조의 bible
- **Let's Talk About the PyTorch Dispatcher** (Edward Yang, 2020)
- **The State of PyTorch in 2024** (Soumith Chintala, 다수 발표)

### 🎮 CUDA Programming
- **CUDA C Programming Guide** (NVIDIA, 최신) — 공식
- **Programming Massively Parallel Processors** (Kirk & Hwu, 4th ed.) — CUDA 교과서
- **Optimizing Parallel Reduction in CUDA** (Mark Harris, NVIDIA) — **7-step reduction 효시**
- **CUDA Best Practices Guide** (NVIDIA) — coalescing · bank conflict
- **Volta / Ampere / Hopper Architecture Whitepapers** (NVIDIA)

### ⚡ Triton & Modern Kernel Languages
- **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** (Tillet, Kung, Cox, 2019) — **Triton 효시**
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022) — Triton 의 대표 응용
- **FlashAttention-2** (Dao, 2023)
- **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention** (Kwon et al., 2023) — Triton 기반 inference

### 🔢 Mixed Precision
- **Mixed Precision Training** (Micikevicius et al., 2018) — **NVIDIA Apex / loss scaling 효시**
- **A Study of BFLOAT16 for Deep Learning Training** (Kalamkar et al., 2019) — **BF16**
- **NVIDIA TensorFloat-32** (NVIDIA, A100 whitepaper) — **TF32**
- **Stochastic Rounding: Implementation, Error Analysis and Applications** (Croci et al., 2022) — **Stochastic rounding**
- **Training Deep Neural Networks with 8-bit Floating Point Numbers** (Wang et al., 2018) — INT8/FP8

### 🚀 PT 2.0 — torch.compile
- **PyTorch 2.0** (Horace He et al., 2023 PyTorch Conference)
- **TorchDynamo** (Jason Ansel, 2022) — bytecode capture
- **AOTAutograd** (Horace He, 2022) — graph-level autograd
- **TorchInductor** (Jason Ansel, Yanan Cao, 2022) — Triton/C++ codegen
- **PEP 523: Adding a frame evaluation API to CPython** (2016)

### 🛠️ Implementation · Libraries
- **PyTorch Source Code** (github.com/pytorch/pytorch) — `aten/`, `c10/`, `torch/csrc/autograd/`
- **NVIDIA Apex** (github.com/NVIDIA/apex) — early AMP / fused optimizer
- **NVIDIA Nsight Compute / Systems** — kernel profiling
- **CUTLASS** (NVIDIA) — composable CUDA templates for GEMM

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`loss.backward()` 를 호출하는 것과 — Baydin 2018 로 reverse-mode AD = $J^\top v$ 의 VJP 가 ML 의 scalar loss·vector params 구조에서 자연스러움을 한 줄씩 유도 · Pearlmutter 1994 로 $Hv = \nabla(\nabla L \cdot v)$ 의 Hessian-free trick 으로 double backward 가 어떻게 가능한지 분석 · Mark Harris 의 7-step reduction 으로 naive ($O(n)$) → tree ($O(\log n)$) → warp shuffle 의 진화를 직접 구현 · Tillet 2019 로 Triton 의 `@triton.jit` block-level abstraction 이 CUDA 보다 단순함을 작성 · Micikevicius 2018 로 loss scaling $L_{\text{scaled}} = L \cdot S$ 가 FP16 의 $6.1 \times 10^{-5}$ underflow threshold 를 어떻게 우회하는지 수치 검증 · PEP 523 + Ansel 2022 로 TorchDynamo 가 CPython bytecode 를 어떻게 다시 쓰고 graph break 가 언제 발생하는지 분석 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
