# 05. Distributed Training 과의 통합

## 🎯 핵심 질문

- **DDP (Distributed Data Parallel)**: Gradient bucket allreduce 가 torch.compile graph 에 어떻게 표현되는가?
- **FSDP (Fully Sharded Data Parallel)**: Parameter sharding 의 all-gather / reduce-scatter 가 compile 시 인식되는가?
- `torch.distributed` collective ops (NCCL backend) 의 async stream 관리는?
- `compiled_ddp_module` 의 구현 원리는? Gradient synchronization 를 graph level 에서 처리할 수 있는가?
- 왜 PT 2.1+ 에서 FSDP + torch.compile 호환성이 본격화되었는가?

---

## 🔍 왜 Distributed + Compile 통합이 중요한가

Large-scale training 에서:

**PT 1.x (Eager mode)**:
- DDP: `reducer.py` 가 backward 후에 gradients 동기화 (수동 scheduling)
- All-reduce 와 computation 이 disjoint (pipeline 불가능)
- Gradient synchronization overhead 가 분명히 보임

**PT 2.0+ (Compiled mode)**:
- Graph capture 시 collective ops 도 node 로 포함
- AOTAutograd: backward 와 all-reduce 를 동시에 생성
- Inductor: backward kernel + all-reduce kernel 을 overlapping 가능
- **Communication-computation overlap** 자동 달성

결과: **2-4GPU cluster 에서 5~10% 추가 speedup** (communication latency 숨김).

---

## 📐 수학적 선행 조건

- **DDP Deep Dive** (distributed-training-deep-dive repo): AllReduce algorithm
- **FSDP**: Parameter sharding, all-gather/reduce-scatter
- **NCCL**: Collective communication on GPU
- **Async streams**: CUDA stream dependency, overlap
- **Ring allreduce**: bandwidth-optimal algorithm

---

## 📖 직관적 이해

### DDP + Compile 의 파이프라인

```
EAGER DDP (PT 1.x)
┌──────────────────────────────────┐
│ Backward pass (single GPU)        │
│ compute gradients → tape          │
└──────────────────┬───────────────┘
                   │ backward 완료
                   ↓
┌──────────────────────────────────┐
│ Allreduce (모든 GPUs)             │
│ sum gradients across N GPUs       │
│ (blocking call)                   │
└──────────────────┬───────────────┘
                   │
                   ↓ (이제야 gradient 동기화)
              Optimizer step
              
Timeline:  [backward..................] [allreduce.....] [optim]
           GPU busy                    GPU idle (waiting for all-reduce)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPILED DDP (PT 2.0+)
┌──────────────────────────────────────────┐
│ Forward + backward + allreduce (1 graph)  │
│                                          │
│ [backward kernel] → [allreduce kernel]   │
│ (same CUDA stream or overlapped streams)  │
└──────────────────┬──────────────────────┘
                   │
                   ↓
              Optimizer step

Timeline:  [backward+allreduce overlapped] [optim]
           GPU utilization higher (less idle time)
           Communication latency 숨김 (hidden)
```

### FSDP 의 구조적 변화

**PT 1.x (Eager)**:
```python
# Forward
x = model(x)  # All parameters 사용

# Backward: parameter 별로 reduce-scatter
grad_x = loss.backward()
# (external scheduler: when to do all-gather? when to do reduce-scatter?)
```

**PT 2.0+ (Compiled with FSDP)**:
```
Graph representation:
[forward with all-gather] → [backward] → [reduce-scatter]
               ↓
        Inductor 가 all-gather, backward, reduce-scatter
        의 relative ordering + stream dependency 최적화

Result: No manual scheduler 필요, automatic fusion possible
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Collective Op 의 Graph Representation

All-Reduce op in FX graph:

$$
\text{allreduce} : \mathbb{R}^{D} \times \mathcal{G} \to \mathbb{R}^D
$$

여기서:
- Input: gradient tensor $g \in \mathbb{R}^D$
- $\mathcal{G}$: process group (0, 1, ..., N-1 GPUs)
- Output: summed gradient $\bar{g} = \sum_{i=0}^{N-1} g_i / N$ (또는 다른 reduction op)

FX graph 에서:

```
FX node: call_function(torch.distributed.all_reduce, [grad_tensor, process_group])
```

AOTAutograd 는 이를 backward graph 에 추가:

$$
G_{\text{bwd}} \ni \{\text{backward ops}, \text{allreduce op}\}
$$

### 정의 5.2 — Ring AllReduce 의 Optimization

Ring allreduce: N GPU 에서 gradient synchronization, bandwidth-optimal.

$$
\text{rounds} = 2(N-1), \quad \text{bandwidth} = \frac{N-1}{N} \times B_{\text{full}}
$$

Compile-time 최적화:

$$
\text{overlap\_factor} = \frac{\text{backward\_compute\_time}}{\text{allreduce\_comm\_time}}
$$

- overlap_factor $> 1$: computation 가 communication 을 숨길 수 있음 (async streams 사용)
- overlap_factor $\leq 1$: communication bottleneck (fusion 도움 안 됨)

### 정의 5.3 — Gradient Bucketing in Compiled DDP

Traditional DDP: 크기별로 bucket 을 만들어 all-reduce (bandwidth 효율, latency hide)

Compiled DDP: FX graph 에서:

$$
\text{bucket}_i = \{\text{params}\ p_j : \text{index}(p_j) \in [k_i, k_{i+1})\}
$$

Each bucket $i$ 는 independent all-reduce node → Inductor 가 backward + all-reduce 를 per-bucket 으로 schedule.

---

## 🔬 정리와 증명

### 정리 5.1 — Communication-Computation Overlap 의 Speedup

Backward time $t_b$, all-reduce time $t_a$, total time $t_b + t_a$.

Overlap (async streams) 가능하면:

$$
t_{\text{overlap}} = \max(t_b, t_a)
$$

Speedup:

$$
\boxed{\text{Speedup} = \frac{t_b + t_a}{\max(t_b, t_a)} = 1 + \min\left(\frac{t_a}{t_b}, \frac{t_b}{t_a}\right)}
$$

최대 speedup: $t_b \approx t_a$ 일 때, 거의 $2 \times$.

**증명**: CUDA stream 은 GPU kernel 을 asynchronously queue 할 수 있음 (host CPU wait 없음). Backward kernel 과 all-reduce kernel 이 다른 stream 에서 동시 실행 가능. 따라서 total time = $\max(t_b, t_a)$. $\square$

### 정리 5.2 — FSDP + Compile 의 메모리 효율

Sharded parameter: 각 GPU 가 full parameter 의 1/N 만 저장.

Compile 시 all-gather timing:

$$
\text{memory\_peak} = \text{model\_size} / N + \text{activation\_size}
$$

(모든 parameter 를 동시에 hold 하지 않음, all-gather 는 필요 시에만)

Non-compiled FSDP 대비:

$$
\text{saving\_factor} = \frac{N}{N-1} \quad (\text{roughly})
$$

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — 기본 DDP + torch.compile

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed (requires torchrun or torch.distributed.launch)
# For testing: MASTER_ADDR=localhost MASTER_PORT=29500 torchrun --nproc_per_node=2 script.py

def setup_ddp():
    dist.init_process_group(backend='nccl')

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    setup_ddp()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    
    # Model and DDP
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
        nn.Linear(1024, 10)
    ).to(device)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    # Compile DDP
    ddp_model = torch.compile(ddp_model, mode='reduce-overhead')
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    x = torch.randn(256, 1024, device=device)
    y_target = torch.randint(0, 10, (256,), device=device)
    
    print(f"[Rank {rank}] Starting training with compiled DDP")
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward + backward
        logits = ddp_model(x)
        loss = loss_fn(logits, y_target)
        loss.backward()
        
        # Gradients already synchronized by DDP wrapper
        # (in PT 2.0+, can happen inside compiled graph)
        
        optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")
    
    cleanup_ddp()

if __name__ == '__main__':
    main()

# Run with:
# MASTER_ADDR=localhost MASTER_PORT=29500 torchrun --nproc_per_node=2 script.py
```

### 실험 2 — DDP Speedup 측정

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

def benchmark_ddp(compiled=False):
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')
    
    model = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048)
    ).to(device)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    if compiled:
        ddp_model = torch.compile(ddp_model, mode='reduce-overhead')
    
    x = torch.randn(128, 2048, device=device, requires_grad=True)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        loss = ddp_model(x).sum()
        loss.backward()
        optimizer.step()
    
    dist.barrier()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    
    num_iters = 50
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = ddp_model(x).sum()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    t = time.time() - t0
    
    return t / num_iters * 1000  # ms per iteration

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    time_eager = benchmark_ddp(compiled=False)
    time_compiled = benchmark_ddp(compiled=True)
    
    if rank == 0:
        print(f"=== DDP Speedup ===")
        print(f"Eager:    {time_eager:.2f}ms/iter")
        print(f"Compiled: {time_compiled:.2f}ms/iter")
        print(f"Speedup:  {time_eager/time_compiled:.2f}×")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

# Expected output (2 GPUs):
# Eager:    25.00ms/iter
# Compiled: 22.50ms/iter
# Speedup:  1.11×  (5-15% typical for DDP)
```

### 실험 3 — FSDP + torch.compile (PT 2.1+)

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')
    
    # Model with FSDP wrapping
    model = nn.Sequential(
        nn.Linear(4096, 8192),
        nn.ReLU(),
        nn.Linear(8192, 4096)
    )
    
    fsdp_model = FSDP(model)
    
    # PT 2.1+: compile works with FSDP
    try:
        fsdp_model = torch.compile(fsdp_model, mode='max-autotune')
        print(f"[Rank {rank}] Compiled FSDP model")
    except Exception as e:
        print(f"[Rank {rank}] FSDP compile not supported: {e}")
        # Fallback to eager FSDP
    
    x = torch.randn(64, 4096, device=device)
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=0.001)
    
    # Forward pass triggers all-gather
    y = fsdp_model(x)
    loss = y.sum()
    
    # Backward + reduce-scatter
    loss.backward()
    optimizer.step()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

# PT 2.0: may not compile FSDP
# PT 2.1+: partial support (all-gather, reduce-scatter 표현)
# PT 2.2+: full support (optimization possible)
```

### 실험 4 — Collective op 직접 사용

```python
import torch
import torch.distributed as dist

def test_allreduce_in_graph():
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')
    
    def allreduce_op(x):
        # Manual all-reduce
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x / dist.get_world_size()
    
    # PT 2.1+: torch.compile can trace all_reduce
    compiled = torch.compile(allreduce_op)
    
    x = torch.randn(1000, 1000, device=device)
    
    # Run
    y = compiled(x)
    
    print(f"[Rank {rank}] All-reduce completed")

# Result:
# - PT 2.0: May not trace dist.all_reduce (fallback to eager)
# - PT 2.1+: Traces as FX node, can optimize scheduling
```

---

## 🔗 실전 활용

### 1. Large-scale training 에서 DDP + compile

```python
# Multi-GPU training template
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train():
    dist.init_process_group(backend='nccl')
    
    # Create model on local GPU
    model = MyModel().to(torch.device(f'cuda:{dist.get_rank()}'))
    
    # Wrap with DDP
    model = DDP(model)
    
    # Compile for performance
    model = torch.compile(model, mode='reduce-overhead')
    
    # Training loop unchanged
    for epoch in range(num_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
    
    dist.destroy_process_group()
```

### 2. Graph break 진단 in distributed setting

```python
import torch._dynamo

# Enable verbose logging for DDP + compile debugging
torch._dynamo.config.verbose = True

model = DDP(model)
compiled = torch.compile(model)

# First forward pass: check for graph breaks
_ = compiled(x)

# If graph breaks in collective ops:
# - PT < 2.1: Expected (collective ops not traced)
# - PT >= 2.1: May still break if FSDP scheduling complex
```

### 3. Manual communication-computation overlap

```python
# For custom all-reduce, manually overlap with backward
def backward_with_allreduce(loss, model, world_size):
    # Start all-reduce in background
    all_reduce_handles = []
    
    loss.backward()
    
    for param in model.parameters():
        if param.grad is not None:
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True  # Non-blocking
            )
            all_reduce_handles.append(handle)
    
    # Wait for all-reduces
    for handle in all_reduce_handles:
        handle.wait()
    
    # Gradient synchronization complete
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Synchronized forward-backward iterations | Stragglers: some GPUs slower (rare in practice) |
| Static parameter sharding (FSDP) | Dynamic sharding: not yet supported |
| NCCL backend | Other backends (gloo, mpi): less optimized for compile |
| Fast inter-GPU communication (NVLink) | Slow network (ethernet): overlap marginal |
| Deterministic collective ops | Debugging non-determinism: harder in compiled graph |

---

## 📌 핵심 정리

$$\boxed{\text{Compile + DDP} = \text{Backward} + \text{AllReduce (overlapped)}}$$

$$\boxed{\text{Compile + FSDP} = \text{AllGather} + \text{Backward} + \text{ReduceScatter}}$$

| 기법 | 소개 | 이점 |
|------|------|------|
| **DDP + compile** | Gradient bucket + all-reduce fused | 5~10% 속도 향상 |
| **FSDP + compile** | Parameter all-gather in graph | Memory 절약 + schedule opt |
| **Collective ops tracing** | All-reduce, broadcast in FX graph | Fusion 가능 |
| **Async stream mgmt** | Backward ∥ all-reduce | Latency hiding |

---

## 🤔 생각해볼 문제

**문제 1** (기초): DDP 에서 all-reduce 가 필요한 이유를 설명하라. Compiled mode 에서 이를 graph 에 포함하면 어떤 이점이 있는가?

<details>
<summary>해설</summary>

**DDP 의 all-reduce**:
- 각 GPU 에서 독립적으로 backward 계산 (local gradient)
- 모든 GPU 의 gradient 를 합산 (동기화)
- 동일한 updated parameter 로 step (consistency)

**Compile 에 포함**:
- Graph 에서 backward + all-reduce 를 sequential 이 아닌 overlapping 으로 scheduling 가능 (async streams)
- HBM ↔ network roundtrip 을 backward computation 과 hide
- Result: ~1.1× speedup (communication latency 절반 정도 숨김)

**Eager 비교**: Backward 완료 → explicit all-reduce call → blocking. Overlap 불가능.  $\square$

</details>

**문제 2** (심화): FSDP 에서 all-gather (forward) 와 reduce-scatter (backward) 의 순서가 중요한 이유는?

<details>
<summary>해설</summary>

**FSDP 의 메모리 최적화**:

```
Forward:
  [param shards on all GPUs]
  → all-gather → full parameter (compute)
  → discard after use (memory freed)

Backward:
  All-gradients 계산 (각 GPU 의 local shard 에 대해)
  → reduce-scatter (모든 gradients 를 합산 후 shard)
  → 각 GPU 는 자신의 shard 의 gradient 만 keep
```

**Order importance**:
- All-gather before forward: parameter 필요
- Reduce-scatter after backward: gradient 동기화

반대 순서면 all-gather 후 parameter 가 남아있어 메모리 waste.

**Compile 의 역할**: Forward 와 backward 를 동시에 보므로, all-gather timing 과 reduce-scatter timing 을 최적 위치에 자동 삽입 가능 (eager 는 manual).  $\square$

</details>

**문제 3** (논문 비평): Ring AllReduce 의 bandwidth optimality 를 증명하고, Compile 이 이를 improve 할 수 있는가?

<details>
<summary>해설</summary>

**Ring AllReduce optimality**:

Total data moved: $N \times D$ (N GPU, D = parameter size)
Optimal bandwidth usage: $\frac{N-1}{N} \times B_{\text{peak}}$ (한 GPU 제외, always sending/receiving)

Ring 은 이를 달성 (bandwidth-optimal).

**Compile improvement**:

Ring AllReduce itself 는 이미 optimal, but:
- **Computation overlap** 가능: forward/backward 와 동시 all-reduce
- **Asynchronous scheduling**: 여러 all-reduce (multi-ring) 을 interleave
- **Gradient bucketing** 최적화: bucket 크기 auto-tuning

따라서 **sequential composition 의 overlap** 에서만 improvement. All-reduce 자체는 이미 optimal.  $\square$

</details>

---

<div align="center">

[◀ 이전](./04-torchinductor.md) | [📚 README](../README.md) | [다음 ▶](../README.md)

</div>
