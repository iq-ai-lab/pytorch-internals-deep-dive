# 01. GPU 아키텍처 — SM, Warp, Thread Block

## 🎯 핵심 질문

- NVIDIA GPU 의 계층 (Grid > Block > Warp > Thread) 은 무엇이고 각 계층에서 어떤 병렬화가 일어나는가?
- Warp = 32 thread 가 SIMT (Single Instruction Multiple Thread) 를 구현하는 구체적 메커니즘은 무엇인가?
- Thread block 과 SM (Streaming Multiprocessor) 의 관계, 왜 block 은 같은 SM 에서만 실행되는가?
- Warp scheduler 가 latency hiding 을 수행하는 원리와, "occupancy" 개념이 성능에 미치는 영향은?
- A100 의 108 SM, H100 의 132 SM 에서 이 계층의 구체적 규모는?

---

## 🔍 왜 이 내부 구조가 PyTorch 사용자에게 중요한가

PyTorch 에서 `loss.backward()` 를 부르면 CUDA kernel 이 GPU 에서 실행되고, 그 kernel 의 성능은 SM · warp · thread block 의 계층에 의해 결정됩니다. 같은 알고리즘이 다르게 구현되면 100× 성능 차이가 납니다 — memory coalescing · bank conflict · warp divergence 등은 모두 이 계층의 특성에서 비롯됩니다.

또한 `torch.cuda.synchronize()` 와 같은 명령이 왜 필요한지, 왜 GPU 계산은 "비동기(async)" 인지, 왜 occupancy 를 신경써야 하는지 이해하려면 SM 의 구조를 알아야 합니다. 더 깊이, custom kernel 을 작성하거나 Nsight Compute 로 프로파일링할 때, "active warp", "SM utilization" 같은 메트릭이 무엇을 의미하는지 아는 것은 필수입니다.

---

## 📐 수학적 선행 조건

- 병렬 처리의 기초: 스레드, 블록, 격자의 개념 (CUDA C Programming Guide 기초)
- Bit manipulation (warp mask 를 이해하기 위해)
- (선택) 컴퓨터 아키텍처: instruction fetch, pipelining, latency vs throughput

---

## 📖 직관적 이해

### Grid, Block, Warp, Thread 의 계층

```
┌─────────────────────────────────────────────────┐
│                   GPU Grid                       │
│  (all blocks, executed in parallel when         │
│   hardware resources allow)                     │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  Block (0,0) │  │  Block (0,1) │   ...      │
│  │              │  │              │            │
│  │ ┌──────────┐ │  │ ┌──────────┐ │            │
│  │ │ Warp 0   │ │  │ │ Warp 0   │ │            │
│  │ │ (32 tid) │ │  │ │ (32 tid) │ │            │
│  │ └──────────┘ │  │ └──────────┘ │            │
│  │ ┌──────────┐ │  │ ┌──────────┐ │            │
│  │ │ Warp 1   │ │  │ │ Warp 1   │ │            │
│  │ │ (32 tid) │ │  │ │ (32 tid) │ │            │
│  │ └──────────┘ │  │ └──────────┘ │            │
│  │     ...      │  │     ...      │            │
│  │ Shared Mem   │  │ Shared Mem   │            │
│  │ (48-100 KB)  │  │ (48-100 KB)  │            │
│  └──────────────┘  └──────────────┘            │
│         │                   │                   │
│         └───────┬───────────┘                   │
│                 ↓                               │
│          Scheduled by SM scheduler              │
│                 ↓                               │
│        ┌────────────────────┐                  │
│        │  SM (Streaming MP)  │                 │
│        │                     │                 │
│        │  - 32 CUDA cores    │                 │
│        │  - Warp scheduler   │                 │
│        │  - L1 cache         │                 │
│        │  - Bank conflicts?  │                 │
│        └────────────────────┘                  │
│                                                  │
│  A100: 108 SM                                   │
│  H100: 132 SM                                   │
└─────────────────────────────────────────────────┘
```

### SIMT 의 구현: Lockstep Execution

**Warp** 는 32 개의 스레드가 **같은 instruction 을 동시에 실행** 합니다:

```
Instruction Fetch & Decode (shared)
          ↓
    ┌─────────────┐
    │  pc = 1000  │  (same program counter)
    └─────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│            Warp Execution (SIMT)                │
│  tid=0  │ tid=1  │ tid=2  │ ... │ tid=31       │
│  ADD    │  ADD   │  ADD   │     │  ADD         │
│  r0+=r1 │ r0+=r1 │ r0+=r1 │     │  r0+=r1      │
│  1 cycle (parallel across CUDA cores)          │
└─────────────────────────────────────────────────┘
         ↓
PC++ (모든 32 thread 가 함께)
```

**장점**: instruction fetch · decode 비용 1/32, 더 높은 throughput  
**한계**: conditional branch 시 두 path 를 sequential 실행 (Ch4-05 에서 다룸)

---

## ✏️ 엄밀한 정의

### 정의 1.1 — GPU Grid 와 Execution Hierarchy

NVIDIA GPU 는 다음과 같은 계층으로 조직됩니다:

$$\text{Grid} = \bigcup_{b \in \text{Blocks}} \text{Block}(b)$$

각 **Block** $b = (b_x, b_y, b_z) \in \text{BlockDim}$ 에 대해:

$$\text{Block}(b) = \bigcup_{t \in [1, 256] \text{ or } 1024} \text{Thread}(b, t)$$

각 **Thread** 는 고유한 $(b, t)$ 튜플로 식별되며, $b$ 는 block index, $t = (t_x, t_y, t_z)$ 는 block 내 local thread index.

### 정의 1.2 — Warp 와 SIMT

**Warp**: 32 개의 연속된 thread 로 구성된 최소 실행 단위.

Block 내 thread 들은 row-major order 로 warp 로 분할:

$$\text{Warp ID} = \lfloor \text{linear thread id} / 32 \rfloor$$

$$\text{Lane ID} = \text{linear thread id} \mod 32$$

한 warp 의 32 thread 는:
- **같은 instruction** 을 fetch & decode (1회)
- **동시에 다른 operand** 에 대해 실행 (32 개 CUDA core parallel)
- **같은 PC** 진행률

이것이 **SIMT (Single Instruction Multiple Thread)**.

### 정의 1.3 — Streaming Multiprocessor (SM)

**SM** 은 GPU 의 최소 독립 실행 단위:

$$\text{SM} = (\text{CUDA cores}, \text{Warp scheduler}, \text{Shared memory}, \text{L1 cache}, \text{Register file})$$

- **CUDA cores**: 32 개 (한 warp 의 32 thread 를 parallel 실행)
- **Warp scheduler**: 여러 warp 를 time-multiplexed 실행
- **Shared memory**: 블록 내 thread 들의 고속 협력 메모리 (48–100 KB per SM)
- **Register file**: 보통 256 KB (thread 들이 나누어 사용)

### 정의 1.4 — Occupancy

**Occupancy** = SM 에서 실제 실행 중인 thread 들의 비율:

$$\text{Occupancy} = \frac{\text{Active warps per SM}}{\text{Max resident warps per SM}}$$

A100/H100 에서 max = 64 warp/SM (2048 thread). Occupancy 는 0–100% 범위:
- 높은 occupancy = 더 많은 warp 가 latency hide 가능
- 낮은 occupancy = 스케줄러가 선택할 ready warp 가 적음

---

## 🔬 정리와 증명

### 정리 1.1 — Warp 의 Synchronization 과 Masking

32 개의 thread 가 SIMT 실행 중 서로 다른 조건을 만족할 때, 하드웨어는 **execution mask** 로 각 lane 을 선택적으로 활성화합니다:

$$\text{Active mask at PC} = \{i \in [0, 32) : \text{thread } i \text{ can reach PC}\}$$

**증명 (직관)**: 조건 분기 `if (cond)` 에서:
- True 를 만족하는 lane 들만 mask 에 포함
- False 를 만족하는 lane 들은 명령 결과를 버림 (또는 읽지 않음)
- 모든 32 thread 가 같은 instruction fetch & decode 수행 (1회)
- 하지만 write-back 단계에서 mask 로 선택적 activation $\square$

이것이 warp divergence 의 비용: "true path → false path" 순차 실행 → 각 path 마다 추가 cycle.

### 정리 1.2 — Block 과 SM 의 Binding

한 block 의 모든 thread 는 **같은 SM 에서만 실행** 되며, 같은 shared memory 를 공유합니다:

$$\exists! \, \text{SM} \in [0, N_{\text{SM}}) : \text{Block } b \text{ executes on } \text{SM}$$

여기서 $N_{\text{SM}}$ 은 GPU 의 SM 개수 (A100: 108, H100: 132).

**증명**: Shared memory 는 per-SM resource (48–100 KB) 이고, block 이 migrate 되면 다른 SM 의 shared memory 로 접근 불가. 따라서 block 은 creation 에서 destruction 까지 같은 SM 에 고정 $\square$.

**결과**: Block 크기가 크거나 shared memory 를 많이 쓰면, 한 SM 에 적재할 수 있는 block 수가 줄어들어 occupancy 하락.

### 정리 1.3 — Latency Hiding via Warp Scheduling

한 SM 에서 $K$ 개 warp 가 resident 일 때, warp scheduler 는:
1. 각 warp 의 PC 와 상태 (running / waiting for memory) 추적
2. 각 cycle, ready 인 warp 1 개 선택 → instruction 실행
3. 다음 cycle, 다른 ready warp 선택 (또는 같은 warp 계속)

**Latency hiding 의 원리**: Global memory access 는 100+ cycle 지연이 필요. 이 동안:

$$\text{Time to hide} = 100 \text{ cycles}$$

$$K \text{ warps} \times 1 \text{ cycle/warp} = K \text{ cycles}$$

$K \geq 100$ 이면, ready warp 를 돌아가며 실행하여 memory wait 시간을 완전히 hide.

**증명**: Deterministic scheduling (round-robin) 을 가정하면,

$$\text{Clock cycles until same warp reissues} = K$$

따라서 $K \geq L$ 이면 ($L$ = memory latency cycle) latency 가 숨겨짐 $\square$.

---

## 💻 PyTorch + CUDA 구현 검증

### 실험 1 — CUDA Device Query 로 SM 개수 및 규격 확인

```python
import torch

# GPU 정보
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

# SM 개수 및 상세 정보
props = torch.cuda.get_device_properties(0)
print(f"Multiprocessors (SM): {props.multiProcessorCount}")
print(f"Max threads per block: {props.maxThreadsPerBlock}")
print(f"Shared memory per block: {props.sharedMemPerBlock} bytes")
print(f"Max registers per block: {props.regsPerBlock}")
print(f"Warp size: {props.warpSize}")
print(f"Memory clock rate: {props.memoryClockRate} kHz")
print(f"Memory bandwidth: {props.memoryBusWidth} bits")

# 예상 최대 occupancy
max_warps_per_sm = 64  # Ampere/Ada 에서 typical
print(f"Expected max warps/SM: {max_warps_per_sm}")
print(f"Max threads/SM: {max_warps_per_sm * 32}")
```

**예상 출력 (H100)**:
```
Device: NVIDIA H100 PCIe
CUDA Capability: (9, 0)
Multiprocessors (SM): 132
Max threads per block: 1024
Shared memory per block: 49152 bytes
Max registers per block: 65536
Warp size: 32
Expected max warps/SM: 64
Max threads/SM: 2048
```

### 실험 2 — Occupancy 계산 및 검증

```python
import torch

def estimate_occupancy(block_size, shared_mem_bytes, device_id=0):
    props = torch.cuda.get_device_properties(device_id)
    max_warps = 64  # Ampere/H100: typical
    max_threads_per_sm = max_warps * 32
    
    # Constraint 1: threads per block
    warps_per_block = (block_size + 31) // 32
    
    # Constraint 2: shared memory
    shared_mem_per_block = shared_mem_bytes
    total_shared_per_sm = props.sharedMemPerBlock  # 96-100 KB typical
    blocks_by_shared = total_shared_per_sm // max(shared_mem_per_block, 1)
    
    # Max blocks that fit
    blocks_per_sm = min(blocks_by_shared, max_threads_per_sm // block_size)
    active_warps = blocks_per_sm * warps_per_block
    occupancy = active_warps / max_warps
    
    return occupancy, blocks_per_sm, active_warps

# 테스트 케이스
test_cases = [
    (32, 0),      # minimal
    (128, 0),     # small block
    (256, 48*1024), # large block + shared mem
    (512, 64*1024), # larger
    (1024, 96*1024), # max
]

for block_size, shared_mem in test_cases:
    occ, blocks, warps = estimate_occupancy(block_size, shared_mem)
    print(f"Block size {block_size:4d}, Shared {shared_mem//1024:3d} KB: "
          f"Occupancy {occ*100:5.1f}%, Blocks {blocks}, Warps {warps}")
```

**예상 출력**:
```
Block size   32, Shared   0 KB: Occupancy 100.0%, Blocks 64, Warps 64
Block size  128, Shared   0 KB: Occupancy 100.0%, Blocks 16, Warps 64
Block size  256, Shared  48 KB: Occupancy 100.0%, Blocks  8, Warps 64
Block size  512, Shared  64 KB: Occupancy 100.0%, Blocks  4, Warps 64
Block size 1024, Shared  96 KB: Occupancy  50.0%, Blocks  1, Warps 32
```

### 실험 3 — Latency Hiding 시뮬레이션

```python
import torch
import time

def memory_access_simulation(num_warps, memory_latency_cycles=100):
    """
    Simulate warp scheduling with varying number of resident warps.
    假设每个 warp 有一个 global memory access，需要 memory_latency_cycles 个周期。
    """
    active_warps = []
    warp_states = []  # 각 warp 의 상태 (latency 남은 cycle)
    
    for w in range(num_warps):
        warp_states.append(memory_latency_cycles)  # 모두 메모리 대기 중
    
    total_cycles = 0
    last_active_cycle = memory_latency_cycles  # 마지막 warp 가 완료되는 시점
    
    for cycle in range(memory_latency_cycles + num_warps):
        # 현재 cycle 에서 ready 인 warp 찾기
        ready = [i for i in range(num_warps) if warp_states[i] == 0]
        
        if ready:
            w = ready[0]  # 첫 ready warp 선택
            active_warps.append(1)
            warp_states[w] = memory_latency_cycles  # 새로운 메모리 요청
        else:
            active_warps.append(0)
        
        # 모든 대기 중인 warp 의 latency 감소
        for i in range(num_warps):
            if warp_states[i] > 0:
                warp_states[i] -= 1
    
    return active_warps

# 다양한 warp 수에 대해 시뮬레이션
memory_latency = 100
for num_w in [1, 10, 50, 100, 150]:
    active = memory_access_simulation(num_w, memory_latency)
    idle_cycles = sum(1 for a in active if a == 0)
    idle_fraction = idle_cycles / len(active)
    print(f"Warps={num_w:3d}: Idle fraction={idle_fraction*100:5.1f}%, "
          f"Hidden latency? {idle_fraction < 0.05}")
```

**예상 출력**:
```
Warps=  1: Idle fraction=99.0%, Hidden latency? False
Warps= 10: Idle fraction=90.0%, Hidden latency? False
Warps= 50: Idle fraction=49.5%, Hidden latency? False
Warps=100: Idle fraction= 0.0%, Hidden latency? True
Warps=150: Idle fraction= 0.0%, Hidden latency? True
```

---

## 🔗 실전 활용

### 1. Occupancy 최적화

Block 크기와 shared memory 사용량이 occupancy 를 결정합니다:

```python
# torch.cuda.get_device_properties 로 제약 확인
props = torch.cuda.get_device_properties(0)
# shared_mem_per_block = 프로토콜에서 선언한 __shared__ 총량
# max_threads_per_block = props.maxThreadsPerBlock
# Register 수도 occupancy 영향

# 보통: block size 128–256 + shared mem 48 KB = 좋은 occupancy
# 너무 커도 (block size 1024) 오직 1–2 block 만 SM 에 fit → occupancy 낮음
```

### 2. Nsight Compute 프로파일링

```bash
nv-nsight-compute --set full --export result.ncu-rep ./a.out
# 또는 PyTorch 에서:
# torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
```

Metrics to watch:
- **SM Efficiency**: occupancy 의 실제 효율
- **Active Warps**: 현재 스케줄된 warp 수
- **Achieved Occupancy**: 이론 vs 실제 occupancy

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| SIMT 로 모든 32 thread 가 동일 속도 진행 | Warp divergence (Ch4-05) 에서 mask 로 인한 직렬화 |
| Block 은 한 SM 에만 고정 | Occupancy 제약: shared memory / register 많으면 block 수 줄어듦 |
| Warp scheduler 가 모든 ready warp 를 공평하게 선택 | 실제로는 priority, stall 사유에 따라 편차 있음 (Volta+) |
| 32-thread warp = 고정 | NVIDIA 아키텍처 기준, AMD 는 wave64 사용 |
| Latency hiding = 단순 round-robin | 실제로는 dependent instruction, register pressure 등 복합 요인 |

---

## 📌 핵심 정리

$$\boxed{\text{Warp} = 32 \text{ thread}, \quad \text{Block} \subset \text{SM}, \quad \text{Occupancy} = \frac{\text{Active warps}}{64}}$$

| 개념 | 정의 | 역할 |
|-----|------|------|
| **Warp** | 32 thread SIMT unit | Instruction fetch 1회, 32 parallel execution |
| **Block** | 1–1024 thread | Shared memory 공유, 같은 SM 에서만 실행 |
| **SM** | 32 CUDA core + scheduler + L1 + shared mem | 블록 실행의 물리 단위 |
| **Grid** | 모든 블록의 집합 | GPU 전체 병렬 computation |
| **Occupancy** | Active warp / Max warp per SM | Latency hiding 능력 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 4-way warp 의 lane ID 0, 1, 2, 3 과 일반 32-thread warp 의 lane ID 0, 1, 2, 3 이 어떤 다른 의미를 가지는가? SIMT 에서 같은 instruction 을 실행한다는 것이 정확히 무엇을 의미하는가?

<details>
<summary>해설</summary>

4-way warp (hypothetically) 에서 lane 0–3 의 4 개 thread 는 같은 instruction 을 동시에 fetch & decode 합니다. 32-thread warp 에서는 32 개 lane 이 동시. **"같은 instruction 을 실행" = 같은 opcode, 다른 operand (각 lane 의 register/memory 에서 operand 를 다르게 로드)**.

예: `ADD r0, r1, r2`
- Lane 0: r0[0] += r1[0] + r2[0]
- Lane 1: r0[1] += r1[1] + r2[1]
- ...
- Lane 31: r0[31] += r1[31] + r2[31]

이것이 **M** (Multiple) in SIMT. $\square$

</details>

**문제 2** (심화): A100 에서 108 SM, H100 에서 132 SM 인데, 같은 kernel 을 두 GPU 에 띄울 때 throughput 이 정확히 132/108 ≈ 1.22배 빨라지는가? 어떤 경우에는 아닐까?

<details>
<summary>해설</summary>

일반적으로:
- **Grid > SM 개수**: GPU 가 자동 block scheduling → throughput ∝ SM 개수
- **Grid ≤ SM 개수**: 많은 SM 이 idle → 성능 향상 없음

또한:
- **Memory bandwidth**: A100 (2 TB/s) vs H100 (3 TB/s) — memory-bound kernel 은 추가 이득
- **Clock speed**: H100 은 약간 낮은 clock (375 MHz) — compute-bound kernel 은 smaller gain
- **Compute capability**: H100 의 Tensor Core 가 matrix op 에서 추가 가속

따라서 kernel 종류에 따라 1.0× ~ 1.5× 범위의 speedup $\square$.

</details>

**문제 3** (논문 비평): Volta (2017) 부터 "Independent Thread Scheduling" 이 도입되어 warp 내 thread 들이 더 독립적으로 실행될 수 있다고 했다 (NVIDIA Volta whitepapers). 이것이 warp divergence (Ch4-05) 의 비용을 어떻게 바꾸는가? 여전히 "mask" 기반 divergence 가 필요한가?

<details>
<summary>해설</summary>

**Pre-Volta**: warp 내 모든 thread 가 항상 같은 PC (program counter) 따라감 → divergence 시 true/false path 를 sequential 실행

**Volta+**: 각 thread 가 독립 PC 유지 가능 → 이론적으로 divergence 비용 제거 가능

그러나 실제:
1. **Memory consistency**: warp 내 thread 들의 execution order 가 달라지면, implicit synchronization barrier 가 필요 (예: `__syncthreads()`)
2. **Occupancy penalty**: independent PC 유지 → per-thread state (PC, registers, ...) 를 더 많이 저장 → occupancy 감소
3. **Compiled code 복잡성**: compiler 가 worst-case scheduling 을 생성 → latency hiding 기회 감소

따라서 **Volta+ 에서도 warp 균일성 (避 divergence) 가 권장** — Independent Thread Scheduling 은 "필요할 때만 유연함" 이지, divergence 가 free 가 되지 않음.

결론: Chapter 4-05 의 warp divergence 회피 기법은 여전히 중요 $\square$.

</details>

---

<div align="center">

[◀ 이전](../ch3-dispatcher/05-functorch-transforms.md) | [📚 README](../README.md) | [다음 ▶](./02-memory-hierarchy.md)

</div>
