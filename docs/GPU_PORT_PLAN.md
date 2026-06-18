# GPU Port Plan & Golden Baseline Guide

This document explains **how we are porting the LGP evaluation pipeline to CUDA**
and **how the golden baseline keeps that port honest**. Read the golden section
first if you've never used it — the whole port leans on it.

---

## Part 1 — What the golden baseline is

### The one-sentence idea

> The golden is a **frozen snapshot of the program's output**, committed to git.
> It is an **answer key**: you generate it once from code you trust, then after
> every change you re-run and check that the current code still produces the
> same answers.

### Why we need it

We are about to restructure the evaluation code heavily (CPU → CUDA). The danger
of any port is **silently changing behavior** while moving code around. The golden
catches that: if a "pure refactor" accidentally changes a result, the golden check
turns red immediately, and we know *that commit* did it.

### What's actually stored

`tools/make_golden.sh` runs a fixed set of `(target, seed)` cells and saves their
output under `golden/`:

| File | What it captures | Why it matters |
|------|------------------|----------------|
| `hist_<target>_s<seed>.csv` | per-generation trace (best/mean fitness + lengths, all 1000 gens) | the **trajectory fingerprint** — most sensitive to drift |
| `prog_<target>_s<seed>.txt` | the evolved best program | catches "same fitness, different program won" |
| `SHA256SUMS` | one fingerprint line per file | compact manifest for cross-machine/repo comparison |

Currently keyed on: targets `nguyen1, nguyen2, nguyen3` × seeds `0, 1, 2` (9 cells).

### Why this works: determinism

Given a fixed `LGPConfig::SEED` and a fixed dataset, the run is **fully
deterministic** — same input → byte-identical output, every time. That is what
makes an exact-match answer key meaningful. (Verified: re-running the same cell
produces identical bytes.)

---

## Part 2 — How to use the golden

There are exactly **two scripts**, and the difference between them is the thing to
internalize:

```
tools/make_golden.sh   →  CREATE the answer key   (run RARELY, on purpose)
tools/check_golden.sh  →  GRADE against the key    (run AFTER EVERY change)
```

### Day-to-day loop (during the port)

```
1. make a change (e.g. a Phase 1 refactor)
2. tools/check_golden.sh
      ALL MATCH  → behavior preserved, safe to commit
      MISMATCH   → you changed behavior, go look at what broke
3. repeat
```

You do **not** regenerate the golden during the port. If you did, you'd be
comparing the new code to itself, and the check would always pass — useless. The
key must stay frozen to be a real reference.

### The golden guards *behavior*, not *code*

You are **supposed** to change the code on `main` — that's the point of the port.
The golden holds those changes accountable to producing the *same behavior*. So:

| You changed... | Should output change? | A `FAIL` means... |
|----------------|----------------------|-------------------|
| Code structure (refactor, backend split) | **No** | 🚨 a bug — you broke behavior |
| `ELITES`, `POPULATION_SIZE`, `SEED`, datasets | **Yes** | ✅ expected — re-key the golden |

### When you *intend* to change behavior (re-keying)

Changing a config constant like `ELITES` or `POPULATION_SIZE` **will** make
`check_golden.sh` fail — and that is correct, not breakage. The answer key is for
**one specific configuration**; change the config and the old key no longer applies.
To adopt the new behavior as the reference:

```bash
# 1. edit LGPConfig.h (e.g. change ELITES)
# 2. eyeball that the new run is sane (R² still climbing, etc.)
tools/make_golden.sh        # re-key: new config becomes the new reference
git add golden && git commit -m "Re-key golden for ELITES=..."
```

### The golden rule that keeps you out of trouble

> **Never change config and refactor in the same commit.**

If you bump `POPULATION_SIZE` *and* split the evaluator at once, a `FAIL` can't tell
you whether the diff came from the (expected) config change or a (bug) refactor
mistake. Change one thing at a time so the signal stays clean.

### Things the golden quietly assumes

A spurious `FAIL` is *usually* one of these, not a real bug:

1. **Toolchain / machine.** Generated with `g++ -O2` on this box. Different
   compiler, `-ffast-math`, or different hardware can shift float results even with
   identical source.
2. **The CPU→GPU jump (important).** The exact-match golden will **not** survive
   moving to the GPU. GPU float reductions sum in a different order (float math is
   non-associative), so results differ in the last few digits — not a bug. That's
   why the GPU phase switches to a **tolerance** check against the CPU oracle
   (`max|Δ| < 1e-4`) instead of byte-exact diff.

---

## Part 3 — The pipeline we're porting

The CPU call chain today:

```
evolve_sr → evaluate_all_sr → [loop over programs]
                                  view_program + Evaluator::evaluate_sr_r2
                                      → Interpreter::run_stateless
```

Two nested parallel axes, both currently sequential loops:

- **Outer:** program (0..POPULATION_SIZE) → maps to a **warp** on the GPU.
- **Inner:** context (0..NUM_CONTEXTS=32) → maps to **lanes/threads** in the warp.

`evaluate_all_sr` is the **seam**: everything above it (evolution loop, selection,
variation, stats) stays host C++; everything below it (per-program R², the 32-context
interpreter) becomes one CUDA kernel launch.

### Target API (whole-population, one call)

```cpp
// the stable interface both CPU and CUDA backends implement
void evaluate_population(
    const uint32_t* instructions,  // program-major flat buffer
    const uint8_t*  lengths,       // [n_programs]
    float*          fitness_out,   // [n_programs]  <- output
    int             n_programs,
    const Dataset&  ds);           // inputs, targets, N, num_inputs, ss_tot
```

### Elites: a caller-side pointer offset

Elites always occupy the contiguous front block `[0, ELITE_COUNT)` of the buffers,
and their fitness is carried over from the previous generation. So the kernel never
needs to know elites exist — the caller just offsets the pointers to skip them:

```cpp
evaluate_population(
    instructions + ELITE_COUNT * MAX_PROGRAM_SIZE,
    lengths      + ELITE_COUNT,
    fitness_out  + ELITE_COUNT,
    NON_ELITE_COUNT,
    ds);
```

Two caveats: (a) elite fitnesses must still *exist* in `fitness_out[0, ELITE_COUNT)`
(selection reads all of them — they're carried over, not recomputed); (b) **gen 0
has no elites**, so the first call evaluates the whole population with no offset.

### GPU launch shape (avoid micro-blocks)

Do **not** launch one warp per block (`<<<n, 32>>>`) — the SM's resident-block limit
(16–32) caps you far below its warp limit (48–64), starving occupancy. Instead pack
several warps per block, one program per warp:

```cpp
constexpr int WARPS_PER_BLOCK = 8;            // tune; or use cudaOccupancyMaxPotentialBlockSize
int blocks = (n_programs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
evaluate_kernel<<<blocks, WARPS_PER_BLOCK * 32>>>(...);
```

In-kernel: `prog = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x/32`, `lane = threadIdx.x % 32`.
Per-lane private `float reg[NUM_REGISTERS]` (not a shared 32×8 array — this is the big
simplification the port buys). Warp-reduce SSE with `__shfl_down_sync`; no
`__syncthreads` needed since warps are independent.

---

## Part 4 — The port, phase by phase

Golden rule: **never change behavior and structure in the same commit.** Each phase
ends with a check so a regression is always isolated to one move.

| Phase | What | How it's verified |
|-------|------|-------------------|
| **0** ✅ | Golden baseline + check harness; verified identical to sequential repo | tag `golden-baseline` |
| **1** | Add `EvalBackend.h` interface; wrap current evaluator as `CpuEvaluator`; route `evaluate_all_sr` through it (keep per-program NaN skip) | `check_golden.sh` → ALL MATCH |
| **2** | Switch to the elite-offset logic (gen0 full / later gens skip front) | `check_golden.sh` → ALL MATCH (elite fitness carried over either way) |
| **3** | Make `ISA.h` device-callable (`__host__ __device__` guards, `sinf`) | CPU build unaffected → ALL MATCH |
| **4** | Build plumbing (nvcc, `make gpu`) + trivial kernel to prove malloc/memcpy/launch/link | runs on GPU hardware |
| **5** | Real kernel, **one program first**; per-lane regs, grid-stride over N, warp-reduce, R² | diff that one fitness vs `CpuEvaluator` |
| **6** | Whole population; pack warps/block, tail guard | **tolerance** diff CPU vs GPU (`max|Δ| < 1e-4`) in `test_bed.cpp` |
| **7** | Occupancy + throughput tuning (`WARPS_PER_BLOCK` sweep, occupancy API, benchmark) | perf only; correctness already locked |
| **8** (later) | Keep population resident on device; move variation/selection onto GPU so it never leaves device memory | separate effort |

Throughline: **0–3** change structure with the golden proving zero drift; **4–6** add
the GPU with the CPU as the correctness oracle; **7–8** are pure performance.

---

## Part 5 — Proposed file layout

```
src/
  core/                  # backend-agnostic, included by host AND device
    LGPConfig.h
    ISA.h                # + __host__ __device__ on inline fns (Phase 3)
    Dataset.h / .cpp
    GenerationStats.h
  engine/
    LGPEngine.h / .cpp    # evolution loop; no GPU knowledge
  eval/
    EvalBackend.h         # the stable interface both backends implement
    cpu/
      EvaluatorCpu.cpp    # today's Evaluator + Interpreter (the ORACLE)
      Interpreter.h / .cpp
    cuda/
      EvaluatorCuda.h / .cu   # device buffers, upload/download, launch
      eval_kernel.cuh         # __global__ kernel + __device__ interpreter
      DeviceDataset.h
      cuda_check.h
tools/
  make_golden.sh          # create the answer key  (run rarely)
  check_golden.sh         # grade against it        (run constantly)
golden/                   # the committed answer key
```

Keep a `make cpu` target (no nvcc) so the reference build works on GPU-less machines —
that's what the golden runs against. The CPU evaluator stays as the permanent oracle;
do not delete it after the kernel works.

---

## Quick reference

```bash
tools/make_golden.sh     # (re)create the baseline — only when behavior change is intended
tools/check_golden.sh    # verify current code reproduces the baseline — after every change
git show golden-baseline:golden/hist_nguyen1_s0.csv   # peek at the original reference
```
```
ALL MATCH  → safe
MISMATCH   → during a refactor: a bug.  after a config change: expected, re-key.
```
