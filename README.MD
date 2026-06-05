# Ribox — CURRENT STATE: Parallel LGP GPU SR Implementation (DELIVERABLE 2) 

A fast, single-threaded C++ implementation of Linear Genetic Programming (LGP),
built as the CPU reference and algorithmic foundation for a GPU-oriented LGP
system. This is **Deliverable 1** of a three-part project.

## The larger goal

Ribox tests a systems hypothesis about scaling Genetic Programming on commodity
GPUs. The argument (from the position paper *In Search of Larger Populations*,
Hill Roy et al.) is that for sequential, trajectory-dependent, weakly-batched GP
workloads — especially agent-centric reinforcement-learning settings — the GPU
bottleneck is **not** accelerating the evaluation of a single program. It is
**sustaining a large concurrently-active population**. When the data axis is
weak, the population axis becomes the primary source of scalable parallelism.

The near-term hypothesis under test is **warp-level execution**: map one GP
individual to one CUDA warp of 32 threads, with all lanes executing the same
program in lockstep while each lane evaluates a different context (a fitness
case in symbolic regression; an episode or environment instance in RL). This
sits between coarse block-level mappings (which limit how many distinct
individuals stay resident) and fully thread-level mappings (which expose heavy
divergence). Array-based LGP is a natural substrate because its flat,
contiguous executable form maps cleanly onto accelerator memory and execution
constraints.

The three deliverables:

1. **Sequential C++ LGP reference (this repo).** A robust single-threaded CPU
   implementation that serves two purposes: a correctness oracle for validating
   the GPU version, and the algorithmic blueprint the GPU code is derived from.
2. **GPU symbolic-regression prototype.** A minimal CUDA implementation of
   warp-level LGP evaluation, testing the core design choices (representation,
   memory layout, kernel structure) in a controlled, data-rich setting.
3. **GPU RL evaluation.** Extension to agent-centric, trajectory-dependent
   workloads with GPU-resident environments — the regime where the
   population-scaling benefits of warp-level execution should be most
   pronounced.

## Design philosophy: hardware-aware software

Every decision in the CPU reference is made so it translates cleanly to GPU.
Choices that work on CPU but would not survive the port are avoided from the
start:

- **GPU-friendly memory layout from day one.** Flat contiguous arrays, no
  pointer chasing, no nested `vector<vector<T>>`. Programs are stored in one
  flat instruction buffer; per-program length is tracked in a parallel array.
- **No runtime polymorphism on hot paths.** No virtual functions in code that
  will execute on or near the device. The evaluator is a plain function, not an
  abstract base class — virtuals do not port to GPU well, and templating for a
  handful of evaluator types would be overengineering.
- **Compile-time constants for anything affecting layout.** Register count, max
  program size, number of contexts. These let the compiler size and unroll
  optimally now, and become template parameters or `__constant__` memory on GPU.
- **CPU and GPU structure mirror each other.** The interpreter's
  "decode-once, apply-across-contexts" loop on CPU maps directly onto warp
  execution on GPU: same logical structure, different expression of parallelism.

## Architecture

### Instruction representation

Each instruction is a 32-bit bit-packed word, four 8-bit fields:

```
  bit:  31 ........ 24 | 23 ........ 16 | 15 .......... 8 | 7 ........... 0
       [    SRC2     ] [    SRC1      ] [     DEST       ] [      OP      ]
       [mode | index ] [  reg index   ] [   reg index    ] [   opcode     ]
```

`SRC2`'s high bit is a mode flag (0 = register, 1 = constant); its low bits are
an index into the register file or the constant pool. The invariant
`NUM_REGISTERS == NUM_CONSTANTS` lets one mask handle both modes. Only the low
`log2(N)` bits of each field are meaningful; the rest are reserved for future
widening.

### Instruction set

Eight operations: `ADD SUB MUL DIV SIN COS LT GT`.

- `DIV` is protected (returns 1.0 on divide-by-zero).
- `LT`/`GT` write 1.0 / 0.0 into a float register.
- `SIN`/`COS` are unary; they read `SRC2` (so a mutated unary instruction can
  still reach both registers and constants) and ignore `SRC1`.

### Interpreter

A stateless free function: `(program, inputs) -> outputs`. The register file is
a flat 1-D layout `registers[c * NUM_REGISTERS + i]` matching the future GPU
layout. The loop decodes each instruction once, then applies it across all
`NUM_CONTEXTS` contexts — the structure that becomes warp broadcast on GPU.
Inputs initialize `r0..r_{n-1}`; scratch registers start at zero; the output of
each context is `r0` after the final instruction.

### Evaluation

The dataset pads its fitness-case count up to a multiple of `NUM_CONTEXTS` so
the chunked evaluator can always issue a full-width interpreter call without
reading past the array. Fitness for symbolic regression is MSE, accumulated in
a `double` for precision, then mapped to a higher-is-better score via
`1 / (1 + mse)` so selection has a single fixed convention (higher wins).
Non-finite results map to the worst score (0).

### Evolutionary loop

- **Selection:** k-tournament with replacement, returns one parent index.
- **Crossover:** single-point. One random cut, two contiguous copies per child.
  Children inherit different parents' lengths, giving length variation for free.
- **Mutation:** micro-mutation (regenerate one field of one instruction, with
  correct field-width masking) plus optional tail insert/delete for length
  drift. Each operator fires independently per offspring.
- **Elitism:** the top-K programs are copied verbatim into the next generation
  with fitness carried forward (skipped during re-evaluation).
- **Double-buffering:** two population buffers ping-pong each generation; a flag
  selects which is live. No per-generation reallocation or copying — only the
  flag flips. This mirrors how the GPU version will manage device buffers.

## Building

```
make            # builds both binaries
make lgp_test   # unit-test binary only
make lgp_run    # evolution-run binary only
make clean
```

Requires a C++20 compiler. The build uses `-MMD -MP` for automatic header
dependency tracking.

## Running

```
./lgp_test      # runs the full unit-test suite
./lgp_run       # runs evolution on the sanity and quadratic targets
```

`lgp_test` covers instruction encode/decode, every opcode, the interpreter
contract (zero-init, context independence, determinism), dataset construction
and padding, chunk-boundary evaluation, crossover (length, conservation,
boundary cases), mutation (single-field change, field-width safety, length
bounds), and full-generation variation stability across many generations.

`lgp_run` evolves against `f(x) = x` (a trivial sanity check that should reach
near-perfect fitness almost immediately) and `f(x) = x² + x`, printing
per-generation best/mean fitness as CSV plus a decode of the best evolved
program.

## Configuration

All tunable parameters live in `LGPConfig.h` as `constexpr` values: population
size, max/starting program size, register and constant counts, the constant
pool, mutation and crossover rates, elite fraction, tournament size, generation
cap, and `NUM_CONTEXTS` (compile-time-fixed at the warp width of 32). Changing
any of these requires a rebuild, in exchange for the compiler being able to size
and unroll around them.

## Status and open questions

Deliverable 1 is functionally complete: the full evolutionary loop runs and is
covered by the test suite. Items being carried into the GPU work:

- **Validation strategy.** Bit-exact CPU/GPU agreement is off the table
  (floating-point non-associativity, different RNGs, SFU-approximate
  transcendentals). The reference contract is per-program-per-input agreement
  within tolerance, which is the test that ports cleanly.
- **Stateful interpreter.** Deferred until RL; the execution loop will be
  factored into a shared helper at that point.
- **Variable program length under warp execution.** Works (all 32 lanes run the
  same program, so lane counts match), but its interaction with crossover across
  the population is worth revisiting on GPU.
