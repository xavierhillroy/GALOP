// =============================================================================
// cuda_diff.cu — differential test: GPU kernel vs CPU oracle.
//
// This is the GPU's equivalent of check_golden.sh. The golden (byte-exact)
// check does NOT apply to the GPU: float reductions reorder and host/device
// sinf differ by a few ULP, so the result matches only within a TOLERANCE.
// And the full evolve_sr TRAJECTORY diverges (chaos), so we never compare runs
// here -- we compare ONE evaluation of ONE population, both backends, and
// require max|Δ| < TOL.
//
// Run after every kernel change.
//
// Build:
//   nvcc -std=c++20 --expt-relaxed-constexpr -Isrc \
//        tests/cuda_diff.cu src/EvaluatorCuda.cu src/EvaluatorCpu.cpp \
//        src/Evaluator.cpp src/Interpreter.cpp src/Dataset.cpp src/LGPEngine.cpp \
//        -o cuda_diff
//   ./cuda_diff
// =============================================================================
#include "LGPEngine.h"
#include "EvaluatorCpu.h"
#include "EvaluatorCuda.h"
#include "Dataset.h"
#include "LGPConfig.h"
#include "cuda_errors.h"
#include <vector>
#include <cstdio>
#include <cmath>

int main() {
    Is_GPU_present();

    // ---- a realistic random population (every opcode exercised) ------------
    // init_evolution() fills the live buffer with random valid programs --
    // the exact distribution real runs start from. (All length 8 here; to
    // stress varied lengths later, evolve a few generations first.)
    LGPEngine engine(42);
    engine.init_evolution();

    const auto& data    = engine.get_data();
    const int   buf     = engine.current_buffer_index();
    const uint32_t* instr   = data.instructions_buf[buf].data();
    const uint8_t*  lengths = data.program_lengths_buf[buf].data();

    // ---- a real dataset (ss_tot computed, so R² is meaningful) -------------
    Dataset ds = load_csv_1d("datasets/nguyen1_train.csv");
    const int n = LGPConfig::POPULATION_SIZE;
    printf("Population: %d programs | dataset N=%d num_inputs=%d\n",
           n, ds.N, ds.num_inputs);

    // ---- evaluate the SAME population on both backends ---------------------
    std::vector<float> fit_cpu(n), fit_gpu(n);
    CpuEvaluator{}.evaluate_population(instr, lengths, fit_cpu.data(), n, ds);
    CudaEvaluator{}.evaluate_population(instr, lengths, fit_gpu.data(), n, ds);

    // ---- compare with tolerance (the "variance baked in") ------------------
    const float TOL = 1e-4f;
    float max_diff = 0.0f;
    int   worst = -1, n_fail = 0;
    for (int i = 0; i < n; ++i) {
        const float d = fabsf(fit_cpu[i] - fit_gpu[i]);
        if (d > max_diff) { max_diff = d; worst = i; }
        if (d > TOL) ++n_fail;
    }

    printf("max |Δ| = %.3e at program %d  (cpu=%.7f  gpu=%.7f)\n",
           max_diff, worst, fit_cpu[worst], fit_gpu[worst]);
    printf("%d / %d programs exceed TOL=%.0e\n", n_fail, n, TOL);

    if (n_fail == 0) {
        printf("PASS: GPU agrees with CPU oracle within tolerance.\n");
        return 0;
    }
    printf("FAIL: GPU diverges from CPU oracle.\n");
    return 1;
}
