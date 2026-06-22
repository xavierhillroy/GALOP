#include "EvaluatorCuda.h"
#include "Dataset.h"
#include "LGPConfig.h"
#include "cuda_errors.h"
#include <vector>
#include <cstdio>
#include <cmath>

int main() {
    Is_GPU_present();                       // prints device + maxThreadsPerBlock

    // minimal valid 1-input dataset (kernel ignores it, but upload_dataset reads it)
    Dataset ds;
    ds.N = 100; ds.num_inputs = 1; ds.ss_tot = 1.0;
    int padded = ds.padded_N();
    ds.inputs.assign(padded * ds.num_inputs, 0.5f);
    ds.targets.assign(padded, 1.0f);

    // fake population with KNOWN lengths
    const int n = LGPConfig::POPULATION_SIZE;
    std::vector<uint32_t> instr(LGPConfig::TOTAL_INSTRUCTIONS, 0);
    std::vector<uint8_t>  lens(n);
    std::vector<float>    fit(n, NAN);
    for (int i = 0; i < n; ++i) lens[i] = (uint8_t)(i % 50 + 1);

    CudaEvaluator gpu;
    gpu.evaluate_population(instr.data(), lens.data(), fit.data(), n, ds);

    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (fit[i] != (float)lens[i]) {
            if (bad < 5) printf("MISMATCH i=%d fit=%f len=%d\n", i, fit[i], lens[i]);
            ++bad;
        }
    printf(bad ? "FAIL: %d mismatches\n" : "PASS: GPU plumbing works (fitness==length)\n", bad);
    return bad ? 1 : 0;
}