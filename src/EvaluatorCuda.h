#ifndef EVALUATOR_CUDA_H
#define EVALUATOR_CUDA_H
#include "EvalBackend.h"
#include "Dataset.h"
#include <cstdint>

class CudaEvaluator : public EvalBackend {
public:
    CudaEvaluator();             // allocates population buffers ONCE
    ~CudaEvaluator() override;   // frees everything

    void evaluate_population(const uint32_t* instructions,
                             const uint8_t*  lengths,
                             float*          fitness_out,
                             int             n_programs,
                             const Dataset&  ds) override;

private:
    void upload_dataset(const Dataset& ds);   // runs once, behind the guard

    // ---- population buffers: allocated once, reused every generation ----
    uint32_t* d_instructions = nullptr;   // [TOTAL_INSTRUCTIONS]
    uint8_t*  d_lengths      = nullptr;   // [POPULATION_SIZE]
    float*    d_fitness      = nullptr;   // [POPULATION_SIZE]

    // ---- dataset: uploaded once, then resident for the whole run ----
    float*    d_inputs       = nullptr;   // [padded_N * num_inputs]
    float*    d_targets      = nullptr;   // [padded_N]
    int       ds_N           = 0;         // scalars travel as kernel args,
    int       ds_num_inputs  = 0;         // not device memory
    double    ds_ss_tot      = 0.0;

    const Dataset* uploaded_ds_ = nullptr;  // the guard: "which dataset is resident"
};
#endif