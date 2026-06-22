#include "EvaluatorCuda.h"
#include "LGPConfig.h"
#include "cuda_errors.h"

// ---------------------------------------------------------------------------
// PHASE 4 TRIVIAL KERNEL — plumbing proof only. One thread per program,
// writes the program's length into its fitness slot. This is NOT fitness.
// Phase 5 replaces this body with the warp-per-program interpreter (yours).
// ---------------------------------------------------------------------------
__global__ void evaluate_kernel(const uint8_t* lengths,
                                float*         fitness_out,
                                int            n_programs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_programs) {
        fitness_out[i] = static_cast<float>(lengths[i]);
    }
}

// ---- allocate ONCE (full size, so it serves gen-0 full + later non-elite) --
CudaEvaluator::CudaEvaluator() {
    ERR(cudaMalloc(&d_instructions,
                          LGPConfig::TOTAL_INSTRUCTIONS * sizeof(uint32_t)));
    ERR(cudaMalloc(&d_lengths,
                          LGPConfig::POPULATION_SIZE * sizeof(uint8_t)));
    ERR(cudaMalloc(&d_fitness,
                          LGPConfig::POPULATION_SIZE * sizeof(float)));
}

CudaEvaluator::~CudaEvaluator() {
    cudaFree(d_instructions);
    cudaFree(d_lengths);
    cudaFree(d_fitness);
    cudaFree(d_inputs);
    cudaFree(d_targets);
}

// ---- dataset upload: called once, guarded by object identity --------------
void CudaEvaluator::upload_dataset(const Dataset& ds) {
    const int padded = ds.padded_N();
    // NOTE: case-major upload is coalesced only because num_inputs==1.
    // For multi-input datasets, transpose to input-major (SoA) here so each
    // per-dimension warp load stays coalesced. See GPU_PORT_PLAN.
    ERR(cudaMalloc(&d_inputs,  padded * ds.num_inputs * sizeof(float)));
    ERR(cudaMalloc(&d_targets, padded * sizeof(float)));
    ERR(cudaMemcpy(d_inputs,  ds.inputs.data(),
                          padded * ds.num_inputs * sizeof(float),
                          cudaMemcpyHostToDevice));
    ERR(cudaMemcpy(d_targets, ds.targets.data(),
                          padded * sizeof(float), cudaMemcpyHostToDevice));
    ds_N          = ds.N;
    ds_num_inputs = ds.num_inputs;
    ds_ss_tot     = ds.ss_tot;
}

void CudaEvaluator::evaluate_population(const uint32_t* instructions,
                                        const uint8_t*  lengths,
                                        float*          fitness_out,
                                        int             n_programs,
                                        const Dataset&  ds)
{
    // 1) dataset — upload the first time we see it, then it stays resident
    if (uploaded_ds_ != &ds) {
        upload_dataset(ds);
        uploaded_ds_ = &ds;
    }

    // 2) population UP. The caller (evaluate_range) already offset these
    //    pointers past the elites, so we copy n_programs worth from the FRONT
    //    of the host pointers into the FRONT of the device buffers. The device
    //    works in a 0-based window; the host offset handles placement.
    //    (The trivial kernel ignores d_instructions — uploaded anyway so the
    //     per-gen copy is already wired for Phase 5.)
    ERR(cudaMemcpy(d_instructions, instructions,
                          n_programs * LGPConfig::MAX_PROGRAM_SIZE * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    ERR(cudaMemcpy(d_lengths, lengths,
                          n_programs * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    // 3) launch — trivial geometry: one thread per program (warp-per-program
    //    packing is Phase 5/6, not now)
    int threads = 256;
    int blocks  = (n_programs + threads - 1) / threads;
    evaluate_kernel<<<blocks, threads>>>(d_lengths, d_fitness, n_programs);
    ERR(cudaGetLastError());        // catch launch-config errors
    ERR(cudaDeviceSynchronize());   // wait + catch runtime errors

    // 4) fitness DOWN — into the (already-offset) host pointer
    ERR(cudaMemcpy(fitness_out, d_fitness,
                          n_programs * sizeof(float),
                          cudaMemcpyDeviceToHost));
}