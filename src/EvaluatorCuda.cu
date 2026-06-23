#include "EvaluatorCuda.h"
#include "LGPConfig.h"
#include "cuda_errors.h"
#include "ISA.h"       
#include "LGPConfig.h"
#include <cassert>

__constant__ float d_constants[LGPConfig::NUM_CONSTANTS];

// ---- allocate ONCE (full size, so it serves gen-0 full + later non-elite) --
CudaEvaluator::CudaEvaluator() {
    ERR(cudaMalloc(&d_instructions,
                          LGPConfig::TOTAL_INSTRUCTIONS * sizeof(uint32_t)));
    ERR(cudaMalloc(&d_lengths,
                          LGPConfig::POPULATION_SIZE * sizeof(uint8_t)));
    ERR(cudaMalloc(&d_fitness,
                          LGPConfig::POPULATION_SIZE * sizeof(float)));

// once, in the CudaEvaluator constructor:
    ERR(cudaMemcpyToSymbol(d_constants, LGPConfig::CONSTANTS.data(),
                       LGPConfig::NUM_CONSTANTS * sizeof(float)));
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
    // per-dimension warp load stays coalesced. See GPU_PORT_PLAN. ~ basically would create one array for each inpuy 
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



// INTERPRETER FOR WARP 
__device__ double program_partial_sse(
        const uint32_t* prog, int len, int lane,
        const float* __restrict__ inputs, const float* __restrict__ targets,
        int N, int num_inputs)
{
    double sse_lane = 0.0;                      // double to match the CPU oracle
    for (int base = 0; base < N; base += 32) { // multiples of 32 warps (warp def not best appraoch when SR > 32 LOL  )
        const int caseIdx = base + lane; // current lane 
        if (caseIdx >= N) continue;            // replaces the CPU's `valid` clamp

        float reg[LGPConfig::NUM_REGISTERS];
        assert(num_inputs<= LGPConfig::NUM_REGISTERS);
        // -------------------------------------------------------------
        // store inputs 
        for (int inp = 0; inp <num_inputs; inp ++){
            reg[inp] = inputs[caseIdx * num_inputs + inp];
        }
        // zero out the rest of regs
        for (int inp = num_inputs; inp < LGPConfig::NUM_REGISTERS; inp ++){
            reg[inp] = 0.0f;
        }
        // executing instruction

        // loop through instructions 
        for (int pc = 0; pc < len; ++pc){
            // decdode instruction
            uint32_t cur_instruction = prog[pc];
            uint8_t dest_index = ISA::get_dest_index(cur_instruction);
            uint8_t op = ISA::get_op(cur_instruction);
            uint8_t src1_index = ISA::get_src1_index(cur_instruction);
            uint8_t src2_index = ISA::get_src2_index(cur_instruction);
            bool src2_const = ISA::is_src2_constant(cur_instruction);

            // apply
            float a = reg[src1_index];
            float b = src2_const ? d_constants[src2_index] : reg[src2_index];
            reg[dest_index] = ISA::apply_op(op,a,b);
            
        }// loops through instructions 

        const double err = (double)reg[0] - (double)targets[caseIdx];
        sse_lane += err * err;
    }
    return sse_lane;
}
// big daddy population kernel 
__global__ void evaluate_kernel(
        const uint32_t* __restrict__ instructions,
        const uint8_t*  __restrict__ lengths,
        float*          __restrict__ fitness_out,
        const float*    __restrict__ inputs,
        const float*    __restrict__ targets,
        int n_programs, int N, int num_inputs, double ss_tot)
{
    const int warps_per_block = blockDim.x >> 5; // block dim ID /32 
    const int lane = threadIdx.x & 31; // Threadidx %32
    const int prog = blockIdx.x * warps_per_block + (threadIdx.x >> 5); // which blocks came before * number of waprs in the block + threadID/ 32- this decides what warp I am in the instruction thing
    if (prog >= n_programs) return;            // warp-uniform: all 32 lanes share prog - if more warps then progs

    double sse = program_partial_sse(
        instructions + prog * LGPConfig::MAX_PROGRAM_SIZE,
        lengths[prog], lane, inputs, targets, N, num_inputs);// device func the interpreter path

    // ---- warp reduction: 32 partials -> lane 0 ----
    for (int off = 16; off > 0; off >>= 1)
        sse += __shfl_down_sync(0xffffffffu, sse, off); // this is the tree reduction 

    if (lane == 0) {                           // must replicate evaluate_sr_r2 + r2_to_fitness - produces refined fitness not just raw perfectly comparable
        float fit;
        if (ss_tot <= 0.0 || !isfinite(sse)) fit = 0.0f;
        else { float r2 = float(1.0 - sse / ss_tot); fit = r2 < 0.0f ? 0.0f : r2; }
        fitness_out[prog] = fit;
    }
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
    int threads = 256;                                   // multiple of 32
    int warps_per_block = threads / 32;
    int blocks = (n_programs + warps_per_block - 1) / warps_per_block;  // ceil


    evaluate_kernel<<<blocks, threads>>>(d_instructions,d_lengths,d_fitness,d_inputs, d_targets, n_programs,ds_N, ds_num_inputs, ds_ss_tot);
    ERR(cudaGetLastError());        // catch launch-config errors
    ERR(cudaDeviceSynchronize());   // wait + catch runtime errors

    // 4) fitness DOWN — into the (already-offset) host pointer
    ERR(cudaMemcpy(fitness_out, d_fitness,
                          n_programs * sizeof(float),
                          cudaMemcpyDeviceToHost));
}