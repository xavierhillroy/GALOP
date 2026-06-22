// This is an interface for the egine to interact with the evaluator backend
// entire population fitness evaluation, CPU and CUDA both implement thsi 

#ifndef EVAL_BACKEND_H
#define EVAL_BACKEND_H
#include "Dataset.h"
#include <cstdint>
struct EvalBackend{
    // evaluates programs from [0,n_programs] outputs fitness from [0,n_programs] ;)
    // instructions at program major flat buffer (program i at i * MAX_PROGRAM SIZE)
    virtual void evaluate_population(
        const uint32_t* instructions,
        const uint8_t* lengths,
        float* fitness_out,
        int n_programs,
        const Dataset& ds
    ) =0;// pure virtual
    virtual ~EvalBackend() =default;
};
#endif