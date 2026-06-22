// EvaluatorCpu.h
#ifndef EVALUATOR_CPU_H
#define EVALUATOR_CPU_H
#include "EvalBackend.h"
struct CpuEvaluator : EvalBackend {
    void evaluate_population(const uint32_t* instructions, const uint8_t* lengths,
                             float* fitness_out, int n_programs, const Dataset& ds) override;
};
#endif