// EvaluatorCpu.cpp
#include "EvaluatorCpu.h"
#include "Evaluator.h"
#include "LGPConfig.h"
#include <cmath>

void CpuEvaluator::evaluate_population(const uint32_t* instructions, const uint8_t* lengths,
                                       float* fitness_out, int n_programs, const Dataset& ds) {
    for (int i = 0; i < n_programs; ++i) {
        if (std::isnan(fitness_out[i])) {                      // same NaN-skip as today
            ProgramView prog{ instructions + i * LGPConfig::MAX_PROGRAM_SIZE,
                              static_cast<int>(lengths[i]) };
            fitness_out[i] = Fitness::r2_to_fitness(Evaluator::evaluate_sr_r2(prog, ds));
        }
    }
}