#ifndef EVALUATOR_H
#define EVALUATOR_H
#include "Dataset.h"
#include <cstdint>
// =============================================================================
// Evaluator: pure free functions that compute fitness for one program.
//
// Kept outside LGPEngine so they can be unit-tested in isolation. Engine's
// evaluate_all() is just a population-level loop that calls into here.
//
// Convention: MSE, lower-is-better. Non-finite or extreme MSE clamps to
// SENTINEL_BAD_FITNESS so selection never sees NaN. The clamp is applied
// to the *final* MSE only -- per-context outputs are not clamped, which
// would distort the fitness landscape arbitrarily.
// =============================================================================

// Evaluator loop will dissapear in GPU version ;) - programs get mapped to unique warps 

struct ProgramView{
    const uint32_t* instructions; // points int flat populattion buffer, current instruction stream for program 
    int length; // current prog length 

};

namespace Evaluator {
    constexpr float  WORST_FITNESS = 0.0f; 
    float evaluate_sr_mse(const ProgramView& prog, const Dataset& dataset);
    float evaluate_sr_r2(const ProgramView& prog, const Dataset& dataset);
}

namespace Fitness {
    float mse_to_fitness(float mse);
    float r2_to_fitness(float r2);
}
#endif
