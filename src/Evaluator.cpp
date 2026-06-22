#include "Evaluator.h"
#include "Interpreter.h"
#include "LGPConfig.h"
#include <algorithm> 
#include <cmath> 

namespace Evaluator{
    // new evaoluators will be added here
    // will need to carefully handle reduction here x ;-0 
    // Sr evaluator 
    float evaluate_sr_mse(const ProgramView& prog, const Dataset& dataset){
        constexpr int CHUNK = LGPConfig::NUM_CONTEXTS; // chunk is num contexts whichh for our case will be warp 
        const int N = dataset.N;
        const int num_inputs = dataset.num_inputs;
        // might consider chanign since we have smaller fitness cases 
        // Double accumulator: float SSE over 1000+ cases loses precision fast,
        // and an off-by-LSB MSE makes selection nondeterministic-looking when
        // two programs tie. Cheap to use double here; final cast is the cast.

        double sse = 0.0; // Sum squared error 
        float outputs[CHUNK]; // outpuct array containing all outputs of prog across dif contextts

        for (int base = 0; base < N; base+= CHUNK){
            // basically the interpreter gets a chunk every time, operates on that. but we only accumulate the valid output cols, the rest are just padded output 
            const int valid = std::min(CHUNK, N-base); // we either take the full chunk as valid, or the valid bits if chunk isnt full 

            Interpreter::run_stateless(
                prog.instructions,
                prog.length,
                &dataset.inputs[base  * num_inputs],  // current inputs for the each context - we mult the base (where we are at in the chunks) * the num inputs. To get the correct starting point for next input set for next chunk 
                num_inputs,
                outputs
            );// interpet one program on NUM contexts (Chunk)

            // sum cur error for chunk 
            for (int c = 0; c < valid ; ++c){
                //err - to get target we must add c to the current base... because it is based on the full sett of fitness cases not just one chhunk 
                const double err = double(outputs[c]) - double(dataset.targets[base + c]); // 
                sse += err * err; // accumulating the err^2
            }

        }
        // now we have gone through all chunks lets get the MSE and properly clamp
        return float(sse /double(N));
    }
    // Returns coefficient-of-determination R^2 = 1 - SS_res/SS_tot.
    // Higher is better, <= 1.0. Can go negative for models worse than the mean.
    // The caller (fitness adapter) is responsible for flooring negatives to 0.
    float evaluate_sr_r2(const ProgramView& prog, const Dataset& dataset){
        constexpr int CHUNK = LGPConfig::NUM_CONTEXTS;
        const int N = dataset.N;
        const int num_inputs = dataset.num_inputs;
        const double ss_tot = dataset.ss_tot;

      

       //degenerate target (constant over sample) -> r^2 undefined
        if (ss_tot <= 0.0) return 0.0f;

        // --- SS_res: your existing chunked SSE loop, unchanged ---
        double sse = 0.0;
        float outputs[CHUNK];
        // running in groups of 32 - pass in the interperter 32 things to work on at once mirrors a warp 
        for (int base = 0; base < N; base += CHUNK){
            const int valid = std::min(CHUNK, N - base);

            Interpreter::run_stateless(
                prog.instructions,
                prog.length,
                &dataset.inputs[base * num_inputs],
                num_inputs,
                outputs
            );

            for (int c = 0; c < valid; ++c){
                const double err = double(outputs[c]) - double(dataset.targets[base + c]);
                sse += err * err;
            }
        }

        // Guard non-finite SSE (program produced inf/nan) before the divide.
        if (!std::isfinite(sse)) return 0.0f;

        return float(1.0 - sse / ss_tot);
    }

}
float Fitness::mse_to_fitness(float mse){ // converts lower is better mse to higher is better for fitness selection
    if (!std::isfinite(mse)) return 0.0f;
    return 1.0f / (1.0f + mse);

}
float Fitness::r2_to_fitness(float r2) {
    if (!std::isfinite(r2)) return 0.0f;
    // R² is already higher-is-better and ~bounded above by 1.
    // Clamp the floor so a terrible model (negative R²) maps to 0, not garbage.
    return r2 < 0.0f ? 0.0f : r2;
}