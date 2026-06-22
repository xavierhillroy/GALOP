#ifndef LGP_ENGINE_H
#define LGP_ENGINE_H
#include "LGPConfig.h"
#include <cstdint>
#include <random>
#include <vector>
#include <limits>
#include "Evaluator.h"
#include "Dataset.h"
#include "GenerationStats.h"
#include <ostream>
#include <iostream>
#include "EvalBackend.h"
#include <memory>

// =============================================================================
// PopulationData: owns all the buffers that describe the population.
//
// This struct is deliberately just data + allocation -- it doesn't implement
// any of the evolutionary loop. LGPEngine operates on it. Keeping the two
// separate means (a) PopulationData can later be swapped for a GPU-backed
// equivalent (same fields, but storage is cudaMalloc'd) without touching
// engine logic, and (b) tests can construct a PopulationData in isolation.
//
// Memory layout recap (see LGPConfig.h for the "why"):
//   - instructions[2][]:    flat buffer of TOTAL_INSTRUCTIONS uint32_t words.
//                        Program i occupies the slice
//                          [i * MAX_PROGRAM_SIZE, i * MAX_PROGRAM_SIZE + program_lengths[i])
//                        Slots past program_lengths[i] exist but aren't
//                        executed.- 
//   - program_lengths[]: one byte per program; gates how far the interpreter
//                        walks into that program's slice.
//
// three parallel "next-gen" buffers implement ping-pong double-buffering for
// variation: children are written into next_gen_* while the current
// generation remains readable. After variation completes, the engine flips
// a pointer/flag  to swap roles rather than copying. This
// avoids allocation churn per generation and mirrors how the GPU version
// will want to work.
// =============================================================================

// Container that contains population information
struct PopulationData {

    // ---- Current generation ------------------------------------------------
    std::vector<uint32_t> instructions_buf[2];      // Flat buffer of all programs'
                                             // instructions; size = TOTAL_INSTRUCTIONS.
    std::vector<uint8_t>  program_lengths_buf[2];   // Active length of each program.
                                             // uint8_t because MAX_PROGRAM_SIZE fits
                                             // comfortably under 255.
    static_assert(LGPConfig::MAX_PROGRAM_SIZE <= LGPConfig::SIZE_SENTINEL, "MAX PROGRAM SIZE LARGER THEN uint8 can store");
    std::vector<float>fitness_scores_buf[2];

    // -------------------------------------------------------------------------
    // Default constructor sizes all buffers from the compile-time constants
    // in LGPConfig, so the struct is always ready-to-use after construction.
    // No manual resize() calls at the use site.
    //
    // Initialization values chosen deliberately:
    //   - instructions[i] -> 0: zero-filled instruction
    //     slots decode to a valid (if meaningless) instruction, so a missed
    //     init() can't produce UB in the interpreter.
    //   - program_lengths -> STARTING_PROGRAM_SIZE: same defensive reasoning.
    //     A missed init() still leaves every program with a valid length.
    //   -
    //   - fitness_scores -> NaN: sentinel for "not yet evaluated". Any
    //     comparison with NaN returns false, so selection run on an
    //     unevaluated population fails visibly rather than silently picking
    //     the zero-fitness winners.
    // -------------------------------------------------------------------------
    PopulationData(){
        for (int b =0; b <2; ++b){
            instructions_buf[b].assign(LGPConfig::TOTAL_INSTRUCTIONS, 0);
            program_lengths_buf[b].assign(LGPConfig::POPULATION_SIZE, LGPConfig::SIZE_SENTINEL);// invalid sentinel is 255 ( largest val we can read in uint8 )
            fitness_scores_buf[b].assign(LGPConfig::POPULATION_SIZE, std::numeric_limits<float>::quiet_NaN());
        }
    }
   
};

// =============================================================================
// LGPEngine: owns the evolutionary loop.
//
// Holds the population data, the RNG, and the generation counter, and
// exposes the operators (init / mutate / crossover / select / vary / evolve)
// that drive the loop. All randomness flows through the engine's single
// mt19937 so that runs are reproducible given LGPConfig::SEED.
// =============================================================================
class LGPEngine {
private:
    // ---- Generation state --------------------------------------------------
    int current_generation;                        // Counter; starts at 0 in ctor.
    int current_buffer;                            // 0 or 1: indicates which of
                                                   // the two buffers in PopulationData
                                                   // is the "live" generation.
                                                   // Flipped after each variation pass.

    // ---- RNG ---------------------------------------------------------------
    std::mt19937 rng;      // Seeded with LGPConfig::SEED
                                                        // in the constructor.
    std::uniform_int_distribution<uint32_t>   dist_32;  // Uniform over the full
                                                        // uint32_t range; used to
                                                        // generate random instruction
                                                        // words.
    //Population distribution, random over 0 -> popsize - 1
    std::uniform_int_distribution<int> dist_pop;

    // Members to add:
    std::uniform_real_distribution<float> dist_unit;       // [0, 1)
    std::uniform_int_distribution<int> dist_field;      // [0, 4]

    // ---- Population --------------------------------------------------------
    PopulationData data;                           // All buffers; default-constructed.
    std::vector<GenerationStats> history;
    std::unique_ptr<EvalBackend> backend; // pointer to backend we use (EIther CPU/ GPU or RL or SR )



public:
    explicit LGPEngine(uint32_t seed = LGPConfig::SEED, std::unique_ptr<EvalBackend> eval = nullptr);
    ~LGPEngine() = default;
    ProgramView view_program(int i) const; // returns a program view object ( cur instruction part and cur length)
    void evaluate_all_sr(const Dataset& dataset); // evluates entire population... this loop will disappear on GPU - be assigned to diff warps 
    void evaluate_all_rl();
    // ---- Public evolutionary interface -------------------------------------
    void init_population();                  // Randomize the initial population.
    void mutate(uint32_t* program, uint8_t& length);
    void crossover(const ProgramView& a, const ProgramView& b,
                          uint32_t* child_a, uint8_t& child_a_len,
                          uint32_t* child_b, uint8_t& child_b_len);
    int  tournament_selection();  // Returns the program index that is selected.
    void vary();                  // Crossover + mutation over the whole population.
    void evolve_sr(const Dataset& dataset); 
    
    // TO DO // Top-level evolutionary loop.
    void evolve_rl();


    const PopulationData& get_data() const { return data; } // access data for testing 
    PopulationData& get_mutable_data() { return data; } // access data for testing 
    uint32_t micro_mutate_instruction(uint32_t instruction);
    
    void vary_pair(int dstA, int dstB);
    
    int current_buffer_index() const { return current_buffer; }
    void init_evolution();
    void print_best_program(std::ostream& os = std::cout) const;
    void print_history() const;
    // void print_best_program() const;
    
    


private:
    // ---- Internal helpers --------------------------------------------------
    uint32_t generate_instruction();  // One random, encoding-valid instruction word.

    // readers
    const std::vector<uint32_t>& cur_instructions() const{ // returns reference to current instruction buffer 
        return data.instructions_buf[current_buffer];
    } 
    const std::vector<uint8_t>& cur_lengths() const {
        return data.program_lengths_buf[current_buffer];
    }
    const std::vector<float>& cur_fitness() const{
        return data.fitness_scores_buf[current_buffer];
    }
    // write accessor - where to write next gen + Current gen fitness  - (KEY THESE GIVE YOU THE NEXT GEN EMBEDDED)
    std::vector<uint32_t>& next_instructions(){
        return data.instructions_buf[1 - current_buffer ]; //if buff is 0, 1 - 0 = 1 flips, if buff is 1, 1-1  = 0, so it flips 
    }
    std::vector<uint8_t>& next_lengths(){
        return data.program_lengths_buf[1 - current_buffer];
    }
    std::vector<float>& next_fitness(){
        return data.fitness_scores_buf[1 - current_buffer];
    }

    std::vector<float>& cur_fitness_mutable(){
        return data.fitness_scores_buf[current_buffer];
    }
    std::vector<uint32_t>& cur_instructions_mutable(){
        return data.instructions_buf[current_buffer];
    }
    std::vector<uint8_t>& cur_lengths_mutable()  {
        return data.program_lengths_buf[current_buffer];
    }
    void flip_generation(){
        current_buffer = 1 - current_buffer;
    }
    uint32_t* next_gen_ptr(int i){ // gives us the pointer to the start of program at index i of next gen ss
        return next_instructions().data()+ i * LGPConfig::MAX_PROGRAM_SIZE;
    }
    std::vector<int> top_k_indices(int k) const;

    void copy_elite_to_next(int srcIdx, int dstIdx); //given idx from source program from cur gen, copy that program to next gen at a the dstIndex - for elites... we copy fitness too!
    
    void reset_buffers_to_sentinels();
    GenerationStats compute_stats() const;
    public:
    float best_train_r2() const;
    float best_test_r2(const Dataset& test) const;
    int best_length() const;
    const std::vector<GenerationStats>& history_view() const { return history; }


};


#endif  // LGP_ENGINE_H