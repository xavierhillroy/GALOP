#include "LGPEngine.h"
#include "ISA.h"
#include "LGPConfig.h"
#include <random>
#include "Evaluator.h"
#include "Dataset.h"
#include <cassert>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "GenerationStats.h"
#include <ostream>
#include "EvaluatorCpu.h"
void LGPEngine::init_evolution(){
    current_buffer = 0;
    current_generation = 0;
    init_population();
    // history 
    history.clear();
    history.reserve(LGPConfig::MAX_GENERATIONS);
}
void LGPEngine::init_population(){
    reset_buffers_to_sentinels();
    // sets all instructions for all programs
    for (int i = 0; i < LGPConfig::TOTAL_INSTRUCTIONS; i ++){
        cur_instructions_mutable()[i] = generate_instruction();
       
    }
       // Set initial program lengths. Using std::fill for clarity vs vector::assign.
    std::fill(cur_lengths_mutable().begin(),
              cur_lengths_mutable().end(),
              static_cast<uint8_t>(LGPConfig::STARTING_PROGRAM_SIZE));

} // V easy GPU transformation 

// Generates instructions
uint32_t LGPEngine::generate_instruction(){
    uint32_t raw_rand = dist_32(rng); // gets random sttream of 32 bits
    return ISA::encode_from_random(raw_rand); // packs and cleans bits (lose some entropy)
}
// constructor- kept explicitly clean- inits current buffer to 0, sets gen to 0, sets up the seed, and distribution, verbose data init
LGPEngine::LGPEngine(uint32_t seed, std::unique_ptr<EvalBackend> eval)
    : current_generation(0), current_buffer(0), rng(seed),
      dist_32(0,UINT32_MAX), dist_pop(0,LGPConfig::POPULATION_SIZE-1),
      dist_unit(0.0f,1.0f), dist_field(0,4), data(),
      backend(eval ? std::move(eval) : std::make_unique<CpuEvaluator>())
{}



// This is program view, a method that returns the program view structure, contains length of cur prog and pointer to the start of it 
ProgramView LGPEngine::view_program(int i) const{
    // because we ping pong betweeen 2 buffers between generations - live bufer selection done in accessor 
    
    return ProgramView{
        cur_instructions().data() + i * LGPConfig::MAX_PROGRAM_SIZE, // here we are getting the raw pointer of instr vector (start) and getting the adress of the prog we are working withh
        static_cast<int>(cur_lengths()[i]) // explicitly casting to int (widening from uint 8)
    };
}
// void LGPEngine::evaluate_all_sr(const Dataset& dataset){
//     for (int agent = 0; agent < LGPConfig::POPULATION_SIZE; ++agent){
//         if(std::isnan(cur_fitness()[agent])){
//             const ProgramView prog = view_program(agent);
//             cur_fitness_mutable()[agent] = Fitness::r2_to_fitness(Evaluator::evaluate_sr_r2(prog, dataset));
//         }
//     }
// }

// void LGPEngine::evaluate_all_sr(const Dataset& dataset){
    
//     backend->evaluate_population(
//         cur_instructions().data(),
//         cur_lengths().data(),
//         cur_fitness_mutable().data(),
//         LGPConfig::POPULATION_SIZE,  // 
//         dataset);
// }
// old version iwht no offset for elites - use NAN check in hotloop

void LGPEngine::evaluate_range(int start, int count, const Dataset& dataset){
    backend->evaluate_population(//skipping elites
        cur_instructions().data()    + start * LGPConfig::MAX_PROGRAM_SIZE,  // stride!
        cur_lengths().data()         + start, 
        cur_fitness_mutable().data() + start, 
        count,
        dataset);
}

// ALL means entire populations
void LGPEngine::evaluate_all_sr(const Dataset& dataset){
    evaluate_range(0, LGPConfig::POPULATION_SIZE, dataset);
}
int LGPEngine::tournament_selection(){
    // select k agents, return the one with the highest fitness 
    //reads from the current buffer fitness fitness_scores

    int best_idx = dist_pop(rng); // get ran agent in population 
    float best_fit = cur_fitness()[best_idx]; // gets their fitness score 

    // make sure that fitness score is not NAN shouldnt be if evaluate all ran before xx 
    assert(!std::isnan(best_fit) && "Selection called before evaluate all (fitness NAN)");

    for (int agent = 1; agent < LGPConfig::TOURNAMENT_SIZE; ++agent){
        int cand = dist_pop(rng); // get ran agent in population 
        float cand_fit = cur_fitness()[cand]; // gets their fitness score 
        assert(!std::isnan(cand_fit) && "Selection called before evaluate all (fitness NAN)");

        if (cand_fit > best_fit){
            //std::cout<<"Candidate fitness ("<<cand_fit<<") > Best fit ("<<best_fit<<")"<<std::endl;
            best_fit = cand_fit;
            best_idx = cand;
        }
    }
    return best_idx;
}
// LGPEngine.cpp
#include <cstring>  // memcpy

void LGPEngine::crossover(const ProgramView& a, const ProgramView& b,
                          uint32_t* child_a, uint8_t& child_a_len,
                          uint32_t* child_b, uint8_t& child_b_len) {
    // Cut point in [1, min(len_a, len_b)). Both children get at least one
    // instruction from each parent. If either parent is length 1, we can't
    // do a meaningful crossover -- just clone both parents.

    // if (dist_unit(rng) > LGPConfig::CROSSOVER_RATE) {return;}
    const int min_len = std::min(a.length, b.length);
    if (min_len < 2) {
        std::memcpy(child_a, a.instructions, a.length * sizeof(uint32_t));
        std::memcpy(child_b, b.instructions, b.length * sizeof(uint32_t));
        child_a_len = static_cast<uint8_t>(a.length);
        child_b_len = static_cast<uint8_t>(b.length);
        return;
    }

    std::uniform_int_distribution<int> dist_cut(1, min_len - 1);
    const int cut = dist_cut(rng);

    // Child A: a's prefix + b's suffix. Final length = b.length.
    std::memcpy(child_a, a.instructions, cut * sizeof(uint32_t));
    std::memcpy(child_a + cut, b.instructions + cut,
                (b.length - cut) * sizeof(uint32_t));
    child_a_len = static_cast<uint8_t>(b.length);

    // Child B: b's prefix + a's suffix. Final length = a.length.
    std::memcpy(child_b, b.instructions, cut * sizeof(uint32_t));
    std::memcpy(child_b + cut, a.instructions + cut,
                (a.length - cut) * sizeof(uint32_t));
    child_b_len = static_cast<uint8_t>(a.length);
}
// Doesnt mirror crossover perfectly cuz probalistic natture baked into method... oops 
// this metthod could cause divergence on GPU 
void LGPEngine::mutate(uint32_t* program, uint8_t& length) {
    if (length == 0) return;

    if (dist_unit(rng) < LGPConfig::MICRO_RATE) {
        std::uniform_int_distribution<int> dist_inst(0, length - 1);
        const int idx = dist_inst(rng);
        program[idx] = micro_mutate_instruction(program[idx]);
    }

    if (dist_unit(rng) < LGPConfig::INSERT_TAIL_RATE) {
        if (length < LGPConfig::MAX_PROGRAM_SIZE) {
            program[length] = generate_instruction();
            ++length;
        }
    }

    if (dist_unit(rng) < LGPConfig::DELETE_TAIL_RATE) {
        if (length > 1) {
            --length;
        }
    }
}
uint32_t LGPEngine::micro_mutate_instruction(uint32_t instr) {
    // Pick one of 5 fields to perturb. Uniform over fields, not bits --
    // gives op-changes and mode-flips comparable mutation pressure even
    // though they occupy different bit widths.
    const int field = dist_field(rng);  // [0, 4]
    const uint32_t r = dist_32(rng);    // source of random bits

    switch (field) {
        // general formula is we are creating a mask and shifting it into its appropriate spot on the ISA (32 bits)
        case 0: {  // op
            const uint32_t mask = LGPConfig::OPERATION_MASK << ISA::OP_SHIFT; 
            const uint32_t new_bits = (r & LGPConfig::OPERATION_MASK) << ISA::OP_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 1: {  // dest
            const uint32_t mask = LGPConfig::REGISTER_MASK << ISA::DEST_SHIFT;
            const uint32_t new_bits = (r & LGPConfig::REGISTER_MASK) << ISA::DEST_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 2: {  // src1
            const uint32_t mask = LGPConfig::REGISTER_MASK << ISA::SRC1_SHIFT;
            const uint32_t new_bits = (r & LGPConfig::REGISTER_MASK) << ISA::SRC1_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 3: {  // src2 index (preserves the mode bit above it)
            const uint32_t mask = ISA::SRC2_INDEX_MASK << ISA::SRC2_SHIFT;
            const uint32_t new_bits = (r & ISA::SRC2_INDEX_MASK) << ISA::SRC2_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 4: {  // src2 mode bit (toggle register/constant)
            return instr ^ (static_cast<uint32_t>(ISA::MASK_MODE_BIT) << ISA::SRC2_SHIFT);
        }
    }
    return instr;  // unreachable
}
void LGPEngine::vary_pair(int dstA, int dstB){
    // get parents 
    int pA_idx = tournament_selection();
    int pB_idx = tournament_selection();
    ProgramView pA = view_program(pA_idx);
    ProgramView pB = view_program(pB_idx);

    // buffers chidren
    uint32_t* childA_buf = next_gen_ptr(dstA);
    uint32_t* childB_buf = next_gen_ptr(dstB);
    uint8_t& childA_length = next_lengths()[dstA];
    uint8_t& childB_length = next_lengths()[dstB];

    

    // if we dont crossover
    if (!(dist_unit(rng) < LGPConfig::CROSSOVER_RATE)){
    //clone parents 
        std::memcpy(childA_buf, pA.instructions, pA.length * sizeof(uint32_t) );// copy it exactly, destination, source, size in bytes
        std::memcpy(childB_buf, pB.instructions, pB.length * sizeof(uint32_t));

        childA_length = pA.length;
        childB_length = pB.length;
    }
    else{
        crossover(pA, pB, childA_buf, childA_length, childB_buf, childB_length);
    }
    // now do the mutation
    mutate(childA_buf, childA_length);
    mutate(childB_buf, childB_length);

    // set their fitness as Nan 
    next_fitness()[dstA] = std::numeric_limits<float>::quiet_NaN();
    next_fitness()[dstB] = std::numeric_limits<float>::quiet_NaN();
}
std::vector<int> LGPEngine::top_k_indices(int k) const{
    // create vector of popsize and fill with numfrom 0 to popsize - 1
    std::vector<int> idx(LGPConfig::POPULATION_SIZE);
    std::iota(idx.begin(), idx.end(), 0);

    const auto& fit = cur_fitness(); // get reference to cur fitness vectors 

    // sort by top k indices 
    // get iterator for being, sorts up to iterator + k, sort partial from it begin to it end. 
    // lambda as comparator to sort indices by their fitness
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&fit](int a, int b){return fit[a]> fit[b];}); // uses lambda to sort indices by fitness 
    idx.resize(k); // resize to k elements idx of top k fittest individuals 
    return idx;

}
// WORTH CONSIDERING IN GPU PORT - this COPY Program copies dif amount based on length woould 
void LGPEngine::copy_elite_to_next(int srcIdx, int dstIdx){
    const uint32_t* source = cur_instructions().data() + srcIdx * LGPConfig::MAX_PROGRAM_SIZE; // ptr to start of program at srcIdx  in current buffer (pointer to program at s)
    uint32_t* dest = next_gen_ptr(dstIdx); // pointer to program at destination index in next gen buffer 
    uint8_t length = cur_lengths()[srcIdx]; // lenght of program we are copying 

    // now the memcopy 
    std::memcpy(dest, source, sizeof(uint32_t) * length);
    next_lengths()[dstIdx] = length;
    next_fitness()[dstIdx] = cur_fitness()[srcIdx];
}

void LGPEngine::vary(){
    // fill elites - carries over length and fitness and all that good stuff 
    std::vector<int> topK = top_k_indices(LGPConfig::ELITE_COUNT);

    for (int e = 0; e < LGPConfig::ELITE_COUNT; ++e){ // FILLS first spots 
        copy_elite_to_next(topK[e], e);  // 
    }
    // populate with real variation 
    for (int children = LGPConfig::ELITE_COUNT; children < LGPConfig::POPULATION_SIZE; children +=2){
        vary_pair(children, children +1); // passing in destination indices 
    }
    flip_generation();
}
void LGPEngine::reset_buffers_to_sentinels() {
    for (int b = 0; b < 2; ++b) {
        std::fill(data.fitness_scores_buf[b].begin(),
                  data.fitness_scores_buf[b].end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(data.program_lengths_buf[b].begin(),
                  data.program_lengths_buf[b].end(),
                  static_cast<uint8_t>(LGPConfig::SIZE_SENTINEL));
    }
}
GenerationStats LGPEngine::compute_stats() const {
    const auto& fit = cur_fitness();
    const auto& len = cur_lengths();

    float best = -1.0f;
    float sum = 0.0f;
    int len_sum = 0;
    int best_idx = 0;
    for (int i = 0; i < LGPConfig::POPULATION_SIZE; ++i) {
        len_sum += static_cast<int>(len[i]);
        sum += fit[i];
        if (fit[i] > best) {
            best = fit[i];
            best_idx = i;
        }
    }

    return GenerationStats{
        current_generation,
        best,
        sum / LGPConfig::POPULATION_SIZE,
        static_cast<int>(len[best_idx]),
        static_cast<float>(len_sum)/LGPConfig::POPULATION_SIZE,
        best_idx
    };
}
void LGPEngine::evolve_sr(const Dataset& dataset){
    init_evolution(); // set programs to random instructs, resets all buff and gen counts

    evaluate_all_sr(dataset);
    history.push_back(compute_stats());

    for (int gen = 1; gen < LGPConfig::MAX_GENERATIONS; ++gen){
        current_generation = gen;
        vary(); // this does tthe flip 
        evaluate_range(LGPConfig::ELITE_COUNT,
                   LGPConfig::NON_ELITE_COUNT, dataset); // skips elites 
        history.push_back(compute_stats()); // based on cur gen - if something flips then reading from wrong buffer
    }
}
void LGPEngine::print_history() const {
    std::cout << "gen,best,mean,best_len\n";
    for (const auto& s : history) {
        std::cout << s.generation << ","
                  << s.best_fitness << ","
                  << s.mean_fitness << ","
                  << s.best_length << "," 
                  <<s.mean_length <<"\n";
    }
}
// Helper: opcode -> printable symbol. Lives near print_best_program (or in
// ISA.h if you want it shared). Returns infix symbol for binary ops, a name
// for unary/function ops.
static const char* op_symbol(uint8_t op) {
    switch (op) {
        case ISA::ADD: return "+";
        case ISA::SUB: return "-";
        case ISA::MUL: return "*";
        case ISA::DIV: return "/";
        case ISA::LT:  return "<";
        case ISA::GT:  return ">";
        case ISA::SIN: return "sin";
        case ISA::COS: return "cos";
        default:       return "?";
    }
}

void LGPEngine::print_best_program(std::ostream& os) const {
    const GenerationStats& last = history.back();
    const uint32_t* prog = cur_instructions().data()
                         + last.best_index * LGPConfig::MAX_PROGRAM_SIZE;

    os << "Best program (fitness " << last.best_fitness
              << ", length " << last.best_length << "):\n";

    for (int i = 0; i < last.best_length; ++i) {
        const uint32_t instr = prog[i];
        const uint8_t  op    = ISA::get_op(instr);
        const uint8_t  dest  = ISA::get_dest_index(instr);
        const uint8_t  src1  = ISA::get_src1_index(instr);
        const uint8_t  src2  = ISA::get_src2_index(instr);
        const bool     s2c   = ISA::is_src2_constant(instr);

        os << "  [" << i << "] r" << (int)dest << " = ";

        if (op == ISA::SIN || op == ISA::COS) {
            // Unary: reads src2 (which may be a register or a constant),
            // ignores src1. Print the function form.
            os << op_symbol(op) << "(";
            if (s2c) os << LGPConfig::CONSTANTS[src2];
            else     os<< "r" << (int)src2;
            os << ")";
        } else {
            // Binary: r[src1] <op> (r[src2] or CONSTANTS[src2])
           os<< "r" << (int)src1 << " " << op_symbol(op) << " ";
            if (s2c) os << LGPConfig::CONSTANTS[src2];
            else     os << "r" << (int)src2;
        }

        os << "\n";
    }
}
// Best program's R² on the training data it evolved against.
// This is the stored fitness, which already went through r2_to_fitness
// (floored at 0). So a sub-mean champion reads as 0 here.
float LGPEngine::best_train_r2() const {
    return history.back().best_fitness;
}

// Best program's RAW R² on a different (held-out) dataset.
// Not floored -- a model that fits train but fails to generalize will
// show negative here, which is the honest signal we want for the test column.
float LGPEngine::best_test_r2(const Dataset& test) const {
    const int idx = history.back().best_index;
    const ProgramView prog = view_program(idx);
    return Evaluator::evaluate_sr_r2(prog, test);
}

int LGPEngine::best_length() const {
    return history.back().best_length;
}
