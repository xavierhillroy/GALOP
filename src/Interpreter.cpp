#include "Interpreter.h"
#include "ISA.h"
#include "LGPConfig.h"
#include <cassert>

// complex flattening done here for readability 
inline static float& reg(float* registers, int c, int i ){
    return registers[c* LGPConfig::NUM_REGISTERS + i];

}
inline static void execute(const uint32_t* instructions, int length, float *registers){
    // Goes through a program and executes
    for (int pc = 0; pc < length; ++pc){

        // Decode the instruction 
        uint32_t cur_instruction = instructions[pc];
        uint8_t dest_index = ISA::get_dest_index(cur_instruction);
        uint8_t op = ISA::get_op(cur_instruction);
        uint8_t src1_index = ISA::get_src1_index(cur_instruction);
        uint8_t src2_index = ISA::get_src2_index(cur_instruction);
        bool src2_const = ISA::is_src2_constant(cur_instruction);
        
        //Now apply for each conttext - this will magically dissapear in GPU version ;)
        for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c){
            float a = reg(registers, c, src1_index);
            float b = src2_const ? 
            LGPConfig::CONSTANTS[src2_index] 
            : reg(registers, c, src2_index);

            // now apply Values
            reg(registers, c, dest_index) = ISA::apply_op(op, a, b); // apply the operation
        }
    }
}
namespace Interpreter{
    void run_stateless(const uint32_t* instructions, int inst_length, const float* inputs,  int num_inputs, float* outputs){
            // allocate registers 
            float registers[LGPConfig::NUM_CONTEXTS * LGPConfig::NUM_REGISTERS]; // NUM REGISTERS for each contexts
            assert(num_inputs<= LGPConfig::NUM_REGISTERS);
            // init inputs and 0.0f the rest
            for(int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c){
                // store inputs
                for (int inp = 0; inp < num_inputs; ++inp){
                    reg(registers, c, inp) = inputs[c * num_inputs + inp]; // DIF STride to reg(num inputs stride vs num registers stride) !
                }
                
                for (int inp = num_inputs; inp < LGPConfig::NUM_REGISTERS; ++inp){
                    reg(registers, c, inp) = 0.0f;
                }
            }
            // actual execution of program
            execute(instructions, inst_length, registers);

            // set the outputs 

            for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c){
                outputs[c] = reg(registers, c, 0); // return is r0
            }
    }
}