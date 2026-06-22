#ifndef ISA_H
#define ISA_H
#include <cstdint>
#include "LGPConfig.h"
#include <cmath>

// Let these inline helpers be called from CUDA device code: the GPU kernel
// reuses the EXACT same decode/apply logic as the CPU interpreter, so the two
// can never drift. Under nvcc, decorate with __host__ __device__; under a
// plain host compiler, ISA_HD expands to nothing and the code is unchanged.
#ifdef __CUDACC__
  #define ISA_HD __host__ __device__
#else
  #define ISA_HD
#endif

// =============================================================================
// ISA: Instruction Set Architecture for the LGP system.
//
// An instruction is a single 32-bit word packed as four 8-bit fields:
//
//   bit:  31 ........ 24 | 23 ........ 16 | 15 .......... 8 | 7 ........... 0
//        [    SRC2     ] [    SRC1      ] [     DEST       ] [      OP      ]
//        [mode|index   ] [ reg index    ] [  reg index     ] [   opcode     ]
//        [ 1  |   7    ] [     8        ] [       8        ] [     8        ]
//
// OP, DEST, and SRC1 each occupy a full byte, but only the low
// OPERATIONS_BITS / REGISTER_BITS of each byte are meaningful -- the rest are
// masked out during encoding.
//
// SRC2 is special: its high bit is a mode flag (0 = register, 1 = constant)
// and the low log2(NUM_REGISTERS) bits are the index into whichever pool was selected, remaining bits for future widening. The
// constraint NUM_REGISTERS == NUM_CONSTANTS (asserted in LGPConfig.h) lets
// the same mask handle both cases.
// To make things cleaner although we have 8 bits for dif sources, we usually use log2(NUM_X) for indexing 
//
// Semantic conventions for the instruction:
//   r[DEST] = r[SRC1]  <op>  (r[SRC2_idx] or CONSTANTS[SRC2_idx])
// where the choice between register and constant is controlled by the mode
// bit in SRC2. Unary ops (SIN, COS) and comparison ops (LT, GT) need their
// own handling in the interpreter -- see notes on the opcode enum below.
// =============================================================================

namespace ISA {

    // -------------------------------------------------------------------------
    // Bit positions (shifts) of each field within the 32-bit instruction.
    // Using these named shifts instead of raw numbers means all the bit
    // layout decisions live in one place.
    // -------------------------------------------------------------------------
    constexpr int OP_SHIFT   = 0;
    constexpr int DEST_SHIFT = 8;
    constexpr int SRC1_SHIFT = 16;
    constexpr int SRC2_SHIFT = 24;

    // -------------------------------------------------------------------------
    // Opcode enum. Values are implicit (0, 1, 2, ...) so the ordering here
    // literally defines the numeric opcode. OPCOUNT is a sentinel whose value
    // equals the number of entries above it; the static_assert ties it to
    // LGPConfig::NUM_OPERATIONS so the two can't silently drift apart.
    //
    // Semantic notes for the interpreter (to resolve later):WS
    //   SIN, COS   -- unary; src2 field is unused. Mutation of src2 on these
    //                 opcodes is structurally neutral.
    //   LT, GT     -- produce a float 1.0 / 0.0 into the dest register.
    //   DIV        -- needs protected division (e.g. return 1.0 on b == 0)
    //                 to avoid poisoning registers with inf/nan.
    // -------------------------------------------------------------------------
    enum OpCode : uint8_t {
        ADD, SUB, MUL, DIV, SIN, COS, LT, GT,
        OPCOUNT   // auto = 8; must match NUM_OPERATIONS
    };
    static_assert(OPCOUNT == LGPConfig::NUM_OPERATIONS,
                  "OpCode enum and NUM_OPERATIONS are out of sync");

    // -------------------------------------------------------------------------
    // Masks for the SRC2 byte.
    //
    // MASK_MODE_BIT isolates the top bit of the SRC2 byte: 0 = register,
    // 1 = constant. SRC2_INDEX_MASK isolates the low bits used as an index
    // into whichever pool was selected. Because NUM_REGISTERS ==
    // NUM_CONSTANTS, one mask works for both.
    // -------------------------------------------------------------------------
    constexpr uint8_t MASK_MODE_BIT  = 0x80;  // 1000 0000
    constexpr int     SRC2_INDEX_MASK = LGPConfig::REGISTER_MASK;
                                              // == CONSTANT_MASK by the
                                              // static_assert in LGPConfig.h

    // =========================================================================
    // ENCODING
    // =========================================================================

    // -------------------------------------------------------------------------
    // encode_from_random: take a uniformly random 32-bit word and extract
    // just the bits that represent valid instruction fields, zeroing the
    // rest. Used during population initialization where every program slot
    // is filled with a random valid instruction.
    //
    // The general pattern for each field is:
    //   1. Shift the field's mask to the field's position in the word.
    //   2. AND with raw_rand to keep only those bits.
    //   3. OR into the accumulator.
    // No runtime randomness needed per-field -- one 32-bit draw supplies
    // enough bits to fill every field of the instruction.
    // -------------------------------------------------------------------------
    ISA_HD inline uint32_t encode_from_random(uint32_t raw_rand){
        /**
        Example of what we are doing 
        encoded_instruct = 0000 0000 0000 0000 0000 0000 0000 0000 
        raw_rand = 0010 0111 1111 0010 1010 0011 1110 1001
        let us say we are encoding the reg (dest) portion at this poitn and lets say its 3 bits (8 registers - must use 2^n registers) then the shift mask is 7 (111)cuz 3 1 bits
        dest shift = 8 
        REGISTER_MASK = 0000 0000 0000 0000 0000 0000 0000 0111
        then we shift it by 8 to left 0000 0000 0000 0000 0000 0000 0000 0111 << 8 =  0000 0000 0000 0000 0000 0111 0000 0000
        Now we & raw_rand & shiftted reg MASk:  0010 0111 1111 0010 1010 0011 1110 1001 & 0000 0000 0000 0000 0000 0111 0000 0000 = 0000 0000 0000 0000 0000 `0011` 0000 0000 - leaving only these bits impacted (the ones impacting the dest reg)
        Then we or that with the encoded_instruct stream thus cleaning the uneeded garbage bits ( we could have left it but for legibility we cleaned it )
        */
        uint32_t encoded_instruct = 0;

        // Opcode: low 3 bits of raw_rand land in the OP byte.
        encoded_instruct |= raw_rand & (LGPConfig::OPERATION_MASK << OP_SHIFT);
        // Dest register: 3 bits from raw_rand land in the DEST byte.
        encoded_instruct |= raw_rand & (LGPConfig::REGISTER_MASK  << DEST_SHIFT);
        // Src1 register: 3 bits from raw_rand land in the SRC1 byte.
        encoded_instruct |= raw_rand & (LGPConfig::REGISTER_MASK  << SRC1_SHIFT);

        // SRC2 is two parts. First the index (bottom 7 bits of that byte):
        // this ALSO strips the mode flag, because SRC2_INDEX_MASK has 0 in
        // bit 7. Then we add the mode bit back from raw_rand in a second OR.
        // The static_cast<uint32_t> on MASK_MODE_BIT is important: without
        // it, shifting 0x80 << 24 would land in the sign bit of an `int`
        // and trip signed-shift UB. Promoting to uint32_t first keeps the
        // operation well-defined.
        encoded_instruct |= raw_rand & (SRC2_INDEX_MASK << SRC2_SHIFT);
        encoded_instruct |= raw_rand & (static_cast<uint32_t>(MASK_MODE_BIT) << SRC2_SHIFT);

        return encoded_instruct;
    }

    // -------------------------------------------------------------------------
    // encode_manual: construct an instruction from explicit field values.
    // Used for tests (building known instructions to verify the decoder) and
    // anywhere else we want to synthesize a specific instruction rather than
    // derive it from random bits.
    //
    // Each input is masked before shifting, so passing oversized values is
    // safe -- they get clipped to the valid range rather than corrupting
    // neighbouring fields.
    // -------------------------------------------------------------------------
    ISA_HD inline uint32_t encode_manual(uint8_t op, uint8_t dest, uint8_t src1, uint8_t src2, bool is_constant){
        uint32_t encoded_instruct = 0;
        encoded_instruct |= (static_cast<uint32_t>(op   & LGPConfig::OPERATION_MASK) << OP_SHIFT);
        encoded_instruct |= (static_cast<uint32_t>(dest & LGPConfig::REGISTER_MASK)  << DEST_SHIFT);
        encoded_instruct |= (static_cast<uint32_t>(src1 & LGPConfig::REGISTER_MASK)  << SRC1_SHIFT);

        // src2 is two parts: the 7-bit index and the 1-bit mode flag.
        // Same split as in encode_from_random, just with caller-supplied bits.
        encoded_instruct |= (static_cast<uint32_t>(src2 & SRC2_INDEX_MASK) << SRC2_SHIFT);
       if (is_constant) {
            encoded_instruct |= (static_cast<uint32_t>(MASK_MODE_BIT) << SRC2_SHIFT);
        } // some bit shifting if its constant 0x80 is 1000 0000 - so shifts 1 to top meaning its a const



        return encoded_instruct;
    }

    // =========================================================================
    // DECODING
    //
    // Each getter shifts the target byte into the low position and masks it
    // down to the valid field width. These are the inverse of the encoders
    // and should round-trip for any instruction produced by either encoder.
    // =========================================================================

    ISA_HD inline uint8_t get_op(uint32_t instruction){
        // OP lives in the low byte; no shift needed.
        return (instruction >> OP_SHIFT) & LGPConfig::OPERATION_MASK;
    }

    ISA_HD inline uint8_t get_dest_index(uint32_t instruction){
        return ((instruction >> DEST_SHIFT) & LGPConfig::REGISTER_MASK);
    }

    ISA_HD inline uint8_t get_src1_index(uint32_t instruction){
        return ((instruction >> SRC1_SHIFT) & LGPConfig::REGISTER_MASK);
    }

    // SRC2 decomposes into two pieces, so it gets two accessors: one for
    // the index and one for the mode flag. The interpreter should call
    // is_src2_constant() first to decide which pool (registers vs. CONSTANTS)
    // the index should select into.
    ISA_HD inline uint8_t get_src2_index(uint32_t instruction){
        return (instruction >> SRC2_SHIFT) & SRC2_INDEX_MASK;
    }

    ISA_HD inline bool is_src2_constant(uint32_t instruction){
        // Shift the SRC2 byte into the low position, then test the mode bit.
        // Returns true if src2 should index into CONSTANTS[], false for
        // the register file.
        return ((instruction >> SRC2_SHIFT) & MASK_MODE_BIT) != 0;
    }

    // -------------------------------------------------------------------------
    // apply_op: execute a single opcode on a pair of operand values.
    // 
    // For unary opcodes (SIN, COS), the first operand is ignored. The caller
    // is responsible for fetching operands from registers/constants before
    // calling this; this function is purely semantic.
    //
    // Inlined and header-resident so the compiler can specialize at every
    // call site and (later) so the GPU and CPU interpreters share one
    // canonical definition.
    // -------------------------------------------------------------------------
    ISA_HD inline float apply_op(uint8_t op, float a, float b) {
        switch (op) {
            case ADD: return a + b;
            case SUB: return a - b;
            case MUL: return a * b;
            case DIV: return (b != 0.0f) ? a / b : 1.0f;   // protected division
            case SIN: return std::sin(b);                  // unary: a ignored
            case COS: return std::cos(b);                  // unary: a ignored
            case LT:  return (a < b) ? 1.0f : 0.0f;
            case GT:  return (a > b) ? 1.0f : 0.0f;
            default:  return 0.0f;                          // unreachable
        }
    }

}  // namespace ISA

#endif  // ISA_H