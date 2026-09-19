#include <stdint.h>
//: one dynamic instruction of a traced warp; a memory operand follows as `mref_t` records.
typedef struct {
    uint32_t pc;        //: the instruction's offset in its function
    uint32_t active;    //: the active-lane mask at the call
    uint32_t pred;      //: the guard predicate's ballot over the active lanes
    uint16_t opcode_id;
    uint16_t warp_id;
    uint64_t clock;     //: clock64 at the call (instrumented time: ordering, not cycles)
    uint32_t preds;     //: the first active lane's predicates before the instruction: P0..P6 in bits 0..6, UP0..UP6 in 8..14
    uint32_t reserved;
} instr_rec_t;
typedef struct {
    uint32_t tag;       //: 0xFFFFFFFFu
    uint32_t mref_idx;
    uint64_t addrs[32];
} mref_t;
//: the value a compare read from a shared load's destination (a barrier poll's word): tag 0xFFFE0000 | warp.
typedef struct {
    uint32_t tag;
    uint32_t pc;
    uint32_t value;     //: the first active lane's register value before the compare
    uint32_t reserved;
} value_rec_t;
