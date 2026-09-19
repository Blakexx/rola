#include <stdint.h>
//: one dynamic instruction of a traced warp; a memory operand follows as `mref_t` records.
typedef struct {
    uint32_t pc;        //: the instruction's offset in its function
    uint32_t active;    //: the active-lane mask at the call
    uint32_t pred;      //: the guard predicate's ballot over the active lanes
    uint16_t opcode_id;
    uint16_t warp_id;
    uint64_t clock;     //: clock64 at the call (instrumented time: ordering, not cycles)
} instr_rec_t;
typedef struct {
    uint32_t tag;       //: 0xFFFFFFFFu
    uint32_t mref_idx;
    uint64_t addrs[32];
} mref_t;
