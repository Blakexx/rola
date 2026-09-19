#include <stdint.h>
#include <stdio.h>
#include "utils/utils.h"
#include "utils/channel.hpp"
#include "common.h"

extern "C" __device__ __noinline__ void instrument_instr(int pred, int opcode_id, int pc, int nmref,
                                                         uint64_t addr0, uint64_t addr1,
                                                         uint64_t pchannel_dev, int target_cta, int with_addrs) {
    const int4 cta = get_ctaid();
    if (cta.x != target_cta || cta.y != 0) return;
    const int active = __ballot_sync(__activemask(), 1);
    const int predmask = __ballot_sync(active, pred);
    const int laneid = get_laneid();
    const int first = __ffs(active) - 1;
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    instr_rec_t r;
    r.pc = (uint32_t)pc;
    r.active = (uint32_t)active;
    r.pred = (uint32_t)predmask;
    r.opcode_id = (uint16_t)opcode_id;
    r.warp_id = (uint16_t)(threadIdx.x >> 5);
    r.clock = (uint64_t)clock64();
    if (laneid == first) channel_dev->push(&r, sizeof(instr_rec_t));
    if (!with_addrs) return;
    for (int m = 0; m < nmref; ++m) {
        const uint64_t a = m == 0 ? addr0 : addr1;
        mref_t mr;
        mr.tag = 0xFFFF0000u | (uint32_t)(threadIdx.x >> 5);
        mr.mref_idx = (uint32_t)m;
        for (int i = 0; i < 32; i++) mr.addrs[i] = __shfl_sync(active, a, i);
        if (laneid == first) channel_dev->push(&mr, sizeof(mref_t));
    }
}
