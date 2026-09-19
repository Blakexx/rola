#include "utils/channel.hpp"

/* flush channel */
extern "C" __global__ void flush_channel(ChannelDev* ch_dev) { ch_dev->flush(); }

