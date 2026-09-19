# The warp tracer, an NVBit tool: every dynamic instruction of one CTA's warps to a file. Built by
# tools/warp_trace.py, which passes the pinned NVBit release, the toolkit's nvcc and the device's arch.
NVBIT_PATH ?= ../../../../nvbit_release/core
NVCC ?= nvcc
ARCH ?= sm_86
CXX ?= g++
NVCC_FLAGS = -ccbin=$(CXX) -D_FORCE_INLINES -I$(NVBIT_PATH) -arch=$(ARCH) -Xcompiler -fPIC
CUDA_LIB = $(dir $(NVCC))../lib64
OUT ?= warp_trace.so

$(OUT): warp_trace.o inject_funcs.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 warp_trace.o inject_funcs.o -L$(NVBIT_PATH) -lnvbit -L$(CUDA_LIB) -lcuda -lcudart_static -shared -o $@

warp_trace.o: warp_trace.cu tool_func/flush_channel.c
	$(NVCC) -dc -c -std=c++11 $(NVCC_FLAGS) -Xptxas -cloning=no -O3 warp_trace.cu -o $@

inject_funcs.o: inject_funcs.cu common.h
	$(NVCC) $(NVCC_FLAGS) -Xptxas -astoolspatch --keep-device-functions -c inject_funcs.cu -o $@

tool_func/flush_channel.c: tool_func/flush_channel.cu
	$(NVCC) -arch=$(ARCH) -I$(NVBIT_PATH) -O3 -fatbin $< -o tool_func/flush_channel.fatbin
	$(dir $(NVCC))bin2c -c --name flush_channel_bin tool_func/flush_channel.fatbin > $@

clean:
	rm -f *.o *.so tool_func/*.fatbin tool_func/*.c
