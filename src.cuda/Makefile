default: conv
NVCC = nvcc
CC = gcc-10

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3
OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

# CFLAGS = -O3 -m64 -ffloat-store -w -g
CFLAGS = -O3 -std=c++17 -w -g -fPIC -lm
NVCCFLAGS = -O3 -w -g -c -Xcompiler "$(CFLAGS)"
LDFLAGS = -lOpenCL

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(CUDA_LINK_FLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

hostFE.o: hostFE.cu
	$(NVCC) $(CUDA_COMPILE_FLAGS) hostFE.cu -o hostFE.o

clean:
	rm -f conv *.o output.bmp ref.bmp