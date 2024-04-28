CFLAGS += $(shell pkg-config --cflags starpu-1.3)
LDLIBS += $(shell pkg-config --libs starpu-1.3)
CFLAGS += -Wno-deprecated-declarations
CFLAGS += -DSTARPU_OPENBLAS

CFLAGS += -I/usr/include/x86_64-linux-gnu/openblas-pthread/ 
CFLAGS += -I../../../include 
CFLAGS += -I../common
LDLIBS +=  -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lblas

# List of source files
SRCS = lu_example_double.c dlu.c dlu_pivot.c dlu_kernels.c ../common/blas.c

# Generate a list of object files from source files
OBJS = $(SRCS:.c=.o)

# Rule to build each object file
%.o: %.c
	nvcc $(CFLAGS) $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart $< -c -o $@

all: lu_example_double

# cholesky_tile_tag.o: cholesky_models.c cholesky_kernels.c cholesky_tile_tag.c
# 	nvcc $(CFLAGS) $< -c -o $@ -L$(CUDA_HOME)/lib64 -lcudart

lu_example_double: $(OBJS)
	nvcc $(CFLAGS) $^ $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusolver -lcusparse -lcusolverMg -o $@
#vector_scal_cuda.o vector_scal_opencl.o
#vector_scal_task_insert: LDLIBS+=-lOpenCL
#test: LDLIBS+=-lcudart -lstdc++
#mult: mult.c
#gemm/sgemm: gemm/sgemm.o gemm/common/blas.o
#gemm/dgemm: gemm/dgemm.o gemm/common/blas.o

#gemm/sgemm: LDLIBS+=-lblas
#gemm/sgemm: LDLIBS+=-lcublas -lcudart
#gemm/dgemm: LDLIBS+=-lblas
#gemm/dgemm: LDLIBS+=-lcublas -lcudart

clean:
	rm -f lu_example_double *.o
#	rm -f gemm/sgemm gemm/dgemm gemm/*.o gemm/common/*.o

