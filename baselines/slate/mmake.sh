# Locations of SLATE , BLAS++, LAPACK++ install or build directories.
export SLATE_ROOT=/opt/slate
export BLASPP_ROOT=${SLATE_ROOT} # /build/blaspp # or ${SLATE_ROOT}, if installed
export LAPACKPP_ROOT=${SPACK_DIR} # /build/lapackpp # or ${SLATE_ROOT}, if installed
export SLATE_GPU_AWARE_MPI=1
# export ROCM_PATH=/opt/rocm # wherever ROCm is installed
# Compile the example.

${MPI_HOME}/bin/mpicxx -fopenmp -c lu_slate.cc \
-I../../include \
-I${SLATE_ROOT}/include \
-I${CUDA_HOME}/include # For CUDA
# -I${SLATE_ROOT}/include/blas \
# -I${BLASPP_ROOT}/include \
# -I${LAPACKPP_ROOT}/include \
# -I${ROCM_PATH}/include # For ROCm

${MPI_HOME}/bin/mpicxx -fopenmp -o lu_slate lu_slate.o \
-L${SLATE_ROOT}/lib -Wl,-rpath,${SLATE_ROOT}/lib \
-L${CUDA_HOME}/lib64 -Wl,-rpath,${CUDA_HOME}/lib64 \
-lslate -llapackpp -lblaspp -lcusolver -lcublas -lcudart
# -L${BLASPP_ROOT} -Wl,-rpath,${BLASPP_ROOT}/lib64 \
# -L${LAPACKPP_ROOT} -Wl,-rpath,${LAPACKPP_ROOT}/lib64 \


${MPI_HOME}/bin/mpicxx -fopenmp -c chol_slate.cc \
-I../../include \
-I${SLATE_ROOT}/include \
-I${CUDA_HOME}/include # For CUDA
# -I${SLATE_ROOT}/include/blas \
# -I${BLASPP_ROOT}/include \
# -I${LAPACKPP_ROOT}/include \
# -I${ROCM_PATH}/include # For ROCm

${MPI_HOME}/bin/mpicxx -fopenmp -o chol_slate chol_slate.o \
-L${SLATE_ROOT}/lib -Wl,-rpath,${SLATE_ROOT}/lib \
-L${CUDA_HOME}/lib64 -Wl,-rpath,${CUDA_HOME}/lib64 \
-lslate -llapackpp -lblaspp -lcusolver -lcublas -lcudart

# For ROCm , may need to add:
# -L${ROCM_PATH}/lib -Wl,-rpath ,${ROCM_PATH}/lib \
# -lrocsolver -lrocblas -lamdhip64

# Run the slate_lu executable.
# mpirun -n 4 ./ slate_lu

# Output from the run will be something like the following:
# lu_solve n 5000 , nb 256, p-by-q 2-by-2, residual 8.41e-20, tol 2.22e-16, time 7.65e-01 sec,
# pass
