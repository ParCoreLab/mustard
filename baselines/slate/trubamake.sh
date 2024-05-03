# Locations of SLATE , BLAS++, LAPACK++ install or build directories.
# export SLATE_ROOT=/opt/slate
# export BLASPP_ROOT=${SLATE_ROOT} # /build/blaspp # or ${SLATE_ROOT}, if installed
# export LAPACKPP_ROOT=${SPACK_DIR} # /build/lapackpp # or ${SLATE_ROOT}, if installed
# export SLATE_GPU_AWARE_MPI=1
export SPACK_DIR=/truba/home/iturimbetov/spack/opt/spack/linux-rhel8-haswell/gcc-12.2.0
export SLATE_ROOT=${SPACK_DIR}/slate-2023.11.05-mjtfw3vjcfqqctk52e342hzoeqjywypw
export BLASPP_ROOT=${SPACK_DIR}/blaspp-2023.11.05-qmmuj32txdbzvcn3o53cqhk3m6a4oiuy # or ${SLATE_ROOT}, if installed
export LAPACKPP_ROOT=${SPACK_DIR}/lapackpp-2023.11.05-l7s5iijbq5gmhihem2vh2u563fk4fljl # or ${SLATE_ROOT}, if installed
export CUDA_HOME=${SPACK_DIR}/cuda-12.3.2-si3xh46372xqqv7ilnpgsbpmshivdzrv # wherever CUDA is installed
export MPI_HOME=${SPACK_DIR}/openmpi-5.0.2-fsapuqh6k5fzsaasuqn4a45t4vbywxiu
export SLATE_GPU_AWARE_MPI=1

# export ROCM_PATH=/opt/rocm # wherever ROCm is installed
# Compile the example.
INCLUDES="-I../../include "
INCLUDES+="-I${BLASPP_ROOT}/include "
INCLUDES+="-I${LAPACKPP_ROOT}/include "
INCLUDES+="-I${SLATE_ROOT}/include "
INCLUDES+="-I${CUDA_HOME}/include "

LIBS="-L${BLASPP_ROOT}/lib64 -Wl,-rpath,${BLASPP_ROOT}/lib64 "
LIBS+="-L${LAPACKPP_ROOT}/lib64 -Wl,-rpath,${LAPACKPP_ROOT}/lib64 "
LIBS+="-L${SLATE_ROOT}/lib64 -Wl,-rpath,${SLATE_ROOT}/lib64 "
LIBS+="-L${CUDA_HOME}/lib64 -Wl,-rpath,${CUDA_HOME}/lib64 "
LIBS+="-lslate -llapackpp -lblaspp -lcusolver -lcublas -lcudart "

${MPI_HOME}/bin/mpicxx -fopenmp -c lu_slate.cc ${INCLUDES}
# -I${ROCM_PATH}/include # For ROCm

${MPI_HOME}/bin/mpicxx -fopenmp -o lu_slate lu_slate.o ${LIBS}

${MPI_HOME}/bin/mpicxx -fopenmp -c chol_slate.cc ${INCLUDES}
# -I${ROCM_PATH}/include # For ROCm

${MPI_HOME}/bin/mpicxx -fopenmp -o chol_slate chol_slate.o ${LIBS}

# For ROCm , may need to add:
# -L${ROCM_PATH}/lib -Wl,-rpath ,${ROCM_PATH}/lib \
# -lrocsolver -lrocblas -lamdhip64

# Run the slate_lu executable.
# mpirun -n 4 ./ slate_lu

# Output from the run will be something like the following:
# lu_solve n 5000 , nb 256, p-by-q 2-by-2, residual 8.41e-20, tol 2.22e-16, time 7.65e-01 sec,
# pass
