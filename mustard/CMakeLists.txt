add_executable(cholesky_mustard cholesky_mustard.cu)

add_executable(lu_mustard lu_mustard.cu)
add_executable(p_lu_mustard lu_mustard_p.cu)

find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)
find_package(METIS)

if(METIS_FOUND)
    target_link_libraries(cholesky_mustard PUBLIC METIS::METIS)
    target_link_libraries(lu_mustard PUBLIC METIS::METIS)
else()
    message(WARNING "METIS library not found. METIS support will be disabled.")
endif()

set(LIBS 
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::nvml
    nvshmem::nvshmem
    MPI::MPI_CXX)

target_link_libraries(cholesky_mustard ${LIBS})
target_link_libraries(lu_mustard ${LIBS})
target_link_libraries(p_lu_mustard ${LIBS})

# target_compile_options(cholesky_mustard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#         -D_FORCE_INLINES
#         >)

# target_compile_options(lu_mustard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#         -D_FORCE_INLINES
#         >)

# target_include_directories(cholesky_mustard SYSTEM PRIVATE 
#                             "${CMAKE_CURRENT_SOURCE_DIR}")
