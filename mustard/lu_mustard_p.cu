#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>

#include "argh.h"
#include "mustard.h"

#define MAX_TILE 4000

size_t N = 15 * 1;
size_t B = N / 5;
size_t T = N / B;
size_t local_N;
int myPE;
int nPE;
int verbose = 0;
int workspace = 256;
int smLimit = 20;
int runs = 1;
int numCopyStreams = 32;

int nodeToPrint = 0;
int peToPrint = 1;

// Set upper triangle entries (excluding diagonal entries) in column-major order to zero.
// Then, transpose to row-major order.
void cleanCusolverLUDecompositionResult(double *L, double *U, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            std::swap(L[i + j * n], L[i * n + j]);
            U[i * n + j] = L[i * n + j];
            L[i * n + j] = 0;
        }
        L[i * n + i] = 1;
    }
}

bool verifyLUDecomposition(double *A, double *L, double *U, const int n)
{
    auto newA = std::make_unique<double[]>(n * n);
    memset(newA.get(), 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                newA[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }

    double error = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            error += fabs(A[i * n + j] - newA[i * n + j]);
        }
    }

    if (verbose) {
        printf("A:\n");
        printSquareMatrix(A, n);

        printf("\nnewA:\n");
        printSquareMatrix(newA.get(), n);

        printf("\nL:\n");
        printSquareMatrix(L, n);
        printf("\n");

        printf("\nU:\n");
        printSquareMatrix(U, n);
        printf("\n");
    }

    printf("error = %.6f}\n", error);

    return error <= 1e-6;
}


__global__ void kernel_print_matrix(double *A, const size_t n, const size_t m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (j != 0)
                printf(" ");
            // printf("A[%d]=%.3f", i * m + j, A[i * m + j]);
            printf("%.6f", A[i * m + j]);
        }
        printf("\n");
    }
}


__global__ void kernel_print_matrix(double *A, const size_t n, const size_t m, int node)
{
    printf("%d\n", node);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (j != 0)
                printf(" ");
            // printf("A[%d]=%.3f", i * m + j, A[i * m + j]);
            printf("%.3f", A[i * m + j]);
        }
        printf("\n");
    }
}


__global__ void kernel_print_submatrix(double *A, const size_t n, const size_t m, const size_t pitch, int node)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("%d\n", node);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (j != 0)
                    printf(" ");
                // printf("A[%d]=%.3f", i * m + j, A[i * m + j]);
                printf("%.3f", A[i * pitch + j]);
            }
            printf("\n");
        }
    }
}

//TODO: create a separate class for kernels, it will have all the kernels and cublas related stuff 
//?: maybe it can be shared for LU and Cholesky, or 2 children of a parent class
// void GemmLU() 
// void TRSMLowerLU()
// void TRSMUpperLU()
// void GetrfLU()

void tiledLUPart(bool verify, bool dot)
{
    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    double *d_matrix, *d_buffer;
    if (myPE == 0)
        printMatrix(originalMatrix.get(), N, N);

    size_t sliceCount = T / nPE;
    size_t sliceMax = T / nPE;
    size_t sliceSize = B*B * T;
    size_t rem = T % nPE;
    if (rem > 0)
        sliceMax++;
    local_N = sliceMax*B;
    if (rem > myPE)
        sliceCount++;
    std::cout << T << " slices. PE " << myPE << " has " << sliceCount << " . Max=" << sliceMax << std::endl;

    auto getMatrixBlock = [&](double* matrix, int i, int j, int width=local_N)
    {
        return matrix + i * B + j * B * width;
    }; 

    // Does it only work with symmetric?
    // tile_size is B*B; a column is tile_size*T; and there are T / nPE of them, if evenly divided;
    d_matrix = (double *) nvshmem_malloc(sliceSize * sliceMax * sizeof(double));
    // TODO: optimally, buffer needs to be of sliceSize, and 
    // !: extra copies should be removed
    d_buffer = (double *) nvshmem_malloc(sliceSize * sliceMax * sizeof(double));

    for (int i = 0; i < sliceCount; i++)
        checkCudaErrors(cudaMemcpy2D(getMatrixBlock(d_matrix, i, 0), 
                                    sizeof(double) * local_N,
                                    getMatrixBlock(originalMatrix.get(), i*nPE + myPE, 0, N), 
                                    sizeof(double) * N, 
                                    sizeof(double) * B, 
                                    N, cudaMemcpyHostToDevice)); 

    // Copy to device
    for (int pe = 0; pe < nPE; pe++) {
        nvshmem_barrier_all();
        if (myPE == pe) {
            std::cout << " PE " << myPE << std::endl;
            kernel_print_matrix<<<1, 1>>>(d_matrix, N, local_N);
            cudaDeviceSynchronize();
        }
    }
    
    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));
    // checkCudaErrors(cublasLoggerConfigure(verbose, verbose, 0, NULL));

    // Prepare constants
    double one = 1.0;
    double minusOne = -1.0;

    // Prepare buffer for potrf
    int workspaceInBytesOnDevice;
    checkCudaErrors(cusolverDnDgetrf_bufferSize(
                    cusolverDnHandle,
                    B,
                    B,
                    d_matrix,
                    N,
                    &workspaceInBytesOnDevice));

    // void *h_workspace, *d_workspace_cusolver;
    double *d_workspace_cusolver;
    int workspaces = T*T; //! can be optimized to sliceCount at least
    int *d_info;
    void **d_workspace_cublas = (void **)malloc(sizeof(void *)*workspaces);
    workspaceInBytesOnDevice*=8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024*workspace; // (B/256+1)*B*256*4;

    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    }
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = T;
    for (int k = 0; k < T; k++)
        for (int i = k + 1; i < T; i++)
            totalNodes += 2 + (T-(k+1));

    if (verbose) {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
        std::cout << "workspaces=" << workspaces << std::endl;
    }

    // setup streams and events for graph construction
    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    cudaStream_t copyStreams[numCopyStreams];
    cudaEvent_t copyEvents[numCopyStreams+1];
    for (int streamID = 0; streamID < numCopyStreams; streamID++) {
        checkCudaErrors(cudaStreamCreate(&copyStreams[streamID]));
        checkCudaErrors(cudaEventCreate(&copyEvents[streamID]));
    }
    checkCudaErrors(cudaEventCreate(&copyEvents[numCopyStreams]));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    int *h_dependencies;
    int *d_dependencies = (int *) nvshmem_malloc(sizeof(int) * totalNodes);
    checkCudaErrors(cudaMallocHost(&h_dependencies, sizeof(int) * totalNodes));
    for (int i = 0; i < totalNodes; i++)
        h_dependencies[i] = 0;

    auto tiledLUGraphCreator = std::make_unique<mustard::TiledGraphCreator>(s, graph, false, totalNodes);

    int nodeIndex = 0;
    int neighborPE = (myPE+1)%nPE;
    std::cout << "PE " << myPE << ". My neighbor is " << neighborPE << std::endl;
    for (int k = 0; k < T; k++)
    {
        //* A[k][k] = GETRF(A[k][k])
        //* L[k][k]*U[k][k] = A[k][k]
        
        bool mySlice = (k % nPE == myPE);
        int myNodeIndex = 0;
        // int local_k = max(0, k - myPE)/nPE;
        int local_k = k / nPE;
        
        if (mySlice) {
            // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[myNodeIndex++], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, k),
                {std::make_pair(k, k)});
            checkCudaErrors(cusolverDnDgetrf(
                cusolverDnHandle,
                B,
                B,
                getMatrixBlock(d_matrix, local_k, k),
                local_N, // ?: N or local_N
                d_workspace_cusolver,
                NULL,
                d_info));
            // nvshmemx_quiet_on_stream(s);

            // *: current design - check the next comment block
            if (k+1 < T) // if there is a neighbor on the right
                mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);
            
            if (myPE == peToPrint && nodeIndex == nodeToPrint)
                kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_k, k), 
                                                            B, B, local_N, nodeIndex);

            tiledLUGraphCreator->endCaptureOperation();
        } else if (k+myPE < T && k+1 < T) { // if there are neighbor tiles  //! make sure it works
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, k),
                {std::make_pair(k, k)});
            // !: alternative design - add conditional while nodes OR use dependency waits on the remote
            // !: also, need to update local dependencies on the same GPU without communicating with others
            // *: current design - chain broadcast (works because immediate neighbor's tiles are more important)
            // *also distributes dependency update load and does not require busy-waiting on remote memory
            mustard::kernel_dep_wait<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, myPE);
            // nvshmemx_quiet_on_stream(s);
            h_dependencies[nodeIndex]++;
            if (k+myPE+1 < T && (k % nPE) != neighborPE) // if there's a neighbor and it's not the parent
                mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);
            tiledLUGraphCreator->endCaptureOperation();
        }
        nodeIndex++; 

        for (int i = k + 1; i < T; i++)
        {
            //* L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seemed like only these needed a separate workspace previously, //! not sure now
            //* no need for local_i version because everything here is local
            //* as all the tiles belong to the same column and therefore same PE

            if (mySlice) {
                // ?: there seemed to be some issue with this workspace thingy
                if (myPE == peToPrint)
                    printf("%d workspace \n\n", k*sliceMax+i);
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[k*sliceMax+i], 
                                                   cublasWorkspaceSize));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, i),
                    {std::make_pair(k, k), std::make_pair(k, i)});
                checkCudaErrors(cublasDtrsm(
                    cublasHandle,
                    CUBLAS_SIDE_LEFT, // used to be right for cholesky
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,// CUBLAS_OP_T for cholesky
                    CUBLAS_DIAG_UNIT, // CUBLAS_DIAG_NON_UNIT for cholesky
                    B, B,
                    &one,
                    getMatrixBlock(d_matrix, local_k, k), local_N,
                    getMatrixBlock(d_matrix, local_k, i), local_N)); 
                // nvshmemx_quiet_on_stream(s);

                if (myPE == peToPrint && nodeIndex == nodeToPrint)
                    kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_k, i), 
                                                                B, B, local_N, nodeIndex);

                if (k+1 < T) // if there is a neighbor on the right
                    mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);
                tiledLUGraphCreator->endCaptureOperation();
                
            } else if (k+myPE < T) { // if there are neighbor tiles
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, i),
                    {std::make_pair(k, k), std::make_pair(k, i)});
                mustard::kernel_dep_wait<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, myPE);
                // nvshmemx_quiet_on_stream(s);
                h_dependencies[nodeIndex]++;
                // if there's a neighbor and it's not the parent, update their dependencies too (chain)
                if (k+myPE+1 < T && (k % nPE) != neighborPE) 
                    mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, 
                                                                              neighborPE, myPE);
                tiledLUGraphCreator->endCaptureOperation();
            }
            nodeIndex++; 
        }

        for (int i = k + 1; i < T; i++)
        {
            //* U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            
            mySlice = (i % nPE == myPE);
            int local_i = i / nPE;
            
            if (mySlice) {
                if (myPE == peToPrint)
                    printf("%d workspace \n\n", i*sliceMax+k);
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i*sliceMax+k], 
                                cublasWorkspaceSize));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, k),
                    {std::make_pair(k, k), std::make_pair(i, k)});

                // *: required memcpy for TRSM_RIGHT: needs the GETRF output
                if (k % nPE != myPE) { // this should only be copied if the parent is remote
                    nvshmemx_quiet_on_stream(s);
                    checkCudaErrors(cudaEventRecord(copyEvents[numCopyStreams], s));
                        for (int row = 0; row < B; row++) {
                            int idx = row%numCopyStreams;
                            checkCudaErrors(cudaStreamWaitEvent(copyStreams[idx], copyEvents[numCopyStreams], 0));
                            nvshmemx_double_get_on_stream(getMatrixBlock(d_buffer, local_k, k)+local_N*row,
                                                        getMatrixBlock(d_matrix, local_k, k)+local_N*row, 
                                                        B, k % nPE, copyStreams[idx]);
                            checkCudaErrors(cudaEventRecord(copyEvents[idx], copyStreams[idx]));
                            checkCudaErrors(cudaStreamWaitEvent(s, copyEvents[idx], 0));
                        }
                    checkCudaErrors(cublasDtrsm(
                        cublasHandle,
                        CUBLAS_SIDE_RIGHT, 
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, 
                        CUBLAS_DIAG_NON_UNIT, 
                        B, B,
                        &one,
                       
                        getMatrixBlock(d_buffer, local_k, k), local_N,  // ! (d_buffer, 0, k) + B when optimized;
                        getMatrixBlock(d_matrix, local_i, k), local_N));   
                } else {
                    checkCudaErrors(cublasDtrsm(
                        cublasHandle,
                        CUBLAS_SIDE_RIGHT, 
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, 
                        CUBLAS_DIAG_NON_UNIT, 
                        B, B,
                        &one,
                        getMatrixBlock(d_matrix, local_k, k), local_N, // k + k * N; 
                        getMatrixBlock(d_matrix, local_i, k), local_N)); // (i + B) + k * N;   
                }
                if (myPE == peToPrint && nodeIndex == nodeToPrint) { 
                    kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_i, k), 
                                                                B, B, local_N, nodeIndex);
                }
                tiledLUGraphCreator->endCaptureOperation();
            } 
            nodeIndex++;

            for (int j = k + 1; j < T; j++)
            {
                //* A[j][i] = GEMM(A[j][k], A[i][k])
                //* A[j][i] = A[j][i] - L[j][k] * L[i][k]^T

                if (mySlice) {
                    if (myPE == peToPrint)
                        printf("%d workspace \n\n", i*sliceMax+j);
                    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i*sliceMax+j], 
                                    cublasWorkspaceSize));
                    tiledLUGraphCreator->beginCaptureOperation(
                        std::make_pair(i, j),
                        {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)}); 
                    // *: required memcpy for GEMM: needs the TRSM_LEFT output (TRSM_RIGHT is local)
                    if (k % nPE != myPE) { // this should only be copied if the parent is remote
                        nvshmemx_quiet_on_stream(s);
                        checkCudaErrors(cudaEventRecord(copyEvents[numCopyStreams], s));
                        for (int row = 0; row < B; row++) {
                            int idx = row%numCopyStreams;
                            checkCudaErrors(cudaStreamWaitEvent(copyStreams[idx], copyEvents[numCopyStreams], 0));
                            // ! (d_buffer, 0, k) + B when optimized
                            nvshmemx_double_get_on_stream(getMatrixBlock(d_buffer, local_k, j)+local_N*row, 
                                                        getMatrixBlock(d_matrix, local_k, j)+local_N*row, 
                                                        B, k % nPE, copyStreams[idx]);
                            checkCudaErrors(cudaEventRecord(copyEvents[idx], copyStreams[idx]));
                            checkCudaErrors(cudaStreamWaitEvent(s, copyEvents[idx], 0));
                        }
                        // if (myPE == peToPrint && nodeIndex == nodeToPrint) { 
                        //     kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_buffer, local_i, j), 
                        //                                                 B, B, local_N, nodeIndex);
                        // }
                        checkCudaErrors(cublasGemmEx(
                            cublasHandle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            B, B, B,
                            &minusOne,
                            getMatrixBlock(d_matrix, local_i, k), CUDA_R_64F, local_N,
                            getMatrixBlock(d_buffer, local_k, j), CUDA_R_64F, local_N, //! (0, j) when optimized, B
                            &one,
                            getMatrixBlock(d_matrix, local_i, j), CUDA_R_64F, local_N,
                            CUBLAS_COMPUTE_64F,
                            CUBLAS_GEMM_DEFAULT));
                        if (myPE == peToPrint && nodeIndex == nodeToPrint) { 
                            kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_buffer, local_k, j), 
                                                                        B, B, local_N, 1000+nodeIndex);
                        }
                        if (myPE == peToPrint && nodeIndex == nodeToPrint) { 
                            kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_i, k), 
                                                                        B, B, local_N, 1000+nodeIndex);
                        }
                    } else {
                        checkCudaErrors(cublasGemmEx(
                            cublasHandle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            B, B, B,
                            &minusOne,
                            getMatrixBlock(d_matrix, local_i, k), CUDA_R_64F, local_N,
                            getMatrixBlock(d_matrix, local_k, j), CUDA_R_64F, local_N, 
                            &one,
                            getMatrixBlock(d_matrix, local_i, j), CUDA_R_64F, local_N,
                            CUBLAS_COMPUTE_64F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                    if (myPE == peToPrint && nodeIndex == nodeToPrint) { 
                        kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_i, j), 
                                                                    B, B, local_N, nodeIndex);
                    }
                    tiledLUGraphCreator->endCaptureOperation();
                } 
                nodeIndex++;   
            }
        }
        printf("NEW ITER \n\n\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    
    if (verbose) std::cout << "Printing .dot graphs..." << std::endl;

    for (int pe = 0; pe < nPE; pe++) {
        nvshmem_barrier_all();
        if (myPE == pe) {
            std::cout << " PE " << myPE << std::endl;
            char filename1[20];
            sprintf(filename1, "./graph_%d_PE%d.dot", 0, myPE);
            if (dot)
                checkCudaErrors(cudaGraphDebugDotPrint(tiledLUGraphCreator->graph, filename1, 0));
        }
    }     
    
    for (int i = 0; i < totalNodes; i++)
    {
        // std::cout << i << ":" << h_dependencies[i] << std::endl;
        if (h_dependencies[i] > 0) 
            std::cout << i << ":" << h_dependencies[i] << std::endl;
    }

    checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies, 
                            sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
                            
    if (verbose) std::cout << "Instantiate graph..." << std::endl;
    cudaGraphExec_t graphExec;
    CudaEventClock clock;
    double totalTime = 0.0;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, tiledLUGraphCreator->graph, NULL));

    if (verbose) showMemUsage();
    if (verbose) std::cout << "Launching..." << std::endl;

    for (int i = 0; i < runs; i++) {
        // for (int pe = 0; pe < nPE; pe++) {
        //     nvshmem_barrier_all();
        //     if (myPE == pe) {
        //         std::cout << " PE " << myPE << std::endl;
        //         kernel_print_matrix<<<1, 1>>>(d_matrix, N, local_N);
        //         cudaDeviceSynchronize();
        //     }
        // }
    
        nvshmem_barrier_all();
        clock.start(s);
        checkCudaErrors(cudaGraphLaunch(graphExec, s));
        checkCudaErrors(cudaStreamSynchronize(s));
        clock.end(s);
        checkCudaErrors(cudaDeviceSynchronize());
        nvshmem_barrier_all();     

        checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies, 
                                sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        float time = clock.getTimeInSeconds();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;

        for (int pe = 0; pe < nPE; pe++) {
            nvshmem_barrier_all();
            if (myPE == pe) {
                std::cout << " PE " << myPE << std::endl;
                kernel_print_matrix<<<1, 1>>>(d_matrix, N, local_N);
                cudaDeviceSynchronize();
            }
        }
    }
    if (verbose) std::cout << "Done" << std::endl;

    nvshmem_barrier_all();
    if (verify) {
        // TODO: need to collect data from all the matrices on all PEs
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        // copy the local tiles
        for (int i = 0; i < sliceCount; i++) {
            checkCudaErrors(cudaMemcpy2D(getMatrixBlock(h_L, i*nPE + myPE, 0, N), 
                                        sizeof(double) * N, 
                                        getMatrixBlock(d_matrix, i, 0), 
                                        sizeof(double) * local_N, 
                                        sizeof(double) * B, 
                                        N, cudaMemcpyDeviceToHost)); 
        }
        std::cout << std::endl << "After local copy:" << std::endl;
        printSquareMatrix(h_L, N);
        // copy the remote tiles 
        for (int iPE = 1; iPE < nPE; iPE++) {

            // std::cout << "buffer of  PE " << myPE << std::endl;
            // kernel_print_matrix<<<1, 1>>>(d_buffer, N, local_N);
            // cudaDeviceSynchronize();

            nvshmem_double_get(d_buffer, d_matrix, local_N * N, iPE);
            int remoteSliceCount = T / nPE + (T % nPE > iPE);
            std::cout << " Collect " << remoteSliceCount << " slices from PE " << iPE << std::endl;
            for (int i = 0; i < remoteSliceCount; i++) {
                checkCudaErrors(cudaMemcpy2D(getMatrixBlock(h_L, i*nPE + iPE, 0, N), 
                                            sizeof(double) * N, 
                                            getMatrixBlock(d_buffer, i, 0), 
                                            sizeof(double) * local_N, 
                                            sizeof(double) * B, 
                                            N, cudaMemcpyDeviceToHost)); 
                // std::cout << std::endl << "After slice " << i*nPE + iPE << " :" << std::endl;
                // printSquareMatrix(h_L, N);
            }

            // std::cout << "buffer of  PE " << myPE << std::endl;
            // kernel_print_matrix<<<1, 1>>>(d_buffer, N, local_N);
            // cudaDeviceSynchronize();
        }
        // checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(originalMatrix.get(), h_L, h_U, N));

        free(h_L);
        free(h_U);
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    nvshmem_free(d_matrix);
    nvshmem_free(d_dependencies);
    checkCudaErrors(cudaFreeHost(h_dependencies));
    checkCudaErrors(cudaFree(d_info));
    // checkCudaErrors(cudaFree(h_workspace));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
}


int main(int argc, char **argv)
{
    auto cmdl = argh::parser(argc, argv);

    if (!(cmdl({"N", "n"}, N) >> N)) {
        std::cerr << "Must provide a valid N value! Got '" << cmdl({"N", "n"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"t", "T"}, T) >> T)) {
        std::cerr << "Must provide a valid T value! Got '" << cmdl({"T", "t"}).str() << "'" << std::endl;
        return 0;
    }
    if (N % T > 0) {
        std::cerr << "N must be divisible by T! Got 'N=" << N << " & T=" << T << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"sm", "SM", "smLimit"}, smLimit) >> smLimit) || smLimit > 108 || smLimit < 1) {
        std::cerr << "Must provide a valid SM Limit value! Got '" << cmdl({"sm", "SM", "smLimit"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"workspace", "ws", "w", "W"}, workspace) >> workspace) || workspace > 1024*1024 || workspace < 1) {
        std::cerr << "Must provide a valid workspace (in kBytes) value! Got '" << cmdl({"workspace", "ws", "w"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"run", "runs", "r", "R"}, runs) >> runs) || runs < 1) {
        std::cerr << "Must provide a valid number of runs! Got '" << cmdl({"run", "r", "R"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"p", "P"}, nodeToPrint) >> nodeToPrint) || nodeToPrint < -1) {
        std::cerr << "Unknown" << std::endl;
        return 0;
    }

    nvshmem_init();

    myPE = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    nPE = nvshmem_n_pes();
    checkCudaErrors(cudaSetDevice(myPE));

    int gpusAvailable = -1;
    checkCudaErrors(cudaGetDeviceCount(&gpusAvailable));
    verbose = cmdl[{"v", "verbose"}] && myPE == 0;

    if (verbose) {
        printf("Hello from NVSHMEM_PE=%d/%d\n", myPE, nPE);
        printf("%d GPUs detected, asked to use use %d GPUs\n", gpusAvailable, nPE);
    }
    if (gpusAvailable < nPE)
        nPE = gpusAvailable;

    B = N / T;
    
    if (myPE == 0) {
        std::cout << " with N=" << N << " (" << T << " of " << B << "x" << B << " tiles)" << std::endl;
        std::cout << "SM Limit per kernel = " << smLimit << std::endl;
        std::cout << "cuBLAS workspace = " << workspace << " kB" << std::endl;
    }

    if (nPE < 2) {
        std::cout << "ONLY WORKS FOR MULTI-GPU! RUN WITH MULTIPLE MPI PROCESSES" << std::endl;
        return 0;
    }

    tiledLUPart(cmdl["verify"] && myPE==0, cmdl["dot"]);
    // testLUPart(cmdl["verify"] && myPE==0, cmdl["dot"]);
    
    nvshmem_finalize();

    return 0;
}

// nvcc lu_partg_nvshmem.cu -I${NVSHMEM_PATH}/include -L${NVSHMEM_PATH}/lib -lcublas -lcusolver -lnvshmem -lnvidia-ml -o nvshmem_lu