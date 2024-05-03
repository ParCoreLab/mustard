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

size_t N = 15 * 1;
size_t B = N / 5;
size_t T = N / B;
int myPE;
int verbose = 0;
int workspace = 1;
int smLimit = 10;
int runs = 1;

// Set upper triangle entries (excluding diagonal entries) in column-major order to zero.
// Then, transpose to row-major order.
void cleanCusolverCholeskyDecompositionResult(double *L, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            L[i + j * n] = 0;
            std::swap(L[i + j * n], L[i * n + j]);
        }
    }
}

bool verifyCholeskyDecomposition(double *A, double *L, const int n)
{
    auto newA = std::make_unique<double[]>(n * n);
    memset(newA.get(), 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                newA[i * n + j] += L[i * n + k] * L[k + j * n];
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
    }

    printf("error = %.6f}\n", error);

    return error <= 1e-6;
}

void trivialCholesky(bool verify)
{
    // Initialize libaries
    cusolverDnHandle_t cusolverDnHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));

    cusolverDnParams_t cusolverDnParams;
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    // Initialize data
    double *h_A = (double *)malloc(N * N * sizeof(double));
    generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

    double *d_A;
    checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;

    checkCudaErrors(cusolverDnXpotrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        CUBLAS_FILL_MODE_LOWER,
        N,
        CUDA_R_64F,
        d_A,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    void *h_workspace = malloc(workspaceInBytesOnHost);

    void *d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

    int *d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    CudaEventClock clock;

    clock.start();
    double totalTime = 0.0;

    // Calculate
    for (int i = 0; i < runs; i++) {
        checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        clock.start();
        checkCudaErrors(cusolverDnXpotrf(
            cusolverDnHandle,
            cusolverDnParams,
            CUBLAS_FILL_MODE_LOWER,
            N,
            CUDA_R_64F,
            d_A,
            N,
            CUDA_R_64F,
            d_workspace,
            workspaceInBytesOnDevice,
            h_workspace,
            workspaceInBytesOnHost,
            d_info));
        checkCudaErrors(cudaStreamSynchronize(0));
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemset(d_workspace, 0, workspaceInBytesOnDevice));
        float time = clock.getTimeInSeconds();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;
    }

    clock.end();

    // Check
    int h_info = 0;
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
    {
        std::cout << "Unsuccessful potrf execution\n\n"
                  << "d_info = " << h_info << "\n\n";
    }

    // Verify
    if (verify)
    {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n", verifyCholeskyDecomposition(h_A, h_L, N));
        free(h_L);
    }

    printf("Total time used (s): %4.4f\n", totalTime);
    // Clean
    free(h_A);
    free(h_workspace);
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}

__global__ void kernel_populate_queue(BrokerWorkDistributor queue, int *dependencies, int totalNodes)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < totalNodes; i = i + gridDim.x * blockDim.x)
    {
        if (dependencies[i] == 0) {
            // printf("Inserting %d", i);
            queue.enqueue(i, 0);
        }
    }
}

__global__ void kernel_test_dequeue(BrokerWorkDistributor queue)
{
    unsigned int placeholder = UINT32_MAX;
    bool placeholder_bool = false;
    while (queue.size(0) > 0)
    {
        queue.dequeue(placeholder_bool, placeholder, 0);
        printf("Dequeued %d\n", placeholder);
    }
}

void tiledCholesky(bool verify, bool subgraph, bool dot)
{
    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy to device
    double *d_matrix;
    double *d_matrices;
    double *d_matrix_remote;
    volatile int *d_flags;
    if (subgraph) {
        d_flags = (volatile int *) nvshmem_malloc(sizeof(int) * 32);
        d_matrices = (double *) nvshmem_malloc(N * N * sizeof(double));
        d_matrix = (double *) nvshmem_ptr(d_matrices, myPE);
    } else {
        checkCudaErrors(cudaMalloc(&d_matrix, N * N * sizeof(double)));
    }
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0) 
        d_matrix_remote = (double *) nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double* matrix, int i, int j)
    {
        return matrix + i * B + j * B * N;
    };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    // checkCudaErrors(cublasLoggerConfigure(verbose, verbose, 0, NULL));
    // checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));

    // Prepare constants
    double one = 1.0;
    double minusOne = -1.0;

    // Prepare buffer for potrf
    // size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    int workspaceInBytesOnDevice;
        
    // checkCudaErrors(cusolverDnXpotrf_bufferSize(
    //     cusolverDnHandle,
    //     cusolverDnParams,
    //     CUBLAS_FILL_MODE_LOWER,
    //     B,
    //     CUDA_R_64F,
    //     d_matrix,
    //     N,
    //     CUDA_R_64F,
    //     &workspaceInBytesOnDevice,
    //     &workspaceInBytesOnHost));
        

    checkCudaErrors(cusolverDnDpotrf_bufferSize(
                    cusolverDnHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    B,
                    d_matrix,
                    N,
                    &workspaceInBytesOnDevice));

    // void *h_workspace, *d_workspace_cusolver;
    double *d_workspace_cusolver;
    int workspaces = T*T;//(T-1)*(T-1);
    void **d_workspace_cublas = (void **)malloc(sizeof(void *)*workspaces);
    int *d_info;
    workspaceInBytesOnDevice*=8;
    // checkCudaErrors(cudaMalloc(&h_workspace, workspaceInBytesOnHost));
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024*workspace; // (B/256+1)*B*256*4;

    // while (cublasWorkspaceSize * T >= 1024*1024*2 - workspaceInBytesOnDevice) {
    //     cublasWorkspaceSize = cublasWorkspaceSize >> 1;
    // }

    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    }
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = T;
    
    for (int k = 0; k < T; k++) 
        for (int i = k + 1; i < T; i++) 
            totalNodes += 2 + (T-(i+1));

    if (verbose) {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
    }

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    auto tiledCholeskyGraphCreator = std::make_unique<mustard::TiledGraphCreator>(s, graph, subgraph, totalNodes);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledCholeskyGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)});
        if (subgraph) {
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, k, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
        }
        checkCudaErrors(cusolverDnDpotrf(
            cusolverDnHandle,
            CUBLAS_FILL_MODE_LOWER,
            B,
            getMatrixBlock(d_matrix, k, k),
            N,
            d_workspace_cusolver,
            workspaceInBytesOnDevice,
            d_info));
        // checkCudaErrors(cusolverDnXpotrf(
        //     cusolverDnHandle,
        //     cusolverDnParams,
        //     CUBLAS_FILL_MODE_LOWER,
        //     B,
        //     CUDA_R_64F,
        //     getMatrixBlock(d_matrix, k, k),
        //     N,
        //     CUDA_R_64F,
        //     d_workspace,
        //     workspaceInBytesOnDevice,
        //     NULL,
        //     0,
        //     d_info));
        if (subgraph) {
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix, k, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        }
        tiledCholeskyGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seems like only these need a separate workspace
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)});
            if (subgraph) {
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0 && k != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, i, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, k, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
            }
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N, // k + k * N;
                getMatrixBlock(d_matrix, i, k), N)); // k + (i + B) * N;
            if (subgraph) {
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix, i, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledCholeskyGraphCreator->endCaptureOperation();

        }
        // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i + T], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, i),
                {std::make_pair(i, i), std::make_pair(i, k)});
            
            if (subgraph) {
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, i, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, i), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, i, i), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
            }
            checkCudaErrors(cublasDsyrk(
                cublasHandle,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                B, B,
                &minusOne, getMatrixBlock(d_matrix, i, k), N,
                &one, getMatrixBlock(d_matrix, i, i), N));
            if (subgraph) {
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, i), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix, i, i), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledCholeskyGraphCreator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[2*T+ (i-1)*T + (j-1)], cublasWorkspaceSize));
                tiledCholeskyGraphCreator->beginCaptureOperation(
                    std::make_pair(j, i),
                    {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)});
                if (subgraph) {
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                    if (myPE != 0) {
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, j, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, i), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, j, i), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                    }
                }
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(d_matrix, j, k), CUDA_R_64F, N,
                    getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N, 
                    &one,
                    getMatrixBlock(d_matrix, j, i), CUDA_R_64F, N, 
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                if (subgraph) {
                    if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, j, i), 
                                                    sizeof(double) * N,
                                                    getMatrixBlock(d_matrix, j, i), 
                                                    sizeof(double) * N, 
                                                    sizeof(double) * B, 
                                                    B, cudaMemcpyDeviceToDevice, s);
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                }
                tiledCholeskyGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
        
    cudaGraphExec_t graphExec;
    CudaEventClock clock;
    double totalTime = 0.0;
    
    if (subgraph) {
        if (verbose)
            tiledCholeskyGraphCreator->printDeps();
        
        // volatile int *d_flags;
        int *h_dependencies; //, *d_dependencies;
        const int queue_size = totalNodes * 2;
        if (verbose) std::cout << "Creating queue..." << std::endl;
        BrokerWorkDistributor queue(queue_size);
        if (verbose) std::cout << "Allocating memory..." << std::endl;

        // checkCudaErrors(cudaMalloc(&d_flags, sizeof(int) * 32));
        // checkCudaErrors(cudaMalloc(&d_dependencies, sizeof(int) * totalNodes));

        int *d_dependencies = (int *) nvshmem_malloc(sizeof(int) * totalNodes);
        checkCudaErrors(cudaMallocHost(&h_dependencies, sizeof(int) * totalNodes));
        if (verbose) std::cout << "Setting dependencies..." << std::endl;

        
        for (int i = 0; i < totalNodes; i++)
        {
            h_dependencies[i] = tiledCholeskyGraphCreator->subgraphDependencies[i].size();
            // std::cout << h_dependencies[i] << " ";
        }
        if (verbose) std::cout << "Populating the queue..." << std::endl;
        // std::cout << std::endl;

        checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies, 
                                sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
        // if (myPE == 0)
        //     kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
        // if (myPE == 0)
        //     kernel_test_dequeue<<<1, 1>>>(queue);
        if (myPE == 0)
            kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
        checkCudaErrors(cudaDeviceSynchronize());
        if (verbose) std::cout << "Inserting dependency kernels..." << std::endl;

        for (int dst = 0; dst < totalNodes; dst++)
            for (int src_ind = 0; src_ind < h_dependencies[dst]; src_ind++) 
                tiledCholeskyGraphCreator->insertDependencyKernel(tiledCholeskyGraphCreator->subgraphDependencies[dst][src_ind], 
                                                            dst, queue, d_dependencies);//, smLimit, d_flags);
        // checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Uploading graphs..." << std::endl;

        cudaGraphExec_t *h_subgraphsExec = new cudaGraphExec_t[totalNodes];
        cudaGraphExec_t *d_subgraphsExec;
        for (int i = 0; i < totalNodes; i++)
        {
            char filename[20];
            sprintf(filename, "./graph_%d.dot", i);
            if (dot)
                checkCudaErrors(cudaGraphDebugDotPrint(tiledCholeskyGraphCreator->subgraphs[i], filename, 0));
            checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i], tiledCholeskyGraphCreator->subgraphs[i], cudaGraphInstantiateFlagDeviceLaunch));
            cudaGraphUpload(h_subgraphsExec[i], s);
        }
        checkCudaErrors(cudaMalloc(&d_subgraphsExec, sizeof(cudaGraphExec_t) * totalNodes));
        checkCudaErrors(cudaMemcpy((void *)d_subgraphsExec, (void *)h_subgraphsExec,
                                    sizeof(cudaGraphExec_t) * totalNodes, cudaMemcpyHostToDevice));

        if (verbose) std::cout << "Initializing scheduler..." << std::endl;
        cudaGraph_t schedulerGraph;
        cudaGraphExec_t schedulerExec;
        checkCudaErrors(cudaGraphCreate(&schedulerGraph, 0));
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        mustard::kernel_scheduler<<<1, 1, 0, s>>>(queue, d_flags, d_subgraphsExec, totalNodes, myPE);
        cudaStreamEndCapture(s, &schedulerGraph);
        checkCudaErrors(cudaGraphInstantiate(&schedulerExec, schedulerGraph, cudaGraphInstantiateFlagDeviceLaunch));
        checkCudaErrors(cudaDeviceSynchronize());
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Launching..." << std::endl;

        for (int i = 0; i < runs; i++) {
            checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
            nvshmem_barrier_all();
            clock.start(s);
            checkCudaErrors(cudaGraphLaunch(schedulerExec, s));
            checkCudaErrors(cudaStreamSynchronize(s));
            clock.end(s);
            checkCudaErrors(cudaDeviceSynchronize());
            nvshmem_barrier_all();
            if (myPE == 0) {
                checkCudaErrors(cudaMemset((void *)d_flags, 0, sizeof(int) * 32));
                checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies, 
                                        sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
                kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
                checkCudaErrors(cudaDeviceSynchronize());
            }
            float time = clock.getTimeInSeconds();
            printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
            totalTime += time;
        }
        if (verbose) std::cout << "Done" << std::endl;
        
        free(h_subgraphsExec);
        checkCudaErrors(cudaFreeHost(h_dependencies));
        checkCudaErrors(cudaFree(d_subgraphsExec));
        nvshmem_free(d_dependencies);
        nvshmem_free((void*)d_flags);
        queue.free_mem();
    } else {
        if (dot)
            checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        for (int i = 0; i < runs; i++) {
            checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
            clock.start(s);
            checkCudaErrors(cudaGraphLaunch(graphExec, s));
            checkCudaErrors(cudaStreamSynchronize(s));
            clock.end(s);
            checkCudaErrors(cudaDeviceSynchronize());
            float time = clock.getTimeInSeconds();
            printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
            totalTime += time;
        }
    }

    if (verify) {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n", verifyCholeskyDecomposition(originalMatrix.get(), h_L, N));

        free(h_L);
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    if (!subgraph) checkCudaErrors(cudaFree(d_matrix));
    else nvshmem_free(d_matrices);
    checkCudaErrors(cudaFree(d_info));
    // checkCudaErrors(cudaFree(h_workspace));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
    // (*queue).free_mem();
    // delete queue;
}

void Cholesky(bool tiled, bool verify, bool subgraph, bool dot)
{
    if (tiled && myPE == 0)
        tiledCholesky(verify, subgraph, dot);
    else if (subgraph)
        tiledCholesky(verify, subgraph, dot);
    else if (myPE == 0)
        trivialCholesky(verify);
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

    nvshmem_init();

    myPE = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    checkCudaErrors(cudaSetDevice(myPE));

    int gpusAvailable = -1;
    checkCudaErrors(cudaGetDeviceCount(&gpusAvailable));
    verbose = cmdl[{"v", "verbose"}] && myPE == 0;

    if (verbose) {
        printf("Hello from NVSHMEM_PE=%d/%d\n", myPE, nvshmem_n_pes());
        printf("%d GPUs detected, asked to use use %d GPUs\n", gpusAvailable, nvshmem_n_pes());
    }

    if (!(cmdl["tiled"] || cmdl["subgraph"]))
        T = 1;
    B = N / T;
    
    if (myPE == 0) {
        if (cmdl["tiled"])
            std::cout << "TILED";
        else if (cmdl["subgraph"])
            std::cout << "SUBGRAPH";
        else {
            std::cout << "Single-kernel";
        }
        std::cout << " with N=" << N << " (" << T << " of " << B << "x" << B << " tiles)" << std::endl;

        if (cmdl[{"subgraph", "tiled"}]) {
            std::cout << "SM Limit per kernel = " << smLimit << std::endl;
            std::cout << "cuBLAS workspace = " << workspace << " kB" << std::endl;
        }
    }

    Cholesky(cmdl["tiled"], cmdl["verify"] && myPE==0, cmdl["subgraph"], cmdl["dot"]);
    
    nvshmem_finalize();

    return 0;
}

// nvcc lu_partg_nvshmem.cu -I${NVSHMEM_PATH}/include -L${NVSHMEM_PATH}/lib -lcublas -lcusolver -lnvshmem -lnvidia-ml -o nvshmem_lu