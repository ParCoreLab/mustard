#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <nvshmem.h>
#include <nvshmemx.h>
// #include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "broker_queue.h"
// #include "../utilities/cudaUtilities.hpp"

#define FLAGS_SUBG_COUNT 0
#define FLAGS_OCCUP 4

size_t N = 15 * 1;
size_t B = N / 5;
size_t T = N / B;
int myPE;
int verbose = 0;
int workspace = 1;
int smLimit = 10;
int runs = 1;


template <typename T>
void __check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) __check((val), #val, __FILE__, __LINE__)

void showMemUsage()
{
    // show memory usage of GPU
    size_t free_byte;
    size_t total_byte;

    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status)
    {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

__global__ void warmUp()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + static_cast<float>(tid);
}

void warmUpCudaDevice()
{
    warmUp<<<32, 32>>>();
    cudaDeviceSynchronize();
}

void initializeCudaDevice(bool displayDeviceInfo)
{
    // checkCudaErrors(cudaSetDevice(0));

    if (displayDeviceInfo)
    {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
        printf("GPU Device %d: %s\n", 0, deviceProp.name);
        printf("Compute Capability: %d.%d\n\n", deviceProp.major, deviceProp.minor);
    }

    warmUpCudaDevice();
}

class CudaEventClock
{
public:
    CudaEventClock();
    ~CudaEventClock();
    void start(cudaStream_t stream = 0);
    void end(cudaStream_t stream = 0);
    float getTimeInSeconds();

private:
    cudaEvent_t startEvent, endEvent;
};

CudaEventClock::CudaEventClock()
{
    checkCudaErrors(cudaEventCreate(&this->startEvent));
    checkCudaErrors(cudaEventCreate(&this->endEvent));
}

CudaEventClock::~CudaEventClock()
{
    checkCudaErrors(cudaEventDestroy(this->startEvent));
    checkCudaErrors(cudaEventDestroy(this->endEvent));
}

void CudaEventClock::start(cudaStream_t stream)
{
    checkCudaErrors(cudaEventRecord(this->startEvent, stream));
}

void CudaEventClock::end(cudaStream_t stream)
{
    checkCudaErrors(cudaEventRecord(this->endEvent, stream));
}

float CudaEventClock::getTimeInSeconds()
{
    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, this->startEvent, this->endEvent));
    return time * 1e-3f;
}


// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n)
{
    // srand(time(NULL));
    srand(420);

    double *h_A_temp = (double *)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_A_temp[i * n + j] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_A[i * n + j] = 0.5 * (h_A_temp[i * n + j] + h_A_temp[j * n + i]);

    for (int i = 0; i < n; i++)
        h_A[i * n + i] = h_A[i * n + i] + n;
}

void printSquareMatrix(double *h_A, const size_t n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (j != 0)
                std::cout << " ";
            std::cout << std::setw(6) << std::setprecision(3) << h_A[i * n + j];
        }
        std::cout << std::endl;
    }
}

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

    // printf("A:\n");
    // printSquareMatrix(A, n);

    // printf("\nnewA:\n");
    // printSquareMatrix(newA.get(), n);

    // printf("\nL:\n");
    // printSquareMatrix(L, n);
    // printf("\n");

    // printf("\nU:\n");
    // printSquareMatrix(U, n);
    // printf("\n");


    printf("error = %.6f}\n", error);

    return error <= 1e-6;
}

void trivialLU(bool verify)
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

    checkCudaErrors(cusolverDnXgetrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        N,
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
    double totalTime = 0.0;

    for (int i = 0; i < runs; i++) {
        checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        clock.start();
        checkCudaErrors(cusolverDnXgetrf(
            cusolverDnHandle,
            cusolverDnParams,
            N,
            N,
            CUDA_R_64F,
            d_A,
            N,
            NULL, // no pivoting
            CUDA_R_64F,
            d_workspace,
            workspaceInBytesOnDevice,
            NULL,
            0,
            d_info));
        checkCudaErrors(cudaStreamSynchronize(0));
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemset(d_workspace, 0, workspaceInBytesOnDevice));
        float time = clock.getTimeInSeconds();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;
    }
    // Calculate

    // Check
    int h_info = 0;
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
    {
        std::cout << "Unsuccessful potrf execution\n\n"
                  << "d_info = " << h_info << "\n\n";
    }

    // Verify
    if (verify) {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(h_A, h_L, h_U, N));

        // Clean
        free(h_L);
        free(h_U);
    }
    
    printf("Total time used (s): %4.4f\n", totalTime);

    free(h_A);
    free(h_workspace);
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}
    
__global__ void kernel_dependency_update(BrokerWorkDistributor queue, int *dependencies, int nodeIndex)
{
    // int old = atomicAdd(dependencies + nodeIndex, -1);
    int old = nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, 0); // on PE 0
    if (old == 1) { // && dependencies[nodeIndex] == 0) {
        queue.enqueue(nodeIndex, 0);
    }
    // printf("Updating dependency of NODE %d, from %d to %d\n", nodeIndex, old, nvshmem_int_atomic_fetch(dependencies + nodeIndex, 0));
}
    
// __global__ void kernel_dep_occup_update(BrokerWorkDistributor queue, int *dependencies, int nodeIndex, int sm_count, volatile int *flags)
// {
//     // int old = atomicAdd(dependencies + nodeIndex, -1);
//     int old = nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, 0); // on PE 0
//     if (old == 1) { // && dependencies[nodeIndex] == 0) {
//         queue.enqueue(nodeIndex, 0);
//     }
//     atomicAdd((int *)&flags[FLAGS_OCCUP], -sm_count);
//     printf("Updating dependency of NODE %d, from %d to %d | occup = %d\n", nodeIndex, old, nvshmem_int_atomic_fetch(dependencies + nodeIndex, 0), flags[FLAGS_OCCUP]);
// }

// only 1 thread runs this
// thread_count is negative when occupancy is reduced after a kernel has been completed
__global__ void kernel_occupancy_update(int sm_count, volatile int *flags)
{
    // atomicAdd((int *)&flag, thread_count);
    atomicAdd((int *)&flags[FLAGS_OCCUP], sm_count);
    // int old = atomicAdd((int *)&all_flags[device_id][FLAGS_OCCUP], thread_count);
    // printf("Updating occupancy of GPU %d, from %d to %d\n", device_id, old, all_flags[device_id][FLAGS_OCCUP]);
    // printf("Updating occupancy to %d\n", flags[FLAGS_OCCUP]);
}


typedef std::pair<int, int> MatrixTile;

class TiledLUGraphCreator
{
public:
    cudaGraph_t *subgraphs;
    std::vector<std::vector<int>> subgraphDependencies;

    TiledLUGraphCreator(cudaStream_t stream, cudaGraph_t graph, bool subgraph = false, int totalNodes = 1) : stream(stream), graph(graph)
    {
        this->lastModifiedTile = std::make_pair(-1, -1);
        this->subgraph = subgraph;
        this->subgraphs = new cudaGraph_t[totalNodes];
        this->subgraphDependencies.resize(totalNodes);
        this->index_counter = 0;
    }

    void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead)
    {            
        auto tiles = std::vector<MatrixTile>(tilesToRead);
        tiles.push_back(tileToWrite);

        this->lastModifiedTile = tileToWrite;

        if (!this->subgraph) {
            auto dependencies = this->getDependencies(tiles);
            this->lastDependencies = dependencies;
            checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), 
                                                          nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
        } else {
            // auto dependencies = this->getSubgraphDependencies(tiles);
            this->subgraphDependencies[index_counter] = this->getSubgraphDependencies(tiles);
            checkCudaErrors(cudaGraphCreate(&this->subgraphs[index_counter], 0)); 
            // std::cout << "Start capturing subgraph " << index_counter << std::endl;
            checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->subgraphs[index_counter], nullptr, 
                                                          nullptr, 0, cudaStreamCaptureModeGlobal));
        }
    }

    void endCaptureOperation()
    {
        assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
        if (!this->subgraph) {
            checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
            this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
        } else {
            checkCudaErrors(cudaStreamEndCapture(this->stream, &(this->subgraphs[index_counter])));
            this->tileIndexByMap[this->lastModifiedTile] = this->index_counter;
            // std::cout << "End capturing subgraph " << index_counter << std::endl;
            this->index_counter++;
        }
        this->lastModifiedTile = std::make_pair(-1, -1);
    };
    
    void printDeps()
    {
        for (int i = 0; i < this->subgraphDependencies.size(); i++)
        {
            std::vector<int> deps = this->subgraphDependencies[i];
            std::cout << i << ":";
            for (int j = 0; j < deps.size(); j++)
            {
                std::cout << " " << deps[j];
            }
            std::cout << std::endl;
        }
    }

    // can merge only for the first src update
    void insertDependencyKernel(int src, int dst, BrokerWorkDistributor queue, int* d_dependencies) //, int sm_count=0, volatile int * flags=NULL)
    {
        cudaGraphNode_t dependencyUpdateNode;
        cudaKernelNodeParams params = {0};
        params.gridDim = dim3(1, 1, 1);
        params.blockDim = dim3(1, 1, 1);
        params.extra = NULL;
        // if (sm_count <= 0 || flags == NULL) {
        params.func = (void *)kernel_dependency_update;
        void *kernelArgs[3] = {&queue, &d_dependencies/*[0]*/, &dst}; 
        params.kernelParams = kernelArgs;
        // } else {
        //     params.func = (void *)kernel_dep_occup_update;
        //     void *kernelArgs[5] = {&queue, &d_dependencies/*[0]*/, &dst, &sm_count, &flags}; 
        //     params.kernelParams = kernelArgs;
        // }
        std::vector<cudaGraphNode_t> deps;
        deps.push_back(getTail(this->subgraphs[src]));
        checkCudaErrors(cudaGraphAddKernelNode(&dependencyUpdateNode, this->subgraphs[src], deps.data(),
                                                deps.size(), &params));
    }

private:
    std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
    std::map<MatrixTile, int> tileIndexByMap;
    std::map<cudaGraphNode_t, bool> visited;
    // std::map<int, bool> occupancy_update_created;
    cudaStream_t stream;
    cudaGraph_t graph;
    MatrixTile lastModifiedTile;
    std::vector<cudaGraphNode_t> lastDependencies;
    int index_counter;
    bool subgraph;

    // auto getDependencies(std::vector<MatrixTile> tiles, bool) {
    // }

    std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles)
    {
        std::vector<cudaGraphNode_t> dependencies;
        for (auto tile : tiles)
        {
            auto it = this->tileLastModifiedByMap.find(tile);
            if (it != this->tileLastModifiedByMap.end())
            {
                dependencies.push_back(it->second);
            }
        }

        auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

        return dependencies;
    }

    std::vector<int> getSubgraphDependencies(std::vector<MatrixTile> tiles)
    {
        std::vector<int> dependencies;
        for (auto tile : tiles)
        {
            auto it = this->tileIndexByMap.find(tile);
            if (it != this->tileIndexByMap.end())
            {
                dependencies.push_back(it->second);
            }
        }

        auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

        return dependencies;
    }

    cudaGraphNode_t getTail(cudaGraph_t graph){
        size_t numEdges;
        checkCudaErrors(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
        if (numEdges == 0) 
        {
            size_t numNodes = 1;
            auto nodes = std::make_unique<cudaGraphNode_t[]>(1);
            checkCudaErrors(cudaGraphGetNodes(graph, nodes.get(), &numNodes));
            return nodes[0];
        }
        auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
        auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
        checkCudaErrors(cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

        std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
        std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
        for (int i = 0; i < numEdges; i++)
        {
            hasOutGoingEdge[from[i]] = true;
            noOutGoingEdgeNodes.erase(from[i]);
            if (!hasOutGoingEdge[to[i]])
                noOutGoingEdgeNodes.insert(to[i]);
        }

        assert(noOutGoingEdgeNodes.size() == 1);

        return *noOutGoingEdgeNodes.begin();
    }

    cudaGraphNode_t getTailOfLastCapturedNodeChain()
    {
        if (lastDependencies.size() == 0)
        {
            return getTail(this->graph);
        }
        else
        {
            auto nodeBeforeChain = lastDependencies[0];
            size_t numDependentNodes;
            checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr, &numDependentNodes));

            assert(numDependentNodes > 0);

            auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
            checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

            cudaGraphNode_t chainBeginningNode;
            for (int i = 0; i < numDependentNodes; i++)
            {
                if (!visited[dependentNodes[i]])
                {
                    chainBeginningNode = dependentNodes[i];
                    break;
                }
            }

            auto u = chainBeginningNode;
            while (true)
            {
                visited[u] = true;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
                if (numDependentNodes == 0)
                    break;

                assert(numDependentNodes == 1);

                cudaGraphNode_t v;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(u, &v, &numDependentNodes));
                u = v;
            }

            return u;
        }
    }
};

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

__global__ void kernel_scheduler(
    BrokerWorkDistributor queue,
    volatile int *flags,
    cudaGraphExec_t *subgraphs,
    int totalSubgraphs,
    int device)
{
    unsigned int placeholder = UINT32_MAX;
    bool placeholder_bool = false;

    // printf("%d dev. Hello from scheduler starting with flags[FLAGS_SUBG_COUNT]=%d. Total subgraphs=%d\n", device, flags[0], totalSubgraphs);

    //while (flags[FLAGS_SUBG_COUNT] < totalSubgraphs)
    while (nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) < totalSubgraphs)
    {
        if (flags[FLAGS_OCCUP] < (107) && queue.size(0) > 0)
        {
            queue.dequeue(placeholder_bool, placeholder, 0);
            if (placeholder_bool) {
                // printf("%d dev. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0));
                cudaGraphLaunch(subgraphs[placeholder], cudaStreamGraphFireAndForget);
                nvshmem_int_atomic_inc((int *)&flags[FLAGS_SUBG_COUNT], 0);
            } 
            // else printf("ERR with: %d dev. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0));
            // atomicAdd((int *)&flags[FLAGS_SUBG_COUNT], 1);
            // flags[FLAGS_SUBG_COUNT]++;
            // printf("%d dev AFTER. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) );
        }
        // cur_iter++;
    }

    // printf("%d dev. Exit from scheduler with flags[0]=%d\n", device, flags[FLAGS_SUBG_COUNT]);

    // if (flags[0] < 2) {
    //     printf("Scheduler self-relaunch with flags[0]=%d\n", all_flags[0][0]);
    //     // Query the current graph handle so we can relaunch it
    //     cudaGraphExec_t currentGraph = cudaGetCurrentGraphExec();
    //     cudaGraphLaunch(currentGraph, cudaStreamGraphTailLaunch);
    // }
}

void tiledLU(bool verify, bool subgraph, bool dot)
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
    checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));

    // Prepare constants
    double one = 1.0;
    double minusOne = -1.0;

    // Prepare buffer for potrf
    // size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    int workspaceInBytesOnDevice;
        
    // checkCudaErrors(cusolverDnXgetrf_bufferSize(
    //     cusolverDnHandle,
    //     cusolverDnParams,
    //     B,
    //     B,
    //     CUDA_R_64F,
    //     d_matrix,
    //     N,
    //     CUDA_R_64F,
    //     &workspaceInBytesOnDevice,
    //     &workspaceInBytesOnHost));
        

    checkCudaErrors(cusolverDnDgetrf_bufferSize(
                    cusolverDnHandle,
                    B,
                    B,
                    d_matrix,
                    N,
                    &workspaceInBytesOnDevice));

    // void *h_workspace, *d_workspace_cusolver;
    double *d_workspace_cusolver;
    int workspaces = T;//(T-1)*(T-1);
    void **d_workspace_cublas = (void **)malloc(sizeof(void *)*workspaces);
    int *d_info;
    // checkCudaErrors(cudaMalloc(&h_workspace, workspaceInBytesOnHost));
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice*1024));
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
    }

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    auto tiledLUGraphCreator = std::make_unique<TiledLUGraphCreator>(s, graph, subgraph, totalNodes);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)});
        if (myPE != 0 && subgraph) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), 
                                        sizeof(double) * N,
                                        getMatrixBlock(d_matrix_remote, k, k), 
                                        sizeof(double) * N, 
                                        sizeof(double) * B, 
                                        B, cudaMemcpyDeviceToDevice, s);
        checkCudaErrors(cusolverDnDgetrf(
            cusolverDnHandle,
            B,
            B,
            getMatrixBlock(d_matrix, k, k),
            N,
            d_workspace_cusolver,
            NULL,
            d_info));
        // checkCudaErrors(cusolverDnXgetrf(
        //     cusolverDnHandle,
        //     cusolverDnParams,
        //     B,
        //     B,
        //     CUDA_R_64F,
        //     getMatrixBlock(d_matrix, k, k),
        //     N,
        //     NULL, // no pivoting
        //     CUDA_R_64F,
        //     d_workspace_cusolver,
        //     workspaceInBytesOnDevice,
        //     NULL,
        //     0,
        //     d_info));
        if (myPE != 0 && subgraph) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), 
                                        sizeof(double) * N,
                                        getMatrixBlock(d_matrix, k, k), 
                                        sizeof(double) * N, 
                                        sizeof(double) * B, 
                                        B, cudaMemcpyDeviceToDevice, s);
        tiledLUGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seems like only these need a separate workspace
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, i),
                {std::make_pair(k, k), std::make_pair(k, i)});
            if (subgraph) {
                kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0 && k != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, i), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, k, i), 
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
                CUBLAS_SIDE_LEFT, // used to be right for cholesky
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,// CUBLAS_OP_T for cholesky
                CUBLAS_DIAG_UNIT, // CUBLAS_DIAG_NON_UNIT for cholesky
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N, // k + k * N;
                getMatrixBlock(d_matrix, k, i), N)); // k + (i + B) * N;
            if (subgraph) {
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, i), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix, k, i), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledLUGraphCreator->endCaptureOperation();

        }
        checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i-1 + T], workspaceInBytesOnDevice));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)});
            
            if (subgraph) {
                kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
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
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, 
                CUBLAS_DIAG_NON_UNIT, 
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N, // k + k * N;
                getMatrixBlock(d_matrix, i, k), N)); // (i + B) + k * N;
            if (subgraph) {
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix, i, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledLUGraphCreator->endCaptureOperation();

            for (int j = k + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[(i-1)*T + (j-1)], workspaceInBytesOnDevice));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)});
                if (subgraph) {
                    kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                    if (myPE != 0) {
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, j), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, k, j), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, j), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, j), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                    }
                }
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N, // CUBLAS_OP_T
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N, // i + k * N
                    getMatrixBlock(d_matrix, k, j), CUDA_R_64F, N, // j + i * N
                    &one,
                    getMatrixBlock(d_matrix, i, j), CUDA_R_64F, N, // k + i * N
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                if (subgraph) {
                    if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, j), 
                                                    sizeof(double) * N,
                                                    getMatrixBlock(d_matrix, i, j), 
                                                    sizeof(double) * N, 
                                                    sizeof(double) * B, 
                                                    B, cudaMemcpyDeviceToDevice, s);
                    kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                }
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
        
    cudaGraphExec_t graphExec;
    CudaEventClock clock;
    double totalTime = 0.0;
    
    if (subgraph) {
        if (dot)
            tiledLUGraphCreator->printDeps();
        
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
            h_dependencies[i] = tiledLUGraphCreator->subgraphDependencies[i].size();
            //std::cout << h_dependencies[i] << " ";
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
                tiledLUGraphCreator->insertDependencyKernel(tiledLUGraphCreator->subgraphDependencies[dst][src_ind], 
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
                checkCudaErrors(cudaGraphDebugDotPrint(tiledLUGraphCreator->subgraphs[i], filename, 0));
            checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i], tiledLUGraphCreator->subgraphs[i], cudaGraphInstantiateFlagDeviceLaunch));
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
        kernel_scheduler<<<1, 1, 0, s>>>(queue, d_flags, d_subgraphsExec, totalNodes, myPE);
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
        double *h_U = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(originalMatrix.get(), h_L, h_U, N));

        free(h_L);
        free(h_U);
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

void LU(bool tiled, bool verify, bool subgraph, bool dot)
{
    if (tiled && myPE == 0)
        tiledLU(verify, subgraph, dot);
    else if (subgraph)
        tiledLU(verify, subgraph, dot);
    else if (myPE == 0)
        trivialLU(verify);
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

    LU(cmdl["tiled"], cmdl["verify"] && myPE==0, cmdl["subgraph"], cmdl["dot"]);
    
    nvshmem_finalize();

    return 0;
}

// nvcc lu_partg_nvshmem.cu -I${NVSHMEM_PATH}/include -L${NVSHMEM_PATH}/lib -lcublas -lcusolver -lnvshmem -lnvidia-ml -o nvshmem_lu