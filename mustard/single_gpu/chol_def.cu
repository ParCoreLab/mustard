#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
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
// #include "../utilities/cudaUtilities.hpp"

constexpr size_t N = 8;
constexpr size_t B = N / 2;
constexpr size_t T = N / B;

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
    checkCudaErrors(cudaSetDevice(0));

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
    srand(time(NULL));

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

    printf("A:\n");
    printSquareMatrix(A, n);

    printf("\nnewA:\n");
    printSquareMatrix(newA.get(), n);

    printf("\nL:\n");
    printSquareMatrix(L, n);
    printf("\n");

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

    // Calculate
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

    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());
    // Clean
    free(h_A);
    free(h_workspace);
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}

typedef std::pair<int, int> MatrixTile;

class TiledCholeskyGraphCreator
{
public:
    TiledCholeskyGraphCreator(cudaStream_t stream, cudaGraph_t graph) : stream(stream), graph(graph)
    {
        this->lastModifiedTile = std::make_pair(-1, -1);
    }
    void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead)
    {
        auto tiles = std::vector<MatrixTile>(tilesToRead);
        tiles.push_back(tileToWrite);
        auto dependencies = this->getDependencies(tiles);

        this->lastModifiedTile = tileToWrite;
        this->lastDependencies = dependencies;

        checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
    }

    void endCaptureOperation()
    {
        assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
        checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
        this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
        this->lastModifiedTile = std::make_pair(-1, -1);
    };

private:
    std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
    std::map<cudaGraphNode_t, bool> visited;
    cudaStream_t stream;
    cudaGraph_t graph;
    MatrixTile lastModifiedTile;
    std::vector<cudaGraphNode_t> lastDependencies;

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

    cudaGraphNode_t getTailOfLastCapturedNodeChain()
    {
        if (lastDependencies.size() == 0)
        {
            size_t numEdges;
            checkCudaErrors(cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges));
            auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
            auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
            checkCudaErrors(cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges));

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

void tiledCholesky(bool verify)
{
    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy to device
    double *d_matrix;
    checkCudaErrors(cudaMallocManaged(&d_matrix, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    auto getMatrixBlock = [&](int i, int j)
    {
        return d_matrix + i * B + j * B * N;
    };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));

    // Prepare constants
    double *one, *minusOne;
    checkCudaErrors(cudaMallocManaged(&one, sizeof(double)));
    checkCudaErrors(cudaMallocManaged(&minusOne, sizeof(double)));
    *one = 1.0;
    *minusOne = -1.0;

    // Prepare buffer for potrf
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    checkCudaErrors(cusolverDnXpotrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        CUBLAS_FILL_MODE_LOWER,
        B,
        CUDA_R_64F,
        d_matrix,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));
    void *h_workspace, *d_workspace;
    int *d_info;
    checkCudaErrors(cudaMallocManaged(&h_workspace, workspaceInBytesOnHost));
    checkCudaErrors(cudaMallocManaged(&d_workspace, workspaceInBytesOnDevice));
    checkCudaErrors(cudaMallocManaged(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));

    auto tiledCholeskyGraphCreator = std::make_unique<TiledCholeskyGraphCreator>(s, graph);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = POTRF(A[k][k])
        // L[k][k] = POTRF(A[k][k])
        tiledCholeskyGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)});
        checkCudaErrors(cusolverDnXpotrf(
            cusolverDnHandle,
            cusolverDnParams,
            CUBLAS_FILL_MODE_LOWER,
            B,
            CUDA_R_64F,
            getMatrixBlock(k, k),
            N,
            CUDA_R_64F,
            d_workspace,
            workspaceInBytesOnDevice,
            h_workspace,
            workspaceInBytesOnHost,
            d_info));
        tiledCholeskyGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // A[i][k] = TRSM(A[k][k], A[i][k])
            // L[i][k] * L[k][k]^T = A[i][k]
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)});
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                B, B,
                one,
                getMatrixBlock(k, k), N,
                getMatrixBlock(i, k), N));
            tiledCholeskyGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            // A[i][i] = SYRK(A[i][k], A[i][i])
            // A[i][i] = A[i][i] - L[i][k] * L[i][k]^T
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, i),
                {std::make_pair(i, i), std::make_pair(i, k)});
            checkCudaErrors(cublasDsyrk(
                cublasHandle,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                B, B,
                minusOne, getMatrixBlock(i, k), N,
                one, getMatrixBlock(i, i), N));
            tiledCholeskyGraphCreator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                tiledCholeskyGraphCreator->beginCaptureOperation(
                    std::make_pair(j, i),
                    {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)});
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    B, B, B,
                    minusOne,
                    getMatrixBlock(j, k), CUDA_R_64F, N,
                    getMatrixBlock(i, k), CUDA_R_64F, N,
                    one,
                    getMatrixBlock(j, i), CUDA_R_64F, N,
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                tiledCholeskyGraphCreator->endCaptureOperation();
            }
        }
    }

    CudaEventClock clock;

    checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.start();
    checkCudaErrors(cudaGraphLaunch(graphExec, s));
    checkCudaErrors(cudaStreamSynchronize(s));
    clock.end();

    checkCudaErrors(cudaDeviceSynchronize());

    cleanCusolverCholeskyDecompositionResult(d_matrix, N);
    if (verify)
        printf("Result passes verification: %d\n", verifyCholeskyDecomposition(originalMatrix.get(), d_matrix, N));

    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());

    free(h_workspace);
    cudaFree(d_matrix);
    cudaFree(d_workspace);
}

void cholesky(bool tiled, bool verify)
{
    if (tiled)
    {
        tiledCholesky(verify);
    }
    else
    {
        trivialCholesky(verify);
    }
}

int main(int argc, char **argv)
{
    auto cmdl = argh::parser(argc, argv);

    cholesky(cmdl["tiled"], cmdl["verify"]);

    return 0;
}
