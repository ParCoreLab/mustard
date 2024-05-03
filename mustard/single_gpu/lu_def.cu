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

size_t N = 15 * 1;
size_t B = N / 5;
size_t T = N / B;


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

    // Calculate
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
    
    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());

    free(h_A);
    free(h_workspace);
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}

typedef std::pair<int, int> MatrixTile;

class TiledLUGraphCreator
{
public:
    TiledLUGraphCreator(cudaStream_t stream, cudaGraph_t graph) : stream(stream), graph(graph)
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

void tiledLU(bool verify)
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
    // double *one, *minusOne;
    // checkCudaErrors(cudaMallocManaged(&one, sizeof(double)));
    // checkCudaErrors(cudaMallocManaged(&minusOne, sizeof(double)));
    double one = 1.0;
    double minusOne = -1.0;

    // Prepare buffer for potrf
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
        
    checkCudaErrors(cusolverDnXgetrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        B,
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
    // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace, workspaceInBytesOnDevice));

    auto tiledLUGraphCreator = std::make_unique<TiledLUGraphCreator>(s, graph);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)});
        checkCudaErrors(cusolverDnXgetrf(
            cusolverDnHandle,
            cusolverDnParams,
            B,
            B,
            CUDA_R_64F,
            getMatrixBlock(k, k),
            N,
            NULL, // no pivoting
            CUDA_R_64F,
            d_workspace,
            workspaceInBytesOnDevice,
            h_workspace,
            workspaceInBytesOnHost,
            d_info));
        tiledLUGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
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
                getMatrixBlock(k, k), N, // k + k * N;
                getMatrixBlock(k, i), N)); // k + (i + B) * N;
            tiledLUGraphCreator->endCaptureOperation();

        }

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)});
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT, 
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, 
                CUBLAS_DIAG_NON_UNIT, 
                B, B,
                &one,
                getMatrixBlock(k, k), N, // k + k * N;
                getMatrixBlock(i, k), N)); // (i + B) + k * N;
            tiledLUGraphCreator->endCaptureOperation();

            for (int j = k + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)});
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N, // CUBLAS_OP_T
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(i, k), CUDA_R_64F, N, // i + k * N
                    getMatrixBlock(k, j), CUDA_R_64F, N, // j + i * N
                    &one,
                    getMatrixBlock(i, j), CUDA_R_64F, N, // k + i * N
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    CudaEventClock clock;
    
    checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    clock.start(s);
    checkCudaErrors(cudaGraphLaunch(graphExec, s));
    checkCudaErrors(cudaStreamSynchronize(s));
    clock.end(s);
    checkCudaErrors(cudaDeviceSynchronize());

    if (verify) {
        double *h_U = (double *)malloc(N * N * sizeof(double));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(d_matrix, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(originalMatrix.get(), d_matrix, h_U, N));

        free(h_U);
    }
    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());

    free(h_workspace);
    cudaFree(d_matrix);
    cudaFree(d_workspace);
}

void LU(bool tiled, bool verify)
{
    if (tiled)
    {
        tiledLU(verify);
    }
    else
    {
        trivialLU(verify);
    }
}

int main(int argc, char **argv)
{
    argh::parser cmdl({"n", "N", "t", "T"});
    cmdl.parse(argc, argv);

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
    B = N / T;
    
    if (cmdl["tiled"])
        std::cout << "TILED ";
    else
        std::cout << "Single-kernel ";
    std::cout << "with 'N=" << N << " & T=" << T << " & B=" << B << "'" << std::endl;

    LU(cmdl["tiled"], cmdl["verify"]);

    return 0;
}