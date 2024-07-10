
void testLUPart(bool verify, bool dot)
{
    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    double *d_matrix;

    auto getMatrixBlock = [&](double* matrix, int i, int j)
    {
        return matrix + i * B + j * B * N;
    };

    // Does it only work with symmetric?
    // tile_size is B*B; a column is tile_size*T; and there are T / nPE of them, if evenly divided;
    checkCudaErrors(cudaMalloc(&d_matrix, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));

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
    int workspaces = 1;
    int *d_info;
    void **d_workspace_cublas = (void **)malloc(sizeof(void *)*workspaces);
    workspaceInBytesOnDevice*=8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024*workspace; // (B/256+1)*B*256*4;

    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    }
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    int nodeIndex = 0;
    cudaStreamSynchronize(s);
    kernel_print_matrix<<<1, 1, 0, s>>>(d_matrix, N, N, nodeIndex);
    cudaStreamSynchronize(s);
    for (int k = 0; k < T; k++)
    {
        //* A[k][k] = GETRF(A[k][k])
        //* L[k][k]*U[k][k] = A[k][k]
        checkCudaErrors(cusolverDnDgetrf(
            cusolverDnHandle,
            B,
            B,
            getMatrixBlock(d_matrix, k, k),
            N, 
            d_workspace_cusolver,
            NULL,
            d_info));
        cudaStreamSynchronize(s);
        kernel_print_matrix<<<1, 1, 0, s>>>(d_matrix, N, N, nodeIndex);
        cudaStreamSynchronize(s);
        nodeIndex++; 

        for (int i = k + 1; i < T; i++)
        {
            //* L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_LEFT, // used to be right for cholesky
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,// CUBLAS_OP_T for cholesky
                CUBLAS_DIAG_UNIT, // CUBLAS_DIAG_NON_UNIT for cholesky
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N,  // k + k * N;
                getMatrixBlock(d_matrix, k, i), N)); // k + (i + B) * N; 
            cudaStreamSynchronize(s);
            kernel_print_matrix<<<1, 1, 0, s>>>(d_matrix, N, N, nodeIndex);
            cudaStreamSynchronize(s);
            nodeIndex++; 
        }

        for (int i = k + 1; i < T; i++)
        {
            //* U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
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
            cudaStreamSynchronize(s);
            kernel_print_matrix<<<1, 1, 0, s>>>(d_matrix, N, N, nodeIndex);
            cudaStreamSynchronize(s);
            nodeIndex++; 

            for (int j = k + 1; j < T; j++)
            {
                //* A[j][i] = GEMM(A[j][k], A[i][k])
                //* A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N,
                    getMatrixBlock(d_matrix, k, j), CUDA_R_64F, N, 
                    &one,
                    getMatrixBlock(d_matrix, i, j), CUDA_R_64F, N,
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                cudaStreamSynchronize(s);
                kernel_print_matrix<<<1, 1, 0, s>>>(d_matrix, N, N, nodeIndex);
                cudaStreamSynchronize(s);
                nodeIndex++;   
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
                   
    if (verbose) std::cout << "Done" << std::endl;

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

    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_matrix));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
}