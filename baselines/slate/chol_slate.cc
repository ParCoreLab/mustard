// slate_lu.cc
// Getting started

/// !!!   Lines between `//---------- begin label`          !!!
/// !!!             and `//---------- end label`            !!!
/// !!!   are included in the SLATE Users' Guide.           !!!

//---------- begin sec1
#include <slate/slate.hh>
#include <blas.hh>
#include <mpi.h>
#include <stdio.h>

#include "argh.h"
#include "slate_util.hh"

// Forward function declarations
template <typename scalar_type>
void lu_example(int64_t n, int64_t nrhs, int64_t nb, int p, int q, int runs);

// template <typename matrix_type>
// void random_matrix( matrix_type& A );

int main(int argc, char **argv)
{
    // Initialize MPI, requiring MPI_THREAD_MULTIPLE support.
    int err = 0, mpi_provided = 0;
    err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    if (err != 0 || mpi_provided != MPI_THREAD_MULTIPLE)
    {
        throw std::runtime_error("MPI_Init failed");
    }

    // Call the LU example.
    int64_t n = 12000, nrhs = 1, nb = 2000 /* tile size */, p = 1, q = 1, runs = 10;
    auto cmdl = argh::parser(argc, argv);

    if (!(cmdl({"N", "n"}, n) >> n))
    {
        std::cerr << "Must provide a valid N value! Got '" << cmdl({"N", "n"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"b", "B"}, nb) >> nb))
    {
        std::cerr << "Must provide a valid B value! Got '" << cmdl({"B", "b"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"run", "runs", "r", "R"}, runs) >> runs) || runs < 1)
    {
        std::cerr << "Must provide a valid number of runs! Got '" << cmdl({"run", "r", "R"}).str() << "'" << std::endl;
        return 0;
    }

    lu_example<double>(n, nrhs, nb, p, q, runs);

    err = MPI_Finalize();
    if (err != 0)
    {
        throw std::runtime_error("MPI_Finalize failed");
    }
    return 0;
}

//---------- end sec1
//---------- begin sec2
// Create matrices, call LU solver, and check result.
template <typename scalar_t>
void lu_example(int64_t n, int64_t nrhs, int64_t nb, int p, int q, int runs)
{
    // Get associated real type, e.g., double for complex<double>.
    using real_t = double; // blas::real_type<scalar_t>;
    using llong = long long; // guaranteed >= 64 bits
    const scalar_t one = 1;
    int err = 0, mpi_size = 0, mpi_rank = 0;

    // Get MPI size. Must be >= p*q for this example.
    err = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (err != 0)
    {
        throw std::runtime_error("MPI_Comm_size failed");
    }
    p = mpi_size;
    // if (mpi_size < p*q) {
    //     printf( "Usage: mpirun -np %d ... # %d ranks hard coded\n",
    //             p*q, p*q );
    //     return;
    // }

    // Get MPI rank
    err = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (err != 0)
    {
        throw std::runtime_error("MPI_Comm_rank failed");
    }

    // Create SLATE matrices A and B. /* \label{line:lu-AB} */
    slate::SymmetricMatrix<scalar_t> A(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    slate::SymmetricMatrix<scalar_t> AH(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> BH(n, nrhs, nb, p, q, MPI_COMM_WORLD);

    // Allocate local space for A, B on distributed nodes. /* \label{line:lu-insert} */
    A.insertLocalTiles(slate::Target::Devices);
    B.insertLocalTiles(slate::Target::Devices);
    AH.insertLocalTiles();
    BH.insertLocalTiles();

    // Set random seed so data is different on each MPI rank.
    srand(100 * mpi_rank);
    // Initialize the data for A, B. /* \label{line:lu-rand} */
    slate::Uplo::Lower, (AH);
    random_matrix_diag_dominant(BH);

    // For residual error check,
    // create A0 as an empty matrix like A and copy A to A0.
    slate::SymmetricMatrix<scalar_t> A0 = A.emptyLike();
    A0.insertLocalTiles();
    slate::copy(AH, A0);
    slate::copy(AH, A); /* \label{line:lu-copy} */
    // Create B0 as an empty matrix like B and copy B to B0.
    slate::Matrix<scalar_t> B0 = B.emptyLike();
    B0.insertLocalTiles();
    slate::copy(BH, B0);
    slate::copy(BH, B);

    // Call the SLATE LU solver.

    for (int run = 0; run < runs; run++)
    {
        slate::Options opts = {/* \label{line:lu-opts} */
                               {slate::Option::Target, slate::Target::Devices}};
        // slate::Pivots pivots;

        double time = omp_get_wtime();
        // slate::lu_solve(A, B, opts); /* \label{line:lu-solve} */
        // slate::lu_factor( A, pivots, opts );
        slate::potrf( A, opts );
        time = omp_get_wtime() - time;
        // slate::lu_solve_using_factor( A, pivots, B, opts );
        slate::potrs( A, B, opts );
        slate::copy(B, BH);

        // Compute residual ||A0 * X  - B0|| / ( ||X|| * ||A0|| * n )  /* \label{line:lu-residual} */
        real_t A_norm = slate::norm(slate::Norm::One, A0);
        real_t X_norm = slate::norm(slate::Norm::One, BH);
        slate::symm(slate::Side::Left, -one, A0, BH, one, B0);
        real_t R_norm = slate::norm(slate::Norm::One, B0);
        real_t residual = R_norm / (X_norm * A_norm * n);
        real_t tol = std::numeric_limits<real_t>::epsilon();
        bool status_ok = (residual < tol);

        if (mpi_rank == 0)
        {
            printf("device potrf n %lld, nb %lld, p-by-q %lld-by-%lld, "
                   "residual %.2e, tol %.2e, time %4.4f sec, %s\n",
                   llong(n), llong(nb), llong(p), llong(q),
                   residual, tol, time,
                   status_ok ? "pass" : "FAILED");
        }

        if (run == runs - 1)
        {
            // slate::copy(A0, AH);
            // slate::copy(B0, BH);
            // // slate::Pivots pivotsH;

            // opts = {/* \label{line:lu-opts} */
            //         {slate::Option::Target, slate::Target::Host}};
            // time = omp_get_wtime();
            // // slate::lu_solve(AH, BH, opts); /* \label{line:lu-solve} */
            // // slate::lu_factor( AH, pivotsH, opts );
            // slate::potrf( AH, opts );
            // time = omp_get_wtime() - time;
            // slate::potrs( AH, BH, opts );
            // // slate::lu_solve_using_factor( AH, pivotsH, BH, opts );

            // // Compute residual ||A0 * X  - B0|| / ( ||X|| * ||A0|| * n )  /* \label{line:lu-residual} */
            // A_norm = slate::norm(slate::Norm::One, A0);
            // X_norm = slate::norm(slate::Norm::One, BH);
            // slate::symm(-one, A0, BH, one, B0);
            // R_norm = slate::norm(slate::Norm::One, B0);
            // residual = R_norm / (X_norm * A_norm * n);
            // tol = std::numeric_limits<real_t>::epsilon();
            // status_ok = (residual < tol);

            // if (mpi_rank == 0)
            // {
            //     printf("host potrf n %lld, nb %lld, p-by-q %lld-by-%lld, "
            //            "residual %.2e, tol %.2e, time %4.4f sec, %s\n",
            //            llong(n), llong(nb), llong(p), llong(q),
            //            residual, tol, time,
            //            status_ok ? "pass" : "FAILED");
            // }
        } else {
            slate::copy(A0, A);
            slate::copy(B0, B);
        }
    }
}
