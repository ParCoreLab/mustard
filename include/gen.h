#include <stdlib.h>

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n)
{
    // srand(time(NULL));
    srand(420);

    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            h_A[i * n + j] = (double)rand() / (double)RAND_MAX;

    for (int i = 0; i < n; i++)
        for (int j = i; j >= 0; j--)
            h_A[i * n + j] = h_A[j * n + i];

    for (int i = 0; i < n; i++)
        h_A[i * n + i] = h_A[i * n + i] + n;
}
