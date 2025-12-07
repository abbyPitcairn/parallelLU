#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "matrix.h"
#include "lu.h"

int main(int argc, char **argv) {
    int n = 100;
    int num_threads = 4;

    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        num_threads = atoi(argv[2]);
    }

    omp_set_num_threads(num_threads);
    printf("OpenMP LU, n = %d, threads = %d\n", n, num_threads);

    Matrix A = create_matrix(n);
    fill_random(A, (unsigned int)time(NULL));

    int *P = (int *)malloc((size_t)n * sizeof(int));

    double start = omp_get_wtime();
    int status = lu_decompose_openmp(A, P);
    double end = omp_get_wtime();

    if (status != 0) {
        fprintf(stderr, "LU failed\n");
        return 1;
    }

    printf("OpenMP LU completed in %.6f seconds\n", end - start);

    if (n <= 8) {
        print_matrix(A, "LU (OpenMP)", n);
    }

    free(P);
    free_matrix(&A);
    return 0;
}
