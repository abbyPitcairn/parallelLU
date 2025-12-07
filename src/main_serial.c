#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "lu.h"

int main(int argc, char **argv) {
    int n = 100;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    printf("Serial LU, n = %d\n", n);

    Matrix A = create_matrix(n);
    fill_random(A, (unsigned int)time(NULL));

    int *P = (int *)malloc((size_t)n * sizeof(int));

    clock_t start = clock();
    int status = lu_decompose_serial(A, P);
    clock_t end = clock();

    if (status != 0) {
        fprintf(stderr, "LU failed\n");
        return 1;
    }

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Serial LU completed in %.6f seconds\n", elapsed);

    if (n <= 8) {
        print_matrix(A, "LU", n);
    }

    free(P);
    free_matrix(&A);
    return 0;
}
