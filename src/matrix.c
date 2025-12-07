#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

Matrix create_matrix(int n) {
    Matrix A;
    A.n = n;
    A.data = (double *)malloc((size_t)n * n * sizeof(double));
    if (!A.data) {
        fprintf(stderr, "Failed to allocate matrix of size %d\n", n);
        A.n = 0;
    }
    return A;
}

void free_matrix(Matrix *A) {
    if (A && A->data) {
        free(A->data);
        A->data = NULL;
        A->n = 0;
    }
}

void fill_random(Matrix A, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) {
            MAT(A, i, j) = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1,1]
        }
    }
}

/* Simple deterministic matrix for debugging (non-singular) */
void fill_test_matrix(Matrix A) {
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) {
            MAT(A, i, j) = (i == j) ? 4.0 : 1.0;
        }
    }
}

void print_matrix(const Matrix A, const char *name, int max_n) {
    int n = A.n;
    if (n > max_n) n = max_n;

    printf("%s (showing %d x %d of %d x %d):\n", name, n, n, A.n, A.n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%8.4f ", MAT(A, i, j));
        }
        if (A.n > n) printf(" ...");
        printf("\n");
    }
    if (A.n > n) {
        printf("...\n");
    }
}
