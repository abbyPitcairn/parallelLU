#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    int n;
    double *data;
} Matrix;

Matrix create_matrix(int n);
void free_matrix(Matrix *A);

/* Access macro: A(i,j) */
#define MAT(A, i, j) ((A).data[(size_t)(i) * (A).n + (j)])

void fill_random(Matrix A, unsigned int seed);
void fill_test_matrix(Matrix A);

void print_matrix(const Matrix A, const char *name, int max_n);

#endif
