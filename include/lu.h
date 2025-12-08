#ifndef LU_H
#define LU_H

#include "matrix.h"

/* Serial LU with partial pivoting: A -> LU in-place, P = pivot vector */
int lu_decompose_serial(Matrix A, int *P);

/* Solve Ax = b using LU + pivot vector P */
void lu_solve(const Matrix LU, const int *P, const double *b, double *x);

/* OpenMP-parallel LU (same interface) */
int lu_decompose_openmp(Matrix A, int *P);

/* MPI-parallel LU with block row distribution
 * Distributes matrix A in block rows across MPI processes and performs
 * LU decomposition with partial pivoting in parallel. Each process owns
 * a contiguous block of rows and updates them during elimination.
 * Returns 0 on success, -1 on failure (singular matrix).
 * Note: MPI_Init must be called before this function.
 */
int lu_decompose_mpi(Matrix *A, int **P, int argc, char **argv);

#endif
