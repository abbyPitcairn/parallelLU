#ifndef LU_H
#define LU_H

#include "matrix.h"

/* Serial LU with partial pivoting: A -> LU in-place, P = pivot vector */
int lu_decompose_serial(Matrix A, int *P);

/* Solve Ax = b using LU + pivot vector P */
void lu_solve(const Matrix LU, const int *P, const double *b, double *x);

/* OpenMP-parallel LU (same interface) */
int lu_decompose_openmp(Matrix A, int *P);

/* MPI-parallel LU skeleton (block row distribution)
 * Returns 0 on success. For now, this will just gather on rank 0 and call serial,
 * then broadcast result.
 */
int lu_decompose_mpi(Matrix *A, int **P, int argc, char **argv);

#endif
