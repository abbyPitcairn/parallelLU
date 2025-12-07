#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "lu.h"

int lu_decompose_openmp(Matrix A, int *P) {
    int n = A.n;

    for (int i = 0; i < n; ++i) {
        P[i] = i;
    }

    for (int k = 0; k < n; ++k) {
        /* Parallel pivot search */
        int pivot = k;
        double max_val = 0.0;

        #pragma omp parallel
        {
            int local_pivot = -1;
            double local_max = max_val;

            #pragma omp for nowait
            for (int i = k; i < n; ++i) {
                double val = fabs(MAT(A, i, k));
                if (val > local_max) {
                    local_max = val;
                    local_pivot = i;
                }
            }

            #pragma omp critical
            {
                if (local_pivot != -1 && local_max > max_val) {
                    max_val = local_max;
                    pivot = local_pivot;
                }
            }
        }

        if (max_val == 0.0) {
            fprintf(stderr, "Matrix is singular at column %d\n", k);
            return -1;
        }

        /* Swap rows (single-threaded to keep it simple) */
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                double tmp = MAT(A, k, j);
                MAT(A, k, j) = MAT(A, pivot, j);
                MAT(A, pivot, j) = tmp;
            }
            int tmpi = P[k];
            P[k] = P[pivot];
            P[pivot] = tmpi;
        }

        double akk = MAT(A, k, k);

        /* Parallel elimination on trailing rows / columns */
        #pragma omp parallel for
        for (int i = k + 1; i < n; ++i) {
            double factor = MAT(A, i, k) / akk;
            MAT(A, i, k) = factor;
            for (int j = k + 1; j < n; ++j) {
                MAT(A, i, j) -= factor * MAT(A, k, j);
            }
        }
    }
    return 0;
}
