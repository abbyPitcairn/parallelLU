#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lu.h"

int lu_decompose_serial(Matrix A, int *P) {
    int n = A.n;

    for (int i = 0; i < n; ++i) {
        P[i] = i;
    }

    for (int k = 0; k < n; ++k) {
        /* Find pivot row */
        int pivot = k;
        double max_val = fabs(MAT(A, k, k));
        for (int i = k + 1; i < n; ++i) {
            double val = fabs(MAT(A, i, k));
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }

        if (max_val == 0.0) {
            fprintf(stderr, "Matrix is singular at column %d\n", k);
            return -1;
        }

        /* Swap rows in A and pivot vector */
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

        /* Elimination */
        for (int i = k + 1; i < n; ++i) {
            double factor = MAT(A, i, k) / MAT(A, k, k);
            MAT(A, i, k) = factor;  // store L
            for (int j = k + 1; j < n; ++j) {
                MAT(A, i, j) -= factor * MAT(A, k, j);
            }
        }
    }
    return 0;
}

void lu_solve(const Matrix LU, const int *P, const double *b, double *x) {
    int n = LU.n;

    /* Apply permutation to b: Pb */
    double *y = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        y[i] = b[P[i]];
    }

    /* Forward substitution: Ly = Pb */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            y[i] -= MAT(LU, i, j) * y[j];
        }
    }

    /* Back substitution: Ux = y */
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            y[i] -= MAT(LU, i, j) * x[j];
        }
        x[i] = y[i] / MAT(LU, i, i);
    }

    free(y);
}
