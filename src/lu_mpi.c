#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "lu.h"
#include "matrix.h"

int lu_decompose_mpi(Matrix *A, int **P_out, int argc, char **argv) {
    int rank, size;
    int n = A->n;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("MPI LU: n = %d, size = %d\n", n, size);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        *A = create_matrix(n);
    }

    int rows_per_proc = n / size;
    int remainder = n % size;
    int local_start = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int local_end = local_start + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_rows = local_end - local_start;

    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int i_start = i * rows_per_proc + (i < remainder ? i : remainder);
            int i_end = i_start + rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = (i_end - i_start) * n;
            displs[i] = i_start * n;
        }
    }

    double *local_data = (double *)malloc((size_t)local_rows * n * sizeof(double));
    MPI_Scatterv(rank == 0 ? A->data : NULL, sendcounts, displs, MPI_DOUBLE,
                 local_data, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }

    int *P = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    double *pivot_row = (double *)malloc((size_t)n * sizeof(double));

    for (int k = 0; k < n; k++) {
        int local_pivot = -1;
        double local_max = 0.0;

        for (int i = local_start; i < local_end; i++) {
            if (i >= k) {
                int local_i = i - local_start;
                double val = fabs(local_data[local_i * n + k]);
                if (val > local_max) {
                    local_max = val;
                    local_pivot = i;
                }
            }
        }

        struct {
            double max_val;
            int rank;
        } local_candidate, global_candidate;

        local_candidate.max_val = local_max;
        local_candidate.rank = (local_pivot != -1) ? rank : -1;

        MPI_Allreduce(&local_candidate, &global_candidate, 1, MPI_DOUBLE_INT,
                     MPI_MAXLOC, MPI_COMM_WORLD);

        if (global_candidate.max_val == 0.0 || global_candidate.rank == -1) {
            if (rank == 0) {
                fprintf(stderr, "Matrix is singular at column %d\n", k);
            }
            free(local_data);
            free(pivot_row);
            free(P);
            return -1;
        }

        int global_pivot = -1;
        if (global_candidate.rank == rank && local_pivot != -1) {
            global_pivot = local_pivot;
        }
        MPI_Bcast(&global_pivot, 1, MPI_INT, global_candidate.rank, MPI_COMM_WORLD);

        if (global_candidate.rank == rank) {
            int local_pivot_idx = global_pivot - local_start;
            for (int j = 0; j < n; j++) {
                pivot_row[j] = local_data[local_pivot_idx * n + j];
            }
        }
        MPI_Bcast(pivot_row, n, MPI_DOUBLE, global_candidate.rank, MPI_COMM_WORLD);

        if (global_pivot != k) {
            int tmp = P[k];
            P[k] = P[global_pivot];
            P[global_pivot] = tmp;

            double *row_k = (double *)malloc((size_t)n * sizeof(double));
            if (k >= local_start && k < local_end) {
                int local_k = k - local_start;
                for (int j = 0; j < n; j++) {
                    row_k[j] = local_data[local_k * n + j];
                }
            }
            int row_k_owner = -1;
            for (int r = 0; r < size; r++) {
                int r_start = r * rows_per_proc + (r < remainder ? r : remainder);
                int r_end = r_start + rows_per_proc + (r < remainder ? 1 : 0);
                if (k >= r_start && k < r_end) {
                    row_k_owner = r;
                    break;
                }
            }
            MPI_Bcast(row_k, n, MPI_DOUBLE, row_k_owner, MPI_COMM_WORLD);

            if (k >= local_start && k < local_end) {
                int local_k = k - local_start;
                for (int j = 0; j < n; j++) {
                    local_data[local_k * n + j] = pivot_row[j];
                }
            }
            if (global_pivot >= local_start && global_pivot < local_end && global_pivot != k) {
                int local_pivot_idx = global_pivot - local_start;
                for (int j = 0; j < n; j++) {
                    local_data[local_pivot_idx * n + j] = row_k[j];
                }
            }
            free(row_k);
        }

        double akk = pivot_row[k];
        if (akk == 0.0) {
            if (rank == 0) {
                fprintf(stderr, "Matrix is singular at column %d\n", k);
            }
            free(local_data);
            free(pivot_row);
            free(P);
            return -1;
        }

        for (int i = local_start; i < local_end; i++) {
            if (i > k) {
                int local_i = i - local_start;
                double factor = local_data[local_i * n + k] / akk;
                local_data[local_i * n + k] = factor;
                for (int j = k + 1; j < n; j++) {
                    local_data[local_i * n + j] -= factor * pivot_row[j];
                }
            }
        }
    }

    if (rank == 0) {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int i_start = i * rows_per_proc + (i < remainder ? i : remainder);
            int i_end = i_start + rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = (i_end - i_start) * n;
            displs[i] = i_start * n;
        }
    }

    MPI_Gatherv(local_data, local_rows * n, MPI_DOUBLE,
                rank == 0 ? A->data : NULL, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Bcast(A->data, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(P, n, MPI_INT, 0, MPI_COMM_WORLD);

    free(local_data);
    free(pivot_row);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }

    *P_out = P;
    return 0;
}
