#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "lu.h"
#include "matrix.h"

/* Starting point: rank 0 owns full A, does serial LU, then broadcast.
 * --> Replace this with a true block-row distribution.
 */
int lu_decompose_mpi(Matrix *A, int **P_out, int argc, char **argv) {
    int rank, size;
    int n = A->n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("MPI LU: n = %d, size = %d\n", n, size);
    }

    /* For now, assume rank 0 has the matrix; others allocate matching storage */
    if (rank != 0) {
        *A = create_matrix(n);
    }

    /* Broadcast matrix size and data */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(A->data, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *P = (int *)malloc((size_t)n * sizeof(int));

    /* Only rank 0 performs LU; others just receive LU result after */
    if (rank == 0) {
        if (lu_decompose_serial(*A, P) != 0) {
            fprintf(stderr, "LU factorization failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Broadcast LU factors and pivot vector from rank 0 */
    MPI_Bcast(A->data, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(P, n, MPI_INT, 0, MPI_COMM_WORLD);

    *P_out = P;

    /* don't finalize here; main_mpi.c should call MPI_Finalize(). */
    return 0;
}
