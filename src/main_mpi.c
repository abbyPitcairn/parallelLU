#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "matrix.h"
#include "lu.h"

int main(int argc, char **argv) {
    int n = 100;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix A;
    if (rank == 0) {
        A = create_matrix(n);
        fill_random(A, (unsigned int)time(NULL));
    } else {
        A.n = n;
        A.data = NULL;  // will be allocated inside lu_decompose_mpi
    }

    int *P = NULL;

    double t0 = MPI_Wtime();
    lu_decompose_mpi(&A, &P, argc, argv);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("MPI LU (stage-1 skeleton) completed in %.6f seconds\n", t1 - t0);
    }

    if (A.data) free_matrix(&A);
    if (P) free(P);

    MPI_Finalize();
    return 0;
}
