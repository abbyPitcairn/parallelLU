#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "matrix.h"
#include "lu.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* Forward declarations */
void write_csv_header(FILE *fp);
void write_csv_row(FILE *fp, const char *test_name, int n, const char *pattern, int num_procs, 
                   const char *status, double error, double tolerance, double exec_time, const char *result);

/* Create permutation matrix from pivot vector */
void create_permutation_matrix(const int *P, int n, Matrix *P_mat) {
    *P_mat = create_matrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            MAT(*P_mat, i, j) = (P[i] == j) ? 1.0 : 0.0;
        }
    }
}

/* Extract L and U from combined LU matrix */
void extract_LU(const Matrix LU, Matrix *L, Matrix *U) {
    int n = LU.n;
    *L = create_matrix(n);
    *U = create_matrix(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) {
                MAT(*L, i, j) = MAT(LU, i, j);
                MAT(*U, i, j) = 0.0;
            } else if (i == j) {
                MAT(*L, i, j) = 1.0;
                MAT(*U, i, j) = MAT(LU, i, j);
            } else {
                MAT(*L, i, j) = 0.0;
                MAT(*U, i, j) = MAT(LU, i, j);
            }
        }
    }
}

/* Matrix multiplication: C = A * B */
void matrix_multiply(const Matrix A, const Matrix B, Matrix *C) {
    int n = A.n;
    *C = create_matrix(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += MAT(A, i, k) * MAT(B, k, j);
            }
            MAT(*C, i, j) = sum;
        }
    }
}

/* Compute PA - LU and return maximum error */
double verify_lu(const Matrix A_orig, const Matrix LU, const int *P) {
    int n = A_orig.n;
    
    Matrix P_mat, L, U, PA, LU_result, diff;
    create_permutation_matrix(P, n, &P_mat);
    extract_LU(LU, &L, &U);
    
    /* Compute PA */
    matrix_multiply(P_mat, A_orig, &PA);
    
    /* Compute LU */
    matrix_multiply(L, U, &LU_result);
    
    /* Compute difference: PA - LU */
    diff = create_matrix(n);
    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            MAT(diff, i, j) = MAT(PA, i, j) - MAT(LU_result, i, j);
            double error = fabs(MAT(diff, i, j));
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    
    free_matrix(&P_mat);
    free_matrix(&L);
    free_matrix(&U);
    free_matrix(&PA);
    free_matrix(&LU_result);
    free_matrix(&diff);
    
    return max_error;
}

/* Write matrix sample to file */
void write_matrix_sample(FILE *fp, const Matrix A, int max_sample) {
    int n = A.n;
    int sample_size = (n > max_sample) ? max_sample : n;
    
    fprintf(fp, "Matrix sample (first %d x %d of %d x %d):\n", sample_size, sample_size, n, n);
    for (int i = 0; i < sample_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            fprintf(fp, "%.6f ", MAT(A, i, j));
        }
        if (n > sample_size) {
            fprintf(fp, "...");
        }
        fprintf(fp, "\n");
    }
    if (n > sample_size) {
        fprintf(fp, "...\n");
    }
}

/* Test serial LU decomposition */
int test_serial(int n, int verbose, int csv_format, FILE *fp) {
    if (!csv_format) {
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Test: Serial LU Decomposition\n");
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Input Values:\n");
        fprintf(fp, "  Matrix size: %d x %d\n", n, n);
        fprintf(fp, "  Matrix pattern: Diagonal = 4.0, Off-diagonal = 1.0\n");
    }
    
    Matrix A = create_matrix(n);
    fill_test_matrix(A);
    
    if (!csv_format && verbose && n <= 10) {
        fprintf(fp, "  Input matrix:\n");
        write_matrix_sample(fp, A, n);
    }
    
    Matrix A_orig = create_matrix(n);
    memcpy(A_orig.data, A.data, (size_t)n * n * sizeof(double));
    
    int *P = (int *)malloc((size_t)n * sizeof(int));
    
    /* Time the decomposition */
    clock_t start = clock();
    int status = lu_decompose_serial(A, P);
    clock_t end = clock();
    double execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double error = 0.0;
    double tolerance = 1e-10;
    int passed = 0;
    
    if (status != 0) {
        if (csv_format) {
            write_csv_row(fp, "Serial LU", n, "Diagonal=4.0,Off-diagonal=1.0", 1, 
                         "FAILED", 0.0, tolerance, execution_time, "FAILED");
        } else {
            fprintf(fp, "\nOutput Values:\n");
            fprintf(fp, "  Decomposition status: FAILED\n");
            fprintf(fp, "  Test Result: FAILED\n");
            fprintf(fp, "  Execution time: %.6f seconds\n", execution_time);
            fprintf(fp, "  Error: Decomposition returned non-zero status\n\n");
        }
        free(P);
        free_matrix(&A);
        free_matrix(&A_orig);
        return 1;
    }
    
    error = verify_lu(A_orig, A, P);
    passed = (error <= tolerance);
    
    if (csv_format) {
        write_csv_row(fp, "Serial LU", n, "Diagonal=4.0,Off-diagonal=1.0", 1,
                     "SUCCESS", error, tolerance, execution_time, passed ? "PASSED" : "FAILED");
    } else {
        fprintf(fp, "\nOutput Values:\n");
        fprintf(fp, "  Decomposition status: SUCCESS\n");
        fprintf(fp, "  Maximum verification error: %.6e\n", error);
        fprintf(fp, "  Error tolerance: %.6e\n", tolerance);
        fprintf(fp, "  Execution time: %.6f seconds\n", execution_time);
        
        if (verbose && n <= 10) {
            fprintf(fp, "  Output LU matrix:\n");
            write_matrix_sample(fp, A, n);
        }
        
        fprintf(fp, "\nTest Result: %s\n", passed ? "PASSED" : "FAILED");
        if (!passed) {
            fprintf(fp, "  Reason: Maximum error (%.6e) exceeds tolerance (%.6e)\n", error, tolerance);
        }
        fprintf(fp, "\n");
    }
    
    free(P);
    free_matrix(&A);
    free_matrix(&A_orig);
    return passed ? 0 : 1;
}

/* Test OpenMP LU decomposition */
int test_openmp(int n, int verbose, int csv_format, FILE *fp) {
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
    
    if (!csv_format) {
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Test: OpenMP LU Decomposition\n");
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Input Values:\n");
        fprintf(fp, "  Matrix size: %d x %d\n", n, n);
        fprintf(fp, "  Matrix pattern: Diagonal = 4.0, Off-diagonal = 1.0\n");
        fprintf(fp, "  Number of threads: %d\n", num_threads);
    }
    
    Matrix A = create_matrix(n);
    fill_test_matrix(A);
    
    if (!csv_format && verbose && n <= 10) {
        fprintf(fp, "  Input matrix:\n");
        write_matrix_sample(fp, A, n);
    }
    
    Matrix A_orig = create_matrix(n);
    memcpy(A_orig.data, A.data, (size_t)n * n * sizeof(double));
    
    int *P = (int *)malloc((size_t)n * sizeof(int));
    
    /* Time the decomposition */
#ifdef _OPENMP
    double start = omp_get_wtime();
    int status = lu_decompose_openmp(A, P);
    double end = omp_get_wtime();
    double execution_time = end - start;
#else
    clock_t start = clock();
    int status = lu_decompose_openmp(A, P);
    clock_t end = clock();
    double execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
#endif
    
    double error = 0.0;
    double tolerance = 1e-10;
    int passed = 0;
    
    if (status != 0) {
        if (csv_format) {
            write_csv_row(fp, "OpenMP LU", n, "Diagonal=4.0,Off-diagonal=1.0", num_threads,
                         "FAILED", 0.0, tolerance, execution_time, "FAILED");
        } else {
            fprintf(fp, "\nOutput Values:\n");
            fprintf(fp, "  Decomposition status: FAILED\n");
            fprintf(fp, "  Test Result: FAILED\n");
            fprintf(fp, "  Execution time: %.6f seconds\n", execution_time);
            fprintf(fp, "  Error: Decomposition returned non-zero status\n\n");
        }
        free(P);
        free_matrix(&A);
        free_matrix(&A_orig);
        return 1;
    }
    
    error = verify_lu(A_orig, A, P);
    passed = (error <= tolerance);
    
    if (csv_format) {
        write_csv_row(fp, "OpenMP LU", n, "Diagonal=4.0,Off-diagonal=1.0", num_threads,
                     "SUCCESS", error, tolerance, execution_time, passed ? "PASSED" : "FAILED");
    } else {
        fprintf(fp, "\nOutput Values:\n");
        fprintf(fp, "  Decomposition status: SUCCESS\n");
        fprintf(fp, "  Maximum verification error: %.6e\n", error);
        fprintf(fp, "  Error tolerance: %.6e\n", tolerance);
        fprintf(fp, "  Execution time: %.6f seconds\n", execution_time);
        
        if (verbose && n <= 10) {
            fprintf(fp, "  Output LU matrix:\n");
            write_matrix_sample(fp, A, n);
        }
        
        fprintf(fp, "\nTest Result: %s\n", passed ? "PASSED" : "FAILED");
        if (!passed) {
            fprintf(fp, "  Reason: Maximum error (%.6e) exceeds tolerance (%.6e)\n", error, tolerance);
        }
        fprintf(fp, "\n");
    }
    
    free(P);
    free_matrix(&A);
    free_matrix(&A_orig);
    return passed ? 0 : 1;
}

/* Test MPI LU decomposition */
int test_mpi(int n, int verbose, int csv_format, int argc, char **argv, FILE *fp) {
#ifdef HAVE_MPI
    int mpi_initialized = 0;
    int should_finalize = 0;
    
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(&argc, &argv);
        should_finalize = 1;
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0 && !csv_format) {
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Test: MPI LU Decomposition\n");
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Input Values:\n");
        fprintf(fp, "  Matrix size: %d x %d\n", n, n);
        fprintf(fp, "  Matrix pattern: Diagonal = 4.0, Off-diagonal = 1.0\n");
        fprintf(fp, "  Number of MPI processes: %d\n", size);
    }
    
    Matrix A = create_matrix(n);
    if (rank == 0) {
        fill_test_matrix(A);
        if (!csv_format && verbose && n <= 10) {
            fprintf(fp, "  Input matrix:\n");
            write_matrix_sample(fp, A, n);
        }
    }
    
    Matrix A_orig = create_matrix(n);
    if (rank == 0) {
        memcpy(A_orig.data, A.data, (size_t)n * n * sizeof(double));
    }
    
    int *P = NULL;
    
    /* Time the decomposition */
    double start = MPI_Wtime();
    int status = lu_decompose_mpi(&A, &P, argc, argv);
    double end = MPI_Wtime();
    double execution_time = end - start;
    
    double error = 0.0;
    double tolerance = 1e-10;
    int passed = 0;
    
    if (rank == 0) {
        if (status != 0) {
            if (csv_format) {
                write_csv_row(fp, "MPI LU", n, "Diagonal=4.0,Off-diagonal=1.0", size,
                             "FAILED", 0.0, tolerance, execution_time, "FAILED");
            } else {
                fprintf(fp, "\nOutput Values:\n");
                fprintf(fp, "  Decomposition status: FAILED\n");
                fprintf(fp, "  Test Result: FAILED\n");
                fprintf(fp, "  Execution time: %.6f seconds\n", execution_time);
                fprintf(fp, "  Error: Decomposition returned non-zero status\n\n");
            }
            free_matrix(&A);
            free_matrix(&A_orig);
            if (should_finalize) {
                MPI_Finalize();
            }
            return 1;
        }
        
        error = verify_lu(A_orig, A, P);
        passed = (error <= tolerance);
        
        if (csv_format) {
            write_csv_row(fp, "MPI LU", n, "Diagonal=4.0,Off-diagonal=1.0", size,
                         "SUCCESS", error, tolerance, execution_time, passed ? "PASSED" : "FAILED");
        } else {
            fprintf(fp, "\nOutput Values:\n");
            fprintf(fp, "  Decomposition status: SUCCESS\n");
            fprintf(fp, "  Maximum verification error: %.6e\n", error);
            fprintf(fp, "  Error tolerance: %.6e\n", tolerance);
            fprintf(fp, "  Execution time: %.6f seconds\n", execution_time);
            
            if (verbose && n <= 10) {
                fprintf(fp, "  Output LU matrix:\n");
                write_matrix_sample(fp, A, n);
            }
            
            fprintf(fp, "\nTest Result: %s\n", passed ? "PASSED" : "FAILED");
            if (!passed) {
                fprintf(fp, "  Reason: Maximum error (%.6e) exceeds tolerance (%.6e)\n", error, tolerance);
            }
            fprintf(fp, "\n");
        }
    }
    
    free(P);
    free_matrix(&A);
    free_matrix(&A_orig);
    
    if (should_finalize) {
        MPI_Finalize();
    }
    return (rank == 0 && passed) ? 0 : 1;
#else
    if (csv_format) {
        write_csv_row(fp, "MPI LU", n, "Diagonal=4.0,Off-diagonal=1.0", 0,
                     "NOT_AVAILABLE", 0.0, 1e-10, 0.0, "FAILED");
    } else {
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Test: MPI LU Decomposition\n");
        fprintf(fp, "===========================================================\n");
        fprintf(fp, "Input Values:\n");
        fprintf(fp, "  Matrix size: %d x %d\n", n, n);
        fprintf(fp, "\nOutput Values:\n");
        fprintf(fp, "  Decomposition status: NOT_AVAILABLE\n");
        fprintf(fp, "  Execution time: 0.000000 seconds\n");
        fprintf(fp, "\nTest Result: FAILED\n");
        fprintf(fp, "  Reason: MPI support not compiled\n");
        fprintf(fp, "\n");
    }
    return 1;
#endif
}

/* Write CSV header */
void write_csv_header(FILE *fp) {
    fprintf(fp, "Test Name,Matrix Size,Input Pattern,Num Threads/Processes,Decomposition Status,Max Error,Error Tolerance,Execution Time (seconds),Test Result\n");
}

/* Write CSV row for a test */
void write_csv_row(FILE *fp, const char *test_name, int n, const char *pattern, int num_procs, 
                   const char *status, double error, double tolerance, double exec_time, const char *result) {
    fprintf(fp, "%s,%d,%s,%d,%s,%.6e,%.6e,%.6f,%s\n", 
            test_name, n, pattern, num_procs, status, error, tolerance, exec_time, result);
}

int main(int argc, char **argv) {
    int n = 5000;
    int verbose = 0;
    int test_serial_flag = 1;
    int test_openmp_flag = 1;
    int test_mpi_flag = 0;
    int csv_format = 0;
    const char *output_file = "test.txt";
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--serial-only") == 0) {
            test_openmp_flag = 0;
            test_mpi_flag = 0;
        } else if (strcmp(argv[i], "--openmp-only") == 0) {
            test_serial_flag = 0;
            test_mpi_flag = 0;
        } else if (strcmp(argv[i], "--mpi-only") == 0) {
            test_serial_flag = 0;
            test_openmp_flag = 0;
            test_mpi_flag = 1;
        } else if (strcmp(argv[i], "--all") == 0) {
            test_serial_flag = 1;
            test_openmp_flag = 1;
            test_mpi_flag = 1;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0) {
            csv_format = 1;
            if (strcmp(output_file, "test.txt") == 0) {
                output_file = "test.csv";
            }
        }
    }
    
    /* Open output file */
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open output file '%s' for writing\n", output_file);
        return 1;
    }
    
    if (csv_format) {
        /* Write CSV header */
        write_csv_header(fp);
    } else {
        /* Write text header */
        fprintf(fp, "LU Decomposition Test Suite Results\n");
        fprintf(fp, "====================================\n\n");
        fprintf(fp, "Test Configuration:\n");
        fprintf(fp, "  Matrix size: %d x %d\n", n, n);
        fprintf(fp, "  Verbose mode: %s\n", verbose ? "enabled" : "disabled");
        fprintf(fp, "  Output file: %s\n", output_file);
        fprintf(fp, "\n");
        fprintf(fp, "Test Summary:\n");
        fprintf(fp, "  Serial tests: %s\n", test_serial_flag ? "enabled" : "disabled");
        fprintf(fp, "  OpenMP tests: %s\n", test_openmp_flag ? "enabled" : "disabled");
        fprintf(fp, "  MPI tests: %s\n", test_mpi_flag ? "enabled" : "disabled");
        fprintf(fp, "\n");
        fprintf(fp, "====================================\n\n");
    }
    
    int failed = 0;
    
    if (test_serial_flag) {
        if (test_serial(n, verbose, csv_format, fp) != 0) {
            failed = 1;
        }
    }
    
    if (test_openmp_flag) {
        if (test_openmp(n, verbose, csv_format, fp) != 0) {
            failed = 1;
        }
    }
    
    if (test_mpi_flag) {
        if (test_mpi(n, verbose, csv_format, argc, argv, fp) != 0) {
            failed = 1;
        }
    }
    
    /* Write summary */
    if (!csv_format) {
        fprintf(fp, "====================================\n");
        fprintf(fp, "Overall Test Suite Result: %s\n", failed ? "FAILED" : "PASSED");
        fprintf(fp, "====================================\n");
    }
    
    fclose(fp);
    
    printf("Test results written to: %s\n", output_file);
    
    return failed ? 1 : 0;
}
