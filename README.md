# Parallel LU Decomposition for Linear Algebra

### COS 574 - Graduate Project

### Author: Abigail Pitcairn {abigail.pitcairn@maine.edu}

### Version: December 7, 2025

## Project Description

This project implements LU decomposition with partial pivoting using three different parallelization approaches: serial, OpenMP, and MPI. LU decomposition is a fundamental linear algebra operation that factors a matrix A into the product of a lower triangular matrix L and an upper triangular matrix U, such that PA = LU, where P is a permutation matrix representing the row exchanges from partial pivoting.

The project provides:
- **Serial Implementation**: A baseline sequential implementation for comparison
- **OpenMP Implementation**: Shared-memory parallelization using OpenMP directives for multi-threaded execution
- **MPI Implementation**: Distributed-memory parallelization using MPI for multi-process execution with block row distribution

Each implementation performs the same mathematical operation but uses different parallelization strategies, allowing for performance comparison across different hardware configurations and problem sizes. The project includes comprehensive testing utilities to verify correctness and measure performance across all three implementations.

## Directory Overview

```
parallelLU/
├── include/              # Header files
│   ├── lu.h             # LU decomposition function declarations
│   └── matrix.h         # Matrix data structure and utility functions
├── src/                 # Source code files
│   ├── matrix.c         # Matrix operations (creation, printing, utilities)
│   ├── lu_serial.c      # Serial LU decomposition implementation
│   ├── lu_openmp.c      # OpenMP-parallel LU decomposition
│   ├── lu_mpi.c         # MPI-parallel LU decomposition
│   ├── main_serial.c    # Serial program entry point
│   ├── main_openmp.c    # OpenMP program entry point
│   ├── main_mpi.c       # MPI program entry point
│   └── test_lu.c        # Comprehensive test suite for all implementations
├── report/              # Project documentation
│   ├── main.tex         # LaTeX report source
│   ├── references.bib   # Bibliography
│   └── report.pdf       # Compiled report
├── test_result/         # Test output examples
│   └── test_result_example.txt
├── Makefile            # Build configuration
└── README.md           # This file
```

## How to Run

### Quick Start - Simple Test Results

**1. Build the project:**
```bash
make
```

**2. Run the test suite:**
```bash
./test_lu
```

This will run tests on the serial and OpenMP implementations with default settings and display the results.

**3. For MPI testing (requires MPI runtime):**
```bash
mpirun -np 4 ./test_lu --all
```

This runs all three implementations (serial, OpenMP, and MPI) with 4 MPI processes.

### Additional Options

- Test with a specific matrix size: `./test_lu -n 50`
- View detailed output for small matrices: `./test_lu -n 5 -v`
- Test individual implementations: `./test_lu --serial-only` or `./test_lu --openmp-only`

---
