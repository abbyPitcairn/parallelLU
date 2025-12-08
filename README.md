# Parallel LU Decomposition for Linear Algebra

### COS 574 - Graduate Project

### Author: Abigail Pitcairn {abigail.pitcairn@maine.edu}

### Version: December 7, 2025

#### Description

This project runs an LU decomposition using parallel algorithms from OpenMP and MPI. 

#### How to run

**Build all targets:**
```bash
make
```

**Run tests:**
```bash
# Test serial and OpenMP implementations
./test_lu

# Test with custom matrix size
./test_lu -n 20

# Test with verbose output (shows matrices for small n)
./test_lu -n 5 -v

# Test only serial
./test_lu --serial-only

# Test only OpenMP
./test_lu --openmp-only

# Test MPI (requires mpirun)
mpirun -np 4 ./test_lu --mpi-only

# Test all implementations including MPI
mpirun -np 4 ./test_lu --all
```

**Run individual programs:**
```bash
# Serial
./serial [n]

# OpenMP
./openmp [n] [num_threads]

# MPI
mpirun -np 4 ./mpi [n]
```

---
