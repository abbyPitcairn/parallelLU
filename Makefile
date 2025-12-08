CC      = mpicc
CFLAGS  = -O2 -Wall -Iinclude
LDFLAGS = 

# Detect macOS and set OpenMP flags accordingly
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OPENMP_CFLAGS = -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp)/include
    OPENMP_LDFLAGS = -L$(shell brew --prefix libomp)/lib -lomp
else
    OPENMP_CFLAGS = -fopenmp
    OPENMP_LDFLAGS = -fopenmp
endif

SRC_COMMON = src/matrix.c src/lu_serial.c src/lu_openmp.c src/lu_mpi.c
SRC_SERIAL = src/matrix.c src/lu_serial.c
SRC_MPI = src/matrix.c src/lu_serial.c src/lu_mpi.c

all: serial openmp mpi test_lu

serial: src/main_serial.c $(SRC_SERIAL)
	$(CC) $(CFLAGS) -o serial $^ $(LDFLAGS)

openmp: src/main_openmp.c $(SRC_COMMON)
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -o openmp $^ $(OPENMP_LDFLAGS)

mpi: src/main_mpi.c $(SRC_MPI)
	$(CC) $(CFLAGS) -o mpi $^ $(LDFLAGS)

test_lu: src/test_lu.c $(SRC_COMMON)
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -DHAVE_MPI -o test_lu $^ $(OPENMP_LDFLAGS)

# Test target that works without MPI/OpenMP (serial only)
test_lu_serial: src/test_lu.c src/matrix.c src/lu_serial.c
	@if command -v gcc >/dev/null 2>&1; then \
		CC=gcc; \
	elif command -v clang >/dev/null 2>&1; then \
		CC=clang; \
	else \
		echo "Error: No suitable compiler found"; exit 1; \
	fi; \
	echo "Compiling test_lu (serial only, no MPI/OpenMP)..."; \
	$$CC -O2 -Wall -Iinclude -o test_lu_serial src/test_lu.c src/matrix.c src/lu_serial.c

# Run test suite with 4 MPI processes
test: test_lu
	mpirun -np 4 ./test_lu --all

clean:
	rm -f serial openmp mpi test_lu test_lu_serial test_lu_no_mpi *.o
