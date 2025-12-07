CC      = mpicc
CFLAGS  = -O2 -Wall -Iinclude -fopenmp
LDFLAGS = -fopenmp

SRC_COMMON = src/matrix.c src/lu_serial.c src/lu_openmp.c src/lu_mpi.c

all: serial openmp mpi

serial: src/main_serial.c $(SRC_COMMON)
	$(CC) $(CFLAGS) -o serial $^ $(LDFLAGS)

openmp: src/main_openmp.c $(SRC_COMMON)
	$(CC) $(CFLAGS) -o openmp $^ $(LDFLAGS)

mpi: src/main_mpi.c $(SRC_COMMON)
	$(CC) $(CFLAGS) -o mpi $^ $(LDFLAGS)

clean:
	rm -f serial openmp mpi *.o
