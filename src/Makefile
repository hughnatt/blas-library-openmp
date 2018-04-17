# une bibliotheque statique (libmnblas.a)
# une bibliotheque dynamique (libmnblasdyn.so)

all: libmnblas.a libmnblasdyn.so

LIB_DIR=../lib
INC_DIR=../include

OPTIM=-msse4.2 -fopenmp -O2
OPTIONS=-fPIC $(OPTIM) -I$(INC_DIR) -g

SRCS= asum.c axpy.c copy.c dot.c dotc.c nrm2.c gemm.c gemv.c
OBJS= $(SRCS:.c=.o)

# Bibliothèque statique

libmnblas.a: $(OBJS)
	rm -f libmnblas.a $(LIB_DIR)/libmnblas.a
	ar -r libmnblas.a $(OBJS)
	cp libmnblas.a $(LIB_DIR)

# Bibliothèque dynamique

libmnblasdyn.so: $(OBJS)
	rm -f libmnblasdyn.so $(LIB_DIR)/libmnblasdyn.so
	gcc -shared -o libmnblasdyn.so $(OBJS)
	cp libmnblasdyn.so $(LIB_DIR)

# BLAS 1

asum.o : asum.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS) -c asum.c

axpy.o : axpy.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS) -c axpy.c

copy.o : copy.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS) -c copy.c

dot.o : dot.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c dot.c

dotc.o : dotc.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c dotc.c

nrm2.o : nrm2.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c nrm2.c


# BLAS 2

gemv.o : gemv.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c gemv.c

# BLAS 3

gemm.o : gemm.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c gemm.c

# CLEAN

clean:
	rm -f *.o *.a *.so *~  $(INC_DIR)/*~ $(LIB_DIR)/libmnblasdyn.so  $(LIB_DIR)/libmnblas.a 

