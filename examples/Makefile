all: mntest mntestdyn

INC_DIR=../include

LIB_DIR=../lib

LIBST=-lmnblas
LIBDYN=-lmnblasdyn

OPTIONS_COMPIL  =-O2 -fPIC -I$(INC_DIR) -lm -g

OPTIONS_LINK_STATIC  =   -L$(LIB_DIR) $(LIBST)  -lgsl -lgslcblas -lm -fopenmp -msse4.2
OPTIONS_LINK_DYN  = -L$(LIB_DIR) $(LIBDYN)  -lgsl -lgslcblas -lm -fopenmp -msse4.2

SRCS= mntest-asum.c mntest-axpy.c mntest-copy.c mntest-dot.c mntest-dotc.c mntest-gemm.c mntest-gemv.c mntest-nrm2.c
OBJS= $(SRCS:.c=.o)

mntest: mntest.o $(LIB_DIR)/libmnblas.a $(OBJS)
	gcc -o mntest mntest.o $(OBJS) $(OPTIONS_LINK_STATIC)

mntestdyn: mntest.o $(LIB_DIR)/libmnblasdyn.so $(OBJS)
	gcc -o mntestdyn mntest.o $(OBJS) $(OPTIONS_LINK_DYN)

mntest.o: mntest.c $(INC_DIR)/mnblas.h $(INC_DIR)/mntest.h
	gcc $(OPTIONS_COMPIL) -c mntest.c

%.o : %.c $(INC_DIR)/mnblas.h $(INC_DIR)/mntest.h
	gcc $(OPTIONS_COMPIL) -c $<

clean:
	rm -f *.o mntest mntestdyn *~
