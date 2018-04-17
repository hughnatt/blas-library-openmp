#include <stdio.h>
// ### Fonctions BLAS Intel
#include <cblas.h>
// ### Fonctions BLAS MN Perso
#include "mnblas.h"

#include "mntest.h"
// ### Mesures des cycles
#include <x86intrin.h>


void mn_axpy_test_all(){

    int exp;
    unsigned long long start, end;
    unsigned long long residu;
    unsigned long long int av ;

    // Calcul du residu de la mesure
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;

    vfloat V1,V2;
    vfloat_rand(V1);
    vfloat_rand(V2);

    
    // @@@@@@@@@@@@@@@@@@@@@
    // @@@ ROUTINES AXPY @@@
    // @@@@@@@@@@@@@@@@@@@@@

    printf("========== AXPY ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_saxpy0(VECSIZE,3.2,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Axpy: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_saxpy1(VECSIZE,3.2,V1,1,V2,1);     
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Axpy: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_saxpy2(VECSIZE,3.2,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Axpy: \t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_saxpy3(VECSIZE,3.2,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Axpy: \t\t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_saxpy4(VECSIZE,3.2,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Axpy: \t\t\t\t %Ld cycles\n", av - residu);
}