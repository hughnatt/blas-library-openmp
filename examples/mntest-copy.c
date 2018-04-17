#include <stdio.h>
// ### Fonctions BLAS Intel
#include <cblas.h>
// ### Fonctions BLAS MN Perso
#include "mnblas.h"

#include "mntest.h"

// ### Mesures des cycles
#include <x86intrin.h>



void mn_copy_test_all(){

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
    // @@@ ROUTINES COPY @@@
    // @@@@@@@@@@@@@@@@@@@@@

    
    printf("========== COPY ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_scopy0(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Copy: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_scopy1(VECSIZE,V1,1,V2,1);     
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Copy: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_scopy2(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Copy: \t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_scopy3(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Copy: \t\t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_scopy4(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Copy: \t\t\t\t %Ld cycles\n", av - residu);

}