#include <stdio.h>
// ### Fonctions BLAS Intel
#include <cblas.h>
// ### Fonctions BLAS MN Perso
#include "mnblas.h"

#include "mntest.h"

// ### Mesures des cycles
#include <x86intrin.h>


void mn_nrm2_test_all(){

    int exp;
    unsigned long long start, end;
    unsigned long long residu;
    unsigned long long int av;

    float r;

    // Calcul du residu de la mesure
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;

    vfloat V1;
    vfloat_rand(V1);

    // @@@@@@@@@@@@@@@@@@@@
    // @@@ ROUTINE nrm2 @@@
    // @@@@@@@@@@@@@@@@@@@@

   printf("========== NRM2 ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_snrm2_0(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Nrm2: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_snrm2_1(VECSIZE,V1,1); 
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Nrm2: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_snrm2_2(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Nrm2: r %5.2f \t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_snrm2_3(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Nrm2: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_snrm2_4(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Nrm2: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);
}