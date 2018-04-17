#include <stdio.h>
// ### Fonctions BLAS Intel
#include <cblas.h>
// ### Fonctions BLAS MN Perso
#include "mnblas.h"

// ### Mesures des cycles
#include <x86intrin.h>

#include "mntest.h"

void mn_asum_test_all()
{
    int exp;
    unsigned long long start, end;
    unsigned long long residu;
    unsigned long long int av ;
    float r;

    // Calcul du residu de la mesure
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;

    vfloat V1;
    vfloat_rand(V1);

    // @@@@@@@@@@@@@@@@@@@@
    // @@@ ROUTINE ASUM @@@
    // @@@@@@@@@@@@@@@@@@@@

    printf("========== ASUM ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sasum0(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Asum: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sasum1(VECSIZE,V1,1);        
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Asum: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sasum2(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Asum: r %5.2f  \t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sasum3(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Asum: r %5.2f  \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sasum4(VECSIZE,V1,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Asum: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

}