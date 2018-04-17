#include <stdio.h>
// ### Fonctions BLAS Intel
#include <cblas.h>
// ### Fonctions BLAS MN Perso
#include "mnblas.h"

#include "mntest.h"

// ### Mesures des cycles
#include <x86intrin.h>

void mn_dot_test_all()
{

    int exp;
    unsigned long long start, end;
    unsigned long long residu;
    unsigned long long int av;
    float r;

    // Calcul du residu de la mesure
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;

    vfloat V1, V2;
    vfloat_rand(V1);
    vfloat_rand(V2);

    // @@@@@@@@@@@@@@@@@@@@@
    // @@@ ROUTINES DOT  @@@
    // @@@@@@@@@@@@@@@@@@@@@

    printf("========== DOT ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sdot0(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Dot: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sdot1(VECSIZE,V1,1,V2,1);    
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Dot: r %5.2f \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sdot2(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Dot: r %5.2f  \t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sdot3(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Dot: r %5.2f  \t\t\t %Ld cycles\n",r, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        r = mncblas_sdot4(VECSIZE,V1,1,V2,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Dot: r %5.2f  \t\t\t %Ld cycles\n",r, av - residu);

}