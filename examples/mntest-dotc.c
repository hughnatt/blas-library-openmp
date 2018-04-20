#include <stdio.h>
// ### Fonctions BLAS Intel
#include <cblas.h>
// ### Fonctions BLAS MN Perso
#include "mnblas.h"

// ### Mesures des cycles
#include <x86intrin.h>

#include "mntest.h"


void mn_dotc_test_all(){

    int exp;
    unsigned long long start, end;
    unsigned long long residu;
    unsigned long long int av;

    // Calcul du residu de la mesure
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;

    vcplx V1,V2;
    vcplx_rand(V1);
    vcplx_rand(V2);

    Complex_c r;

    // @@@@@@@@@@@@@@@@@@@@@
    // @@@ ROUTINES DOTC @@@
    // @@@@@@@@@@@@@@@@@@@@@

    printf("========== DOTC ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_cdotc_sub0(VECSIZE,V1,1,V2,1,&r);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Dotc: r %5.2f %5.2f \t\t %Ld cycles\n",r.re,r.im, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_cdotc_sub1(VECSIZE,V1,1,V2,1,&r);  
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Dotc: r %5.2f %5.2f \t\t %Ld cycles\n",r.re,r.im, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_cdotc_sub2(VECSIZE,V1,1,V2,1,&r);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Dotc: r %5.2f %5.2f \t %Ld cycles\n",r.re,r.im, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_cdotc_sub3(VECSIZE,V1,1,V2,1,&r);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Dotc: r %5.2f %5.2f \t\t %Ld cycles\n",r.re,r.im, av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_cdotc_sub4(VECSIZE,V1,1,V2,1,&r);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Dotc: r %5.2f %5.2f\t\t %Ld cycles\n",r.re,r.im, av - residu);
}