#include <stdio.h>
// ### Fonctions BLAS Intel
#include <cblas.h>
// ### Fonctions BLAS MN Perso
#include "mnblas.h"

#include "mntest.h"

// ### Mesures des cycles
#include <x86intrin.h>


static const int m = MATSIZE;
static const int n = MATSIZE;
static const int k = MATSIZE;

void mn_gemv_test_all(){

    int exp;
    unsigned long long start, end;
    unsigned long long residu;
    unsigned long long int av ;
    float r;

    // Calcul du residu de la mesure
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;

    // --- saxpy
    float *my_matA = malloc(m * k * sizeof(float));
    vfloat xx;
    vfloat yy;
    vfloat_rand(xx);
    vfloat_rand(yy);
    

    const float alpha = 3.0;
    const float beta = 3.0;
    

    int i, j;
    for (i = 0; i < MATSIZE; i++)
    {
        for (j = 0; j < MATSIZE; j++)
        {
            my_matA[k * i + j] = (float) randfloat();
        }
    }


    // @@@@@@@@@@@@@@@@@@@@@
    // @@@ ROUTINES GEMV @@@
    // @@@@@@@@@@@@@@@@@@@@@

    printf("========== GEMV ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemv0(MNCblasRowMajor,MNCblasNoTrans,m,n,alpha,my_matA,m,xx,1,beta,yy,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Gemv: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemv1(MNCblasRowMajor,MNCblasNoTrans,m,n,alpha,my_matA,m,xx,1,beta,yy,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Gemv: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemv2(MNCblasRowMajor,MNCblasNoTrans,m,n,alpha,my_matA,m,xx,1,beta,yy,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Gemv:  \t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemv3(MNCblasRowMajor,MNCblasNoTrans,m,n,alpha,my_matA,m,xx,1,beta,yy,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Gemv:  \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemv4(MNCblasRowMajor,MNCblasNoTrans,m,n,alpha,my_matA,m,xx,1,beta,yy,1);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Gemv: \t\t\t\t %Ld cycles\n", av - residu);

}