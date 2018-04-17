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

void mn_gemm_test_all(){
    int exp;
    unsigned long long start, end;
    unsigned long long residu;
    unsigned long long int av ;
    float r;

    // Calcul du residu de la mesure
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;


    float *my_matA = malloc(m * k * sizeof(float));
    float *my_matB = malloc(k * n * sizeof(float));
    float *my_matC = malloc(m * n * sizeof(float));
    

    const float alpha = 3.0;
    const float beta = 2.0;

    int i, j;
    for (i = 0; i < MATSIZE; i++)
    {
        for (j = 0; j < MATSIZE; j++)
        {
            my_matA[k * i + j] = (float) randfloat();
            my_matB[n * i + j] = (float) randfloat();
            my_matC[n * i + j] = (float) randfloat();
        }
    }



    //mncblas_sgemm(MNCblasRowMajor,MNCblasNoTrans,MNCblasNoTrans,m,n,k,alpha,my_matA,m,my_matB,n,beta,my_matC,k);

    // @@@@@@@@@@@@@@@@@@@@@
    // @@@ ROUTINES GEMM @@@
    // @@@@@@@@@@@@@@@@@@@@@

    printf("========== GEMM ==========\n");

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemm0(MNCblasRowMajor,MNCblasNoTrans,MNCblasNoTrans,m,n,k,alpha,my_matA,m,my_matB,n,beta,my_matC,k);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Sequential Gemm: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemm1(MNCblasRowMajor,MNCblasNoTrans,MNCblasNoTrans,m,n,k,alpha,my_matA,m,my_matB,n,beta,my_matC,k);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Gemm: \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemm2(MNCblasRowMajor,MNCblasNoTrans,MNCblasNoTrans,m,n,k,alpha,my_matA,m,my_matB,n,beta,my_matC,k);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("OMP Static Unrolled Gemm:  \t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemm3(MNCblasRowMajor,MNCblasNoTrans,MNCblasNoTrans,m,n,k,alpha,my_matA,m,my_matB,n,beta,my_matC,k);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored Gemm:  \t\t\t\t %Ld cycles\n", av - residu);

    for (exp = 0; exp < NBEXPERIMENTS; exp++)
    {
        start = _rdtsc();
        mncblas_sgemm4(MNCblasRowMajor,MNCblasNoTrans,MNCblasNoTrans,m,n,k,alpha,my_matA,m,my_matB,n,beta,my_matC,k);
        end = _rdtsc();
        experiments[exp] = end - start;
    }
    av = average(experiments);
    printf("Vectored OMP Gemm: \t\t\t\t %Ld cycles\n", av - residu);
}