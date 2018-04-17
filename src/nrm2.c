/*
 * RICM3 - Méthodes Numériques - 2018
 * ANCRENAZ Ariane - SAUTON Tanguy
 * C Implementation of BLAS routines
 */

#include "math.h"
#include "mnblas.h"
#include <omp.h>
#include <smmintrin.h>

/*
 * cblas_?nrm2
 * Computes the Euclidean norm of a vector
 * res = ||x||
 *
 * Types = s / d / sc / dz
 */

float mncblas_snrm2_0(const int n, const float *x, const int incX)
{
    float nrm = 0.0;
    register unsigned int i;

    for (i = 0; i < n; i++)
    {
        nrm += x[i] * x[i];
    }
    return sqrtf(nrm);
}

float mncblas_snrm2_1(const int n, const float *x, const int incX)
{
    float nrm = 0.0;
    register unsigned int i;

#pragma omp parallel for schedule(static) reduction(+ : nrm)
    for (i = 0; i < n; i++)
    {
        nrm += x[i] * x[i];
    }


    return sqrtf(nrm);
}

float mncblas_snrm2_2(const int n, const float *x, const int incX)
{
    float nrm = 0.0;
    register unsigned int i;

#pragma omp parallel for schedule(static) reduction(+:nrm)
    for (i = 0; i < n; i += 4)
    {
        nrm += x[i] * x[i];
        nrm += x[i + 1] * x[i + 1];
        nrm += x[i + 2] * x[i + 2];
        nrm += x[i + 3] * x[i + 3];
    }

    return sqrtf(nrm);
}

float mncblas_snrm2_3(const int n, const float *x, const int incX)
{
    float nrm = 0.0;
    register unsigned int i;
    float tmp[4] __attribute__((aligned(16))) ;
    

    __m128 v1, res;
#pragma omp parallel for schedule(static) reduction(+:nrm)
    for (i = 0; i < n; i += 4)
    {   
        v1 = _mm_load_ps(&x[i]);
        res = _mm_dp_ps(v1,v1,0xFF);
        _mm_store_ps(tmp,res);
        nrm += tmp[0];
    }

    return sqrtf(nrm);
}

float mncblas_snrm2_4(const int n, const float *x, const int incX)
{
    float nrm = 0.0;
    register unsigned int i;
    float tmp[4] __attribute__((aligned(16))) ;
    

    __m128 v1, res;
    for (i = 0; i < n; i += 4)
    {   
        v1 = _mm_load_ps(&x[i]);
        res = _mm_dp_ps(v1,v1,0xFF);
        _mm_store_ps(tmp,res);
        nrm += tmp[0];
    }

    return sqrtf(nrm);
}