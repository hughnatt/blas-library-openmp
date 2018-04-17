/*
 * RICM3 - Méthodes Numériques - 2018
 * ANCRENAZ Ariane - SAUTON Tanguy
 * C Implementation of BLAS routines
 */

#include "mnblas.h"
#include <omp.h>
#include <smmintrin.h>

/*
 * cblas_?dot
 * Computes a vector-vector dot product
 * res = SUM(i from 1 to N){Xi * Yi}
 * 
 * Types : s / d
 */

float mncblas_sdot0(const int N, const float *X, const int incX,
                   const float *Y, const int incY)
{
    register float dot = 0.0;
    register unsigned int i;
    
    for (i = 0; i < N; i ++)
    {
        dot += X[i] * Y[i];
    }

    return dot;
}

float mncblas_sdot1(const int N, const float *X, const int incX,
                   const float *Y, const int incY)
{
    register float dot;
    register unsigned int i;

    dot = 0.0;
#pragma omp parallel for schedule(static) reduction(+:dot)
    for (i = 0; i < N; i ++)
    {
        dot += X[i] * Y[i];
    }

    return dot;
}

float mncblas_sdot2(const int N, const float *X, const int incX,
                   const float *Y, const int incY)
{
    register float dot = 0.0;
    register unsigned int i;

#pragma omp parallel for schedule(static) reduction(+:dot)
    for (i = 0; i < N; i += 4)
    {
        dot += X[i] * Y[i];
        dot += X[i + 1] * Y[i + 1];
        dot += X[i + 2] * Y[i + 2];
        dot += X[i + 3] * Y[i + 3];
    }
    return dot;
}

float mncblas_sdot3(const int N, const float *X, const int incX,
                   const float *Y, const int incY)
{
    register float dot = 0.0;
    float tmp[4] __attribute__((aligned(16))) ;
    register unsigned int i;

    __m128 v1,v2, res;
    for (i = 0; i < N; i += 4)
    {
        v1 = _mm_load_ps(&X[i]);
        v2 = _mm_load_ps(&Y[i]);
        res = _mm_dp_ps(v1,v2,0xFF);
        _mm_store_ps(tmp,res);
        dot += tmp[0];
    }

    return dot;
}

float mncblas_sdot4(const int N, const float *X, const int incX,
                   const float *Y, const int incY)
{
    register float dot = 0.0;
    float tmp[4] __attribute__((aligned(16)));
    register unsigned int i;

    __m128 v1,v2, res;
#pragma omp parallel for schedule(static) reduction(+:dot) reduction(+:tmp)
    for (i = 0; i < N; i += 4)
    {
        v1 = _mm_load_ps(&X[i]);
        v2 = _mm_load_ps(&Y[i]);
        res = _mm_dp_ps(v1,v2,0xFF);
        _mm_store_ps(tmp,res);
        dot += tmp[0];
    }

    return dot;
}