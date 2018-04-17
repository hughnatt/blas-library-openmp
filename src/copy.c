/*
 * RICM3 - Méthodes Numériques - 2018
 * ANCRENAZ Ariane - SAUTON Tanguy
 * C Implementation of BLAS routines
 */

#include "mnblas.h"
#include <omp.h>
#include <xmmintrin.h>

/*
 * cblas_?copy
 * Copies vector to another vector
 * Y := X
 * 
 * Types : s / d / c / z 
 */

void mncblas_scopy0(const int N, const float *X, const int incX,
                   float *Y, const int incY)
{
    register unsigned int i;

    for (i = 0; i < N; i ++)
    {
        Y[i] = X[i];
    }
}

void mncblas_scopy1(const int N, const float *X, const int incX,
                   float *Y, const int incY)
{
    register unsigned int i;

#pragma omp parallel for schedule(static)
    for (i = 0; i < N; i ++)
    {
        Y[i] = X[i];
    }
}

void mncblas_scopy2(const int N, const float *X, const int incX,
                   float *Y, const int incY)
{
    register unsigned int i;

#pragma omp parallel for schedule(static)
    for (i = 0; i < N; i += 4)
    {
        Y[i] = X[i];
        Y[i + 1] = X[i + 1];
        Y[i + 2] = X[i + 2];
        Y[i + 3] = X[i + 3];
    }
}

void mncblas_scopy3(const int N, const float *X, const int incX,
                   float *Y, const int incY)
{
    register unsigned int i;

    __m128 v1;
    for (i = 0; i < N; i += 4)
    {
        v1 = _mm_load_ps(&X[i]);
        _mm_store_ps(&Y[i],v1);
    }
}

void mncblas_scopy4(const int N, const float *X, const int incX,
                   float *Y, const int incY)
{
    register unsigned int i;

    __m128 v1;
#pragma omp parallel for schedule(static)
    for (i = 0; i < N; i += 4)
    {
        v1 = _mm_load_ps(&X[i]);
        _mm_store_ps(&Y[i],v1);
    }
}