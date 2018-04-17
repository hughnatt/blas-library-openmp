/*
 * RICM3 - Méthodes Numériques - 2018
 * ANCRENAZ Ariane - SAUTON Tanguy
 * C Implementation of BLAS routines
 */

#include "mnblas.h"
#include <omp.h>
#include <xmmintrin.h>

/*
 * cblas_?axpy
 * Computes a vector-scalar product and adds the result to a vector
 * Y := a*X + Y
 * 
 * Types : s / d / c / z
 */

void mncblas_saxpy0(const int n, const float a, const float *x,
                   const int incx, float *y, const int incy)
{
    register unsigned int i;
    
    for (i = 0; i < n ; i ++)
    {
        y[i] += a * x[i];
    }
}

void mncblas_saxpy1(const int n, const float a, const float *x,
                   const int incx, float *y, const int incy)
{
    register unsigned int i;

#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i ++)
    {
        y[i] += a * x[i];
    }
}

void mncblas_saxpy2(const int n, const float a, const float *x,
                   const int incx, float *y, const int incy)
{
    register unsigned int i;

#pragma omp parallel for schedule(static)
    for (i = 0; i < n ; i += 4)
    {
        y[i] += a * x[i];
        y[i + 1] += a * x[i + 1];
        y[i + 2] += a * x[i + 2];
        y[i + 3] += a * x[i + 3];
    }
}

void mncblas_saxpy3(const int n, const float a, const float *x,
                   const int incx, float *y, const int incy)
{
    register unsigned int i;

    __m128 v1,v2,scalar,tmp;
    for (i = 0; i < n ; i += 4)
    {
        v1 = _mm_load_ps(&x[i]);
        v2 = _mm_load_ps(&y[i]);
        scalar = _mm_set1_ps(a);
        tmp = _mm_mul_ps(v1,scalar);
        v2 = _mm_add_ps(v2,tmp);
        _mm_store_ps(&y[i],v2);
    }
}

void mncblas_saxpy4(const int n, const float a, const float *x,
                   const int incx, float *y, const int incy)
{
    register unsigned int i;

    __m128 v1,v2,scalar,tmp;

#pragma omp parallel for schedule(static)
    for (i = 0; i < n ; i += 4)
    {
        v1 = _mm_load_ps(&x[i]);
        v2 = _mm_load_ps(&y[i]);
        scalar = _mm_set1_ps(a);
        tmp = _mm_mul_ps(v1,scalar);
        v2 = _mm_add_ps(v2,tmp);
        _mm_store_ps(&y[i],v2);
    }
}