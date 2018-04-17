/*
 * RICM3 - Méthodes Numériques - 2018
 * ANCRENAZ Ariane - SAUTON Tanguy
 * C Implementation of BLAS routines
 */

#include "mnblas.h"
#include <omp.h>
#include <smmintrin.h>

/*
 * cblas_?dotc
 * Computes a dot product of a conjugated vector with another vector
 * res = SUM(i from 1 to n){ conjg(Xi) * Yi}
 *
 * Types : c / z
 */

void mncblas_cdotc_sub0(const int N, const void *X, const int incX,
                        const void *Y, const int incY, void *dotc) {
  cplx_t *xx = (cplx_t *)X;
  cplx_t *yy = (cplx_t *)Y;

  register unsigned int i;
  float re = 0;
  float im = 0;

  for (i = 0; i < N; i++) {
    re += xx[i].re * yy[i].re + xx[i].im * yy[i].im;
    im += xx[i].re * yy[i].im - xx[i].im * yy[i].re;
  }

  ((cplx_t *)dotc)->re = re;
  ((cplx_t *)dotc)->im = im;
}

void mncblas_cdotc_sub1(const int N, const void *X, const int incX,
                        const void *Y, const int incY, void *dotc) {
  cplx_t *xx = (cplx_t *)X;
  cplx_t *yy = (cplx_t *)Y;

  register unsigned int i;
  float re = 0;
  float im = 0;

#pragma omp parallel for schedule(static) reduction(+:re) reduction(+ : im)
  for (i = 0; i < N; i++) {
    re += xx[i].re * yy[i].re + xx[i].im * yy[i].im;
    im += xx[i].re * yy[i].im - xx[i].im * yy[i].re;
  }

  ((cplx_t *)dotc)->re = re;
  ((cplx_t *)dotc)->im = im;
}

void mncblas_cdotc_sub2(const int N, const void *X, const int incX,
                        const void *Y, const int incY, void *dotc) {
  cplx_t *xx = (cplx_t *)X;
  cplx_t *yy = (cplx_t *)Y;

  register unsigned int i;

  float re = 0;
  float im = 0;

#pragma omp parallel for schedule(static) reduction(+ : re) reduction(+ : im)
  for (i = 0; i < N; i += 4) {
    re += xx[i].re * yy[i].re + xx[i].im * yy[i].im;
    im += xx[i].re * yy[i].im - xx[i].im * yy[i].re;

    re += xx[i + 1].re * yy[i + 1].re + xx[i + 1].im * yy[i + 1].im;
    im += xx[i + 1].re * yy[i + 1].im - xx[i + 1].im * yy[i + 1].re;

    re += xx[i + 2].re * yy[i + 2].re + xx[i + 2].im * yy[i + 2].im;
    im += xx[i + 2].re * yy[i + 2].im - xx[i + 2].im * yy[i + 2].re;

    re += xx[i + 3].re * yy[i + 3].re + xx[i + 3].im * yy[i + 3].im;
    im += xx[i + 3].re * yy[i + 3].im - xx[i + 3].im * yy[i + 3].re;
  }

  ((cplx_t *)dotc)->re = re;
  ((cplx_t *)dotc)->im = im;
}

void mncblas_cdotc_sub3(const int N, const void *X, const int incX,
                        const void *Y, const int incY, void *dotc) {
/*   cplx_t *xx = (cplx_t *)X;
  cplx_t *yy = (cplx_t *)Y;

  register unsigned int i;

  float re = 0;
  float im = 0;

  float tmp[4] __attribute__((aligned(16)));

  __m128 v1, v2, mul;
  for (i = 0; i < N; i += 2) {
    v1 = _mm_load_pd(&X[i]);
    v2 = _mm_load_pd(&Y[i]);

    //res = _mm_dp_pd(v1,v2,0xFF);
    // = mm_dp_pd();

  //    A0        A1      A2        A3
  //  V1[0].re V1[0].im V1[1].re V1[1].im
  //  V2[0].re V2[0].im V2[1].re V2[1].im 
  //    B0        B1      B2        B3

    re += xx[i].re * yy[i].re + xx[i].im * yy[i].im;
  //       A0 * B0 + A1 * B1 + A2 * B2 + A3 * B3             
    re = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    im += xx[i].re * yy[i].im - xx[i].im * yy[i].re;
  //       A0 * B1 - A1 * B0
    

    v1 = _mm_load_ps(&X[i]);
    v2 = _mm_load_ps(&Y[i]);
    
    mul = _mm_mul_ps(v1,v2);
    


//_mm_hadd_ps();
//_mm_addsub();

    _mm_store_ps(tmp,mul);

    tmp[0] + tmp[1] + tmp[2] + tmp[3];

  }

  ((cplx_t *)dotc)->re = re;
  ((cplx_t *)dotc)->im = im;  */
}

void mncblas_cdotc_sub4(const int N, const void *X, const int incX,
                        const void *Y, const int incY, void *dotc) {
}