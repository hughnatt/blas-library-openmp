#include <stdio.h>
#include <time.h>

// ### Fonctions BLAS MN Perso
#include "mnblas.h"

// ### Mesures des cycles
#include <x86intrin.h>

#include "mntest.h"

/*
 * Initialisation d'un @vector avec une valeur fixe x
 */
void vfloat_init(vfloat V, float x)
{
    register unsigned int i;
    for (i = 0; i < VECSIZE; i++)
        V[i] = x;
}
void vdouble_init(vdouble V, double x)
{
    register unsigned int i;
    for (i = 0; i < VECSIZE; i++)
        V[i] = x;
}
void vcplx_init(vcplx V, Complex_c x)
{
    for (int i = 0; i < VECSIZE; i++)
    {
        V[i].re = x.re;
        V[i].im = x.im;
    }
}
void vcplxd_init(vcplxd V, Complex_z x)
{
    for (int i = 0; i < VECSIZE; i++)
    {
        V[i].re = x.re;
        V[i].im = x.im;
    }
}

/*
 * Type vecteur : @vfloat 
 */
typedef float vfloat[VECSIZE];
vfloat vec_1, vec_2;

typedef cplx_t vcomplex[VECSIZE];
vcomplex vec1, vec2;

/*
 * Génère des nombres flottants entre 0 et 10
 */
float randfloat()
{
    float x = ((float)rand() / (float)(RAND_MAX)) * 10;
    return x;
}
double randdouble(){
    double x = ((double)rand() / (double)(RAND_MAX)) * 10;
    return x;
}

/*
 * Initialisation d'un @vfloat avec une valeur fixe x
 */
void vfloat_rand(vfloat V)
{
    register unsigned int i;
    for (i = 0; i < VECSIZE; i++)
        V[i] = (float)randfloat();
}
void vdouble_rand(vdouble V)
{
    register unsigned int i;
    for (i = 0; i < VECSIZE; i++)
        V[i] = (double)randfloat();
}
void vcplx_rand(vcplx V)
{
    for (int i = 0; i < VECSIZE; i++)
    {
        V[i].re = (float)randfloat();
        V[i].im = (float)randfloat();
    }
}
void vcplxd_rand(vcplxd V)
{
    for (int i = 0; i < VECSIZE; i++)
    {
        V[i].re = (double)randfloat();
        V[i].im = (double)randfloat();
    }
}

/*
 * Affichage d'un @vfloat en console
 */
void vector_print(cplx_t *V, int n)
{
    register unsigned int i;

    for (i = 0; i < n; i++)
        printf("%f + i%f--", V[i].re, V[i].im);
    printf("\n");

    return;
}


long long unsigned int average (long long unsigned int *exps)
{
  unsigned int i ;
  long long unsigned int s = 0 ;

  for (i = 2; i < (NBEXPERIMENTS-2); i++)
    {
      s = s + exps [i] ;
    }

  return s / (NBEXPERIMENTS-2) ;
}

/*
 * Entry Point
 */
int main(int argc, char **argv)
{
    int nthreads, maxnthreads ;
  
    int tid;

    srand(time(NULL));


    mn_asum_test_all();
    mn_axpy_test_all();
    mn_copy_test_all();
    mn_dot_test_all();
    mn_dotc_test_all();
    mn_nrm2_test_all();
    

    mn_gemv_test_all();
    
    mn_gemm_test_all();
}
