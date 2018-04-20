#include <stdio.h>
#include <smmintrin.h>


typedef struct {
    float re;
    float im;
} Complex;

typedef Complex Complex4[4] __attribute__((aligned(16)));

int main(int argc, char **argv)
{
    Complex4 a;
    a[0].re = 1.0;
    a[0].im = 5.0;
    a[1].re = 2.0;
    a[1].im = 6.0;
    a[2].re = 3.0;
    a[2].im = 7.0;
    a[3].re = 4.0;
    a[3].im = 8.0;
    Complex4 b;
    b[0].re = 3.0;
    b[0].im = 7.0;
    b[1].re = 4.0;
    b[1].im = 8.0;
    b[2].re = 0.0;
    b[2].im = 0.0;
    b[3].re = 0.0;
    b[3].im = 0.0;
    Complex4 dp;


    __m128 va1, va2, re,im;
    
    va1 = _mm_load_ps((float*) &a[0]);
    va2 = _mm_load_ps((float*) &a[2]);

    re = _mm_shuffle_ps(va1,va2,0x88);
    im = _mm_shuffle_ps(va1,va2,0xDD);

    _mm_store_ps((float*) &dp[0],re);

    printf("Re Load = %f %f %f %f\n",dp[0].re,dp[0].im,dp[1].re,dp[1].im);

    _mm_store_ps((float*) &dp[0],im);


    printf("Im Load = %f %f %f %f\n",dp[0].re,dp[0].im,dp[1].re,dp[1].im);
    return 0;
}