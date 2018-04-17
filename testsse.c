#include <stdio.h>
#include <smmintrin.h>

typedef float float4[4] __attribute__((aligned(16)));

int main(int argc, char **argv)
{
    float4 a = {2.0f, 2.0f, 3.0f, 4.0f};
    float4 b = {2.0f, 2.0f, 3.0f, 4.0f};
    float4 dp;

    __m128 v1, v2, res;
    v1 = _mm_load_ps(a);
    v2 = _mm_load_ps(b);

    res = _mm_dp_ps(v1,v2,0xFF);

    _mm_store_ps(dp,res);

    printf("Dot Prod = %f %f %f %f\n",dp[0],dp[1],dp[2],dp[3]);
    return 0;
}