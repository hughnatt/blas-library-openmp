
typedef enum { MNCblasRowMajor = 101,
               MNCblasColMajor = 102 } MNCBLAS_LAYOUT;
typedef enum { MNCblasNoTrans = 111,
               MNCblasTrans = 112,
               MNCblasConjTrans = 113 } MNCBLAS_TRANSPOSE;
typedef enum { MNCblasUpper = 121,
               MNCblasLower = 122 } MNCBLAS_UPLO;
typedef enum { MNCblasNonUnit = 131,
               MNCblasUnit = 132 } MNCBLAS_DIAG;
typedef enum { MNCblasLeft = 141,
               MNCblasRight = 142 } MNCBLAS_SIDE;

typedef struct
{
  float re;
  float im;
} cplx_t;

typedef struct
{
  double re;
  double im;
} cplxd_t;

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions 
 * ===========================================================================
 */

/*
  BLAS copy
*/

void mncblas_scopy0(const int N, const float *X, const int incX, float *Y, const int incY);
void mncblas_scopy1(const int N, const float *X, const int incX, float *Y, const int incY);
void mncblas_scopy2(const int N, const float *X, const int incX, float *Y, const int incY);
void mncblas_scopy3(const int N, const float *X, const int incX, float *Y, const int incY);
void mncblas_scopy4(const int N, const float *X, const int incX, float *Y, const int incY);

void mncblas_dcopy(const int N, const double *X, const int incX,
                   double *Y, const int incY);

void mncblas_ccopy(const int N, const void *X, const int incX,
                   void *Y, const int incY);

void mncblas_zcopy(const int N, const void *X, const int incX,
                   void *Y, const int incY);

/*
  end COPY BLAS
*/

/*
  BLAS SWAP
*/

void mncblas_sswap(const int N, float *X, const int incX,
                   float *Y, const int incY);

void mncblas_dswap(const int N, double *X, const int incX,
                   double *Y, const int incY);

void mncblas_cswap(const int N, void *X, const int incX,
                   void *Y, const int incY);

void mncblas_zswap(const int N, void *X, const int incX,
                   void *Y, const int incY);

/*
  END SWAP
*/

/*

  BLAS DOT

*/

float mncblas_sdot0(const int N, const float *X, const int incX, const float *Y, const int incY);
float mncblas_sdot1(const int N, const float *X, const int incX, const float *Y, const int incY);
float mncblas_sdot2(const int N, const float *X, const int incX, const float *Y, const int incY);
float mncblas_sdot3(const int N, const float *X, const int incX, const float *Y, const int incY);
float mncblas_sdot4(const int N, const float *X, const int incX, const float *Y, const int incY);

double mncblas_ddot(const int N, const double *X, const int incX,
                    const double *Y, const int incY);

void mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu);

void mncblas_cdotc_sub0(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc);
void mncblas_cdotc_sub1(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc);
void mncblas_cdotc_sub2(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc);
void mncblas_cdotc_sub3(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc);
void mncblas_cdotc_sub4(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc);

void mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu);
void mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc);

/*
  END BLAS DOT
*/

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (s, d, c, z)
 */

void mncblas_saxpy0(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);
void mncblas_saxpy1(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);
void mncblas_saxpy2(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);
void mncblas_saxpy3(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);
void mncblas_saxpy4(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);
void mncblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);
void mncblas_caxpy(const int N, const void *alpha, const void *X, const int incX, void *Y, const int incY);
void mncblas_zaxpy(const int N, const void *alpha, const void *X, const int incX, void *Y, const int incY);

float mncblas_sasum0(const int n, const float *x, const int incx);
float mncblas_sasum1(const int n, const float *x, const int incx);
float mncblas_sasum2(const int n, const float *x, const int incx);
float mncblas_sasum3(const int n, const float *x, const int incx);
float mncblas_sasum4(const int n, const float *x, const int incx);

float mncblas_scasum(const int n, const void *x, const int incx);
double mncblas_dasum(const int n, const double *x, const int incx);
double mncblas_dzasum(const int n, const void *x, const int incx);

float mncblas_snrm2_0(const int n, const float *x, const int incx);
float mncblas_snrm2_1(const int n, const float *x, const int incx);
float mncblas_snrm2_2(const int n, const float *x, const int incx);
float mncblas_snrm2_3(const int n, const float *x, const int incx);
float mncblas_snrm2_4(const int n, const float *x, const int incx);

double mncblas_dnrm2(const int n, const double *x, const int incx);
float mncblas_scnrm2(const int n, const void *x, const int incx);
double mncblas_dznrm2(const int n, const void *x, const int incx);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (S, D, C, Z)
 */

void mncblas_sgemv0(const MNCBLAS_LAYOUT layout,
                    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                    const float alpha, const float *A, const int lda,
                    const float *X, const int incX, const float beta,
                    float *Y, const int incY);
void mncblas_sgemv1(const MNCBLAS_LAYOUT layout,
                    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                    const float alpha, const float *A, const int lda,
                    const float *X, const int incX, const float beta,
                    float *Y, const int incY);
void mncblas_sgemv2(const MNCBLAS_LAYOUT layout,
                    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                    const float alpha, const float *A, const int lda,
                    const float *X, const int incX, const float beta,
                    float *Y, const int incY);
void mncblas_sgemv3(const MNCBLAS_LAYOUT layout,
                    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                    const float alpha, const float *A, const int lda,
                    const float *X, const int incX, const float beta,
                    float *Y, const int incY);
void mncblas_sgemv4(const MNCBLAS_LAYOUT layout,
                    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                    const float alpha, const float *A, const int lda,
                    const float *X, const int incX, const float beta,
                    float *Y, const int incY);

void mncblas_dgemv(MNCBLAS_LAYOUT layout,
                   MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                   const double alpha, const double *A, const int lda,
                   const double *X, const int incX, const double beta,
                   double *Y, const int incY);

void mncblas_cgemv(MNCBLAS_LAYOUT layout,
                   MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *X, const int incX, const void *beta,
                   void *Y, const int incY);

void mncblas_zgemv(MNCBLAS_LAYOUT layout,
                   MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *X, const int incX, const void *beta,
                   void *Y, const int incY);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (S, D, C, Z)
 */

void mncblas_sgemm0(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A,
                   const int lda, const float *B, const int ldb,
                   const float beta, float *C, const int ldc);
void mncblas_sgemm1(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A,
                   const int lda, const float *B, const int ldb,
                   const float beta, float *C, const int ldc);
void mncblas_sgemm2(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A,
                   const int lda, const float *B, const int ldb,
                   const float beta, float *C, const int ldc);
void mncblas_sgemm3(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A,
                   const int lda, const float *B, const int ldb,
                   const float beta, float *C, const int ldc);
void mncblas_sgemm4(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A,
                   const int lda, const float *B, const int ldb,
                   const float beta, float *C, const int ldc);

void mncblas_dgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const double alpha, const double *A,
                   const int lda, const double *B, const int ldb,
                   const double beta, double *C, const int ldc);

void mncblas_cgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const void *alpha, const void *A,
                   const int lda, const void *B, const int ldb,
                   const void *beta, void *C, const int ldc);

void mncblas_zgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const void *alpha, const void *A,
                   const int lda, const void *B, const int ldb,
                   const void *beta, void *C, const int ldc);
