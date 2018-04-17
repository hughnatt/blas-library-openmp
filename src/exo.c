#include <omp.h>
#include <stdio.h>

#include <x86intrin.h>

#define NBEXPERIMENTS 32
static long long unsigned int experiments[NBEXPERIMENTS];

// Type complexe

typedef struct {
  double real;
  double imaginary;
} complexe;

#define VECTOR_SIZE 8192
#define MATRIX_SIZE 256

#define TILE 16

typedef double vector[VECTOR_SIZE];

typedef complexe vector_complexe[VECTOR_SIZE];

typedef double matrix[MATRIX_SIZE][MATRIX_SIZE];

static vector a, b, c;
static vector_complexe d, e, f;

static matrix M1, M2, M3;

long long unsigned int average(long long unsigned int *exps) {
  unsigned int i;
  long long unsigned int s = 0;

  for (i = 2; i < (NBEXPERIMENTS - 2); i++) {
    s = s + exps[i];
  }

  return s / (NBEXPERIMENTS - 2);
}

void init_vector_complexe(complexe *t, double re, double im) {
  int i;

  for (i = 0; i < VECTOR_SIZE; i++) {
    t[i].real = re;
    t[i].imaginary = im;
  }
}

void init_vector(vector X, const double val) {
  register unsigned int i;

  for (i = 0; i < VECTOR_SIZE; i++)
    X[i] = val;

  return;
}

void init_matrix(matrix X, const double val) {
  register unsigned int i, j;

  for (i = 0; i < MATRIX_SIZE; i++) {
    for (j = 0; j < MATRIX_SIZE; j++) {
      X[i][j] = val;
    }
  }
}

void print_vectors(vector X, vector Y) {
  register unsigned int i;

  for (i = 0; i < VECTOR_SIZE; i++)
    printf(" X [%d] = %le Y [%d] = %le\n", i, X[i], i, Y[i]);

  return;
}

void add_vectors0(vector X, vector Y, vector Z) {
  register unsigned int i;

  for (i = 0; i < VECTOR_SIZE; i++)
    X[i] = Y[i] + Z[i];

  return;
}

void add_vectors1(vector X, vector Y, vector Z) {
  register unsigned int i;

#pragma omp parallel for schedule(static, 64)
  for (i = 0; i < VECTOR_SIZE; i++)
    X[i] = Y[i] + Z[i];

  return;
}

void add_vectors2(vector X, vector Y, vector Z) {
  register unsigned int i;

#pragma omp parallel for schedule(dynamic, 64)
  for (i = 0; i < VECTOR_SIZE; i++)
    X[i] = Y[i] + Z[i];

  return;
}

double dot0(vector X, vector Y) {
  register unsigned int i;
  register double dot;

  dot = 0.0;
  for (i = 0; i < VECTOR_SIZE; i++)
    dot += X[i] * Y[i];

  return dot;
}

double dot1(vector X, vector Y) {
  register unsigned int i;
  register double dot;

  dot = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dot)
  for (i = 0; i < VECTOR_SIZE; i++)
    dot += X[i] * Y[i];

  return dot;
}

double dot2(vector X, vector Y) {
  register unsigned int i;
  register double dot;

  dot = 0.0;
#pragma omp parallel for schedule(dynamic, 32) reduction(+ : dot)
  for (i = 0; i < VECTOR_SIZE; i++)
    dot += X[i] * Y[i];

  return dot;
}

double dot3(vector X, vector Y) {
  register unsigned int i;
  register double dot;

  dot = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dot)
  for (i = 0; i < VECTOR_SIZE; i = i + 8) {
    dot += X[i] * Y[i];
    dot += X[i + 1] * Y[i + 1];
    dot += X[i + 2] * Y[i + 2];
    dot += X[i + 3] * Y[i + 3];

    dot += X[i + 4] * Y[i + 4];
    dot += X[i + 5] * Y[i + 5];
    dot += X[i + 6] * Y[i + 6];
    dot += X[i + 7] * Y[i + 7];
  }

  return dot;
}

complexe dotc0(vector_complexe X, vector_complexe Y) {
  int i;
  double re, im;
  complexe resultat;

  re = 0.0;
  im = 0.0;

  for (i = 0; i < VECTOR_SIZE; i++) {
    re += (X[i].real * Y[i].real) - (X[i].imaginary * Y[i].imaginary);
    im += (X[i].real * Y[i].imaginary) + (X[i].imaginary * Y[i].real);
  }

  resultat.real = re;
  resultat.imaginary = im;

  return resultat;
}

complexe dotc1(vector_complexe X, vector_complexe Y) {
  int i;
  double re, im;
  complexe resultat;

  re = 0.0;
  im = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : re) reduction(+ : im)
  for (i = 0; i < VECTOR_SIZE; i++) {
    re += (X[i].real * Y[i].real) - (X[i].imaginary * Y[i].imaginary);
    im += (X[i].real * Y[i].imaginary) + (X[i].imaginary * Y[i].real);
  }

  resultat.real = re;
  resultat.imaginary = im;

  return resultat;
}

void mat_mult0(matrix A, matrix B, matrix C) {
  int i, j, k;
  double r;

  for (i = 0; i < MATRIX_SIZE; i++) {
    for (j = 0; j < MATRIX_SIZE; j++) {
      r = 0.0;
      for (k = 0; k < MATRIX_SIZE; k++) {
        r = r + (A[i][k] * B[k][j]);
      }
      C[i][j] = r;
    }
  }
  return;
}

void mat_mult1(matrix A, matrix B, matrix C) {
  int i, j, k;
  double r;

#pragma omp parallel for private(j, k, r)
  for (i = 0; i < MATRIX_SIZE; i++) {
    for (j = 0; j < MATRIX_SIZE; j++) {
      r = 0.0;
      for (k = 0; k < MATRIX_SIZE; k++) {
        r = r + (A[i][k] * B[k][j]);
      }
      C[i][j] = r;
    }
  }
  return;
}

int main() {
  int nthreads, maxnthreads;

  int tid;

  unsigned long long int start, end;
  unsigned long long int residu;

  unsigned long long int av;

  double r;

  complexe produit_scalaire_complexe;

  int exp;

  /*
     rdtsc: read the cycle counter
  */

  start = _rdtsc();
  end = _rdtsc();
  residu = end - start;

  /*
    Vector Initialization
  */

  printf("=============== ADD ==========================================\n");

  init_vector(a, 1.0);
  init_vector(b, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    add_vectors0(c, a, b);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("sequential loop \t %Ld cycles\n", av - residu);

  /*
    print_vectors (a, b) ;
  */

  init_vector(a, 1.0);
  init_vector(b, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    add_vectors1(c, a, b);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("OpenMP static loop \t %Ld cycles\n", av - residu);

  init_vector(a, 1.0);
  init_vector(b, 2.0);

  /*
    print_vectors (a, b) ;
  */

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    add_vectors2(c, a, b);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("OpenMP dynamic loop \t %Ld cycles\n", av - residu);

  printf("==============================================================\n");

  printf("====================DOT =====================================\n");

  init_vector(a, 1.0);
  init_vector(b, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    r = dot0(a, b);

    end = _rdtsc();

    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("Sequential dot: r %5.2f \t\t\t %Ld cycles\n", r, av - residu);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    r = dot1(a, b);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("OpenMP static loop dot: r %5.2f \t\t %Ld cycles\n", r, av - residu);

  init_vector(a, 1.0);
  init_vector(b, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    r = dot2(a, b);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("OpenMP dynamic loop dot: r %5.2f \t\t %Ld cycles\n", r, av - residu);

  init_vector(a, 1.0);
  init_vector(b, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    r = dot3(a, b);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("OpenMP static unrolled loop dot: r %5.2f \t %Ld cycles\n", r,
         av - residu);

  printf("====================DOT COMPLEXE "
         "=====================================\n");

  init_vector_complexe(d, 2.0, 3.0);
  init_vector_complexe(e, 1.0, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    produit_scalaire_complexe = dotc0(d, e);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("dotc0 complexe: %5.2f %5.2f \t\t %Ld cycles\n",
         produit_scalaire_complexe.real, produit_scalaire_complexe.imaginary,
         av - residu);

  init_vector_complexe(d, 2.0, 3.0);
  init_vector_complexe(e, 1.0, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    produit_scalaire_complexe = dotc1(d, e);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("OpenMP dotc1 complexe: %5.2f %5.2f \t %Ld cycles\n",
         produit_scalaire_complexe.real, produit_scalaire_complexe.imaginary,
         av - residu);

  printf(
      "====================MAT_MULT  =====================================\n");
  init_matrix(M1, 1.0);
  init_matrix(M2, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    mat_mult0(M1, M2, M3);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("sequential matmult0: \t\t %Ld cycles\n", av - residu);

  init_matrix(M1, 1.0);
  init_matrix(M2, 2.0);

  for (exp = 0; exp < NBEXPERIMENTS; exp++) {
    start = _rdtsc();

    mat_mult1(M1, M2, M3);

    end = _rdtsc();
    experiments[exp] = end - start;
  }

  av = average(experiments);

  printf("OpenMP matmult1: \t\t %Ld cycles\n", av - residu);

  return 0;
}
