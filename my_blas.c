/* Упрощённая BLAS библиотека для сравнения производительности */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Константы, совместимые с CBLAS */
#define CBLAS_ROW_MAJOR 101
#define CBLAS_COL_MAJOR 102
#define CBLAS_LEFT 141
#define CBLAS_RIGHT 142
#define CBLAS_UPPER 121
#define CBLAS_LOWER 122
#define CBLAS_NO_TRANS 111
#define CBLAS_TRANS 112
#define CBLAS_CONJ_TRANS 113
#define CBLAS_UNIT 131
#define CBLAS_NON_UNIT 132

/* ============================================================
   Naive реализация TRMM (простая, без оптимизаций)
   ============================================================ */

void my_strmm_naive(int M, int N, float alpha,
                    const float *A, int lda,
                    float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* ============================================================
   Блочная реализация (для лучшего использования кэша)
   ============================================================ */

void my_strmm_blocked(int M, int N, float alpha,
                      const float *A, int lda,
                      float *B, int ldb) {
    const int BLOCK = 64;
    float *C = (float*)malloc(M * N * sizeof(float));
    
    for (int ib = 0; ib < M; ib += BLOCK) {
        int i_end = (ib + BLOCK < M) ? ib + BLOCK : M;
        for (int jb = 0; jb < N; jb += BLOCK) {
            int j_end = (jb + BLOCK < N) ? jb + BLOCK : N;
            for (int i = ib; i < i_end; i++) {
                for (int j = jb; j < j_end; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k <= i; k++) {
                        sum += A[i * lda + k] * B[k * ldb + j];
                    }
                    C[i * ldb + j] = alpha * sum;
                }
            }
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* ============================================================
   Оптимизированная версия с register blocking
   ============================================================ */

void my_strmm_opt(int M, int N, float alpha,
                  const float *A, int lda,
                  float *B, int ldb) {
    const int MR = 4, NR = 4;
    float *C = (float*)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            for (int ii = i; ii < i + MR && ii < M; ii++) {
                for (int jj = j; jj < j + NR && jj < N; jj++) {
                    float sum = 0.0f;
                    for (int k = 0; k <= ii; k++) {
                        sum += A[ii * lda + k] * B[k * ldb + jj];
                    }
                    C[ii * ldb + jj] = alpha * sum;
                }
            }
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* ============================================================
   Векторизованная версия (имитация SIMD)
   ============================================================ */

void my_strmm_simd(int M, int N, float alpha,
                   const float *A, int lda,
                   float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M; i++) {
        int j = 0;
        /* Векторизованный цикл (имитация AVX) */
        for (; j + 3 < N; j += 4) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k <= i; k++) {
                float ak = A[i * lda + k];
                s0 += ak * B[k * ldb + j];
                s1 += ak * B[k * ldb + j + 1];
                s2 += ak * B[k * ldb + j + 2];
                s3 += ak * B[k * ldb + j + 3];
            }
            C[i * ldb + j] = alpha * s0;
            C[i * ldb + j + 1] = alpha * s1;
            C[i * ldb + j + 2] = alpha * s2;
            C[i * ldb + j + 3] = alpha * s3;
        }
        /* Остаток */
        for (; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Обёртка для совместимости с cblas_strmm */
void cblas_strmm_naive(int Order, int Side, int Uplo,
                       int TransA, int Diag,
                       int M, int N,
                       float alpha, const float *A, int lda,
                       float *B, int ldb) {
    if (Order == CBLAS_ROW_MAJOR && Side == CBLAS_LEFT && 
        Uplo == CBLAS_LOWER && TransA == CBLAS_NO_TRANS) {
        my_strmm_naive(M, N, alpha, A, lda, B, ldb);
    }
}

void cblas_strmm_blocked(int Order, int Side, int Uplo,
                         int TransA, int Diag,
                         int M, int N,
                         float alpha, const float *A, int lda,
                         float *B, int ldb) {
    if (Order == CBLAS_ROW_MAJOR && Side == CBLAS_LEFT && 
        Uplo == CBLAS_LOWER && TransA == CBLAS_NO_TRANS) {
        my_strmm_blocked(M, N, alpha, A, lda, B, ldb);
    }
}

void cblas_strmm_opt(int Order, int Side, int Uplo,
                     int TransA, int Diag,
                     int M, int N,
                     float alpha, const float *A, int lda,
                     float *B, int ldb) {
    if (Order == CBLAS_ROW_MAJOR && Side == CBLAS_LEFT && 
        Uplo == CBLAS_LOWER && TransA == CBLAS_NO_TRANS) {
        my_strmm_opt(M, N, alpha, A, lda, B, ldb);
    }
}

void cblas_strmm_simd(int Order, int Side, int Uplo,
                      int TransA, int Diag,
                      int M, int N,
                      float alpha, const float *A, int lda,
                      float *B, int ldb) {
    if (Order == CBLAS_ROW_MAJOR && Side == CBLAS_LEFT && 
        Uplo == CBLAS_LOWER && TransA == CBLAS_NO_TRANS) {
        my_strmm_simd(M, N, alpha, A, lda, B, ldb);
    }
}
