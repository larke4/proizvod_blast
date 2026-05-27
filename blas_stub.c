/* Базовая BLAS-подобная библиотека для сравнения */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Константы */
#define CBLAS_ROW_MAJOR 101
#define CBLAS_COL_MAJOR 102
#define CBLAS_LEFT 141
#define CBLAS_RIGHT 142
#define CBLAS_UPPER 121
#define CBLAS_LOWER 122
#define CBLAS_NO_TRANS 111
#define CBLAS_TRANS 112

/* Референсная реализация TRMM для сравнения */
void ref_strmm_row_lower_nont(int M, int N, float alpha,
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

void ref_strmm_row_upper_nont(int M, int N, float alpha,
                               const float *A, int lda,
                               float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = i; k < M; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Упрощённая реализация с блочной обработкой */
void opt_strmm_row_lower_nont(int M, int N, float alpha,
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

/* Оптимизированная версия с register blocking */
void opt_strmm_reg_block(int M, int N, float alpha,
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

/* Простая векторизованная версия (имитация SIMD) */
void simd_strmm_row_lower(int M, int N, float alpha,
                          const float *A, int lda,
                          float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M; i++) {
        int j = 0;
        /* Векторизованный цикл (имитация) */
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

#endif
