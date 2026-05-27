#include "strmm.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================
   Вспомогательные функции для одинарной точности
   ============================================================ */

/* Left side, Lower triangular, NoTrans, NonUnit (RowMajor) */
static void strmm_left_lower_nont_nounit_rm(int M, int N, float alpha,
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

/* Left side, Lower triangular, NoTrans, Unit (RowMajor) */
static void strmm_left_lower_nont_unit_rm(int M, int N, float alpha,
                                          const float *A, int lda,
                                          float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            sum += B[i * ldb + j]; /* Диагональ = 1 */
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Upper triangular, NoTrans, NonUnit (RowMajor) */
static void strmm_left_upper_nont_nounit_rm(int M, int N, float alpha,
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

/* Left side, Upper triangular, NoTrans, Unit (RowMajor) */
static void strmm_left_upper_nont_unit_rm(int M, int N, float alpha,
                                          const float *A, int lda,
                                          float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = i + 1; k < M; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Lower triangular, Trans, NonUnit (RowMajor) */
static void strmm_left_lower_trans_nounit_rm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = i; k < M; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Lower triangular, Trans, Unit (RowMajor) */
static void strmm_left_lower_trans_unit_rm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = i + 1; k < M; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Upper triangular, Trans, NonUnit (RowMajor) */
static void strmm_left_upper_trans_nounit_rm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Upper triangular, Trans, Unit (RowMajor) */
static void strmm_left_upper_trans_unit_rm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Lower triangular, NoTrans, NonUnit (RowMajor): B = alpha * B * A */
static void strmm_right_lower_nont_nounit_rm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = j; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Lower triangular, NoTrans, Unit (RowMajor) */
static void strmm_right_lower_nont_unit_rm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = j + 1; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper triangular, NoTrans, NonUnit (RowMajor) */
static void strmm_right_upper_nont_nounit_rm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper triangular, NoTrans, Unit (RowMajor) */
static void strmm_right_upper_nont_unit_rm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Lower triangular, Trans, NonUnit (RowMajor) */
static void strmm_right_lower_trans_nounit_rm(int M, int N, float alpha,
                                              const float *A, int lda,
                                              float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= j; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Lower triangular, Trans, Unit (RowMajor) */
static void strmm_right_lower_trans_unit_rm(int M, int N, float alpha,
                                            const float *A, int lda,
                                            float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper triangular, Trans, NonUnit (RowMajor) */
static void strmm_right_upper_trans_nounit_rm(int M, int N, float alpha,
                                              const float *A, int lda,
                                              float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = j; k < N; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper triangular, Trans, Unit (RowMajor) */
static void strmm_right_upper_trans_unit_rm(int M, int N, float alpha,
                                            const float *A, int lda,
                                            float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = j + 1; k < N; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* ============================================================
   ColumnMajor реализации для Left side
   ============================================================ */

/* Left side, Lower, NoTrans, NonUnit (ColumnMajor) */
static void strmm_left_lower_nont_nounit_cm(int M, int N, float alpha,
                                            const float *A, int lda,
                                            float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Lower, NoTrans, Unit (ColumnMajor) */
static void strmm_left_lower_nont_unit_cm(int M, int N, float alpha,
                                          const float *A, int lda,
                                          float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Upper, NoTrans, NonUnit (ColumnMajor) */
static void strmm_left_upper_nont_nounit_cm(int M, int N, float alpha,
                                            const float *A, int lda,
                                            float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = i; k < M; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Upper, NoTrans, Unit (ColumnMajor) */
static void strmm_left_upper_nont_unit_cm(int M, int N, float alpha,
                                          const float *A, int lda,
                                          float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = i + 1; k < M; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Lower, Trans, NonUnit (ColumnMajor) */
static void strmm_left_lower_trans_nounit_cm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
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

/* Left side, Lower, Trans, Unit (ColumnMajor) */
static void strmm_left_lower_trans_unit_cm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = i + 1; k < M; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Left side, Upper, Trans, NonUnit (ColumnMajor) */
static void strmm_left_upper_trans_nounit_cm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
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

/* Left side, Upper, Trans, Unit (ColumnMajor) */
static void strmm_left_upper_trans_unit_cm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* ============================================================
   ColumnMajor реализации для Right side
   ============================================================ */

/* Right side, Lower, NoTrans, NonUnit (ColumnMajor) */
static void strmm_right_lower_nont_nounit_cm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = j; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Lower, NoTrans, Unit (ColumnMajor) */
static void strmm_right_lower_nont_unit_cm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = j + 1; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper, NoTrans, NonUnit (ColumnMajor) */
static void strmm_right_upper_nont_nounit_cm(int M, int N, float alpha,
                                             const float *A, int lda,
                                             float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k <= j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper, NoTrans, Unit (ColumnMajor) */
static void strmm_right_upper_nont_unit_cm(int M, int N, float alpha,
                                           const float *A, int lda,
                                           float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Lower, Trans, NonUnit (ColumnMajor) */
static void strmm_right_lower_trans_nounit_cm(int M, int N, float alpha,
                                              const float *A, int lda,
                                              float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k <= j; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Lower, Trans, Unit (ColumnMajor) */
static void strmm_right_lower_trans_unit_cm(int M, int N, float alpha,
                                            const float *A, int lda,
                                            float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper, Trans, NonUnit (ColumnMajor) */
static void strmm_right_upper_trans_nounit_cm(int M, int N, float alpha,
                                              const float *A, int lda,
                                              float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = j; k < N; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* Right side, Upper, Trans, Unit (ColumnMajor) */
static void strmm_right_upper_trans_unit_cm(int M, int N, float alpha,
                                            const float *A, int lda,
                                            float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = j + 1; k < N; k++) {
                sum += B[i * ldb + k] * A[j * lda + k];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(float));
    free(C);
}

/* ============================================================
   Главная функция для одинарной точности
   ============================================================ */
void cblas_strmm(const int Order, const int Side, const int Uplo,
                 const int TransA, const int Diag,
                 const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb) {
    
    if (M <= 0 || N <= 0) return;
    
    if (alpha == 0.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                B[i * ldb + j] = 0.0f;
            }
        }
        return;
    }
    
    if (Order == CblasRowMajor) {
        if (Side == CblasLeft) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_lower_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_lower_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_lower_trans_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_lower_trans_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_upper_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_upper_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_upper_trans_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_upper_trans_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        } else if (Side == CblasRight) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_lower_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_lower_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_lower_trans_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_lower_trans_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_upper_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_upper_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_upper_trans_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_upper_trans_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        }
    } else if (Order == CblasColMajor) {
        if (Side == CblasLeft) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_lower_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_lower_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_lower_trans_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_lower_trans_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_upper_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_upper_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_left_upper_trans_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_left_upper_trans_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        } else if (Side == CblasRight) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_lower_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_lower_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_lower_trans_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_lower_trans_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_upper_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_upper_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                } else if (TransA == CblasTrans) {
                    if (Diag == CblasNonUnit) {
                        strmm_right_upper_trans_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        strmm_right_upper_trans_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        }
    }
}

/* ============================================================
   Вспомогательные функции для двойной точности
   ============================================================ */

static void dtrmm_left_lower_nont_nounit_rm(int M, int N, double alpha,
                                            const double *A, int lda,
                                            double *B, int ldb) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_left_lower_nont_unit_rm(int M, int N, double alpha,
                                          const double *A, int lda,
                                          double *B, int ldb) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_left_upper_nont_nounit_rm(int M, int N, double alpha,
                                            const double *A, int lda,
                                            double *B, int ldb) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = i; k < M; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_left_upper_nont_unit_rm(int M, int N, double alpha,
                                          const double *A, int lda,
                                          double *B, int ldb) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = i + 1; k < M; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_right_lower_nont_nounit_rm(int M, int N, double alpha,
                                             const double *A, int lda,
                                             double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = j; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

static void dtrmm_right_lower_nont_unit_rm(int M, int N, double alpha,
                                           const double *A, int lda,
                                           double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = j + 1; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

static void dtrmm_right_upper_nont_nounit_rm(int M, int N, double alpha,
                                             const double *A, int lda,
                                             double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

static void dtrmm_right_upper_nont_unit_rm(int M, int N, double alpha,
                                           const double *A, int lda,
                                           double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

/* ColumnMajor для двойной точности */
static void dtrmm_left_lower_nont_nounit_cm(int M, int N, double alpha,
                                            const double *A, int lda,
                                            double *B, int ldb) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_left_lower_nont_unit_cm(int M, int N, double alpha,
                                          const double *A, int lda,
                                          double *B, int ldb) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_left_upper_nont_nounit_cm(int M, int N, double alpha,
                                            const double *A, int lda,
                                            double *B, int ldb) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = i; k < M; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_left_upper_nont_unit_cm(int M, int N, double alpha,
                                          const double *A, int lda,
                                          double *B, int ldb) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = i + 1; k < M; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            sum += B[i * ldb + j];
            B[i * ldb + j] = alpha * sum;
        }
    }
}

static void dtrmm_right_lower_nont_nounit_cm(int M, int N, double alpha,
                                             const double *A, int lda,
                                             double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = j; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

static void dtrmm_right_lower_nont_unit_cm(int M, int N, double alpha,
                                           const double *A, int lda,
                                           double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = j + 1; k < N; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

static void dtrmm_right_upper_nont_nounit_cm(int M, int N, double alpha,
                                             const double *A, int lda,
                                             double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

static void dtrmm_right_upper_nont_unit_cm(int M, int N, double alpha,
                                           const double *A, int lda,
                                           double *B, int ldb) {
    double *C = (double*)malloc(M * N * sizeof(double));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += B[i * ldb + k] * A[k * lda + j];
            }
            sum += B[i * ldb + j];
            C[i * ldb + j] = alpha * sum;
        }
    }
    memcpy(B, C, M * N * sizeof(double));
    free(C);
}

/* ============================================================
   Главная функция для двойной точности
   ============================================================ */
void cblas_dtrmm(const int Order, const int Side, const int Uplo,
                 const int TransA, const int Diag,
                 const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb) {
    
    if (M <= 0 || N <= 0) return;
    
    if (alpha == 0.0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                B[i * ldb + j] = 0.0;
            }
        }
        return;
    }
    
    if (Order == CblasRowMajor) {
        if (Side == CblasLeft) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_left_lower_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_left_lower_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_left_upper_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_left_upper_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        } else if (Side == CblasRight) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_right_lower_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_right_lower_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_right_upper_nont_nounit_rm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_right_upper_nont_unit_rm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        }
    } else if (Order == CblasColMajor) {
        if (Side == CblasLeft) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_left_lower_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_left_lower_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_left_upper_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_left_upper_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        } else if (Side == CblasRight) {
            if (Uplo == CblasLower) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_right_lower_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_right_lower_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            } else if (Uplo == CblasUpper) {
                if (TransA == CblasNoTrans) {
                    if (Diag == CblasNonUnit) {
                        dtrmm_right_upper_nont_nounit_cm(M, N, alpha, A, lda, B, ldb);
                    } else {
                        dtrmm_right_upper_nont_unit_cm(M, N, alpha, A, lda, B, ldb);
                    }
                }
            }
        }
    }
}