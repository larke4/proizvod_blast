#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "strmm.h"

#define EPS 1e-5f
#define EPS_D 1e-10

/* Инициализация треугольной матрицы */
void init_triangular(int n, float *A, int lda, int uplo, int diag) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (uplo == CblasLower && j <= i) {
                A[i * lda + j] = (diag == CblasUnit && i == j) ? 1.0f : (float)(i + j + 1);
            } else if (uplo == CblasUpper && j >= i) {
                A[i * lda + j] = (diag == CblasUnit && i == j) ? 1.0f : (float)(i + j + 1);
            } else {
                A[i * lda + j] = 0.0f;
            }
        }
    }
}

void init_triangular_d(int n, double *A, int lda, int uplo, int diag) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (uplo == CblasLower && j <= i) {
                A[i * lda + j] = (diag == CblasUnit && i == j) ? 1.0 : (double)(i + j + 1);
            } else if (uplo == CblasUpper && j >= i) {
                A[i * lda + j] = (diag == CblasUnit && i == j) ? 1.0 : (double)(i + j + 1);
            } else {
                A[i * lda + j] = 0.0;
            }
        }
    }
}

/* Инициализация матрицы B */
void init_matrix(int rows, int cols, float *mat, int lda) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * lda + j] = (float)(i * cols + j + 1);
        }
    }
}

void init_matrix_d(int rows, int cols, double *mat, int lda) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * lda + j] = (double)(i * cols + j + 1);
        }
    }
}

void init_matrix_small(int rows, int cols, float *mat, int lda) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * lda + j] = (float)(i + 1) * 0.1f;
        }
    }
}

/* Печать матрицы (для отладки) */
void print_matrix(const char *name, int rows, int cols, float *mat, int lda) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3f ", mat[i * lda + j]);
        }
        printf("\n");
    }
}

void print_matrix_d(const char *name, int rows, int cols, double *mat, int lda) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%10.6f ", mat[i * lda + j]);
        }
        printf("\n");
    }
}

int almost_equal(float a, float b) {
    return fabsf(a - b) < EPS;
}

int almost_equal_d(double a, double b) {
    return fabs(a - b) < EPS_D;
}

/* ============================================================
   Эталонные реализации для проверки
   ============================================================ */

/* Референс: Left, Lower, NoTrans, NonUnit */
void ref_strmm_left_lower_nont_nounit(int M, int N, float alpha,
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

/* Референс: Left, Lower, NoTrans, Unit */
void ref_strmm_left_lower_nont_unit(int M, int N, float alpha,
                                    const float *A, int lda,
                                    float *B, int ldb) {
    float *C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
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

/* Референс: Left, Upper, NoTrans, NonUnit */
void ref_strmm_left_upper_nont_nounit(int M, int N, float alpha,
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

/* Референс: Right, Upper, NoTrans, NonUnit */
void ref_strmm_right_upper_nont_nounit(int M, int N, float alpha,
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

/* Референс: Right, Lower, NoTrans, NonUnit */
void ref_strmm_right_lower_nont_nounit(int M, int N, float alpha,
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

/* Референс: Left, Lower, Trans, NonUnit (для RowMajor) */
void ref_strmm_left_lower_trans_nounit(int M, int N, float alpha,
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

/* Референс: Left, Upper, Trans, NonUnit (для RowMajor) */
void ref_strmm_left_upper_trans_nounit(int M, int N, float alpha,
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

/* ============================================================
   Тесты
   ============================================================ */

int test_1_left_lower_nont_nounit(void) {
    int M = 4, N = 3;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasLower, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, 1.0f, A, M, B, N);
    ref_strmm_left_lower_nont_nounit(M, N, 1.0f, A, M, B_ref, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int test_2_left_lower_nont_unit(void) {
    int M = 3, N = 4;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasLower, CblasUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                M, N, 2.0f, A, M, B, N);
    
    /* Эталон с временным буфером */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) {
                sum += A[i * M + k] * B_ref[k * N + j];
            }
            sum += B_ref[i * N + j];
            C[i * N + j] = 2.0f * sum;
        }
    }
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], C[i * N + j])) {
                printf("FAIL at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], C[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref); free(C);
    return ok;
}

int test_3_left_upper_nont_nounit(void) {
    int M = 4, N = 3;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasUpper, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                M, N, 1.0f, A, M, B, N);
    ref_strmm_left_upper_nont_nounit(M, N, 1.0f, A, M, B_ref, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int test_4_right_upper_nont_nounit(void) {
    int M = 3, N = 4;
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(N, A, N, CblasUpper, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                M, N, 1.0f, A, N, B, N);
    ref_strmm_right_upper_nont_nounit(M, N, 1.0f, A, N, B_ref, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int test_5_right_lower_nont_nounit(void) {
    int M = 3, N = 4;
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(N, A, N, CblasLower, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, 1.0f, A, N, B, N);
    ref_strmm_right_lower_nont_nounit(M, N, 1.0f, A, N, B_ref, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int test_6_left_lower_trans_nounit(void) {
    int M = 4, N = 3;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasLower, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                M, N, 1.0f, A, M, B, N);
    ref_strmm_left_lower_trans_nounit(M, N, 1.0f, A, M, B_ref, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int test_7_left_upper_trans_nounit(void) {
    int M = 4, N = 3;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasUpper, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                M, N, 1.0f, A, M, B, N);
    ref_strmm_left_upper_trans_nounit(M, N, 1.0f, A, M, B_ref, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int test_8_alpha_zero(void) {
    int M = 3, N = 2;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasLower, CblasNonUnit);
    init_matrix(M, N, B, N);
    
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, 0.0f, A, M, B, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (B[i * N + j] != 0.0f) {
                printf("FAIL alpha=0 at (%d,%d): %f\n", i, j, B[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B);
    return ok;
}

int test_9_colmajor_left_lower(void) {
    int M = 4, N = 3;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasLower, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, 1.0f, A, M, B, N);
    
    /* Референс для ColumnMajor: B[i,j] = alpha * sum_{k=0..i} A[k,i] * B[k,j] */
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                sum += A[k * M + i] * B_ref[k * N + j];
            }
            B_ref[i * N + j] = sum;
        }
    }
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL CM at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int test_10_colmajor_right_upper(void) {
    int M = 3, N = 4;
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(N, A, N, CblasUpper, CblasNonUnit);
    init_matrix(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                M, N, 1.0f, A, N, B, N);
    
    /* Референс для ColumnMajor, Right, Upper с временным буфером */
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k <= j; k++) {
                sum += B_ref[i * N + k] * A[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], C[i * N + j])) {
                printf("FAIL CM Right at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], C[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref); free(C);
    return ok;
}

/* Тест двойной точности */
int test_11_double_left_lower(void) {
    int M = 4, N = 3;
    double *A = (double*)malloc(M * M * sizeof(double));
    double *B = (double*)malloc(M * N * sizeof(double));
    double *B_ref = (double*)malloc(M * N * sizeof(double));
    
    init_triangular_d(M, A, M, CblasLower, CblasNonUnit);
    init_matrix_d(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(double));
    
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, 1.0, A, M, B, N);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) {
                sum += A[i * M + k] * B_ref[k * N + j];
            }
            B_ref[i * N + j] = sum;
        }
    }
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal_d(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL D at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

/* Тест прямоугольной матрицы */
int test_12_rectangular(void) {
    int M = 8, N = 5;
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_ref = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M, CblasLower, CblasNonUnit);
    init_matrix_small(M, N, B, N);
    memcpy(B_ref, B, M * N * sizeof(float));
    
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, 2.5f, A, M, B, N);
    ref_strmm_left_lower_nont_nounit(M, N, 2.5f, A, M, B_ref, N);
    
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            if (!almost_equal(B[i * N + j], B_ref[i * N + j])) {
                printf("FAIL Rect at (%d,%d): my=%f, ref=%f\n", i, j, B[i * N + j], B_ref[i * N + j]);
                ok = 0;
            }
        }
    }
    
    free(A); free(B); free(B_ref);
    return ok;
}

int main(void) {
    int passed = 0, total = 0;
    
    printf("=== TRMM Correctness Tests ===\n\n");
    
    printf("Test 1: Left, Lower, NoTrans, NonUnit... ");
    total++; if (test_1_left_lower_nont_nounit()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 2: Left, Lower, NoTrans, Unit (alpha=2.0)... ");
    total++; if (test_2_left_lower_nont_unit()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 3: Left, Upper, NoTrans, NonUnit... ");
    total++; if (test_3_left_upper_nont_nounit()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 4: Right, Upper, NoTrans, NonUnit... ");
    total++; if (test_4_right_upper_nont_nounit()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 5: Right, Lower, NoTrans, NonUnit... ");
    total++; if (test_5_right_lower_nont_nounit()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 6: Left, Lower, Trans, NonUnit... ");
    total++; if (test_6_left_lower_trans_nounit()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 7: Left, Upper, Trans, NonUnit... ");
    total++; if (test_7_left_upper_trans_nounit()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 8: alpha=0 (обнуление)... ");
    total++; if (test_8_alpha_zero()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 9: ColumnMajor, Left, Lower, NonUnit... ");
    total++; if (test_9_colmajor_left_lower()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 10: ColumnMajor, Right, Upper, NonUnit... ");
    total++; if (test_10_colmajor_right_upper()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 11: Double precision, Left, Lower, NonUnit... ");
    total++; if (test_11_double_left_lower()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("Test 12: Rectangular matrix (M=8, N=5)... ");
    total++; if (test_12_rectangular()) { printf("PASSED\n"); passed++; } else printf("FAILED\n");
    
    printf("\n=== Summary: %d/%d tests passed ===\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}
