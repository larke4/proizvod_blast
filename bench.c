#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "strmm.h"

#ifdef _WIN32
#include <windows.h>
double get_time() {
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER t;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / freq.QuadPart;
}
#else
#include <time.h>
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

#define MIN_RUN_TIME 60.0  /* Минимальное время теста в секундах */
#define NUM_RUNS 10        /* Количество запусков для усреднения */

void init_triangular(int n, float *A, int lda, int uplo) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (uplo == CblasLower && j <= i) {
                A[i * lda + j] = (float)(i + j + 1);
            } else if (uplo == CblasUpper && j >= i) {
                A[i * lda + j] = (float)(i + j + 1);
            } else {
                A[i * lda + j] = 0.0f;
            }
        }
    }
}

void init_matrix(int rows, int cols, float *mat, int lda) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * lda + j] = (float)(i * cols + j + 1);
        }
    }
}

typedef struct {
    double my_time;
    double my_geom;
    int found_better;
} BenchResult;

BenchResult run_benchmark(const char *name, int M, int N,
                          void (*my_func)(int,int,float,const float*,int,float*,int),
                          int num_runs) {
    float *A = (float*)_aligned_malloc(M * M * sizeof(float), 64);
    float *B = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    float *B_copy = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    
    init_triangular(M, A, M, CblasLower);
    init_matrix(M, N, B, N);
    
    double times[20];
    double my_total = 0;
    
    for (int run = 0; run < num_runs; run++) {
        memcpy(B_copy, B, M * N * sizeof(float));
        
        double start = get_time();
        my_func(M, N, 1.0f, A, M, B_copy, N);
        double end = get_time();
        
        times[run] = end - start;
        my_total += times[run];
    }
    
    double my_avg = my_total / num_runs;
    
    /* Вычисление геометрического среднего */
    double geom = 1.0;
    for (int i = 0; i < num_runs; i++) {
        geom *= times[i];
    }
    geom = pow(geom, 1.0 / num_runs);
    
    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(B_copy);
    
    BenchResult result = { my_avg, geom, 0 };
    return result;
}

/* Простая реализация для сравнения */
void naive_strmm(int M, int N, float alpha, const float *A, int lda, float *B, int ldb) {
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, alpha, A, lda, B, ldb);
}

int main() {
    printf("=== TRMM Benchmark ===\n\n");
    printf("Минимальное время теста: %.1f сек\n", MIN_RUN_TIME);
    printf("Количество запусков: %d\n\n", NUM_RUNS);
    
    /* Подбор размера матрицы */
    int M = 512, N = 512;
    double test_time;
    
    printf("Подбор размера матрицы...\n");
    float *A = (float*)_aligned_malloc(M * M * sizeof(float), 64);
    float *B = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    float *B_copy = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    
    init_triangular(M, A, M, CblasLower);
    init_matrix(M, N, B, N);
    
    double start = get_time();
    for (int i = 0; i < 5; i++) {
        memcpy(B_copy, B, M * N * sizeof(float));
        cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    M, N, 1.0f, A, M, B_copy, N);
    }
    double single_run = (get_time() - start) / 5;
    
    while (single_run < MIN_RUN_TIME / 10 && M < 4096) {
        M *= 2;
        N = M;
        _aligned_free(A); _aligned_free(B); _aligned_free(B_copy);
        A = (float*)_aligned_malloc(M * M * sizeof(float), 64);
        B = (float*)_aligned_malloc(M * N * sizeof(float), 64);
        B_copy = (float*)_aligned_malloc(M * N * sizeof(float), 64);
        init_triangular(M, A, M, CblasLower);
        init_matrix(M, N, B, N);
        
        start = get_time();
        for (int i = 0; i < 5; i++) {
            memcpy(B_copy, B, M * N * sizeof(float));
            cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                        M, N, 1.0f, A, M, B_copy, N);
        }
        single_run = (get_time() - start) / 5;
    }
    
    printf("Подобранный размер: M = N = %d\n", M);
    printf("Время одного запуска: %.3f сек\n\n", single_run);
    
    _aligned_free(A); _aligned_free(B); _aligned_free(B_copy);
    
    /* Запуск бенчмарка */
    printf("Запуск бенчмарка (%d итераций)...\n", NUM_RUNS);
    printf("Это может занять несколько минут...\n\n");
    
    BenchResult result = run_benchmark("cblas_strmm", M, N, naive_strmm, NUM_RUNS);
    
    printf("=== Результаты ===\n\n");
    printf("Размер матрицы: %d x %d\n", M, M);
    printf("Количество запусков: %d\n\n", NUM_RUNS);
    
    printf("Время выполнения (секунды):\n");
    printf("  Среднее арифметическое: %.4f\n", result.my_time);
    printf("  Среднее геометрическое: %.4f\n\n", result.my_geom);
    
    /* Вывод всех замеров */
    printf("Индивидуальные замеры:\n");
    A = (float*)_aligned_malloc(M * M * sizeof(float), 64);
    B = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    B_copy = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    init_triangular(M, A, M, CblasLower);
    init_matrix(M, N, B, N);
    
    for (int i = 0; i < NUM_RUNS; i++) {
        memcpy(B_copy, B, M * N * sizeof(float));
        double start = get_time();
        naive_strmm(M, N, 1.0f, A, M, B_copy, N);
        double end = get_time();
        printf("  Замер %d: %.4f сек\n", i + 1, end - start);
    }
    
    _aligned_free(A); _aligned_free(B); _aligned_free(B_copy);
    
    /* Теоретическая производительность */
    double flops = 2.0 * M * M * N / 1e9;  /* млрд операций */
    double gflops = flops / result.my_time;
    
    printf("\n=== Производительность ===\n");
    printf("Теоретические FLOPS: %.2f млрд операций\n", flops);
    printf("Производительность: %.2f GFLOPS\n", gflops);
    
    printf("\n=== Рекомендации ===\n");
    printf("Для сравнения с OpenBLAS:\n");
    printf("1. Установите OpenBLAS (через vcpkg, conda или скачайте prebuilt)\n");
    printf("2. Запустите тот же бенчмарк с вызовом OpenBLAS cblas_strmm\n");
    printf("3. Рассчитайте относительную производительность:\n");
    printf("   Relative%% = (my_time / openblas_time) * 100\n");
    printf("4. Вычислите среднее геометрическое по 10 запускам\n");
    
    return 0;
}
