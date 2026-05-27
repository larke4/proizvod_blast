/* Benchmark comparing different TRMM implementations */
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

#define NUM_RUNS 10

void init_triangular(int n, float *A, int lda) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j <= i) {
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

/* Обёртки для cblas_strmm */
void strmm_wrapper(int M, int N, float alpha, const float *A, int lda, float *B, int ldb) {
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                M, N, alpha, A, lda, B, ldb);
}

/* Прототипы из my_blas.c */
void my_strmm_naive(int M, int N, float alpha, const float *A, int lda, float *B, int ldb);
void my_strmm_blocked(int M, int N, float alpha, const float *A, int lda, float *B, int ldb);
void my_strmm_opt(int M, int N, float alpha, const float *A, int lda, float *B, int ldb);
void my_strmm_simd(int M, int N, float alpha, const float *A, int lda, float *B, int ldb);

typedef struct {
    double avg_time;
    double geom_time;
    double gflops;
} Result;

Result benchmark(const char *name, int M, int N,
                 void (*func)(int,int,float,const float*,int,float*,int),
                 int num_runs) {
    float *A = (float*)malloc(M * M * sizeof(float));
    float *B = (float*)malloc(M * N * sizeof(float));
    float *B_copy = (float*)malloc(M * N * sizeof(float));
    
    init_triangular(M, A, M);
    init_matrix(M, N, B, N);
    
    double times[20];
    
    printf("  %-25s: ", name);
    for (int run = 0; run < num_runs; run++) {
        memcpy(B_copy, B, M * N * sizeof(float));
        
        double start = get_time();
        func(M, N, 1.0f, A, M, B_copy, N);
        double end = get_time();
        
        times[run] = end - start;
    }
    
    double sum = 0, geom = 1;
    for (int i = 0; i < num_runs; i++) {
        sum += times[i];
        geom *= times[i];
    }
    double avg = sum / num_runs;
    geom = pow(geom, 1.0 / num_runs);
    
    double flops = 2.0 * M * M * N / 1e9;
    double gflops = flops / avg;
    
    printf("%.3f sec (%.2f GFLOPS)\n", avg, gflops);
    
    free(A); free(B); free(B_copy);
    
    Result r = { avg, geom, gflops };
    return r;
}

int main() {
    printf("=== TRMM Implementation Comparison ===\n\n");
    
    int M = 2048, N = 2048;
    printf("Matrix size: %d x %d\n", M, M);
    printf("Number of runs: %d\n\n", NUM_RUNS);
    
    Result results[5];
    int idx = 0;
    
    printf("Running tests...\n\n");
    
    results[idx++] = benchmark("Naive", M, N, my_strmm_naive, NUM_RUNS);
    results[idx++] = benchmark("Blocked (64)", M, N, my_strmm_blocked, NUM_RUNS);
    results[idx++] = benchmark("Register Block 4x4", M, N, my_strmm_opt, NUM_RUNS);
    results[idx++] = benchmark("SIMD-like (4-wide)", M, N, my_strmm_simd, NUM_RUNS);
    results[idx++] = benchmark("cblas_strmm (full)", M, N, strmm_wrapper, NUM_RUNS);
    
    printf("\n=== Summary Table ===\n\n");
    printf("| Implementation          | Time (sec)  | GFLOPS    | vs Naive     |\n");
    printf("|-------------------------|-------------|-----------|--------------|\n");
    
    double naive_time = results[0].avg_time;
    for (int i = 0; i < idx; i++) {
        double rel = (naive_time / results[i].avg_time) * 100;
        const char *names[] = {"Naive", "Blocked", "RegBlock 4x4", "SIMD-like", "cblas_strmm"};
        printf("| %-23s | %9.3f | %9.2f | %11.1f%% |\n",
               names[i], results[i].avg_time, results[i].gflops, rel);
    }
    
    printf("\n=== Speedup ===\n");
    printf("Blocked vs Naive:      %.1fx\n", naive_time / results[1].avg_time);
    printf("RegBlock vs Naive:     %.1fx\n", naive_time / results[2].avg_time);
    printf("SIMD vs Naive:         %.1fx\n", naive_time / results[3].avg_time);
    printf("cblas_strmm vs Naive:  %.1fx\n", naive_time / results[4].avg_time);
    
    return 0;
}
