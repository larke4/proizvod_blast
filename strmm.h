#ifndef STRMM_H
#define STRMM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Константы, совместимые с CBLAS */
#define CblasRowMajor 101
#define CblasColMajor 102
#define CblasLeft 141
#define CblasRight 142
#define CblasUpper 121
#define CblasLower 122
#define CblasNoTrans 111
#define CblasTrans 112
#define CblasConjTrans 113
#define CblasUnit 131
#define CblasNonUnit 132

/* Одинарная точность */
void cblas_strmm(const int Order, const int Side, const int Uplo,
                 const int TransA, const int Diag,
                 const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb);

/* Двойная точность */
void cblas_dtrmm(const int Order, const int Side, const int Uplo,
                 const int TransA, const int Diag,
                 const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb);

#ifdef __cplusplus
}
#endif

#endif /* STRMM_H */
