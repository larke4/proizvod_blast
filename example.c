/* Пример использования TRMM */
#include <stdio.h>
#include <stdlib.h>
#include "strmm.h"

void print_matrix(const char *name, int rows, int cols, float *mat, int lda) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3f ", mat[i * lda + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("=== Пример использования cblas_strmm ===\n\n");

    /* Матрица A: 4x4 нижняя треугольная */
    float A[4*4] = {
        2, 0, 0, 0,
        1, 3, 0, 0,
        4, 2, 5, 0,
        1, 1, 1, 2
    };

    /* Матрица B: 4x2 */
    float B[4*2] = {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };

    printf("Исходная матрица A (нижняя треугольная):\n");
    print_matrix("A", 4, 4, A, 4);

    printf("Исходная матрица B:\n");
    print_matrix("B", 4, 2, B, 2);

    /* Вычисляем B = A * B (Left, Lower, NoTrans) */
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower,
                CblasNoTrans, CblasNonUnit,
                4, 2, 1.0f, A, 4, B, 2);

    printf("Результат B = A * B:\n");
    print_matrix("B (результат)", 4, 2, B, 2);

    /* Проверка вручную для первого элемента:
       B[0,0] = A[0,0]*B[0,0] = 2*1 = 2
       B[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 1*1 + 3*3 = 10
       B[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0] = 4*1 + 2*3 + 5*5 = 35
    */
    printf("Ожидаемые значения (первый столбец): 2, 10, 35, ...\n");

    /* Пример с alpha = 2.0 */
    float B2[4*2] = {1, 2, 3, 4, 5, 6, 7, 8};
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower,
                CblasNoTrans, CblasNonUnit,
                4, 2, 2.0f, A, 4, B2, 2);

    printf("Результат B = 2 * A * B:\n");
    print_matrix("B (alpha=2)", 4, 2, B2, 2);

    /* Пример с ColumnMajor */
    /* В ColumnMajor нижняя треугольная матрица A хранится по столбцам */
    float A_col[4*4] = {
        2, 0, 0, 0,   /* столбец 0 */
        1, 3, 0, 0,   /* столбец 1 */
        4, 2, 5, 0,   /* столбец 2 */
        1, 1, 1, 2    /* столбец 3 */
    };
    float B_col[4*2] = {1, 3, 5, 7, 2, 4, 6, 8};

    printf("ColumnMajor пример:\n");
    cblas_strmm(CblasColMajor, CblasLeft, CblasLower,
                CblasNoTrans, CblasNonUnit,
                4, 2, 1.0f, A_col, 4, B_col, 4);

    printf("Результат (ColumnMajor):\n");
    print_matrix("B_col", 4, 2, B_col, 4);

    return 0;
}
