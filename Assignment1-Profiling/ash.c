#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

#define N 2000

typedef struct{
    double **arr1;
    double **arr2;
    double **res;
}Matrices;

void matrix_mul_serial(Matrices *M){
    double start_time, end_time;

    // Start time of the program
    start_time = omp_get_wtime();

    // Matrix multiplication
    for (int i = 0; i < N; i++){
        for (int k = 0; k < N; k++){
            double temp = M->arr1[i][k];
            for (int j = 0; j < N; j++){
                M->res[i][j] += temp * M->arr2[k][j];
            }
        }
    }

    // End time of the program
    end_time = omp_get_wtime();

    printf("Execution time for matrix multiplication (serial code): %f seconds\n", end_time - start_time);
}

void matrix_mul_parallel(Matrices *M) {
    double start_time, end_time;

    // Start time of the program
    start_time = omp_get_wtime();

    // Parallel matrix multiplication
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                #pragma omp atomic
                M->res[i][j] += M->arr1[i][k] * M->arr2[k][j];
            }
        }
    }

    // End time of the program
    end_time = omp_get_wtime();
    printf("Execution time for matrix multiplication (parallel optimized): %f seconds\n", end_time - start_time);
}


int main(void){

    Matrices M;

    // Allocate memory for the matrices dynamically
    M.arr1 = (double **)malloc(N * sizeof(double *));
    M.arr2 = (double **)malloc(N * sizeof(double *));
    M.res = (double **)malloc(N * sizeof(double *));

    for(int i = 0; i < N; i++){
        M.arr1[i] = (double *)malloc(N * sizeof(double));
        M.arr2[i] = (double *)malloc(N * sizeof(double));
        M.res[i] = (double *)malloc(N * sizeof(double));
    }

    srand(time(0));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            M.arr1[i][j] = (double)((rand() / RAND_MAX)) * 1000.0;
            M.arr2[i][j] = (double)((rand() / RAND_MAX)) * 1000.0;
            M.res[i][j] = 0.0;
        }
    }
    matrix_mul_parallel(&M);
    matrix_mul_serial(&M);
    // Free the memory allocated for the matrices
    for(int i = 0; i < N; i++){
        free(M.arr1[i]);
        free(M.arr2[i]);
        free(M.res[i]);
    }

    free(M.arr1);
    free(M.arr2);
    free(M.res);

    return 0;
}