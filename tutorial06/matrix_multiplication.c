#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define N 10000

double **allocate_matrix(){
    double** matrix = (double**)malloc(N*sizeof(double*));
    if(matrix == NULL){
        printf("Memory allocation failed!\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        matrix[i] = (double*) malloc(N*sizeof(double));
        if(matrix[i] == NULL){
            printf("Memory allocation failed!\n");
            exit(1);
        }
    }
    
    return matrix;
}

void initialize_matrix(double **matrix,double value){
    for (int i = 0; i < N; i++)
    {
        for(int j=0;j<N;j++){
            matrix[i][j] = value;
        }
    }
}

void free_matrix(double **matrix){
    for(int i=0;i<N;i++){
        free(matrix[i]);
    }
    free(matrix);
}

void matrix_multiplication_parallel(double** A,double **B,double** C){
    #pragma omp parallel for collapse(2)
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            C[i][j] = 0.0;
            for(int k=0;k<N;k++){
                C[i][j]+= A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiplication_serial(double **A,double **B,double **C){
    for (int i = 0; i < N; i++)
    {
        for(int j=0;j<N;j++){
            C[i][j] = 0;
            for(int k=0;k<N;k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    
}

int main(){
    double **A = allocate_matrix();
    double **B = allocate_matrix();
    double **C = allocate_matrix();

    initialize_matrix(A,2.0);
    initialize_matrix(B,6.0);

    printf("SERIAL CODE : \n");
    printf("----------------------\n");
    double start = omp_get_wtime();
    matrix_multiplication_serial(A,B,C);
    double end = omp_get_wtime();
    printf("Time taken for serial : %0.3f\n",end-start);

    printf("\n\nPARALLEL CODE : \n");
    printf("----------------------");
    
    for(int num_threads = 1;num_threads<=64;num_threads*=2){
        omp_set_num_threads(num_threads);
        double start = omp_get_wtime();
        matrix_multiplication_parallel(A,B,C);
        double end = omp_get_wtime();

        printf("Threads : %d , Time taken : %0.3f\n",num_threads,end-start);

    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}
