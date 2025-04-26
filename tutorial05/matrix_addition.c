#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

#define N 10000 //MatrixSize

double **allocate_matrix(){
    double** matrix = (double**)malloc(N*sizeof(double*));
    for(int i=0;i<N;i++){
        matrix[i] = (double*)malloc(N*sizeof(double));
    }
    return matrix;
}
void initialize_matrix(double **matrix,double value){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            matrix[i][j] = value;
        }
    }
}
void free_matrix(double** matrix){
    for(int i=0;i<N;i++)
        free(matrix[i]);
    free(matrix);
}

void matrix_addition_parallel(double **A,double** B,double** C){
    #pragma omp parallel for collapse(2)
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrix_addition_serial(double **A,double **B,double **C){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}
int main(){
    
    double **A = allocate_matrix();
    double **B = allocate_matrix();
    double **C = allocate_matrix();

    initialize_matrix(A,1.0);
    initialize_matrix(B,2.0);
    initialize_matrix(C,0.0);

    printf("SERIAL CODE: \n");
    printf("----------------------\n");
    double start = omp_get_wtime();
    matrix_addition_serial(A,B,C);
    double end = omp_get_wtime();
    printf("Serial Time taken : %0.3f\n",end-start);
    printf("\n\nPARALLEL CODE :\n");
    printf("----------------------\n");

    for(int num_threads = 1;num_threads<=64 ;num_threads*=2){
        omp_set_num_threads(num_threads);
        double start = omp_get_wtime();
        matrix_addition_parallel(A,B,C);
        double end = omp_get_wtime();

        printf("Thread : %d , Time %0.3f seconds\n",num_threads,end-start);
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    
    return 0;


}