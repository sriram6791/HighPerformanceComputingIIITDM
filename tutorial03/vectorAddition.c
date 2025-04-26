#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define N 100000000
int main(){
    double* arr1 = (double*)malloc(N * sizeof(double));
    double* arr2 = (double*)malloc(N * sizeof(double));
    double* sum = (double*)malloc(N * sizeof(double));

    //filling two arrays with random floating point values
    for(int i=0;i<N;i++){
        double random = (rand()%1000 + 1);
        arr1[i] = random;
        random = (rand()%1000 + 1);
        arr2[i] = random;
    }
    double start_time = omp_get_wtime();
    for(int i=0;i<N;i++){
        sum[i] = arr1[i] + arr2[i];
    }
    double end_time = omp_get_wtime();
    printf("Time Taken for vector addition without vector parallelization: %lf seconds\n",end_time-start_time);
    //adding both the vectors using parallellization
    start_time = omp_get_wtime();
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        sum[i] = arr1[i] + arr2[i];
    }
    end_time = omp_get_wtime();
    printf("Time Taken for vector addition with vector parallelization : %lf seconds\n",end_time-start_time);
}