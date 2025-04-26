#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define N 100000000

int main(){
    double* arr1 = (double*)malloc(N*sizeof(double));
    double* arr2 = (double*)malloc(N*sizeof(double));
    double* mul = (double*)malloc(N*sizeof(double));

    for (int threads = 1; threads <= 64; threads*=2)
    {
       omp_set_num_threads(threads);

       double start_time = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            mul[i] = arr1[i] * arr2[i];
        }
       double end_time = omp_get_wtime();
       printf("Num threads : %d , Time Taken : %lf\n",threads,end_time-start_time);
    }
    
}