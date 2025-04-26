// Should compile gcc -fopenmp reductionConstruct.c -o reductionConstruct
// ./reductionConstruct

// Parallel code using reduction construct
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define N 100000000 //Number of elements in the array

int main(){
    double *arr = (double *)malloc(N * sizeof(double));
    double sum = 0.0;
    double start_time,end_time;

    //Filling hte array with some large double precision values(randomly generated)
    //? in C or C++ data type for double precision floating point numbers is double
    // If float is used it is single precision(32bit) not double precision(64bit)


    for (int i = 0; i < N; i++)
    {
        arr[i] = (double)(rand() % 1000 +1);
    }
    
    // Parallel summing using OpenMP redution
    // Time the execution with varying threads
    for (int threads = 1; threads <= 64; threads *= 2) {
        omp_set_num_threads(threads);

        start_time = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            sum += arr[i];
        }

        end_time = omp_get_wtime();
        printf("Sum : %lf\n",sum);
        printf("Threads: %d, Time: %lf seconds\n", threads, end_time - start_time);
    }

    
    free(arr);
    return 0;
    
}