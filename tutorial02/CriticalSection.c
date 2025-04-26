#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define N 1000000 

int main(){

    double* arr = (double *)malloc(N * sizeof(double));
    double sum =0.0;

    //Fill the array with some large double precision numbers
    for (int i = 0; i < N; i++)
    {
        arr[i] = (double)(rand()%1000 + 1); 
    }

    // Parallel summing using OpenMP critical section
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        #pragma omp critical
        {
            sum += arr[i];
        }
    }
    printf("Sum : %lf\n",sum);
    free(arr);

    return 0;
}
//? #pragma omp critical section ensures that only one thread can acccess and update the sum variable at a time
