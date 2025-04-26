#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#define N 100000000 
int main(){
    double* arr1  =(double *)malloc(N * sizeof(double));
    double* arr2  =(double *)malloc(N * sizeof(double));
    double* mul  =(double *)malloc(N * sizeof(double));

    for(int i=0;i<N;i++){
        double random = (rand()%1000 + 1);
        arr1[i] = random;
        random = (rand()%1000 + 1);
        arr2[i] = random;
    }
    //Without parallelization
        double start_time = omp_get_wtime();
        for(int i=0;i<N;i++){
            mul[i] = arr1[i] * arr2[i];
        }
        double end_time = omp_get_wtime();
        printf("Time taken without parallelization : %lf seconds\n",end_time-start_time);
    
    //With parallelization
        start_time = omp_get_wtime();
        #pragma omp parallel for
        for(int i=0;i<N;i++){
            mul[i] = arr1[i] * arr2[i];
        }
        end_time = omp_get_wtime();
        printf("Time taken with parallelization : %lf seconds\n",end_time-start_time);
        
    
}