#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 100000000

int main()
{
    double *arr1 = (double *)malloc(N * sizeof(double));
    double *arr2 = (double *)malloc(N * sizeof(double));
    double dot_product = 0.0;

    for (int i = 0; i < N; i++)
    {
        arr1[i] = (double)(rand() % 1000 + 1);
        arr2[i] = (double)(rand() % 1000 + 1);
    }
    // without parallelization
    double start_time = omp_get_wtime();
    double serial_dot = 0.0;
    for (int i = 0; i < N; i++)
    {
        serial_dot += arr1[i] * arr2[i];
    }
    double end_time = omp_get_wtime();
    printf("Serial Dot Product: %lf\n", serial_dot);
    printf("Time Taken (Serial): %lf seconds\n", end_time - start_time);

    // Parallel computation using omp critical

    for (int thread = 1; thread <= 64; thread *= 2)
    {
        omp_set_num_threads(thread);
        start_time = omp_get_wtime();
#pragma omp parallel
        {
            double local_sum = 0.0;
#pragma omp for
            for (int i = 0; i < N; i++)
            {
                local_sum += arr1[i] * arr2[i];
            }

#pragma omp critical
            {
                dot_product += local_sum;
            }
        }
        end_time = omp_get_wtime();
        printf("Num threads : %d\n",thread);
        printf("Parallel Dot Product: %lf\n", dot_product);
        printf("Time Taken (Parallel): %lf seconds\n", end_time - start_time);
    }
    // Free memory
    free(arr1);
    free(arr2);

    return 0;
}
