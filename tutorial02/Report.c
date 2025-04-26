#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000000 // Number of elements in the array

int main() {
    double *arr = (double *)malloc(N * sizeof(double));
    double sum = 0.0;
    double start_time, end_time;

    // Fill the array with some large double precision values (randomly generated here)
    for (int i = 0; i < N; i++) {
        arr[i] = (double)(rand() % 1000 + 1);
    }

    // Time the execution with varying threads
    for (int threads = 1; threads <= 64; threads *= 2) {
        omp_set_num_threads(threads);

        start_time = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            sum += arr[i];
        }

        end_time = omp_get_wtime();
        printf("Threads: %d, Time: %lf seconds\n", threads, end_time - start_time);
    }

    free(arr);
    return 0;
}
