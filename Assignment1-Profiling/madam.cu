#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>

#define DIM 1000
#define NB_OF_RUNS 20
#define MAX_ITER 100000
#define T_INITIAL 100.0
#define T_MIN 1e-8
#define COOL_RATE 0.99995
#define LOWER_BOUND -5.12
#define UPPER_BOUND 5.12

_device_ double rastrigin(double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        sum += xi * xi - 10.0 * cos(2.0 * M_PI * xi);
    }
    return 10.0 * n + sum;
}

_global_ void simulated_annealing_kernel(double *results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB_OF_RUNS) return;

    curandState state;
    curand_init(1234, tid, 0, &state);

    double x[DIM], x_new[DIM];
    for (int i = 0; i < DIM; i++) {
        x[i] = LOWER_BOUND + curand_uniform_double(&state) * (UPPER_BOUND - LOWER_BOUND);
    }

    double current_cost = rastrigin(x, DIM);
    double best_cost = current_cost;
    double T = T_INITIAL;

    for (long iter = 0; iter < MAX_ITER && T > T_MIN; iter++) {
        for (int i = 0; i < DIM; i++) x_new[i] = x[i];
        int idx = curand(&state) % DIM;
        x_new[idx] += curand_uniform_double(&state) * 0.2 - 0.1;
        x_new[idx] = fmax(fmin(x_new[idx], UPPER_BOUND), LOWER_BOUND);

        double new_cost = rastrigin(x_new, DIM);
        double diff = new_cost - current_cost;

        if (diff < 0 || exp(-diff / T) > curand_uniform_double(&state)) {
            for (int i = 0; i < DIM; i++) x[i] = x_new[i];
            current_cost = new_cost;
            if (current_cost < best_cost) best_cost = current_cost;
        }
        T *= COOL_RATE;
    }

    results[tid] = best_cost;
}

int main() {
    double *d_results, h_results[NB_OF_RUNS];
    cudaMalloc(&d_results, NB_OF_RUNS * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    simulated_annealing_kernel<<<1, NB_OF_RUNS>>>(d_results);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(h_results, d_results, NB_OF_RUNS * sizeof(double), cudaMemcpyDeviceToHost);

    double best = h_results[0];
    for (int i = 1; i < NB_OF_RUNS; i++) {
        if (h_results[i] < best) best = h_results[i];
    }

    printf("Best cost: %f\n", best);
    printf("CUDA time: %f ms\n", time_ms);

    cudaFree(d_results);
    return 0;
}