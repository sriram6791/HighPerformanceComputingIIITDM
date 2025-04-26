#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix addition
__global__ void matrixAdd(const double *A, const double *B, double *C, int N) {
    // Compute row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        // Use size_t to avoid overflow in index computation
        size_t idx = (size_t)row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Set matrix dimension N x N.
    // Adjust N according to your available GPU memory.
    int N = 20000; // For example: 10,000 x 10,000 matrix
    size_t totalElements = (size_t)N * N;
    size_t size = totalElements * sizeof(double);

    printf("Matrix size: %d x %d, Total elements: %zu, Memory per matrix: %zu MB\n", 
           N, N, totalElements, size / (1024 * 1024));

    // Allocate host memory
    double *h_A = (double*) malloc(size);
    double *h_B = (double*) malloc(size);
    double *h_C = (double*) malloc(size);
    if (!h_A || !h_B || !h_C) {
        printf("Host memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with random numbers in the range [-1000, 1000]
    srand(time(NULL));
    for (size_t i = 0; i < totalElements; i++) {
        h_A[i] = ((double)rand() / RAND_MAX) * 2000.0 - 1000.0;
        h_B[i] = ((double)rand() / RAND_MAX) * 2000.0 - 1000.0;
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Set up CUDA events for timing the GPU kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the matrix addition kernel
    matrixAdd<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Copy the result from device back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Perform CPU matrix addition for comparison (and measure time)
    clock_t cpu_start = clock();
    for (size_t i = 0; i < totalElements; i++) {
        double tmp = h_A[i] + h_B[i];
        // (Optional) You could verify a few elements here.
    }
    clock_t cpu_end = clock();
    double cpuTime = (double)(cpu_end - cpu_start) * 1000.0 / CLOCKS_PER_SEC; // in ms

    // Display timing and speedup results
    printf("CPU time : %f ms\n", cpuTime);
    printf("GPU time : %f ms\n", gpuTime);
    if (gpuTime > 0)
        printf("Speedup  : %f\n", cpuTime / gpuTime);
    else
        printf("Speedup  : CPU time measured as %f ms\n", cpuTime);

    // Clean up resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
