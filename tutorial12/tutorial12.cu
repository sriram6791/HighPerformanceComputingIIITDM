#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(const double *A, const double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // compute row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // compute column index
    if (row < N && col < N) {
        double sum = 0.0;
        // Compute one element of C by accumulating over the row of A and column of B
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Set matrix dimensions. For demonstration adjust N based on available memory.
    int N = 1000;  // e.g., 10,000 x 10,000 matrix
    size_t totalElements = (size_t)N * N;
    size_t size = totalElements * sizeof(double);
    
    printf("Matrix size: %d x %d\nTotal elements: %zu\nMemory per matrix: %zu MB\n", 
           N, N, totalElements, size / (1024 * 1024));

    // Allocate host memory for two input matrices and two result matrices (CPU and GPU results)
    double *h_A = (double*) malloc(size);
    double *h_B = (double*) malloc(size);
    double *h_C_cpu = (double*) malloc(size);    // CPU result
    double *h_C_gpu = (double*) malloc(size);    // GPU result
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        printf("Host memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with random double precision numbers in the range [-1000, 1000]
    srand(time(NULL));
    for (size_t i = 0; i < totalElements; i++) {
        h_A[i] = ((double)rand() / RAND_MAX) * 2000.0 - 1000.0;
        h_B[i] = ((double)rand() / RAND_MAX) * 2000.0 - 1000.0;
    }

    // ======================
    // Serial CPU multiplication
    // ======================
    // WARNING: For very large N, a triple nested loop will take a very long time.
    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_cpu[i * N + j] = sum;
        }
    }
    clock_t cpu_end = clock();
    double cpuTime = ((double)(cpu_end - cpu_start)) * 1000.0 / CLOCKS_PER_SEC; // in ms
    printf("CPU (serial) time : %f ms\n", cpuTime);

    // ======================
    // Parallel GPU multiplication
    // ======================
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host matrices to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions (each thread computes one element)
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Use CUDA events for accurate timing of the GPU kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the GPU matrix multiplication kernel
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("GPU (parallel) time : %f ms\n", gpuTime);

    // Copy the GPU result back to host
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Estimate speedup
    double speedup = cpuTime / gpuTime;
    printf("Speedup : %f\n", speedup);

    // Cleanup device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
