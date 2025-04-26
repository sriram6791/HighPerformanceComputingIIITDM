#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

#define N 10000000  // 1 million elements
#define BLOCK_SIZE 256  // Threads per block

// CUDA Kernel for dot product (Reduction)
__global__ void dotProduct(double *A, double *B, double *result, int n) {
    __shared__ double temp[BLOCK_SIZE]; // Shared memory for reduction

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Each thread computes one product
    temp[tid] = (idx < n) ? A[idx] * B[idx] : 0.0;

    __syncthreads(); // Ensure all threads have written their values

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // First thread in block writes result to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]); // Atomic add to prevent race conditions
    }
}

int main() {
    ifstream file("data.txt");
    vector<double> A(N), B(N);
    double num;

    // Read first N numbers into A and next N numbers into B
    for (int i = 0; i < N && file >> num; i++) A[i] = num;
    for (int i = 0; i < N && file >> num; i++) B[i] = num;
    file.close();

    // Allocate memory on GPU
    double *d_A, *d_B, *d_result;
    cudaMalloc((void **)&d_A, N * sizeof(double));
    cudaMalloc((void **)&d_B, N * sizeof(double));
    cudaMalloc((void **)&d_result, sizeof(double));

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, A.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(double)); // Initialize result to zero

    // Launch CUDA kernel
    auto start = high_resolution_clock::now();
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dotProduct<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_result, N);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();

    // Copy result back to CPU
    double result;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Measure execution time
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "CUDA Dot Product Result: " << result << endl;
    cout << "Time Taken (GPU): " << duration.count() << " ms" << endl;

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    return 0;
}
