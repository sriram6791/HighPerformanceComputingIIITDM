#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

#define N 1000000
#define BLOCK_SIZE 256

// CUDA Kernel for vector multiplication
__global__ void vectorMultiply(double *A, double *B, double *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx]; // Element-wise multiplication
    }
}

int main() {
    ifstream file("data.txt");
    vector<double> A(N), B(N), C(N);
    double num;

    // Read first N numbers into A and next N numbers into B
    for (int i = 0; i < N && file >> num; i++) A[i] = num;
    for (int i = 0; i < N && file >> num; i++) B[i] = num;
    file.close();

    // Allocate memory on GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(double));
    cudaMalloc((void **)&d_B, N * sizeof(double));
    cudaMalloc((void **)&d_C, N * sizeof(double));

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, A.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    auto start = high_resolution_clock::now();
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorMultiply<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();

    // Copy result back to CPU
    cudaMemcpy(C.data(), d_C, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Measure execution time
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "CUDA Vector Multiplication Completed\n";
    cout << "Time Taken (GPU): " << duration.count() << " ms" << endl;

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
