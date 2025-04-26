#include <iostream>
#include<fstream>
#include <vector>
#include<cuda_runtime.h>
#include<chrono>

#define N 10000000
#define BLOCK_SIZE 256

using namespace std;
using namespace std::chrono;

__global__ void parallelSum(double *input,double *result,int n){
    __shared__ double sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    // threadIdx.x  thread's index with in a block
    // blockIdx.x  Block index with in the grid
    // blockDim.x   Number of threads in a block
    // tid = Global thread Id
    // local_tid = Local thread ID

    sharedData[local_tid] = (tid<n) ? input[tid] : 0.0;
    __syncthreads();

    for(int stride = blockDim.x/2 ; stride>0 ; stride >>=1){
        if(local_tid < stride){
            sharedData[local_tid] += sharedData[local_tid + stride];
        }
        __syncthreads();
    }
    
    if(local_tid == 0){
        result[blockIdx.x] = sharedData[0];
    }
}

int main(){
    ifstream file("data.txt");
    vector<double> numbers(N);
    double num;

    for(int i=0;i<N && file >> num;i++){
        numbers[i] = num;
    }
    file.close();

    // To Allocate memory on GPU
    double *d_input ,*d_partial_sums;
    double *h_partial_sums;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void **)&d_input ,N*sizeof(double));
    cudaMalloc((void **)&d_partial_sums,numBlocks*sizeof(double));
    h_partial_sums = new double[numBlocks];

    //COPY DATA TO GPU
    cudaMemcpy(d_input,numbers.data(),N*sizeof(double),cudaMemcpyHostToDevice);
    auto start = high_resolution_clock::now();
    parallelSum<<<numBlocks,BLOCK_SIZE>>>(d_input,d_partial_sums,N);
    cudaMemcpy(h_partial_sums,d_partial_sums,numBlocks * sizeof(double),cudaMemcpyHostToHost);

    double totalSum = 0.0;
    for(int i=0;i<numBlocks;i++){
        totalSum += h_partial_sums[i];
    }
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    cout<< "Cuda Parallel Sum "<<totalSum<<endl;
    cout<< " Time Taken (GPU) :"<<duration.count()<<"ms"<<endl;

    cudaFree(d_input);
    cudaFree(d_partial_sums);
    delete[] h_partial_sums;

    return 0;

}