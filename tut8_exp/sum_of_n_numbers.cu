#include <bits/stdc++.h>
#include <cuda.h>
#include <time.h>

using namespace std;

void serial_sum(float * numbers)
{
    double sum = 0;
    int n = 10000000;
    clock_t start_time = clock();
    for(int i = 0; i<n; i++)
    {
        sum += numbers[i];
    }
    clock_t end_time = clock();
    double execution_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout<<"The sum of N double precision numbers in a serial fashion is : "<<sum <<" and is computed in : "<<execution_time<<"(s)"<<endl;
}

__global__ void parallel_sum(float * numbers, float * result, int N)
{
    __shared__ float sharedSum[512];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedSum[tid] = (i < N) ? numbers[i] : 0.0f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        result[blockIdx.x] = sharedSum[0];
    }
}

int main()
{
    ifstream file("numbers_one.txt");

    if(!file)
    {
        cout<<"Error in opening the file"<<endl;
        return 0;
    }

    float * h_numbers = (float *)malloc(sizeof(float) * 10000000);

    double num;
    int i = 0;
    
    while (file >> num)
    {
        h_numbers[i] = num;
        i++;
    }

    file.close();

    // SERIAL SUM OF N NUMBERS
    serial_sum(h_numbers);

    // PARALLEL SUM OF N NUMBERS
    float * d_numbers, *d_partialSums;

    cudaMalloc(&d_numbers, sizeof(float) * 10000000);
    cudaMemcpy(d_numbers, h_numbers, sizeof(float) * 10000000, cudaMemcpyHostToDevice);

    int threadsperBlock = 256;

    int blocksperGrid = (10000000 + threadsperBlock - 1)/threadsperBlock;
    cudaMalloc(&d_partialSums, sizeof(float) * blocksperGrid);
    clock_t start_time = clock();
    parallel_sum<<<blocksperGrid, threadsperBlock>>>(d_numbers, d_partialSums, 10000000);
    clock_t end_time = clock();
    double execution_time_one = double(end_time - start_time) / CLOCKS_PER_SEC;

    float *h_partialSums = (float *)malloc(sizeof(float) * blocksperGrid);
    cudaMemcpy(h_partialSums, d_partialSums, sizeof(float) * blocksperGrid, cudaMemcpyDeviceToHost);
    
    double finalSum = 0.0;
    start_time = clock();
    for (int j = 0; j < blocksperGrid; j++)
    {
        finalSum += h_partialSums[j];
    }
    end_time = clock();
    double execution_time_two = double(end_time - start_time) / CLOCKS_PER_SEC;

    cout << "The sum of N numbers in parallel fashion is: " << finalSum << " and is computed in "<<execution_time_one+execution_time_two<<"(s)"<<endl;
    
    free(h_numbers);
    free(h_partialSums);
    cudaFree(d_numbers);
    cudaFree(d_partialSums);
    return 0;
}