#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Match this with the GPU code:
#define N 1000000  // 1 million elements

// CPU Function for dot product
double dotProductCPU(const vector<double> &A, const vector<double> &B, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += A[i] * B[i];  // Element-wise multiplication + summation
    }
    return sum;
}

int main() {
    ifstream file("data.txt");
    vector<double> A(N), B(N);
    double num;

    // Read first N numbers into A
    for (int i = 0; i < N && file >> num; i++) {
        A[i] = num;
    }

    // Read next N numbers into B
    for (int i = 0; i < N && file >> num; i++) {
        B[i] = num;
    }
    file.close();

    // Measure CPU execution time
    auto start = high_resolution_clock::now();
    double result = dotProductCPU(A, B, N);
    auto stop = high_resolution_clock::now();

    // Time calculation
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "CPU Dot Product Result: " << result << endl;
    cout << "Time Taken (CPU): " << duration.count() << " ms" << endl;

    return 0;
}
