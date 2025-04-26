// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <chrono>

// using namespace std;
// using namespace chrono;

// #define N 10000000  // 1 million elements

// void vectorAddCPU(const vector<double> &A, const vector<double> &B, vector<double> &C, int n) {
//     for (int i = 0; i < n; i++) {
//         C[i] = A[i] + B[i]; // Element-wise addition
//     }
// }

// int main() {
//     ifstream file("data.txt");
//     vector<double> A(N), B(N), C(N);
//     double num;

//     // Read first N numbers into A and next N numbers into B
//     for (int i = 0; i < N && file >> num; i++) A[i] = num;
//     for (int i = 0; i < N && file >> num; i++) B[i] = num;
//     file.close();

//     // Measure CPU execution time
//     auto start = high_resolution_clock::now();
//     vectorAddCPU(A, B, C, N);
//     auto stop = high_resolution_clock::now();

//     // Time calculation
//     auto duration = duration_cast<milliseconds>(stop - start);
//     cout << "CPU Vector Addition Completed\n";
//     cout << "Time Taken (CPU): " << duration.count() << " ms" << endl;

//     return 0;
// }


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

#define N 10000000  // 1 million elements

void vectorMultiplyCPU(const vector<double> &A, const vector<double> &B, vector<double> &C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] * B[i]; // Element-wise multiplication
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

    // Measure CPU execution time
    auto start = high_resolution_clock::now();
    vectorMultiplyCPU(A, B, C, N);
    auto stop = high_resolution_clock::now();

    // Time calculation
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "CPU Vector Multiplication Completed\n";
    cout << "Time Taken (CPU): " << duration.count() << " ms" << endl;

    return 0;
}
