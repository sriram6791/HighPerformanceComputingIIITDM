#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

int main() {
    srand(time(0));  
    ofstream file("data.txt");

    int N = 10000000;  // Generate 10 million numbers
    for (int i = 0; i < N; i++) {
        // Generate random double in the range [-1000, 1000]
        double num = ((double)rand() / RAND_MAX) * 2000.0 - 1000.0;

        // Write number in scientific notation
        file << scientific << setprecision(18) << num << "\n";
    }

    file.close();
    cout << "Generated " << N << " large double-precision numbers in data.txt\n";
    return 0;
}
