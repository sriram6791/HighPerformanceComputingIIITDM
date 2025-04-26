#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main() {
    ifstream file("data.txt");
    vector<double> numbers;
    double num;

    // Read numbers from file
    while (file >> num) {
        numbers.push_back(num);
    }
    file.close();

    // Serial Summation
    auto start = high_resolution_clock::now();
    double sum = 0.0;
    for (double val : numbers) {
        sum += val;
    }
    auto stop = high_resolution_clock::now();

    // Measure time
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Serial Sum: " << sum << endl;
    cout << "Time Taken (CPU): " << duration.count() << " ms" << endl;

    return 0;
}
