#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    int num_pixels = 784;      // number of pixels per image
    int dataset_size = 100;      // total number of images
    ofstream outfile("dataset.txt");
    if (!outfile.is_open()) {
        cerr << "Error opening file for writing!" << endl;
        return 1;
    }

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Generate dataset: each image is a row of num_pixels values
    for (int i = 0; i < dataset_size; i++) {
        for (int j = 0; j < num_pixels; j++) {
            // Generate a random double in [0, 1]
            double pixel = static_cast<double>(rand()) / RAND_MAX;
            outfile << pixel;
            if (j < num_pixels - 1)
                outfile << " ";  // separate pixels with a space
        }
        outfile << "\n"; // new line for each sample
    }
    outfile.close();
    cout << "Dataset generated and stored in dataset.txt" << endl;
    return 0;
}
