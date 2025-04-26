#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include "MatrixMath.h"
using namespace std;
class utils
{
public:
    double get_random_number()
    {
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        return dis(gen);
    }
    vector<vector<double>> initialize_weights(int dim1, int dim2)
    {
        vector<vector<double>> weights;
        for (int i = 0; i < dim1; i++)
        {
            vector<double> row;
            for (int j = 0; j < dim2; j++)
            {
                double weight = get_random_number();
                row.push_back(weight);
            }
            weights.push_back(row);
        }
        return weights;
    }
};
class NeuralNetwork : public utils
{
public:
    vector<vector<vector<double>>> NetworkWeights; // TO STORE NETWORK WEIGHTS
    int num_layers;                                // STORE NUMBER OF LAYERS INCLUDEING INPUT LAYER SO DONT COUNT FIRST NUMBER
    vector<int> nodes;                             // STORE NUMBER OF NODES IN EACH LAYER
    vector<vector<vector<double>>> layer_outputs;  // STORE OUTPUTS FROM EACH LAYER IN FEED FORWARD NETWORK
    vector<vector<vector<double>>> d_w;            // STORES dW of each layer
    vector<vector<vector<double>>> d_z;            // STORES dz of each layer
    vector<vector<vector<double>>> inputs_for_each_layer;
    double learning_rate = 0.00000001;
    NeuralNetwork(vector<int> nodes_count_in_each_layer)
    {
        /* nodes_count_in_each_layer is a vector containing nodes count in each layer ex:{4,3,3,1} 4 in input feature vector,first layer contains 3 nodes second layer contains 3 nodes and last node containd 1 node*/
        num_layers = nodes_count_in_each_layer.size();
        nodes = nodes_count_in_each_layer;
        for (int i = 1; i < num_layers; i++)
        {
            vector<vector<double>> layer_weights;
            layer_weights = initialize_weights(nodes_count_in_each_layer[i], nodes_count_in_each_layer[i - 1]);
            NetworkWeights.push_back(layer_weights);
        }
    }
    void print_network_weights()
    {
        for (int i = 0; i < num_layers - 1; i++)
        {
            cout << "Layer-" << i << endl;
            cout << "input dim : " << nodes[i] << ", num nodes:" << nodes[i + 1] << endl;
            cout << "Weight dimensions : " << nodes[i] << " X " << nodes[i + 1] << endl;
            cout << "Weights :" << endl;
            print_weights(NetworkWeights[i]);
            cout << endl;
        }
    }
    vector<vector<double>> ReLU(vector<vector<double>> input)
    { // Relu activation function ReLU(x) returns 0 if x<0 else returns x
        int rows = input.size();
        int col = input[0].size();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < col; j++)
            {
                input[i][j] = input[i][j] > 0 ? input[i][j] : 0;
            }
        }
        return input;
    }
    vector<vector<double>> feed_forward(vector<vector<double>> input)
    {
        //! Clear previous layer outputs and inputs
        inputs_for_each_layer.clear();
        layer_outputs.clear();
        //! ALWAYS NEED TO PASS A 2D MATRIX EX: {{1,2,3}}
        // pass the input in {{1,2,3}} format , Here we shall convert it into column vector {{1},{2},{3}}
        vector<vector<double>> result;
        vector<vector<double>> input_to_pass = Transpose(input);
        for (int i = 0; i < num_layers - 1; i++)
        {
            inputs_for_each_layer.push_back(input_to_pass);
            result = Cross_MUl(NetworkWeights[i], input_to_pass);
            // result = ReLU(result);
            layer_outputs.push_back(result); // RESULT WILL ALSO BE A 2D VECTOR AND ALL LAYER RESULTS ARE STORED IN layer_outputs { {{}} , {{}}, {{}} }
            input_to_pass = result;
        }
        return result;
    }
    void back_propogation(vector<vector<double>> predicted, vector<vector<double>> target)
    {
        // Clear previous gradients
        d_w.clear();
        d_z.clear();
        // Number of weight layers (excluding input layer)
        int L = num_layers - 1;
        int number_of_samples = target.size();
        // Convert target into a column vector
        target = Transpose(target);
        // --- 1. Compute gradient for the output layer ---
        // delta for output layer: δ^L = (a^L - y)
        vector<vector<double>> delta = Subtract(predicted, target);
        delta = error_cal(delta, number_of_samples);
        d_z.push_back(delta);
        // The input to the output layer is the last stored input: a^(L-1)
        vector<vector<double>> grad = Cross_MUl(delta, Transpose(inputs_for_each_layer[L - 1]));
        d_w.push_back(grad);
        // --- 2. Backpropagate the error for layers L-1, L-2, ..., 1 ---
        // (Assuming here you want to ignore the activation derivative;
        // if using ReLU you would need to multiply elementwise by g'(z) )
        for (int l = L - 1; l > 0; l--)
        {
            // Note: To compute δ^l, use the weights from the layer l+1 (which is at index l)
            // delta(l) = (W^(l+1))^T * delta(l+1)
            delta = Cross_MUl(Transpose(NetworkWeights[l]), delta);
            d_z.push_back(delta);
            // Compute dW for the current layer l: dW = δ^l * (a^(l-1))^T
            grad = Cross_MUl(delta, Transpose(inputs_for_each_layer[l - 1]));
            d_w.push_back(grad);
        }

        // Since we computed gradients starting from the output and going backwards,
        // reverse the order to match the ordering in NetworkWeights.
        reverse(d_w.begin(), d_w.end());
        reverse(d_z.begin(), d_z.end());
        // --- 3. Update the weights ---
        for (int i = 0; i < d_w.size(); i++)
        {
            NetworkWeights[i] = Subtract(NetworkWeights[i], scalar_mul_with_matrix(d_w[i], learning_rate));
        }
    }
};
// int main()
// {

//     vector<int> layers = {3,3,3,2,2,3,3,3};
//     NeuralNetwork nn(layers);

//     // TEST
//     vector<vector<double>> input = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {1.5, 2.6, 3.8}, {15.0, 25.0, 37.0}};
//     // vector<vector<double>> input = {{4.0,5.0,6.0}};
//     int epochs = 30;
//     for (int i = 1; i <= epochs; i++)
//     {
//         cout << "epoch : " << i << endl;
//         cout << "Before weight update : " << endl;
//         nn.print_network_weights();
//         vector<vector<double>> feed_forward_output = nn.feed_forward(input);
//         /*debug step */ cout << "completed " << i << " feedforward" << endl;
//         print_vec2D(feed_forward_output);
//         nn.back_propogation(feed_forward_output, input);
//         cout << "After weight update : " << endl;
//         nn.print_network_weights();
//         cout << endl;
//     }

//     vector<vector<double>> input2 = {{4.0, 5.0, 6.0}};
//     cout << endl
//          << "TESTING1 : " << endl;
//     vector<vector<double>> test = nn.feed_forward(input2);
//     print_vec2D(test);
//     // cout<<endl<<"TESTING2 : "<<endl;
//     // test = nn.feed_forward(input2);
//     // print_vec2D(test);
// }


// Function to load dataset from file into a vector of vector of double
vector<vector<double>> loadDataset(const string &filename, int num_pixels) {
    vector<vector<double>> data;
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }
    string line;
    while (getline(infile, line)) {
        vector<double> sample;
        istringstream iss(line);
        double val;
        while (iss >> val) {
            sample.push_back(val);
        }
        if (sample.size() != num_pixels) {
            cerr << "Warning: sample size (" << sample.size()
                 << ") does not match expected (" << num_pixels << ")." << endl;
            // Optionally, you can choose to skip this sample:
            // continue;
        }
        data.push_back(sample);
    }
    infile.close();
    return data;
}

int main() {
    // Load the dataset (each sample has 784 pixels)
    int num_pixels = 784;
    vector<vector<double>> input = loadDataset("dataset.txt", num_pixels);
    cout << "Loaded dataset with " << input.size() << " samples." << endl;

    // Initialize your neural network (ensure that the first layer size matches your input dimension if needed)
    vector<int> layers = {784, 128, 64, 128,784};  // Example: input layer of 784, two hidden layers, and 10 output neurons
    NeuralNetwork nn(layers);

    // Training loop (example using 30 epochs)
    int epochs = 1;
    for (int i = 1; i <= epochs; i++) {
        cout << "Epoch: " << i << endl;
        cout << "Before weight update:" << endl;
        // nn.print_network_weights();
        vector<vector<double>> feed_forward_output = nn.feed_forward(input);
        cout << "Completed " << i << " feedforward pass." << endl;
        // print_vec2D(feed_forward_output);
        nn.back_propogation(feed_forward_output, input);
        cout << "After weight update:" << endl;
        // nn.print_network_weights();
        cout << endl;
    }

    // Testing with a single sample (for example, the first sample)
    vector<vector<double>> input2 = {input[0]};
    cout<<"Test Input : "<<endl;
    print_vec2D(input2);
    cout<<endl;
    cout << "\nTESTING:" << endl;
    vector<vector<double>> test = nn.feed_forward(input2);
    print_vec2D(test);

    return 0;
}
