#include<iostream>
#include<vector>
#include<random>
#include<algorithm>

#include "MatrixMath.h"

using namespace std;

class utils{
    public:

    double get_random_number(){
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        return dis(gen);
    }
    vector<vector<double>> initialize_weights(int dim1,int dim2){
        vector<vector<double>>weights;
        for(int i=0;i<dim1;i++){
            vector<double>row;
            for(int j=0;j<dim2;j++){
                double weight = get_random_number();
                row.push_back(weight);
            }
            weights.push_back(row);
        }

        return weights;
    }
};
class NeuralNetwork : public utils {
    public:
    vector<vector<vector<double>>>NetworkWeights; // TO STORE NETWORK WEIGHTS
    int num_layers; // STORE NUMBER OF LAYERS INCLUDEING INPUT LAYER SO DONT COUNT FIRST NUMBER
    vector<int>nodes;// STORE NUMBER OF NODES IN EACH LAYER
    vector<vector<vector<double>>> layer_outputs; // STORE OUTPUTS FROM EACH LAYER IN FEED FORWARD NETWORK
    vector<vector<vector<double>>> d_w; //STORES dW of each layer
    vector<vector<vector<double>>> d_z; //STORES dz of each layer
    vector<vector<vector<double>>> inputs_for_each_layer;
    double learning_rate = 0.01;

    NeuralNetwork(vector<int>nodes_count_in_each_layer){
        /* nodes_count_in_each_layer is a vector containing nodes count in each layer ex:{4,3,3,1} 4 in input feature vector,first layer contains 3 nodes second layer contains 3 nodes and last node containd 1 node*/
        num_layers = nodes_count_in_each_layer.size();
        nodes = nodes_count_in_each_layer;

        for(int i=1;i<num_layers;i++){
            vector<vector<double>>layer_weights;
            layer_weights = initialize_weights(nodes_count_in_each_layer[i-1],nodes_count_in_each_layer[i]);
            NetworkWeights.push_back(layer_weights);
        }
    }   

    void print_network_weights(){
        for(int i=0;i<num_layers-1;i++){
            cout<<"Layer-"<<i<<endl;
            cout<<"input dim : "<<nodes[i]<<", num nodes:"<<nodes[i+1]<<endl;
            cout<<"Weight dimensions : "<<nodes[i]<<" X "<<nodes[i+1]<<endl;
            cout<<"Weights :"<<endl;
            print_weights(NetworkWeights[i]);
            cout<<endl;
        }
    }

    vector<vector<double>> ReLU(vector<vector<double>>input){  //Relu activation function ReLU(x) returns 0 if x<0 else returns x
        int rows = input.size();
        int col = input[0].size();
        for(int i=0;i<rows;i++){
            for(int j=0;j<col;j++){
                input[i][j] = input[i][j] >0 ? input[i][j] : 0;
            }
        }

        return input;
    }
    vector<vector<double>> feed_forward(vector<vector<double>>input){

        //! Clear previous layer outputs and inputs
        inputs_for_each_layer.clear();
        layer_outputs.clear();

        //! ALWAYS NEED TO PASS A 2D MATRIX EX: {{1,2,3}}
        // pass the input in {{1,2,3}} format , Here we shall convert it into column vector {{1},{2},{3}}
        vector<vector<double>> result;
        vector<vector<double>> input_to_pass = Transpose(input);

        for(int i=0;i<num_layers-1;i++){
            inputs_for_each_layer.push_back(input_to_pass);
            result = Cross_MUl(Transpose(NetworkWeights[i]),input_to_pass);
            // result = ReLU(result);
            layer_outputs.push_back(result); //RESULT WILL ALSO BE A 2D VECTOR AND ALL LAYER RESULTS ARE STORED IN layer_outputs { {{}} , {{}}, {{}} }
            input_to_pass = result;
        }

        return result;
    }

    void back_propogation(vector<vector<double>>predicted,vector<vector<double>>target){

        //! Clear previous gradients
        d_w.clear();
        d_z.clear();

        int num_neural_layers = num_layers-1; //Num of actual layers excluding input layer
        target = Transpose(target);
        int weight_index = num_neural_layers-1;
        int dw_index = 0;

        vector<vector<double>>dz_l = Subtract(predicted,target);

        //Printing predicted and target for debugging
        // cout<<endl<<"debugging"<<endl;
        // print_vec2D(predicted);
        // print_vec2D(target);


        vector<vector<double>>dw_l = Cross_MUl(dz_l,Transpose(inputs_for_each_layer[weight_index]));
        d_z.push_back(dz_l);
        d_w.push_back(dw_l);

        weight_index--;
        // inputs_for_each_layer,d_w,d_z,layer_outputs,NetworkWeights

        for(int i=0;i<num_neural_layers-1;i++){
            // GENERAL FORMULA FOR EACH ITERATION : NetworkWeights[weight_index].T x d_z[dw_index] x inputs_for_each_layer[weight_index].T

            vector<vector<double>>temp_dz = Cross_MUl(Transpose(NetworkWeights[weight_index]),d_z[dw_index]);
            d_z.push_back(temp_dz);
            vector<vector<double>>temp_dw = Cross_MUl(temp_dz,Transpose(inputs_for_each_layer[weight_index]));
            d_w.push_back(temp_dw);
            weight_index--;
            dw_index++;

        }

        // reverse the vector
        reverse(d_w.begin(),d_w.end());
        reverse(d_z.begin(),d_z.end());

        for(int i=0;i<num_neural_layers;i++){
            NetworkWeights[i] = Subtract(NetworkWeights[i],(scalar_mul_with_matrix(d_w[i],learning_rate)));
        }

    }



}; 
int main(){
    
    vector<int>layers = {2,2,2,2};
    NeuralNetwork nn(layers);
    

    // TEST
    vector<vector<double>> input = {{1.0,2.0}};
    int epochs = 3;

    for(int i=1;i<=epochs;i++){
    cout<<"epoch : "<<i<<endl;
    cout<<"Before weight update : "<<endl;
    nn.print_network_weights();
    vector<vector<double>> feed_forward_output = nn.feed_forward(input);
    /*debug step */ cout<<"completed "<<i<<" feedforward"<<endl;
    print_vec2D(feed_forward_output);
    nn.back_propogation(feed_forward_output,input);
    cout<<"After weight update : "<<endl;
    nn.print_network_weights();
    cout<<endl;
    }

    cout<<endl<<"TESTING : "<<endl;
    vector<vector<double>> test = nn.feed_forward(input);
    print_vec2D(test);
    
}