#ifndef MATRIXMATH_H
#define MATRIXMATH_H
#include<iostream>
#include<vector>
using namespace std;
void print_vec1D(vector<int>vec);
void print_vec2D(vector<vector<int>>matrix);
void print_vec2D(vector<vector<double>>matrix);
void print_weights(vector<vector<double>>weights);

int dot(vector<int>vec1,vector<int>vec2);
double dot(vector<double>vec1,vector<double>vec2); 
vector<int> return_vertical_col(vector<vector<int>>mat,int col);
vector<vector<int>> Cross_MUl(vector<vector<int>>mat1,vector<vector<int>>mat2);
vector<vector<double>> Cross_MUl(vector<vector<double>>mat1,vector<vector<double>>mat2);
vector<vector<double>> Transpose(const vector<vector<double>>& matrix);
vector<vector<int>> Transpose(const vector<vector<int>>& matrix);
vector<vector<double>> Subtract(vector<vector<double>>mat1,vector<vector<double>>mat2);
vector<vector<double>> scalar_mul_with_matrix(vector<vector<double>>matrix,double learning_rate);
double calculate_error(vector<vector<double>> test,vector<vector<double>> input);
vector<vector<double>> error_cal(vector<vector<double>> delta,int m);
#endif
