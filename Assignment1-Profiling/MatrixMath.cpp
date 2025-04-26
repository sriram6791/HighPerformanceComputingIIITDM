#include <iostream>
#include <vector>
using namespace std;
/* dot function description:
if two vectors are given {1,2,3} and {1,2,3} it returns 14 as a scalar (return only a integer) */
int dot(vector<int> vec1, vector<int> vec2)
{
    int n = vec1.size();
    int m = vec2.size();
    if (n != m)
    {
        // cerr << "Error: cannot do dot product of vectors with different shapes" << endl;
    }
    int dot_sum = 0;
    for (int i = 0; i < n; i++)
    {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}
double dot(vector<double> vec1, vector<double> vec2)
{
    int n = vec1.size();
    int m = vec2.size();
    if (n != m)
    {
        // cerr << "Error: cannot do dot product of vectors with different shapes" << endl;
    }
    double dot_sum = 0;
    for (int i = 0; i < n; i++)
    {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}
/* when passed {1,2,3} prints the vector */
void print_vec1D(vector<int> vec)
{
    int n = vec.size();
    for (int i = 0; i < n; i++)
    {
        cout << vec[i] << " ";
    }
    cout << endl;
}
/* When passed with a 2D matrix, this function prints it */
void print_vec2D(vector<vector<int>> matrix)
{
    int rows = matrix.size();
    int col = matrix[0].size();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
void print_vec2D(vector<vector<double>> matrix)
{
    int rows = matrix.size();
    int col = matrix[0].size();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
// print weights function is written because print_2D function only takes int vectors and prints them
void print_weights(vector<vector<double>> weights)
{
    int rows = weights.size();
    int col = weights[0].size();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << weights[i][j] << " ";
        }
        cout << endl;
    }
}
/*when a matrix is passed
{1,2,3,4}
{4,5,6,7}
{7,8,9,6}
when col = 0 it returns {1,4,7} as a vector */
vector<int> return_vertical_col(vector<vector<int>> mat, int col)
{
    vector<int> column;
    int rows = mat.size();
    for (int i = 0; i < rows; i++)
    {
        column.push_back(mat[i][col]);
    }
    return column;
}
vector<double> return_vertical_col(vector<vector<double>> mat, int col)
{
    vector<double> column;
    int rows = mat.size();
    for (int i = 0; i < rows; i++)
    {
        column.push_back(mat[i][col]);
    }
    return column;
}
/* This function does the Matrix cross mutiplication sizes should be compatible*/
vector<vector<int>> Cross_MUl(vector<vector<int>> mat1, vector<vector<int>> mat2)
{
    int mat1_rows = mat1.size();
    int mat1_col = mat1[0].size();
    int mat2_rows = mat2.size();
    int mat2_col = mat2[0].size();
    if (mat1_col != mat2_rows)
    {
        // cerr << "Error : Mat1 and Mat2 are not compatible for matrix multiplication" << endl;
    }
    vector<vector<int>> result;
    for (int i = 0; i < mat1_rows; i++)
    {
        vector<int> result_row;
        vector<int> curr_row = mat1[i];
        for (int j = 0; j < mat2_col; j++)
        {
            vector<int> curr_col = return_vertical_col(mat2, j);
            int val = dot(curr_row, curr_col);
            result_row.push_back(val);
        }
        result.push_back(result_row);
    }
    return result;
}
vector<vector<double>> Cross_MUl(vector<vector<double>> mat1, vector<vector<double>> mat2)
{
    int mat1_rows = mat1.size();
    int mat1_col = mat1[0].size();
    int mat2_rows = mat2.size();
    int mat2_col = mat2[0].size();
    if (mat1_col != mat2_rows)
    {
        // cerr << "Error : Mat1 and Mat2 are not compatible for matrix multiplication" << endl;
    }
    vector<vector<double>> result;
    for (int i = 0; i < mat1_rows; i++)
    {
        vector<double> result_row;
        vector<double> curr_row = mat1[i];
        for (int j = 0; j < mat2_col; j++)
        {
            vector<double> curr_col = return_vertical_col(mat2, j);
            double val = dot(curr_row, curr_col);
            result_row.push_back(val);
        }
        result.push_back(result_row);
    }
    return result;
}
vector<vector<double>> Transpose(const vector<vector<double>> &matrix)
{
    // ALWAYS NEED TO PASS A 2D MATRIX EX: {{1,2,3}} OR {{1,2,3},{4,5,6}}
    if (matrix.empty())
        return {};
    // int num_rows = matrix.size();
    int num_col = matrix[0].size();

    vector<vector<double>> transposed_matrix;
    for (int i = 0; i < num_col; i++)
    {
        transposed_matrix.push_back(return_vertical_col(matrix, i));
    }
    return transposed_matrix;
}
vector<vector<int>> Transpose(const vector<vector<int>> &matrix)
{
    // ALWAYS NEED TO PASS A 2D MATRIX EX: {{1,2,3}} OR {{1,2,3},{4,5,6}}
    if (matrix.empty())
        return {};
    // int num_rows = matrix.size();
    int num_col = matrix[0].size();

    vector<vector<int>> transposed_matrix;
    for (int i = 0; i < num_col; i++)
    {
        transposed_matrix.push_back(return_vertical_col(matrix, i));
    }
    return transposed_matrix;
}
// Subtract function is used to do element wise subtraction between two given matrices DOES MAT1 - MAT2 (ORDER IS IMPORTANT)
vector<vector<double>> Subtract(vector<vector<double>> mat1, vector<vector<double>> mat2)
{
    int rows = mat1.size();
    int col = mat1[0].size();
    vector<vector<double>> result;
    for (int i = 0; i < rows; i++)
    {
        vector<double> result_row;
        for (int j = 0; j < col; j++)
        {
            result_row.push_back((mat1[i][j] - mat2[i][j]));
        }
        result.push_back(result_row);
    }
    return result;
}
vector<vector<double>> scalar_mul_with_matrix(vector<vector<double>> matrix, double learning_rate)
{
    int rows = matrix.size();
    int columns = matrix[0].size();

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            matrix[i][j] = learning_rate * matrix[i][j];
        }
    }
    return matrix;
}
double calculate_error(vector<vector<double>> test, vector<vector<double>> input)
{
    int rows = test.size();
    int col = test[0].size();
    double error = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < col; j++)
        {
            error += (test[i][j] - input[i][j]) * (test[i][j] - input[i][j]);
        }
    }

    error = error / (rows * col);
    return error;
}
/* The below function error_cal : does aggregate of all errors of all samples
[1.0 2.0 3.0 ]= 2.0
[4.0 5.0 6.0 ]= 12.5
returns
2.0
12.5
*/
vector<vector<double>> error_cal(vector<vector<double>> delta, int m)
{
    int rows = delta.size();
    int col = delta[0].size();
    vector<vector<double>> ans;
    for (int i = 0; i < rows; i++)
    {
        vector<double> curr_row;
        double sum = 0;
        for (int j = 0; j < col; j++)
        {
            sum += delta[i][j];
        }
        sum = sum / m;
        curr_row.push_back(sum);
        ans.push_back(curr_row);
    }
    return ans;
}
// int main(){
//     // TO DEBUG VECTOR DOT PRODUCT
//     // vector<int>v1 = {1,2,3};
//     // vector<int>v2 = {2,3,5};
//     // cout<<dot(v1,v2)<<endl;
//     //TO DEBUG VECTOR COLUMN RETURN
//     // vector<vector<int>> matrix = {{1,2,3},{1,2,3}};
//     // vector<int>col = return_vertical_col(matrix,0);
//     // print_vec1D(col);
//     // print_vec2D(matrix);
//     // vector<vector<int>> matrix1 = {{1,2},{1,2}};
//     // vector<vector<int>> matrix2 = {{1,2},{1,2},{1,2}};
//     // vector<vector<int>>result = Cross_MUl(matrix1,matrix2);
//     // print_vec2D(result);
//     return 0;
// }
