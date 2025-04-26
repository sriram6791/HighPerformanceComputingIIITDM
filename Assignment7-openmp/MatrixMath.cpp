#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

/* dot function for integers:
   If two vectors are given {1,2,3} and {1,2,3} it returns 14 as a scalar (return only an integer) */
int dot(vector<int> vec1, vector<int> vec2) {
    int n = vec1.size();
    int dot_sum = 0;
    #pragma omp parallel for reduction(+:dot_sum)
    for (int i = 0; i < n; i++) {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}

double dot(vector<double> vec1, vector<double> vec2) {
    int n = vec1.size();
    double dot_sum = 0;
    #pragma omp parallel for reduction(+:dot_sum)
    for (int i = 0; i < n; i++) {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}

/* When passed {1,2,3} prints the vector */
void print_vec1D(vector<int> vec) {
    int n = vec.size();
    for (int i = 0; i < n; i++) {
        cout << vec[i] << " ";
    }
    cout << endl;
}

/* When passed a 2D matrix, this function prints it */
void print_vec2D(vector<vector<int>> matrix) {
    int rows = matrix.size();
    int col = matrix[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < col; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void print_vec2D(vector<vector<double>> matrix) {
    int rows = matrix.size();
    int col = matrix[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < col; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// print_weights function for printing weight matrices (left sequential)
void print_weights(vector<vector<double>> weights) {
    int rows = weights.size();
    int col = weights[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < col; j++) {
            cout << weights[i][j] << " ";
        }
        cout << endl;
    }
}

/* When a matrix is passed, for example:
   {1,2,3,4}
   {4,5,6,7}
   {7,8,9,6}
   When col = 0 it returns {1,4,7} as a vector */
// For int
vector<int> return_vertical_col(vector<vector<int>> mat, int col) {
    int rows = mat.size();
    vector<int> column(rows);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        column[i] = mat[i][col];
    }
    return column;
}
// For double
vector<double> return_vertical_col(vector<vector<double>> mat, int col) {
    int rows = mat.size();
    vector<double> column(rows);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        column[i] = mat[i][col];
    }
    return column;
}

/* Matrix multiplication (Cross_MUl) for int matrices */
vector<vector<int>> Cross_MUl(vector<vector<int>> mat1, vector<vector<int>> mat2) {
    int mat1_rows = mat1.size();
    int mat1_col = mat1[0].size();
    int mat2_rows = mat2.size();
    int mat2_col = mat2[0].size();
    if (mat1_col != mat2_rows) {
        // Error handling if needed
    }
    // Preallocate result matrix
    vector<vector<int>> result(mat1_rows, vector<int>(mat2_col, 0));
    // Parallelize over both i and j loops.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat1_rows; i++) {
        for (int j = 0; j < mat2_col; j++) {
            // Compute the j-th column of mat2 (each iteration is independent)
            vector<int> curr_col = return_vertical_col(mat2, j);
            result[i][j] = dot(mat1[i], curr_col);
        }
    }
    return result;
}

/* Matrix multiplication (Cross_MUl) for double matrices */
vector<vector<double>> Cross_MUl(vector<vector<double>> mat1, vector<vector<double>> mat2) {
    int mat1_rows = mat1.size();
    int mat1_col = mat1[0].size();
    int mat2_rows = mat2.size();
    int mat2_col = mat2[0].size();
    if (mat1_col != mat2_rows) {
        // Error handling if needed
    }
    vector<vector<double>> result(mat1_rows, vector<double>(mat2_col, 0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat1_rows; i++) {
        for (int j = 0; j < mat2_col; j++) {
            vector<double> curr_col = return_vertical_col(mat2, j);
            result[i][j] = dot(mat1[i], curr_col);
        }
    }
    return result;
}

/* Transpose for double matrices */
vector<vector<double>> Transpose(const vector<vector<double>> &matrix) {
    if (matrix.empty())
        return {};
    int num_rows = matrix.size();
    int num_cols = matrix[0].size();
    vector<vector<double>> transposed_matrix(num_cols, vector<double>(num_rows, 0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_rows; j++) {
            transposed_matrix[i][j] = matrix[j][i];
        }
    }
    return transposed_matrix;
}

/* Transpose for int matrices */
vector<vector<int>> Transpose(const vector<vector<int>> &matrix) {
    if (matrix.empty())
        return {};
    int num_rows = matrix.size();
    int num_cols = matrix[0].size();
    vector<vector<int>> transposed_matrix(num_cols, vector<int>(num_rows, 0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_rows; j++) {
            transposed_matrix[i][j] = matrix[j][i];
        }
    }
    return transposed_matrix;
}

/* Element-wise subtraction between two matrices: returns (mat1 - mat2) */
vector<vector<double>> Subtract(vector<vector<double>> mat1, vector<vector<double>> mat2) {
    int rows = mat1.size();
    int col = mat1[0].size();
    vector<vector<double>> result(rows, vector<double>(col, 0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < col; j++) {
            result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return result;
}

/* Multiply each element of a matrix by a scalar (learning_rate) */
vector<vector<double>> scalar_mul_with_matrix(vector<vector<double>> matrix, double learning_rate) {
    int rows = matrix.size();
    int columns = matrix[0].size();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] *= learning_rate;
        }
    }
    return matrix;
}

/* Calculate the mean squared error between two matrices */
double calculate_error(vector<vector<double>> test, vector<vector<double>> input) {
    int rows = test.size();
    int col = test[0].size();
    double error = 0;
    #pragma omp parallel for reduction(+:error) collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < col; j++) {
            double diff = test[i][j] - input[i][j];
            error += diff * diff;
        }
    }
    error = error / (rows * col);
    return error;
}

/* Aggregate error calculation over all samples.
   For each row of delta, compute the average error.
   For example, for a sample [1.0,2.0,3.0] it computes 2.0,
   and for [4.0,5.0,6.0] it computes 5.0.
   Returns a matrix with one value per row.
*/
vector<vector<double>> error_cal(vector<vector<double>> delta, int m) {
    int rows = delta.size();
    int col = delta[0].size();
    vector<vector<double>> ans(rows, vector<double>(1, 0));
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < col; j++) {
            sum += delta[i][j];
        }
        ans[i][0] = sum / m;
    }
    return ans;
}

// int main(){
//     // The main function here is only for debugging purposes.
//     // You can call and test your functions as needed.
    
//     // Example: Dot product tests
//     vector<int> v1 = {1,2,3};
//     vector<int> v2 = {1,2,3};
//     cout << "dot(int): " << dot(v1,v2) << endl; // Should output 14

//     vector<double> vd1 = {1.0,2.0,3.0};
//     vector<double> vd2 = {1.0,2.0,3.0};
//     cout << "dot(double): " << dot(vd1,vd2) << endl; // Should output 14.0

//     // Example: Return vertical column for a 2D matrix
//     vector<vector<int>> matrix_int = {{1,2,3}, {4,5,6}, {7,8,9}};
//     vector<int> col0 = return_vertical_col(matrix_int, 0);
//     cout << "Vertical col (int): ";
//     print_vec1D(col0);

//     vector<vector<double>> matrix_double = {{1.0,2.0,3.0}, {4.0,5.0,6.0}, {7.0,8.0,9.0}};
//     vector<double> col0_d = return_vertical_col(matrix_double, 0);
//     cout << "Vertical col (double): ";
//     for(auto d: col0_d)
//         cout << d << " ";
//     cout << endl;

//     // You can similarly test other functions such as Cross_MUl, Transpose, Subtract, etc.

//     return 0;
// }
