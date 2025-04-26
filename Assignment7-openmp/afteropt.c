// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// #include <string.h>
// #include <math.h>
// #include <omp.h>

// /* ------------------------ MATRIX MATH FUNCTIONS ------------------------ */

// /* Dot product for integer vectors */
// double dot_int(const int *vec1, const int *vec2, int n) {
//     int dot_sum = 0;
//     #pragma omp parallel for reduction(+:dot_sum)
//     for (int i = 0; i < n; i++) {
//         dot_sum += (vec1[i] * vec2[i]);
//     }
//     return dot_sum;
// }

// /* Dot product for double vectors */
// double dot_double(const double *vec1, const double *vec2, int n) {
//     double dot_sum = 0;
//     #pragma omp parallel for reduction(+:dot_sum)
//     for (int i = 0; i < n; i++) {
//         dot_sum += (vec1[i] * vec2[i]);
//     }
//     return dot_sum;
// }

// /* Print a 1D integer vector (serial) */
// void print_vec1D_int(int *vec, int n) {
//     for (int i = 0; i < n; i++) {
//         printf("%d ", vec[i]);
//     }
//     printf("\n");
// }

// /* Print a 2D integer matrix (serial) */
// void print_vec2D_int(int **matrix, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%d ", matrix[i][j]);
//         }
//         printf("\n");
//     }
// }

// /* Print a 2D double matrix (serial) */
// void print_vec2D_double(double **matrix, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%lf ", matrix[i][j]);
//         }
//         printf("\n");
//     }
// }

// /* Print a weights matrix (2D double) */
// void print_weights(double **weights, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%lf ", weights[i][j]);
//         }
//         printf("\n");
//     }
// }

// /* Return a vertical column from a 2D int matrix */
// int *return_vertical_col_int(int **mat, int rows, int col) {
//     int *column = (int *)malloc(rows * sizeof(int));
//     #pragma omp parallel for
//     for (int i = 0; i < rows; i++) {
//         column[i] = mat[i][col];
//     }
//     return column;
// }

// /* Return a vertical column from a 2D double matrix */
// double *return_vertical_col_double(double **mat, int rows, int col) {
//     double *column = (double *)malloc(rows * sizeof(double));
//     #pragma omp parallel for
//     for (int i = 0; i < rows; i++) {
//         column[i] = mat[i][col];
//     }
//     return column;
// }

// /* Matrix multiplication for 2D int matrices */
// int **Cross_MUl_int(int **mat1, int r1, int c1, int **mat2, int r2, int c2) {
//     int **result = (int **)malloc(r1 * sizeof(int *));
//     #pragma omp parallel for
//     for (int i = 0; i < r1; i++) {
//         result[i] = (int *)malloc(c2 * sizeof(int));
//         for (int j = 0; j < c2; j++) {
//             int *curr_col = return_vertical_col_int(mat2, r2, j);
//             result[i][j] = dot_int(mat1[i], curr_col, c1);
//             free(curr_col);
//         }
//     }
//     return result;
// }

// /* Matrix multiplication for 2D double matrices */
// double **Cross_MUl_double(double **mat1, int r1, int c1, double **mat2, int r2, int c2) {
//     double **result = (double **)malloc(r1 * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < r1; i++) {
//         result[i] = (double *)malloc(c2 * sizeof(double));
//         for (int j = 0; j < c2; j++) {
//             double *curr_col = return_vertical_col_double(mat2, r2, j);
//             result[i][j] = dot_double(mat1[i], curr_col, c1);
//             free(curr_col);
//         }
//     }
//     return result;
// }

// /* Transpose a 2D int matrix */
// int **Transpose_int(int **matrix, int rows, int cols) {
//     int **transposed = (int **)malloc(cols * sizeof(int *));
//     #pragma omp parallel for
//     for (int i = 0; i < cols; i++) {
//         transposed[i] = return_vertical_col_int(matrix, rows, i);
//     }
//     return transposed;
// }

// /* Transpose a 2D double matrix */
// double **Transpose_double(double **matrix, int rows, int cols) {
//     double **transposed = (double **)malloc(cols * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < cols; i++) {
//         transposed[i] = return_vertical_col_double(matrix, rows, i);
//     }
//     return transposed;
// }

// /* Element-wise subtraction for 2D double matrices: mat1 - mat2 */
// double **Subtract(double **mat1, double **mat2, int rows, int cols) {
//     double **result = (double **)malloc(rows * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < rows; i++) {
//         result[i] = (double *)malloc(cols * sizeof(double));
//         for (int j = 0; j < cols; j++) {
//             result[i][j] = mat1[i][j] - mat2[i][j];
//         }
//     }
//     return result;
// }

// /* Multiply each element of a 2D double matrix by a scalar */
// double **scalar_mul_with_matrix(double **matrix, int rows, int cols, double scalar) {
//     double **result = (double **)malloc(rows * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < rows; i++) {
//         result[i] = (double *)malloc(cols * sizeof(double));
//         for (int j = 0; j < cols; j++) {
//             result[i][j] = scalar * matrix[i][j];
//         }
//     }
//     return result;
// }

// /* Calculate the mean squared error between two 2D double matrices */
// double calculate_error(double **test, double **input, int rows, int cols) {
//     double error = 0;
//     #pragma omp parallel for reduction(+:error)
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             double diff = test[i][j] - input[i][j];
//             error += diff * diff;
//         }
//     }
//     return error / (rows * cols);
// }

// /* For each row in delta, compute the average and return as a column vector */
// double **error_cal(double **delta, int rows, int cols, int m) {
//     double **ans = (double **)malloc(rows * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < rows; i++) {
//         ans[i] = (double *)malloc(sizeof(double));
//         double sum = 0;
//         for (int j = 0; j < cols; j++) {
//             sum += delta[i][j];
//         }
//         ans[i][0] = sum / m;
//     }
//     return ans;
// }

// /* ------------------------ UTILS ------------------------ */

// /* Return a random number in [0,1) */
// double get_random_number() {
//     return (double)rand() / ((double)RAND_MAX);
// }

// /* Initialize weights: returns a 2D double matrix of size dim1 x dim2 with random values.
//    Each weight is scaled by 1/sqrt(dim2) to avoid huge sums.
// */
// double **initialize_weights(int dim1, int dim2) {
//     double **weights = (double **)malloc(dim1 * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < dim1; i++) {
//         weights[i] = (double *)malloc(dim2 * sizeof(double));
//         for (int j = 0; j < dim2; j++) {
//             weights[i][j] = get_random_number() / sqrt((double)dim2);
//         }
//     }
//     return weights;
// }

// /* ------------------------ NEURAL NETWORK ------------------------ */

// typedef struct {
//     double **NetworkWeights[100]; /* Weight matrices */
//     int nodes[100];               /* Number of nodes per layer */
//     int num_layers;               /* Total layers including input */
//     double **layer_outputs[100];  /* Outputs from each layer */
//     double **d_w[100];            /* Weight gradients */
//     double **d_z[100];            /* Pre-activation gradients */
//     double **inputs_for_each_layer[100]; /* Inputs for each layer */
//     double learning_rate;
//     int NetworkWeights_dims[100][2];
//     int layer_outputs_dims[100][2];
//     int d_w_dims[100][2];
//     int d_z_dims[100][2];
//     int inputs_dims[100][2];
// } NeuralNetwork;

// NeuralNetwork *create_NeuralNetwork(int *nodes_count_in_each_layer, int num) {
//     NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
//     nn->num_layers = num;
//     nn->learning_rate = 0.000000000001;  // Adjust if needed
//     for (int i = 0; i < num; i++) {
//         nn->nodes[i] = nodes_count_in_each_layer[i];
//     }
//     for (int i = 1; i < num; i++) {
//         int r = nodes_count_in_each_layer[i];
//         int c = nodes_count_in_each_layer[i - 1];
//         nn->NetworkWeights[i - 1] = initialize_weights(r, c);
//         nn->NetworkWeights_dims[i - 1][0] = r;
//         nn->NetworkWeights_dims[i - 1][1] = c;
//         nn->layer_outputs[i - 1] = NULL;
//         nn->inputs_for_each_layer[i - 1] = NULL;
//         nn->d_w[i - 1] = NULL;
//         nn->d_z[i - 1] = NULL;
//     }
//     return nn;
// }

// void print_network_weights(NeuralNetwork *nn) {
//     int layers = nn->num_layers - 1;
//     for (int i = 0; i < layers; i++) {
//         printf("Layer-%d\n", i);
//         printf("Input dim: %d, Num nodes: %d\n", nn->nodes[i], nn->nodes[i+1]);
//         printf("Weight dimensions: %d x %d\n", nn->nodes[i], nn->nodes[i+1]);
//         printf("Weights:\n");
//         print_weights(nn->NetworkWeights[i], nn->NetworkWeights_dims[i][0], nn->NetworkWeights_dims[i][1]);
//         printf("\n");
//     }
// }

// double **ReLU(double **input, int rows, int cols) {
//     double **output = (double **)malloc(rows * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < rows; i++) {
//         output[i] = (double *)malloc(cols * sizeof(double));
//         for (int j = 0; j < cols; j++) {
//             output[i][j] = (input[i][j] > 0 ? input[i][j] : 0);
//         }
//     }
//     return output;
// }

// double **feed_forward(NeuralNetwork *nn, double **input, int input_rows, int input_cols, int *out_rows, int *out_cols) {
//     int layers = nn->num_layers - 1;
//     int i;
//     double **input_to_pass = Transpose_double(input, input_rows, input_cols);
//     int current_rows = input_cols;
//     int current_cols = input_rows;
//     for (i = 0; i < layers; i++) {
//         nn->inputs_for_each_layer[i] = input_to_pass;
//         nn->inputs_dims[i][0] = current_rows;
//         nn->inputs_dims[i][1] = current_cols;
//         int w_r = nn->NetworkWeights_dims[i][0];
//         int w_c = nn->NetworkWeights_dims[i][1];
//         double **result = Cross_MUl_double(nn->NetworkWeights[i], w_r, w_c, input_to_pass, current_rows, current_cols);
//         result = ReLU(result, w_r, current_cols);  // Apply ReLU activation
//         nn->layer_outputs[i] = result;
//         nn->layer_outputs_dims[i][0] = w_r;
//         nn->layer_outputs_dims[i][1] = current_cols;
//         input_to_pass = result;
//         current_rows = w_r;
//         /* current_cols remains unchanged */
//     }
//     *out_rows = current_rows;
//     *out_cols = current_cols;
//     return input_to_pass;
// }

// void back_propogation(NeuralNetwork *nn, double **predicted, int pred_rows, int pred_cols, double **target, int target_rows, int target_cols) {
//     int layers = nn->num_layers - 1;
//     int number_of_samples = target_rows;
//     double **target_T = Transpose_double(target, target_rows, target_cols);
//     double **delta = Subtract(predicted, target_T, pred_rows, pred_cols);
//     for (int i = 0; i < target_cols; i++) {
//         free(target_T[i]);
//     }
//     free(target_T);
//     double **delta_ec = error_cal(delta, pred_rows, pred_cols, number_of_samples);
//     for (int i = 0; i < pred_rows; i++) {
//         free(delta[i]);
//     }
//     free(delta);
//     nn->d_z[0] = delta_ec;
//     nn->d_z_dims[0][0] = pred_rows;
//     nn->d_z_dims[0][1] = 1;
//     double **inputs_last_T = Transpose_double(nn->inputs_for_each_layer[layers - 1],
//                                                nn->inputs_dims[layers - 1][0],
//                                                nn->inputs_dims[layers - 1][1]);
//     double **grad = Cross_MUl_double(delta_ec, pred_rows, 1,
//                                      inputs_last_T,
//                                      nn->inputs_dims[layers - 1][1],
//                                      nn->inputs_dims[layers - 1][0]);
//     for (int i = 0; i < nn->inputs_dims[layers - 1][1]; i++) {
//         free(inputs_last_T[i]);
//     }
//     free(inputs_last_T);
//     nn->d_w[0] = grad;
//     nn->d_w_dims[0][0] = pred_rows;
//     nn->d_w_dims[0][1] = nn->inputs_dims[layers - 1][0];
//     int l;
//     for (l = layers - 1; l > 0; l--) {
//         double **W_next_T = Transpose_double(nn->NetworkWeights[l],
//                                              nn->NetworkWeights_dims[l][0],
//                                              nn->NetworkWeights_dims[l][1]);
//         double **delta_new = Cross_MUl_double(W_next_T,
//                                               nn->NetworkWeights_dims[l][1],
//                                               nn->NetworkWeights_dims[l][0],
//                                               delta_ec,
//                                               nn->d_z_dims[0][0],
//                                               nn->d_z_dims[0][1]);
//         for (int i = 0; i < nn->NetworkWeights_dims[l][1]; i++) {
//             free(W_next_T[i]);
//         }
//         free(W_next_T);
//         nn->d_z[layers - l] = delta_new;
//         nn->d_z_dims[layers - l][0] = nn->NetworkWeights_dims[l][1];
//         nn->d_z_dims[layers - l][1] = 1;
//         double **input_layer_T = Transpose_double(nn->inputs_for_each_layer[l - 1],
//                                                    nn->inputs_dims[l - 1][0],
//                                                    nn->inputs_dims[l - 1][1]);
//         double **grad_new = Cross_MUl_double(delta_new,
//                                              nn->d_z_dims[layers - l][0],
//                                              1,
//                                              input_layer_T,
//                                              nn->inputs_dims[l - 1][1],
//                                              nn->inputs_dims[l - 1][0]);
//         for (int i = 0; i < nn->inputs_dims[l - 1][1]; i++) {
//             free(input_layer_T[i]);
//         }
//         free(input_layer_T);
//         nn->d_w[layers - l] = grad_new;
//         nn->d_w_dims[layers - l][0] = nn->d_z_dims[layers - l][0];
//         nn->d_w_dims[layers - l][1] = nn->inputs_dims[l - 1][0];
//     }
//     #pragma omp parallel for
//     for (int i = 0; i < layers / 2; i++) {
//         double **temp = nn->d_w[i];
//         int temp_dims0 = nn->d_w_dims[i][0], temp_dims1 = nn->d_w_dims[i][1];
//         nn->d_w[i] = nn->d_w[layers - 1 - i];
//         nn->d_w_dims[i][0] = nn->d_w_dims[layers - 1 - i][0];
//         nn->d_w_dims[i][1] = nn->d_w_dims[layers - 1 - i][1];
//         nn->d_w[layers - 1 - i] = temp;
//         nn->d_w_dims[layers - 1 - i][0] = temp_dims0;
//         nn->d_w_dims[layers - 1 - i][1] = temp_dims1;

//         temp = nn->d_z[i];
//         int temp_z0 = nn->d_z_dims[i][0], temp_z1 = nn->d_z_dims[i][1];
//         nn->d_z[i] = nn->d_z[layers - 1 - i];
//         nn->d_z_dims[i][0] = nn->d_z_dims[layers - 1 - i][0];
//         nn->d_z_dims[i][1] = nn->d_z_dims[layers - 1 - i][1];
//         nn->d_z[layers - 1 - i] = temp;
//         nn->d_z_dims[layers - 1 - i][0] = temp_z0;
//         nn->d_z_dims[layers - 1 - i][1] = temp_z1;
//     }
//     #pragma omp parallel for
//     for (int i = 0; i < layers; i++) {
//         double **scaled = scalar_mul_with_matrix(nn->d_w[i],
//                                                  nn->d_w_dims[i][0],
//                                                  nn->d_w_dims[i][1],
//                                                  nn->learning_rate);
//         double **newW = Subtract(nn->NetworkWeights[i], scaled,
//                                  nn->NetworkWeights_dims[i][0],
//                                  nn->NetworkWeights_dims[i][1]);
//         int r = nn->NetworkWeights_dims[i][0];
//         for (int j = 0; j < r; j++) {
//             free(nn->NetworkWeights[i][j]);
//         }
//         free(nn->NetworkWeights[i]);
//         nn->NetworkWeights[i] = newW;
//         for (int j = 0; j < nn->NetworkWeights_dims[i][0]; j++) {
//             free(scaled[j]);
//         }
//         free(scaled);
//     }
// }

// double **loadDataset(const char *filename, int num_pixels, int *num_samples) {
//     FILE *fp = fopen(filename, "r");
//     if (!fp) {
//         fprintf(stderr, "Error opening file %s\n", filename);
//         exit(1);
//     }
//     char line[10000];
//     int count = 0;
//     while (fgets(line, sizeof(line), fp) != NULL) {
//         if (strlen(line) > 1)
//             count++;
//     }
//     rewind(fp);
//     double **dataset = (double **)malloc(count * sizeof(double *));
//     #pragma omp parallel for
//     for (int i = 0; i < count; i++) {
//         dataset[i] = (double *)malloc(num_pixels * sizeof(double));
//     }
//     int sample_index = 0;
//     while (fgets(line, sizeof(line), fp) != NULL && sample_index < count) {
//         if (strlen(line) <= 1)
//             continue;
//         char *token = strtok(line, " \t\n");
//         int pixel_index = 0;
//         while (token != NULL && pixel_index < num_pixels) {
//             dataset[sample_index][pixel_index] = atof(token);
//             token = strtok(NULL, " \t\n");
//             pixel_index++;
//         }
//         sample_index++;
//     }
//     fclose(fp);
//     *num_samples = count;
//     return dataset;
// }

// int main() {
//     srand((unsigned int)time(NULL));
    
//     int num_pixels = 784;
//     int num_samples = 0;
//     double **input = loadDataset("dataset.txt", num_pixels, &num_samples);
//     printf("Loaded dataset with %d samples.\n", num_samples);
    
//     int layers_arr[] = {784, 128, 64, 128, 784};
//     int num_layers = sizeof(layers_arr) / sizeof(layers_arr[0]);
//     NeuralNetwork *nn = create_NeuralNetwork(layers_arr, num_layers);
    
//     int epochs = 100;
//     for (int i = 1; i <= epochs; i++) {
//         printf("Epoch: %d\n", i);
//         int out_r, out_c;
//         double **feed_forward_output = feed_forward(nn, input, num_samples, num_pixels, &out_r, &out_c);
//         printf("Completed %d feedforward pass.\n", i);
//         back_propogation(nn, feed_forward_output, out_r, out_c, input, num_samples, num_pixels);
//         printf("After weight update.\n\n");
//     }
    
//     double **input2 = (double **)malloc(1 * sizeof(double *));
//     input2[0] = (double *)malloc(num_pixels * sizeof(double));
//     for (int j = 0; j < num_pixels; j++) {
//         input2[0][j] = input[0][j];
//     }
//     printf("\nTESTING:\nTest Input:\n");
//     print_vec2D_double(input2, 1, num_pixels);
//     int test_r, test_c;
//     double **test = feed_forward(nn, input2, 1, num_pixels, &test_r, &test_c);
//     printf("Network Output:\n");
//     print_vec2D_double(test, test_r, test_c);
    
//     return 0;
// }




#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* ------------------------ MATRIX MATH FUNCTIONS ------------------------ */

/* Dot product for integer vectors */
double dot_int(const int *vec1, const int *vec2, int n) {
    int dot_sum = 0;
    #pragma omp parallel for reduction(+:dot_sum)
    for (int i = 0; i < n; i++) {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}

/* Dot product for double vectors */
double dot_double(const double *vec1, const double *vec2, int n) {
    double dot_sum = 0;
    #pragma omp parallel for reduction(+:dot_sum)
    for (int i = 0; i < n; i++) {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}

/* Print a 1D integer vector (serial) */
void print_vec1D_int(int *vec, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

/* Print a 2D integer matrix (serial) */
void print_vec2D_int(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

/* Print a 2D double matrix (serial) */
void print_vec2D_double(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

/* Print a weights matrix (2D double) */
void print_weights(double **weights, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", weights[i][j]);
        }
        printf("\n");
    }
}

/* Return a vertical column from a 2D int matrix */
int *return_vertical_col_int(int **mat, int rows, int col) {
    int *column = (int *)malloc(rows * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        column[i] = mat[i][col];
    }
    return column;
}

/* Return a vertical column from a 2D double matrix */
double *return_vertical_col_double(double **mat, int rows, int col) {
    double *column = (double *)malloc(rows * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        column[i] = mat[i][col];
    }
    return column;
}

/* Matrix multiplication for 2D int matrices */
int **Cross_MUl_int(int **mat1, int r1, int c1, int **mat2, int r2, int c2) {
    int **result = (int **)malloc(r1 * sizeof(int *));
    #pragma omp parallel for
    for (int i = 0; i < r1; i++) {
        result[i] = (int *)malloc(c2 * sizeof(int));
        for (int j = 0; j < c2; j++) {
            int *curr_col = return_vertical_col_int(mat2, r2, j);
            result[i][j] = dot_int(mat1[i], curr_col, c1);
            free(curr_col);
        }
    }
    return result;
}

/* Matrix multiplication for 2D double matrices */
double **Cross_MUl_double(double **mat1, int r1, int c1, double **mat2, int r2, int c2) {
    double **result = (double **)malloc(r1 * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < r1; i++) {
        result[i] = (double *)malloc(c2 * sizeof(double));
        for (int j = 0; j < c2; j++) {
            double *curr_col = return_vertical_col_double(mat2, r2, j);
            result[i][j] = dot_double(mat1[i], curr_col, c1);
            free(curr_col);
        }
    }
    return result;
}

/* Transpose a 2D int matrix */
int **Transpose_int(int **matrix, int rows, int cols) {
    int **transposed = (int **)malloc(cols * sizeof(int *));
    #pragma omp parallel for
    for (int i = 0; i < cols; i++) {
        transposed[i] = return_vertical_col_int(matrix, rows, i);
    }
    return transposed;
}

/* Transpose a 2D double matrix */
double **Transpose_double(double **matrix, int rows, int cols) {
    double **transposed = (double **)malloc(cols * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < cols; i++) {
        transposed[i] = return_vertical_col_double(matrix, rows, i);
    }
    return transposed;
}

/* Element-wise subtraction for 2D double matrices: mat1 - mat2 */
double **Subtract(double **mat1, double **mat2, int rows, int cols) {
    double **result = (double **)malloc(rows * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return result;
}

/* Multiply each element of a 2D double matrix by a scalar */
double **scalar_mul_with_matrix(double **matrix, int rows, int cols, double scalar) {
    double **result = (double **)malloc(rows * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = scalar * matrix[i][j];
        }
    }
    return result;
}

/* Calculate the mean squared error between two 2D double matrices */
double calculate_error(double **test, double **input, int rows, int cols) {
    double error = 0;
    #pragma omp parallel for reduction(+:error)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff = test[i][j] - input[i][j];
            error += diff * diff;
        }
    }
    return error / (rows * cols);
}

/* For each row in delta, compute the average and return as a column vector */
double **error_cal(double **delta, int rows, int cols, int m) {
    double **ans = (double **)malloc(rows * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        ans[i] = (double *)malloc(sizeof(double));
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += delta[i][j];
        }
        ans[i][0] = sum / m;
    }
    return ans;
}

/* ------------------------ UTILS ------------------------ */

/* Return a random number in [0,1) */
double get_random_number() {
    return (double)rand() / ((double)RAND_MAX);
}

/* Initialize weights: returns a 2D double matrix of size dim1 x dim2 with random values.
   Each weight is scaled by 1/sqrt(dim2) to avoid huge sums.
*/
double **initialize_weights(int dim1, int dim2) {
    double **weights = (double **)malloc(dim1 * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < dim1; i++) {
        weights[i] = (double *)malloc(dim2 * sizeof(double));
        for (int j = 0; j < dim2; j++) {
            weights[i][j] = get_random_number() / sqrt((double)dim2);
        }
    }
    return weights;
}

/* ------------------------ NEURAL NETWORK ------------------------ */

typedef struct {
    double **NetworkWeights[100]; /* Weight matrices */
    int nodes[100];               /* Number of nodes per layer */
    int num_layers;               /* Total layers including input */
    double **layer_outputs[100];  /* Outputs from each layer */
    double **d_w[100];            /* Weight gradients */
    double **d_z[100];            /* Pre-activation gradients */
    double **inputs_for_each_layer[100]; /* Inputs for each layer */
    double learning_rate;
    int NetworkWeights_dims[100][2];
    int layer_outputs_dims[100][2];
    int d_w_dims[100][2];
    int d_z_dims[100][2];
    int inputs_dims[100][2];
} NeuralNetwork;

NeuralNetwork *create_NeuralNetwork(int *nodes_count_in_each_layer, int num) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num;
    nn->learning_rate = 0.000000000001;  // Adjust if needed
    for (int i = 0; i < num; i++) {
        nn->nodes[i] = nodes_count_in_each_layer[i];
    }
    for (int i = 1; i < num; i++) {
        int r = nodes_count_in_each_layer[i];
        int c = nodes_count_in_each_layer[i - 1];
        nn->NetworkWeights[i - 1] = initialize_weights(r, c);
        nn->NetworkWeights_dims[i - 1][0] = r;
        nn->NetworkWeights_dims[i - 1][1] = c;
        nn->layer_outputs[i - 1] = NULL;
        nn->inputs_for_each_layer[i - 1] = NULL;
        nn->d_w[i - 1] = NULL;
        nn->d_z[i - 1] = NULL;
    }
    return nn;
}

void print_network_weights(NeuralNetwork *nn) {
    int layers = nn->num_layers - 1;
    for (int i = 0; i < layers; i++) {
        printf("Layer-%d\n", i);
        printf("Input dim: %d, Num nodes: %d\n", nn->nodes[i], nn->nodes[i+1]);
        printf("Weight dimensions: %d x %d\n", nn->nodes[i], nn->nodes[i+1]);
        printf("Weights:\n");
        print_weights(nn->NetworkWeights[i], nn->NetworkWeights_dims[i][0], nn->NetworkWeights_dims[i][1]);
        printf("\n");
    }
}

double **ReLU(double **input, int rows, int cols) {
    double **output = (double **)malloc(rows * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        output[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            output[i][j] = (input[i][j] > 0 ? input[i][j] : 0);
        }
    }
    return output;
}

double **feed_forward(NeuralNetwork *nn, double **input, int input_rows, int input_cols, int *out_rows, int *out_cols) {
    int layers = nn->num_layers - 1;
    int i;
    double **input_to_pass = Transpose_double(input, input_rows, input_cols);
    int current_rows = input_cols;
    int current_cols = input_rows;
    for (i = 0; i < layers; i++) {
        nn->inputs_for_each_layer[i] = input_to_pass;
        nn->inputs_dims[i][0] = current_rows;
        nn->inputs_dims[i][1] = current_cols;
        int w_r = nn->NetworkWeights_dims[i][0];
        int w_c = nn->NetworkWeights_dims[i][1];
        double **result = Cross_MUl_double(nn->NetworkWeights[i], w_r, w_c, input_to_pass, current_rows, current_cols);
        result = ReLU(result, w_r, current_cols);  // Apply ReLU activation
        nn->layer_outputs[i] = result;
        nn->layer_outputs_dims[i][0] = w_r;
        nn->layer_outputs_dims[i][1] = current_cols;
        input_to_pass = result;
        current_rows = w_r;
        /* current_cols remains unchanged */
    }
    *out_rows = current_rows;
    *out_cols = current_cols;
    return input_to_pass;
}

void back_propogation(NeuralNetwork *nn, double **predicted, int pred_rows, int pred_cols, double **target, int target_rows, int target_cols) {
    int layers = nn->num_layers - 1;
    int number_of_samples = target_rows;
    double **target_T = Transpose_double(target, target_rows, target_cols);
    double **delta = Subtract(predicted, target_T, pred_rows, pred_cols);
    for (int i = 0; i < target_cols; i++) {
        free(target_T[i]);
    }
    free(target_T);
    double **delta_ec = error_cal(delta, pred_rows, pred_cols, number_of_samples);
    for (int i = 0; i < pred_rows; i++) {
        free(delta[i]);
    }
    free(delta);
    nn->d_z[0] = delta_ec;
    nn->d_z_dims[0][0] = pred_rows;
    nn->d_z_dims[0][1] = 1;
    double **inputs_last_T = Transpose_double(nn->inputs_for_each_layer[layers - 1],
                                               nn->inputs_dims[layers - 1][0],
                                               nn->inputs_dims[layers - 1][1]);
    double **grad = Cross_MUl_double(delta_ec, pred_rows, 1,
                                     inputs_last_T,
                                     nn->inputs_dims[layers - 1][1],
                                     nn->inputs_dims[layers - 1][0]);
    for (int i = 0; i < nn->inputs_dims[layers - 1][1]; i++) {
        free(inputs_last_T[i]);
    }
    free(inputs_last_T);
    nn->d_w[0] = grad;
    nn->d_w_dims[0][0] = pred_rows;
    nn->d_w_dims[0][1] = nn->inputs_dims[layers - 1][0];
    int l;
    for (l = layers - 1; l > 0; l--) {
        double **W_next_T = Transpose_double(nn->NetworkWeights[l],
                                             nn->NetworkWeights_dims[l][0],
                                             nn->NetworkWeights_dims[l][1]);
        double **delta_new = Cross_MUl_double(W_next_T,
                                              nn->NetworkWeights_dims[l][1],
                                              nn->NetworkWeights_dims[l][0],
                                              delta_ec,
                                              nn->d_z_dims[0][0],
                                              nn->d_z_dims[0][1]);
        for (int i = 0; i < nn->NetworkWeights_dims[l][1]; i++) {
            free(W_next_T[i]);
        }
        free(W_next_T);
        nn->d_z[layers - l] = delta_new;
        nn->d_z_dims[layers - l][0] = nn->NetworkWeights_dims[l][1];
        nn->d_z_dims[layers - l][1] = 1;
        double **input_layer_T = Transpose_double(nn->inputs_for_each_layer[l - 1],
                                                   nn->inputs_dims[l - 1][0],
                                                   nn->inputs_dims[l - 1][1]);
        double **grad_new = Cross_MUl_double(delta_new,
                                             nn->d_z_dims[layers - l][0],
                                             1,
                                             input_layer_T,
                                             nn->inputs_dims[l - 1][1],
                                             nn->inputs_dims[l - 1][0]);
        for (int i = 0; i < nn->inputs_dims[l - 1][1]; i++) {
            free(input_layer_T[i]);
        }
        free(input_layer_T);
        nn->d_w[layers - l] = grad_new;
        nn->d_w_dims[layers - l][0] = nn->d_z_dims[layers - l][0];
        nn->d_w_dims[layers - l][1] = nn->inputs_dims[l - 1][0];
    }
    #pragma omp parallel for
    for (int i = 0; i < layers / 2; i++) {
        double **temp = nn->d_w[i];
        int temp_dims0 = nn->d_w_dims[i][0], temp_dims1 = nn->d_w_dims[i][1];
        nn->d_w[i] = nn->d_w[layers - 1 - i];
        nn->d_w_dims[i][0] = nn->d_w_dims[layers - 1 - i][0];
        nn->d_w_dims[i][1] = nn->d_w_dims[layers - 1 - i][1];
        nn->d_w[layers - 1 - i] = temp;
        nn->d_w_dims[layers - 1 - i][0] = temp_dims0;
        nn->d_w_dims[layers - 1 - i][1] = temp_dims1;

        temp = nn->d_z[i];
        int temp_z0 = nn->d_z_dims[i][0], temp_z1 = nn->d_z_dims[i][1];
        nn->d_z[i] = nn->d_z[layers - 1 - i];
        nn->d_z_dims[i][0] = nn->d_z_dims[layers - 1 - i][0];
        nn->d_z_dims[i][1] = nn->d_z_dims[layers - 1 - i][1];
        nn->d_z[layers - 1 - i] = temp;
        nn->d_z_dims[layers - 1 - i][0] = temp_z0;
        nn->d_z_dims[layers - 1 - i][1] = temp_z1;
    }
    #pragma omp parallel for
    for (int i = 0; i < layers; i++) {
        double **scaled = scalar_mul_with_matrix(nn->d_w[i],
                                                 nn->d_w_dims[i][0],
                                                 nn->d_w_dims[i][1],
                                                 nn->learning_rate);
        double **newW = Subtract(nn->NetworkWeights[i], scaled,
                                 nn->NetworkWeights_dims[i][0],
                                 nn->NetworkWeights_dims[i][1]);
        int r = nn->NetworkWeights_dims[i][0];
        for (int j = 0; j < r; j++) {
            free(nn->NetworkWeights[i][j]);
        }
        free(nn->NetworkWeights[i]);
        nn->NetworkWeights[i] = newW;
        for (int j = 0; j < nn->NetworkWeights_dims[i][0]; j++) {
            free(scaled[j]);
        }
        free(scaled);
    }
}

/* ------------------------ DATASET LOADING ------------------------ */

double **loadDataset(const char *filename, int num_pixels, int *num_samples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }
    char line[10000];
    int count = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strlen(line) > 1)
            count++;
    }
    rewind(fp);
    double **dataset = (double **)malloc(count * sizeof(double *));
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        dataset[i] = (double *)malloc(num_pixels * sizeof(double));
    }
    int sample_index = 0;
    while (fgets(line, sizeof(line), fp) != NULL && sample_index < count) {
        if (strlen(line) <= 1)
            continue;
        char *token = strtok(line, " \t\n");
        int pixel_index = 0;
        while (token != NULL && pixel_index < num_pixels) {
            dataset[sample_index][pixel_index] = atof(token);
            token = strtok(NULL, " \t\n");
            pixel_index++;
        }
        sample_index++;
    }
    fclose(fp);
    *num_samples = count;
    return dataset;
}

/* ------------------------ MAIN FUNCTION ------------------------ */

int main() {
    srand((unsigned int)time(NULL));
    
    int num_pixels = 784;
    int num_samples = 0;
    double **input = loadDataset("dataset.txt", num_pixels, &num_samples);
    printf("Loaded dataset with %d samples.\n", num_samples);
    
    int layers_arr[] = {784, 128, 64, 128, 784};
    int num_layers = sizeof(layers_arr) / sizeof(layers_arr[0]);
    NeuralNetwork *nn = create_NeuralNetwork(layers_arr, num_layers);
    
    int epochs = 100;
    for (int i = 1; i <= epochs; i++) {
        printf("Epoch: %d\n", i);
        int out_r, out_c;
        double **feed_forward_output = feed_forward(nn, input, num_samples, num_pixels, &out_r, &out_c);
        printf("Completed %d feedforward pass.\n", i);
        back_propogation(nn, feed_forward_output, out_r, out_c, input, num_samples, num_pixels);
        printf("After weight update.\n\n");
    }
    
    double **input2 = (double **)malloc(1 * sizeof(double *));
    input2[0] = (double *)malloc(num_pixels * sizeof(double));
    for (int j = 0; j < num_pixels; j++) {
        input2[0][j] = input[0][j];
    }
    printf("\nTESTING:\nTest Input:\n");
    print_vec2D_double(input2, 1, num_pixels);
    int test_r, test_c;
    double **test = feed_forward(nn, input2, 1, num_pixels, &test_r, &test_c);
    printf("Network Output:\n");
    print_vec2D_double(test, test_r, test_c);
    
    /* ------------------- New Snippet: Threads vs Time Reporting ------------------- */
    printf("\n=== Report Threads vs Time ===\n");
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);
    for (int t = 0; t < num_thread_counts; t++) {
        omp_set_num_threads(thread_counts[t]);
        /* Optionally reinitialize the network for fair comparison */
        NeuralNetwork *nn_test = create_NeuralNetwork(layers_arr, num_layers);
        double start_time = omp_get_wtime();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            int out_r, out_c;
            double **ff_output = feed_forward(nn_test, input, num_samples, num_pixels, &out_r, &out_c);
            back_propogation(nn_test, ff_output, out_r, out_c, input, num_samples, num_pixels);
        }
        double end_time = omp_get_wtime();
        printf("Threads: %d, Time taken: %lf seconds\n", thread_counts[t], end_time - start_time);
        /* Memory cleanup for nn_test is omitted for brevity */
    }
    /* ------------------- End Threads vs Time Reporting ------------------- */
    
    return 0;
}
