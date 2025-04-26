#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ------------------------ MATRIX MATH FUNCTIONS ------------------------ */
double dot_int(const int *vec1, const int *vec2, int n) {
    int dot_sum = 0;
    for (int i = 0; i < n; i++) {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}

double dot_double(const double *vec1, const double *vec2, int n) {
    double dot_sum = 0;
    for (int i = 0; i < n; i++) {
        dot_sum += (vec1[i] * vec2[i]);
    }
    return dot_sum;
}

/* Print a 1D int vector */
void print_vec1D_int(int *vec, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

/* Print a 2D int matrix */
void print_vec2D_int(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

/* Print a 2D double matrix */
void print_vec2D_double(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

/* Print weights (a 2D double matrix) */
void print_weights(double **weights, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", weights[i][j]);
        }
        printf("\n");
    }
}

/* Return vertical column from a 2D int matrix */
int *return_vertical_col_int(int **mat, int rows, int col) {
    int *column = (int *)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++) {
        column[i] = mat[i][col];
    }
    return column;
}

/* Return vertical column from a 2D double matrix */
double *return_vertical_col_double(double **mat, int rows, int col) {
    double *column = (double *)malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        column[i] = mat[i][col];
    }
    return column;
}

/* Cross multiplication for 2D int matrices */
int **Cross_MUl_int(int **mat1, int r1, int c1, int **mat2, int r2, int c2) {
    if (c1 != r2) {
        fprintf(stderr, "Error : Mat1 and Mat2 are not compatible for matrix multiplication\n");
    }
    int **result = (int **)malloc(r1 * sizeof(int *));
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

/* Cross multiplication for 2D double matrices */
double **Cross_MUl_double(double **mat1, int r1, int c1, double **mat2, int r2, int c2) {
    if (c1 != r2) {
        fprintf(stderr, "Error : Mat1 and Mat2 are not compatible for matrix multiplication\n");
    }
    double **result = (double **)malloc(r1 * sizeof(double *));
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
    for (int i = 0; i < cols; i++) {
        transposed[i] = return_vertical_col_int(matrix, rows, i);
    }
    return transposed;
}

/* Transpose a 2D double matrix */
double **Transpose_double(double **matrix, int rows, int cols) {
    double **transposed = (double **)malloc(cols * sizeof(double *));
    for (int i = 0; i < cols; i++) {
        transposed[i] = return_vertical_col_double(matrix, rows, i);
    }
    return transposed;
}

/* Element-wise subtraction for 2D double matrices: mat1 - mat2 */
double **Subtract(double **mat1, double **mat2, int rows, int cols) {
    double **result = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        result[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return result;
}

/* Scalar multiplication of a 2D double matrix */
double **scalar_mul_with_matrix(double **matrix, int rows, int cols, double scalar) {
    double **result = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        result[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = scalar * matrix[i][j];
        }
    }
    return result;
}

/* Calculate error over a 2D double matrix */
double calculate_error(double **test, double **input, int rows, int cols) {
    double error = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff = test[i][j] - input[i][j];
            error += diff * diff;
        }
    }
    error = error / (rows * cols);
    return error;
}

/* error_cal: For each row in delta, compute average and return a column vector */
double **error_cal(double **delta, int rows, int cols, int m) {
    double **ans = (double **)malloc(rows * sizeof(double *));
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
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

/* Initialize weights: returns a 2D double matrix of size dim1 x dim2 with random numbers */
double **initialize_weights(int dim1, int dim2) {
    double **weights = (double **)malloc(dim1 * sizeof(double *));
    for (int i = 0; i < dim1; i++) {
        weights[i] = (double *)malloc(dim2 * sizeof(double));
        for (int j = 0; j < dim2; j++) {
            weights[i][j] = get_random_number();
        }
    }
    return weights;
}

/* ------------------------ NEURAL NETWORK ------------------------ */
typedef struct {
    double **NetworkWeights[100]; /* To store network weights (max 100 layers) */
    int nodes[100];                /* Number of nodes in each layer */
    int num_layers;                /* Total number of layers */
    double **layer_outputs[100];   /* Outputs from each layer */
    double **d_w[100];             /* dW for each layer */
    double **d_z[100];             /* dz for each layer */
    double **inputs_for_each_layer[100];
    double learning_rate;
    /* To keep track of dimensions for each stored 2D matrix */
    int NetworkWeights_dims[100][2];
    int layer_outputs_dims[100][2];
    int d_w_dims[100][2];
    int d_z_dims[100][2];
    int inputs_dims[100][2];
} NeuralNetwork;

/* Constructor for NeuralNetwork.
   nodes_count_in_each_layer: array containing number of nodes in each layer.
   num: length of that array.
   For example, {4,3,3,1} means input size 4, first layer 3 nodes, second layer 3 nodes, and output 1 node.
*/
NeuralNetwork *create_NeuralNetwork(int *nodes_count_in_each_layer, int num) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num;
    nn->learning_rate = 0.001;
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

/* Print network weights */
void print_network_weights(NeuralNetwork *nn) {
    int layers = nn->num_layers - 1;
    for (int i = 0; i < layers; i++) {
        printf("Layer-%d\n", i);
        printf("input dim : %d, num nodes:%d\n", nn->nodes[i], nn->nodes[i + 1]);
        printf("Weight dimensions : %d X %d\n", nn->nodes[i], nn->nodes[i + 1]);
        printf("Weights :\n");
        print_weights(nn->NetworkWeights[i], nn->NetworkWeights_dims[i][0], nn->NetworkWeights_dims[i][1]);
        printf("\n");
    }
}

/* ReLU activation: ReLU(x) returns 0 if x < 0 else x */
double **ReLU(double **input, int rows, int cols) {
    double **output = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        output[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            output[i][j] = (input[i][j] > 0 ? input[i][j] : 0);
        }
    }
    return output;
}

/* Feed forward function.
   input is a 2D matrix (samples as rows). */
double **feed_forward(NeuralNetwork *nn, double **input, int input_rows, int input_cols, int *out_rows, int *out_cols) {
    int layers = nn->num_layers - 1;
    int i, j;
    /* Free previous layer_outputs and inputs_for_each_layer if needed (not done here for brevity) */
    /* Transpose the input: from (input_rows x input_cols) to (input_cols x input_rows) */
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
        /* Uncomment next line if ReLU is desired: result = ReLU(result, w_r, current_cols); */
        nn->layer_outputs[i] = result;
        nn->layer_outputs_dims[i][0] = w_r;
        nn->layer_outputs_dims[i][1] = current_cols;
        /* Free the previous input_to_pass if allocated (skip free for brevity) */
        input_to_pass = result;
        current_rows = w_r;
        /* current_cols remains the same */
    }
    *out_rows = current_rows;
    *out_cols = current_cols;
    return input_to_pass;
}

/* Back propagation.
   predicted and target are 2D matrices.
   predicted dimensions should match target after transposition.
*/
void back_propogation(NeuralNetwork *nn, double **predicted, int pred_rows, int pred_cols, double **target, int target_rows, int target_cols) {
    int layers = nn->num_layers - 1;
    int number_of_samples = target_rows;
    double **target_T = Transpose_double(target, target_rows, target_cols);
    double **delta = Subtract(predicted, target_T, pred_rows, pred_cols);
    /* Free target_T after use */
    int i, j;
    for (i = 0; i < target_cols; i++) {
        free(target_T[i]);
    }
    free(target_T);
    double **delta_ec = error_cal(delta, pred_rows, pred_cols, number_of_samples);
    for (i = 0; i < pred_rows; i++) {
        free(delta[i]);
    }
    free(delta);
    /* Store d_z for output layer */
    nn->d_z[0] = delta_ec;
    nn->d_z_dims[0][0] = pred_rows;
    nn->d_z_dims[0][1] = 1;
    double **inputs_last_T = Transpose_double(nn->inputs_for_each_layer[layers - 1], nn->inputs_dims[layers - 1][0], nn->inputs_dims[layers - 1][1]);
    double **grad = Cross_MUl_double(delta_ec, pred_rows, 1, inputs_last_T, nn->inputs_dims[layers - 1][1], nn->inputs_dims[layers - 1][0]);
    for (i = 0; i < nn->inputs_dims[layers - 1][1]; i++) {
        free(inputs_last_T[i]);
    }
    free(inputs_last_T);
    nn->d_w[0] = grad;
    nn->d_w_dims[0][0] = pred_rows;
    nn->d_w_dims[0][1] = nn->inputs_dims[layers - 1][0];
    int l;
    for (l = layers - 1; l > 0; l--) {
        double **W_next_T = Transpose_double(nn->NetworkWeights[l], nn->NetworkWeights_dims[l][0], nn->NetworkWeights_dims[l][1]);
        double **delta_new = Cross_MUl_double(W_next_T, nn->NetworkWeights_dims[l][1], nn->NetworkWeights_dims[l][0], delta_ec, nn->d_z_dims[0][0], nn->d_z_dims[0][1]);
        for (i = 0; i < nn->NetworkWeights_dims[l][1]; i++) {
            free(W_next_T[i]);
        }
        free(W_next_T);
        nn->d_z[layers - l] = delta_new;
        nn->d_z_dims[layers - l][0] = nn->NetworkWeights_dims[l][1];
        nn->d_z_dims[layers - l][1] = 1;
        double **input_layer_T = Transpose_double(nn->inputs_for_each_layer[l - 1], nn->inputs_dims[l - 1][0], nn->inputs_dims[l - 1][1]);
        double **grad_new = Cross_MUl_double(delta_new, nn->d_z_dims[layers - l][0], 1, input_layer_T, nn->inputs_dims[l - 1][1], nn->inputs_dims[l - 1][0]);
        for (i = 0; i < nn->inputs_dims[l - 1][1]; i++) {
            free(input_layer_T[i]);
        }
        free(input_layer_T);
        nn->d_w[layers - l] = grad_new;
        nn->d_w_dims[layers - l][0] = nn->d_z_dims[layers - l][0];
        nn->d_w_dims[layers - l][1] = nn->inputs_dims[l - 1][0];
    }
    /* Reverse d_w and d_z arrays */
    for (i = 0; i < layers / 2; i++) {
        double **temp = nn->d_w[i];
        int temp_dims0 = nn->d_w_dims[i][0], temp_dims1 = nn->d_w_dims[i][1];
        nn->d_w[i] = nn->d_w[layers - 1 - i];
        nn->d_w_dims[i][0] = nn->d_w_dims[layers - 1 - i][0];
        nn->d_w_dims[i][1] = nn->d_w_dims[layers - 1 - i][1];
        nn->d_w[layers - 1 - i] = temp;
        nn->d_w_dims[layers - 1 - i][0] = temp_dims0;
        nn->d_w_dims[layers - 1 - i][1] = temp_dims1;
        temp = nn->d_z[i];
        temp_dims0 = nn->d_z_dims[i][0];
        temp_dims1 = nn->d_z_dims[i][1];
        nn->d_z[i] = nn->d_z[layers - 1 - i];
        nn->d_z_dims[i][0] = nn->d_z_dims[layers - 1 - i][0];
        nn->d_z_dims[i][1] = nn->d_z_dims[layers - 1 - i][1];
        nn->d_z[layers - 1 - i] = temp;
        nn->d_z_dims[layers - 1 - i][0] = temp_dims0;
        nn->d_z_dims[layers - 1 - i][1] = temp_dims1;
    }
    /* Update weights: W = W - learning_rate * d_w */
    for (i = 0; i < layers; i++) {
        double **scaled = scalar_mul_with_matrix(nn->d_w[i], nn->d_w_dims[i][0], nn->d_w_dims[i][1], nn->learning_rate);
        double **newW = Subtract(nn->NetworkWeights[i], scaled, nn->NetworkWeights_dims[i][0], nn->NetworkWeights_dims[i][1]);
        int r = nn->NetworkWeights_dims[i][0];
        for (j = 0; j < r; j++) {
            free(nn->NetworkWeights[i][j]);
        }
        free(nn->NetworkWeights[i]);
        nn->NetworkWeights[i] = newW;
        for (j = 0; j < nn->NetworkWeights_dims[i][0]; j++) {
            free(scaled[j]);
        }
        free(scaled);
    }
}

/* ------------------------ MAIN FUNCTION ------------------------ */
int main() {
    srand((unsigned int)time(NULL));
    int layers[] = {3, 2, 3, 3, 2, 3};
    int num_layers = sizeof(layers)/sizeof(layers[0]);
    NeuralNetwork *nn = create_NeuralNetwork(layers, num_layers);
    /* Prepare input: 5 rows x 3 cols */
    int input_rows = 5, input_cols = 3, i, j;
    double **input = (double **)malloc(input_rows * sizeof(double *));
    double input_values[5][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {1.5, 2.6, 3.8},
        {15.0, 25.0, 37.0}
    };
    for (i = 0; i < input_rows; i++) {
        input[i] = (double *)malloc(input_cols * sizeof(double));
        for (j = 0; j < input_cols; j++) {
            input[i][j] = input_values[i][j];
        }
    }
    int epochs = 3000;

    clock_t start, end;
    start = clock();

    for (i = 1; i <= epochs; i++) {
        printf("epoch : %d\n", i);
        printf("Before weight update : \n");
        print_network_weights(nn);
        int out_r, out_c;
        double **feed_forward_output = feed_forward(nn, input, input_rows, input_cols, &out_r, &out_c);
        printf("completed %d feedforward\n", i);
        print_vec2D_double(feed_forward_output, out_r, out_c);
        back_propogation(nn, feed_forward_output, out_r, out_c, input, input_rows, input_cols);
        printf("After weight update : \n");
        print_network_weights(nn);
        printf("\n");
        /* For simplicity, memory cleanup of intermediate matrices is not shown */
    }

    end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total execution time: %f seconds\n", time_spent);

    double **input2 = (double **)malloc(1 * sizeof(double *));
    input2[0] = (double *)malloc(3 * sizeof(double));
    input2[0][0] = 4.0; input2[0][1] = 5.0; input2[0][2] = 6.0;
    printf("\nTESTING1 : \n");
    int test_r, test_c;
    double **test = feed_forward(nn, input2, 1, 3, &test_r, &test_c);
    print_vec2D_double(test, test_r, test_c);
    /* Free allocated memory (cleanup code omitted for brevity) */
    return 0;
}
