/*
 * Cuda_myproject.cu
 *
 * This code is a CUDA‐accelerated version of your neural network code.
 * It keeps the overall structure (feedforward, backpropagation, weight update)
 * and I/O similar to your original C code but offloads heavy operations (matrix multiplication,
 * subtraction, scalar multiplication) to the GPU.
 *
 * IMPORTANT: In your original code the back-propagation computed an “averaged” delta by calling error_cal,
 * which resulted in a dimension mismatch for computing gradients. Here we replace that call with a GPU‐accelerated
 * scalar multiplication that averages element‐wise without collapsing the matrix shape.
 *
 * To compile:
 *      nvcc Cuda_myproject.cu -o cuda_myproject
 * To run:
 *      ./cuda_myproject
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <cuda_runtime.h>
 
 // Macro for CUDA error checking
 #define CUDA_CHECK(call)                                          \
     do {                                                          \
         cudaError_t err = call;                                   \
         if(err != cudaSuccess) {                                  \
             fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
             exit(EXIT_FAILURE);                                   \
         }                                                         \
     } while(0)
 
 /* ------------------------ GPU KERNELS ------------------------ */
 
 // Matrix multiplication kernel for double precision
 __global__ void matMulKernel(double* A, int A_rows, int A_cols,
                              double* B, int B_cols,
                              double* C) {
     int row = blockIdx.y * blockDim.y + threadIdx.y; // row index for A and C
     int col = blockIdx.x * blockDim.x + threadIdx.x; // col index for B and C
     if(row < A_rows && col < B_cols) {
         double sum = 0.0;
         for (int k = 0; k < A_cols; k++) {
             sum += A[row * A_cols + k] * B[k * B_cols + col];
         }
         C[row * B_cols + col] = sum;
     }
 }
 
 // Kernel for element-wise subtraction: C = A - B
 __global__ void subtractKernel(double* A, double* B, double* C, int total_elements) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < total_elements) {
          C[idx] = A[idx] - B[idx];
     }
 }
 
 // Kernel for scalar multiplication: A = scalar * A
 __global__ void scalarMulKernel(double* A, double scalar, int total_elements) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < total_elements) {
          A[idx] = scalar * A[idx];
     }
 }
 
 /* ------------------------ HELPER FUNCTIONS ------------------------ */
 
 // Flatten a 2D double matrix (stored as double**) into a 1D array (row-major)
 double* flatten_matrix_double(double **matrix, int rows, int cols) {
     double *flat = (double *) malloc(rows * cols * sizeof(double));
     for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
             flat[i * cols + j] = matrix[i][j];
         }
     }
     return flat;
 }
 
 // Unflatten a 1D array into a 2D double matrix (allocated as double**)
 double** unflatten_matrix_double(double *flat, int rows, int cols) {
     double **matrix = (double **) malloc(rows * sizeof(double *));
     for (int i = 0; i < rows; i++) {
         matrix[i] = (double *) malloc(cols * sizeof(double));
         for (int j = 0; j < cols; j++) {
             matrix[i][j] = flat[i * cols + j];
         }
     }
     return matrix;
 }
 
 /* ------------------------ MATRIX MATH FUNCTIONS (CPU for ints and printing) ------------------------ */
 
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
 
 /* Print functions */
 void print_vec1D_int(int *vec, int n) {
     for (int i = 0; i < n; i++) {
         printf("%d ", vec[i]);
     }
     printf("\n");
 }
 
 void print_vec2D_int(int **matrix, int rows, int cols) {
     for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
             printf("%d ", matrix[i][j]);
         }
         printf("\n");
     }
 }
 
 void print_vec2D_double(double **matrix, int rows, int cols) {
     for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
             printf("%lf ", matrix[i][j]);
         }
         printf("\n");
     }
 }
 
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
 
 /* ------------------------ GPU ACCELERATED MATRIX OPERATIONS ------------------------ */
 
 /* Cross multiplication for 2D double matrices using CUDA.
    Computes: result = mat1 (r1 x c1) * mat2 (r2 x c2)
 */
 double **Cross_MUl_double(double **mat1, int r1, int c1,
                            double **mat2, int r2, int c2) {
     if (c1 != r2) {
         fprintf(stderr, "Error: Mat1 and Mat2 are not compatible for matrix multiplication\n");
         // Instead of exiting, we continue (as in your original C code)
     }
     // Flatten matrices
     double *h_A = flatten_matrix_double(mat1, r1, c1);
     double *h_B = flatten_matrix_double(mat2, r2, c2);
     double *h_C = (double *) malloc(r1 * c2 * sizeof(double));
 
     // Allocate device memory
     double *d_A, *d_B, *d_C;
     CUDA_CHECK( cudaMalloc((void**)&d_A, r1 * c1 * sizeof(double)) );
     CUDA_CHECK( cudaMalloc((void**)&d_B, r2 * c2 * sizeof(double)) );
     CUDA_CHECK( cudaMalloc((void**)&d_C, r1 * c2 * sizeof(double)) );
 
     // Copy data to device
     CUDA_CHECK( cudaMemcpy(d_A, h_A, r1 * c1 * sizeof(double), cudaMemcpyHostToDevice) );
     CUDA_CHECK( cudaMemcpy(d_B, h_B, r2 * c2 * sizeof(double), cudaMemcpyHostToDevice) );
 
     // Launch kernel
     dim3 blockDim(16, 16);
     dim3 gridDim((c2 + blockDim.x - 1) / blockDim.x, (r1 + blockDim.y - 1) / blockDim.y);
     matMulKernel<<<gridDim, blockDim>>>(d_A, r1, c1, d_B, c2, d_C);
     CUDA_CHECK( cudaDeviceSynchronize() );
 
     // Copy result back to host
     CUDA_CHECK( cudaMemcpy(h_C, d_C, r1 * c2 * sizeof(double), cudaMemcpyDeviceToHost) );
 
     // Free device and temporary host memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     free(h_A);
     free(h_B);
 
     double **result = unflatten_matrix_double(h_C, r1, c2);
     free(h_C);
     return result;
 }
 
 /* Element-wise subtraction for 2D double matrices (GPU accelerated) */
 double **Subtract(double **mat1, double **mat2, int rows, int cols) {
     int total = rows * cols;
     double *h_A = flatten_matrix_double(mat1, rows, cols);
     double *h_B = flatten_matrix_double(mat2, rows, cols);
     double *h_C = (double *) malloc(total * sizeof(double));
 
     double *d_A, *d_B, *d_C;
     CUDA_CHECK( cudaMalloc((void**)&d_A, total * sizeof(double)) );
     CUDA_CHECK( cudaMalloc((void**)&d_B, total * sizeof(double)) );
     CUDA_CHECK( cudaMalloc((void**)&d_C, total * sizeof(double)) );
 
     CUDA_CHECK( cudaMemcpy(d_A, h_A, total * sizeof(double), cudaMemcpyHostToDevice) );
     CUDA_CHECK( cudaMemcpy(d_B, h_B, total * sizeof(double), cudaMemcpyHostToDevice) );
 
     int threadsPerBlock = 256;
     int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
     subtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, total);
     CUDA_CHECK( cudaDeviceSynchronize() );
 
     CUDA_CHECK( cudaMemcpy(h_C, d_C, total * sizeof(double), cudaMemcpyDeviceToHost) );
 
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     free(h_A);
     free(h_B);
 
     double **result = unflatten_matrix_double(h_C, rows, cols);
     free(h_C);
     return result;
 }
 
 /* Scalar multiplication of a 2D double matrix (GPU accelerated)
    Computes: result = scalar * matrix
 */
 double **scalar_mul_with_matrix(double **matrix, int rows, int cols, double scalar) {
     int total = rows * cols;
     double *h_A = flatten_matrix_double(matrix, rows, cols);
 
     double *d_A;
     CUDA_CHECK( cudaMalloc((void**)&d_A, total * sizeof(double)) );
     CUDA_CHECK( cudaMemcpy(d_A, h_A, total * sizeof(double), cudaMemcpyHostToDevice) );
 
     int threadsPerBlock = 256;
     int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
     scalarMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, scalar, total);
     CUDA_CHECK( cudaDeviceSynchronize() );
 
     CUDA_CHECK( cudaMemcpy(h_A, d_A, total * sizeof(double), cudaMemcpyDeviceToHost) );
     cudaFree(d_A);
 
     double **result = unflatten_matrix_double(h_A, rows, cols);
     free(h_A);
     return result;
 }
 
 /* Calculate error over a 2D double matrix (CPU) */
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
 
 /* ------------------------ NEURAL NETWORK STRUCTURE ------------------------ */
 typedef struct {
     double **NetworkWeights[100]; /* Network weights (max 100 layers) */
     int nodes[100];                /* Number of nodes in each layer */
     int num_layers;                /* Total number of layers */
     double **layer_outputs[100];   /* Outputs from each layer */
     double **d_w[100];             /* Gradients for weights */
     double **d_z[100];             /* Delta for each layer */
     double **inputs_for_each_layer[100];  /* Inputs passed to each layer */
     double learning_rate;
     /* Dimensions for each stored 2D matrix */
     int NetworkWeights_dims[100][2];
     int layer_outputs_dims[100][2];
     int d_w_dims[100][2];
     int d_z_dims[100][2];
     int inputs_dims[100][2];
 } NeuralNetwork;
 
 /* Constructor for NeuralNetwork.
    nodes_count_in_each_layer: array with node counts per layer.
    For example, {3,2,3,3,2,3} means input size 3, first layer 2 nodes, etc.
 */
 NeuralNetwork *create_NeuralNetwork(int *nodes_count_in_each_layer, int num) {
     NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
     nn->num_layers = num;
     nn->learning_rate = 0.00001;
     for (int i = 0; i < num; i++) {
         nn->nodes[i] = nodes_count_in_each_layer[i];
     }
     for (int i = 1; i < num; i++) {
         int r = nodes_count_in_each_layer[i];   // actual weight matrix row = next layer nodes
         int c = nodes_count_in_each_layer[i - 1]; // columns = current layer nodes
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
 
 /* ReLU activation: ReLU(x) returns x if x>0 else 0 */
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
 
 /* Transpose a 2D double matrix (CPU) */
 double **Transpose_double(double **matrix, int rows, int cols) {
     double **transposed = (double **)malloc(cols * sizeof(double *));
     for (int i = 0; i < cols; i++) {
         transposed[i] = return_vertical_col_double(matrix, rows, i);
     }
     return transposed;
 }
 
 /* ------------------------ FEEDFORWARD FUNCTION ------------------------ */
 /* input: samples as rows */
 double **feed_forward(NeuralNetwork *nn, double **input, int input_rows, int input_cols, int *out_rows, int *out_cols) {
     int layers = nn->num_layers - 1;
     // Transpose the input: from (input_rows x input_cols) to (input_cols x input_rows)
     double **input_to_pass = Transpose_double(input, input_rows, input_cols);
     int current_rows = input_cols; // after transpose
     int current_cols = input_rows;
     for (int i = 0; i < layers; i++) {
         nn->inputs_for_each_layer[i] = input_to_pass;
         nn->inputs_dims[i][0] = current_rows;
         nn->inputs_dims[i][1] = current_cols;
         int w_r = nn->NetworkWeights_dims[i][0]; // weight matrix row count = next layer nodes
         int w_c = nn->NetworkWeights_dims[i][1]; // weight matrix col count = current layer nodes
         double **result = Cross_MUl_double(nn->NetworkWeights[i], w_r, w_c, input_to_pass, current_rows, current_cols);
         // Uncomment the next line to apply ReLU activation if desired:
         // result = ReLU(result, w_r, current_cols);
         nn->layer_outputs[i] = result;
         nn->layer_outputs_dims[i][0] = w_r;
         nn->layer_outputs_dims[i][1] = current_cols;
         // In a full implementation, free the previous input_to_pass here if allocated
         input_to_pass = result;
         current_rows = w_r;
         // current_cols remains the same (number of samples)
     }
     *out_rows = current_rows;
     *out_cols = current_cols;
     return input_to_pass;
 }
 
 /* ------------------------ BACKPROPAGATION FUNCTION ------------------------ */
 /* predicted and target are 2D matrices.
    Here we fix the gradient calculation by averaging delta element-wise (without collapsing dims)
    so that dimensions remain compatible.
 */
 void back_propogation(NeuralNetwork *nn, double **predicted, int pred_rows, int pred_cols,
                       double **target, int target_rows, int target_cols) {
     int layers = nn->num_layers - 1;
     int number_of_samples = target_rows;
     // Transpose target: (target_rows x target_cols) -> (target_cols x target_rows)
     double **target_T = Transpose_double(target, target_rows, target_cols);
     // delta = predicted - target_T; dimensions: (pred_rows x pred_cols)
     double **delta = Subtract(predicted, target_T, pred_rows, pred_cols);
     for (int i = 0; i < target_cols; i++) {
         free(target_T[i]);
     }
     free(target_T);
     // Average delta element-wise over samples (without reducing shape)
     double **delta_avg = scalar_mul_with_matrix(delta, pred_rows, pred_cols, 1.0 / number_of_samples);
     for (int i = 0; i < pred_rows; i++) {
         free(delta[i]);
     }
     free(delta);
     // Set d_z for output layer to averaged delta; dimensions: (pred_rows x pred_cols)
     nn->d_z[0] = delta_avg;
     nn->d_z_dims[0][0] = pred_rows;
     nn->d_z_dims[0][1] = pred_cols;
     // Compute gradient for output layer: grad = delta_avg * (inputs_last)^T
     double **inputs_last_T = Transpose_double(nn->inputs_for_each_layer[layers - 1],
                                                nn->inputs_dims[layers - 1][0],
                                                nn->inputs_dims[layers - 1][1]); // dims: (samples x previous_layer_nodes)
     double **grad = Cross_MUl_double(delta_avg, pred_rows, pred_cols,
                                      inputs_last_T, nn->inputs_dims[layers - 1][1],
                                      nn->inputs_dims[layers - 1][0]); // result: (pred_rows x previous_layer_nodes)
     for (int i = 0; i < nn->inputs_dims[layers - 1][1]; i++) {
         free(inputs_last_T[i]);
     }
     free(inputs_last_T);
     nn->d_w[0] = grad;
     nn->d_w_dims[0][0] = pred_rows;
     nn->d_w_dims[0][1] = nn->inputs_dims[layers - 1][0];
     
     // Backpropagate through remaining layers
     for (int l = layers - 1; l > 0; l--) {
         double **W_next_T = Transpose_double(nn->NetworkWeights[l],
                                              nn->NetworkWeights_dims[l][0],
                                              nn->NetworkWeights_dims[l][1]);
         // Compute delta for previous layer: delta_new = W_next_T * current delta_avg
         double **delta_new = Cross_MUl_double(W_next_T,
                                               nn->NetworkWeights_dims[l][1],
                                               nn->NetworkWeights_dims[l][0],
                                               nn->d_z[0],
                                               nn->d_z_dims[0][0],
                                               nn->d_z_dims[0][1]);
         for (int i = 0; i < nn->NetworkWeights_dims[l][1]; i++) {
             free(W_next_T[i]);
         }
         free(W_next_T);
         nn->d_z[layers - l] = delta_new;
         nn->d_z_dims[layers - l][0] = nn->NetworkWeights_dims[l][1];
         nn->d_z_dims[layers - l][1] = pred_cols;  // same number of samples
         double **input_layer_T = Transpose_double(nn->inputs_for_each_layer[l - 1],
                                                   nn->inputs_dims[l - 1][0],
                                                   nn->inputs_dims[l - 1][1]);
         double **grad_new = Cross_MUl_double(delta_new,
                                              nn->d_z_dims[layers - l][0],
                                              nn->d_z_dims[layers - l][1],
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
     // Reverse d_w and d_z arrays (as in your original code)
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
         temp_dims0 = nn->d_z_dims[i][0];
         temp_dims1 = nn->d_z_dims[i][1];
         nn->d_z[i] = nn->d_z[layers - 1 - i];
         nn->d_z_dims[i][0] = nn->d_z_dims[layers - 1 - i][0];
         nn->d_z_dims[i][1] = nn->d_z_dims[layers - 1 - i][1];
         nn->d_z[layers - 1 - i] = temp;
         nn->d_z_dims[layers - 1 - i][0] = temp_dims0;
         nn->d_z_dims[layers - 1 - i][1] = temp_dims1;
     }
     // Update weights: W = W - learning_rate * d_w
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
 
 /* ------------------------ MAIN FUNCTION ------------------------ */
 int main() {
     srand((unsigned int)time(NULL));
     int layers[] = {3, 2, 3, 3, 2, 3};
     int num_layers = sizeof(layers)/sizeof(layers[0]);
     NeuralNetwork *nn = create_NeuralNetwork(layers, num_layers);
     
     /* Prepare input: 5 rows x 3 cols */
     int input_rows = 5, input_cols = 3;
     double **input = (double **)malloc(input_rows * sizeof(double *));
     double input_values[5][3] = {
         {1.0, 2.0, 3.0},
         {4.0, 5.0, 6.0},
         {7.0, 8.0, 9.0},
         {1.5, 2.6, 3.8},
         {15.0, 25.0, 37.0}
     };
     for (int i = 0; i < input_rows; i++) {
         input[i] = (double *)malloc(input_cols * sizeof(double));
         for (int j = 0; j < input_cols; j++) {
             input[i][j] = input_values[i][j];
         }
     }
     
     int epochs = 1;
     clock_t start, end;
     start = clock();
     for (int i = 1; i <= epochs; i++) {
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
         // Note: For brevity, cleanup of intermediate allocations is omitted.
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
     
     // (Memory cleanup for all allocated matrices is omitted for brevity)
     return 0;
 }
 