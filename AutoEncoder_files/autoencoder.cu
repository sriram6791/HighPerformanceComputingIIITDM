/*

#!/usr/bin/env bash
set -e

# 1) Prepare the MNIST data (if you haven’t already):
python3 prepare_mnist.py

# 2) Compile the CUDA autoencoder:
#    -O3          : optimize aggressively
#    -arch=sm_70  : target your GPU’s compute capability (e.g. sm_70 for V100/T4; adjust if needed)
nvcc -O3 -arch=sm_70 -o autoencoder autoencoder.cu

# 3) Run the autoencoder, pointing it at the data folder
./autoencoder ./mnist_data



nvcc -O3 -o autoencoder_cuda autoencoder_cuda.cu
./autoencoder_cuda ./mnist_data


*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// MNIST parameters
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define INPUT_SIZE (IMG_WIDTH * IMG_HEIGHT)

// Network architecture
#define HIDDEN_SIZE 128  // Size of the bottleneck layer
#define BATCH_SIZE 128
#define LEARNING_RATE 0.001f
#define NUM_EPOCHS 50

// Helper functions for CUDA error checking
#define CHECK_CUDA_ERROR(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error)); \
      exit(1); \
    } \
  } while (0)

// Forward declarations
void load_mnist_data(float* train_images, float* test_images, const char* data_path);
void train_autoencoder(float* train_images, float* test_images);
float test_autoencoder(float* test_images, float* weights_encoder, float* weights_decoder);
void save_weights(float* weights_encoder, float* weights_decoder);
void save_reconstructed_images(float* test_images, float* reconstructed_images, int num_images);

// CUDA kernels
__global__ void encoder_forward(const float* input, float* hidden, const float* weights_encoder, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < HIDDEN_SIZE * batch_size) {
        int batch_idx = tid / HIDDEN_SIZE;
        int hidden_idx = tid % HIDDEN_SIZE;

        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[batch_idx * INPUT_SIZE + i] * weights_encoder[i * HIDDEN_SIZE + hidden_idx];
        }

        // ReLU activation
        hidden[tid] = fmaxf(0.0f, sum);
    }
}

__global__ void decoder_forward(const float* hidden, float* output, const float* weights_decoder, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < INPUT_SIZE * batch_size) {
        int batch_idx = tid / INPUT_SIZE;
        int output_idx = tid % INPUT_SIZE;

        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += hidden[batch_idx * HIDDEN_SIZE + i] * weights_decoder[i * INPUT_SIZE + output_idx];
        }

        // Sigmoid activation for output between 0 and 1
        output[tid] = 1.0f / (1.0f + expf(-sum));
    }
}

__global__ void compute_loss_and_gradients(const float* input, const float* output, const float* hidden,
                                          float* d_output, float* d_hidden,
                                          float* d_weights_decoder, float* d_weights_encoder,
                                          const float* weights_decoder, const float* weights_encoder,
                                          int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute output gradients (MSE loss derivative)
    if (tid < INPUT_SIZE * batch_size) {
        int batch_idx = tid / INPUT_SIZE;
        int pixel_idx = tid % INPUT_SIZE;

        // MSE derivative: 2 * (output - target) / batch_size
        d_output[tid] = 2.0f * (output[tid] - input[tid]) / batch_size;

        // Backprop through sigmoid: d_sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        d_output[tid] *= output[tid] * (1.0f - output[tid]);
    }
    __syncthreads();

    // Compute hidden layer gradients
    if (tid < HIDDEN_SIZE * batch_size) {
        int batch_idx = tid / HIDDEN_SIZE;
        int hidden_idx = tid % HIDDEN_SIZE;

        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += d_output[batch_idx * INPUT_SIZE + i] * weights_decoder[hidden_idx * INPUT_SIZE + i];
        }

        // Backprop through ReLU: d_relu(x) = 1 if x > 0, else 0
        d_hidden[tid] = (hidden[tid] > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();

    // Compute hidden layer gradients
    if (tid < HIDDEN_SIZE * batch_size) {
        int batch_idx = tid / HIDDEN_SIZE;
        int hidden_idx = tid % HIDDEN_SIZE;

        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += d_output[batch_idx * INPUT_SIZE + i] * weights_decoder[hidden_idx * INPUT_SIZE + i];
        }

        // Backprop through ReLU: d_relu(x) = 1 if x > 0, else 0
        d_hidden[tid] = (hidden[tid] > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();

    // Compute weight gradients for decoder
    if (tid < HIDDEN_SIZE * INPUT_SIZE) {
        int hidden_idx = tid / INPUT_SIZE;
        int output_idx = tid % INPUT_SIZE;

        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_sum += hidden[b * HIDDEN_SIZE + hidden_idx] * d_output[b * INPUT_SIZE + output_idx];
        }

        d_weights_decoder[tid] = grad_sum;
    }
    __syncthreads();

    // Compute weight gradients for encoder
    if (tid < INPUT_SIZE * HIDDEN_SIZE) {
        int input_idx = tid / HIDDEN_SIZE;
        int hidden_idx = tid % HIDDEN_SIZE;

        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_sum += input[b * INPUT_SIZE + input_idx] * d_hidden[b * HIDDEN_SIZE + hidden_idx];
        }

        d_weights_encoder[tid] = grad_sum;
    }
}

__global__ void update_weights(float* weights, float* gradients, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        weights[tid] -= LEARNING_RATE * gradients[tid];
    }
}

void initialize_weights(float* weights, int input_dim, int output_dim) {
    float scale = sqrtf(2.0f / input_dim);  // He initialization
    for (int i = 0; i < input_dim * output_dim; i++) {
        // Generate random number between -scale and scale
        weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

int main(int argc, char** argv) {
    srand(42);  // For reproducibility

    // Allocate memory for MNIST data
    float* train_images = (float*)malloc(NUM_TRAIN_IMAGES * INPUT_SIZE * sizeof(float));
    float* test_images = (float*)malloc(NUM_TEST_IMAGES * INPUT_SIZE * sizeof(float));

    // Set data path (adjust as needed)
    const char* data_path = (argc > 1) ? argv[1] : "./mnist_data/";

    // Load MNIST data
    printf("Loading MNIST data from %s...\n", data_path);
    load_mnist_data(train_images, test_images, data_path);

    // Train the autoencoder
    printf("Training autoencoder...\n");
    train_autoencoder(train_images, test_images);

    // Free memory
    free(train_images);
    free(test_images);

    return 0;
}

void train_autoencoder(float* train_images, float* test_images) {
    // Allocate and initialize weights on host
    float* h_weights_encoder = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_weights_decoder = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));

    initialize_weights(h_weights_encoder, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(h_weights_decoder, HIDDEN_SIZE, INPUT_SIZE);

    // Allocate memory on device
    float *d_input, *d_hidden, *d_output;
    float *d_weights_encoder, *d_weights_decoder;
    float *d_d_hidden, *d_d_output;
    float *d_d_weights_encoder, *d_d_weights_decoder;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, BATCH_SIZE * INPUT_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights_encoder, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights_decoder, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_output, BATCH_SIZE * INPUT_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_weights_encoder, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_weights_decoder, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));

    // Copy weights to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights_encoder, h_weights_encoder,
                                INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights_decoder, h_weights_decoder,
                                HIDDEN_SIZE * INPUT_SIZE * sizeof(float),
                                cudaMemcpyHostToDevice));

    // Define CUDA thread blocks
    int threads_per_block = 256;
    int blocks_encoder = (BATCH_SIZE * HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    int blocks_decoder = (BATCH_SIZE * INPUT_SIZE + threads_per_block - 1) / threads_per_block;
    int blocks_gradients = max(blocks_encoder, blocks_decoder);
    int blocks_encoder_weights = (INPUT_SIZE * HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    int blocks_decoder_weights = (HIDDEN_SIZE * INPUT_SIZE + threads_per_block - 1) / threads_per_block;

    // Training loop
    int num_batches = NUM_TRAIN_IMAGES / BATCH_SIZE;
    printf("Training with %d batches per epoch for %d epochs\n", num_batches, NUM_EPOCHS);

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            // Copy batch data to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_input,
                                       train_images + batch * BATCH_SIZE * INPUT_SIZE,
                                       BATCH_SIZE * INPUT_SIZE * sizeof(float),
                                       cudaMemcpyHostToDevice));

            // Forward pass
            encoder_forward<<<blocks_encoder, threads_per_block>>>(d_input, d_hidden, d_weights_encoder, BATCH_SIZE);
            decoder_forward<<<blocks_decoder, threads_per_block>>>(d_hidden, d_output, d_weights_decoder, BATCH_SIZE);

            // Backward pass
            compute_loss_and_gradients<<<blocks_gradients, threads_per_block>>>(
              d_input, d_output, d_hidden,
              d_d_output, d_d_hidden,
              d_d_weights_decoder, d_d_weights_encoder,
              d_weights_decoder, d_weights_encoder,
              BATCH_SIZE);

            // Update weights
            update_weights<<<blocks_encoder_weights, threads_per_block>>>(
                d_weights_encoder, d_d_weights_encoder, INPUT_SIZE * HIDDEN_SIZE);
            update_weights<<<blocks_decoder_weights, threads_per_block>>>(
                d_weights_decoder, d_d_weights_decoder, HIDDEN_SIZE * INPUT_SIZE);

            // Compute loss for reporting
            if (batch % 100 == 0) {
                float* h_output = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
                CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output,
                                          BATCH_SIZE * INPUT_SIZE * sizeof(float),
                                          cudaMemcpyDeviceToHost));

                float batch_loss = 0.0f;
                for (int i = 0; i < BATCH_SIZE * INPUT_SIZE; i++) {
                    float diff = h_output[i] - train_images[batch * BATCH_SIZE * INPUT_SIZE + i];
                    batch_loss += diff * diff;
                }
                batch_loss /= (BATCH_SIZE * INPUT_SIZE);
                total_loss += batch_loss;

                free(h_output);

                printf("Epoch %d, Batch %d/%d, Loss: %.6f\n", epoch + 1, batch, num_batches, batch_loss);
            }
        }

        // Copy weights back to host for testing
        CHECK_CUDA_ERROR(cudaMemcpy(h_weights_encoder, d_weights_encoder,
                                    INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
                                    cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_weights_decoder, d_weights_decoder,
                                    HIDDEN_SIZE * INPUT_SIZE * sizeof(float),
                                    cudaMemcpyDeviceToHost));

        // Test the autoencoder
        float test_loss = test_autoencoder(test_images, h_weights_encoder, h_weights_decoder);
        printf("Epoch %d complete, Avg. Training Loss: %.6f, Test Loss: %.6f\n",
               epoch + 1, total_loss / (num_batches / 100), test_loss);

        // Save weights periodically
        if ((epoch + 1) % 10 == 0) {
            save_weights(h_weights_encoder, h_weights_decoder);
        }
    }

    // Save the final reconstructed images
    float* reconstructed_images = (float*)malloc(NUM_TEST_IMAGES * INPUT_SIZE * sizeof(float));

    // Perform reconstruction on test images
    for (int i = 0; i < NUM_TEST_IMAGES; i += BATCH_SIZE) {
        int current_batch_size = min(BATCH_SIZE, NUM_TEST_IMAGES - i);

        // Copy batch to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input,
                                   test_images + i * INPUT_SIZE,
                                   current_batch_size * INPUT_SIZE * sizeof(float),
                                   cudaMemcpyHostToDevice));

        // Forward pass for reconstruction
        encoder_forward<<<blocks_encoder, threads_per_block>>>(d_input, d_hidden, d_weights_encoder, current_batch_size);
        decoder_forward<<<blocks_decoder, threads_per_block>>>(d_hidden, d_output, d_weights_decoder, current_batch_size);

        // Copy reconstructed images back to host
        CHECK_CUDA_ERROR(cudaMemcpy(reconstructed_images + i * INPUT_SIZE,
                                   d_output,
                                   current_batch_size * INPUT_SIZE * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    }

    save_reconstructed_images(test_images, reconstructed_images, 10);  // Save the first 10 test images

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights_encoder);
    cudaFree(d_weights_decoder);
    cudaFree(d_d_hidden);
    cudaFree(d_d_output);
    cudaFree(d_d_weights_encoder);
    cudaFree(d_d_weights_decoder);

    // Free host memory
    free(h_weights_encoder);
    free(h_weights_decoder);
    free(reconstructed_images);
}

float test_autoencoder(float* test_images, float* weights_encoder, float* weights_decoder) {
    // Allocate memory on device
    float *d_input, *d_hidden, *d_output;
    float *d_weights_encoder, *d_weights_decoder;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, BATCH_SIZE * INPUT_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights_encoder, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights_decoder, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));

    // Copy weights to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights_encoder, weights_encoder,
                                INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights_decoder, weights_decoder,
                                HIDDEN_SIZE * INPUT_SIZE * sizeof(float),
                                cudaMemcpyHostToDevice));

    // Define CUDA thread blocks
    int threads_per_block = 256;
    int blocks_encoder = (BATCH_SIZE * HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    int blocks_decoder = (BATCH_SIZE * INPUT_SIZE + threads_per_block - 1) / threads_per_block;

    float total_loss = 0.0f;
    int num_test_batches = (NUM_TEST_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int batch = 0; batch < num_test_batches; batch++) {
        int current_batch_size = min(BATCH_SIZE, NUM_TEST_IMAGES - batch * BATCH_SIZE);

        // Copy batch data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input,
                                   test_images + batch * BATCH_SIZE * INPUT_SIZE,
                                   current_batch_size * INPUT_SIZE * sizeof(float),
                                   cudaMemcpyHostToDevice));

        // Forward pass
        encoder_forward<<<blocks_encoder, threads_per_block>>>(d_input, d_hidden, d_weights_encoder, current_batch_size);
        decoder_forward<<<blocks_decoder, threads_per_block>>>(d_hidden, d_output, d_weights_decoder, current_batch_size);

        // Compute test loss
        float* h_output = (float*)malloc(current_batch_size * INPUT_SIZE * sizeof(float));
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output,
                                   current_batch_size * INPUT_SIZE * sizeof(float),
                                   cudaMemcpyDeviceToHost));

        float batch_loss = 0.0f;
        for (int i = 0; i < current_batch_size * INPUT_SIZE; i++) {
            float diff = h_output[i] - test_images[batch * BATCH_SIZE * INPUT_SIZE + i];
            batch_loss += diff * diff;
        }
        batch_loss /= (current_batch_size * INPUT_SIZE);
        total_loss += batch_loss * current_batch_size;

        free(h_output);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights_encoder);
    cudaFree(d_weights_decoder);

    return total_loss / NUM_TEST_IMAGES;
}

// Replace the stub load_mnist_data function with this implementation
void load_mnist_data(float* train_images, float* test_images, const char* data_path) {
    char train_path[256];
    char test_path[256];

    snprintf(train_path, 256, "%s/mnist_train.bin", data_path);
    snprintf(test_path, 256, "%s/mnist_test.bin", data_path);

    FILE* train_file = fopen(train_path, "rb");
    FILE* test_file = fopen(test_path, "rb");

    if (!train_file || !test_file) {
        printf("Error: Could not open MNIST data files. Using random data instead.\n");
        printf("Make sure to run the Python script to download and prepare MNIST data first.\n");

        // Generate random data as fallback
        for (int i = 0; i < NUM_TRAIN_IMAGES * INPUT_SIZE; i++) {
            train_images[i] = (float)rand() / RAND_MAX;
        }

        for (int i = 0; i < NUM_TEST_IMAGES * INPUT_SIZE; i++) {
            test_images[i] = (float)rand() / RAND_MAX;
        }

        if (train_file) fclose(train_file);
        if (test_file) fclose(test_file);
        return;
    }

    size_t train_read = fread(train_images, sizeof(float), NUM_TRAIN_IMAGES * INPUT_SIZE, train_file);
    size_t test_read = fread(test_images, sizeof(float), NUM_TEST_IMAGES * INPUT_SIZE, test_file);

    printf("Loaded %zu training and %zu test images\n", train_read / INPUT_SIZE, test_read / INPUT_SIZE);

    fclose(train_file);
    fclose(test_file);
}

void save_weights(float* weights_encoder, float* weights_decoder) {
    FILE* fp_encoder = fopen("weights_encoder.bin", "wb");
    FILE* fp_decoder = fopen("weights_decoder.bin", "wb");

    if (fp_encoder && fp_decoder) {
        fwrite(weights_encoder, sizeof(float), INPUT_SIZE * HIDDEN_SIZE, fp_encoder);
        fwrite(weights_decoder, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, fp_decoder);

        fclose(fp_encoder);
        fclose(fp_decoder);
        printf("Weights saved to weights_encoder.bin and weights_decoder.bin\n");
    } else {
        printf("Error: Could not save weights\n");
    }
}

void save_reconstructed_images(float* test_images, float* reconstructed_images, int num_images) {
    // This function would save the original and reconstructed images side by side
    // For simplicity, we'll just save the raw data
    FILE* fp = fopen("reconstructed_images.bin", "wb");

    if (fp) {
        for (int i = 0; i < num_images; i++) {
            fwrite(test_images + i * INPUT_SIZE, sizeof(float), INPUT_SIZE, fp);
            fwrite(reconstructed_images + i * INPUT_SIZE, sizeof(float), INPUT_SIZE, fp);
        }

        fclose(fp);
        printf("Reconstructed images saved to reconstructed_images.bin\n");
    } else {
        printf("Error: Could not save reconstructed images\n");
    }

}
