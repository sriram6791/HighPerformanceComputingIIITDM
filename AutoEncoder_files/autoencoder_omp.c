// autoencoder_omp.c
// -------------------------------------------
// Commands to run
/*
python3 prepare_mnist.py
gcc -O3 -fopenmp -o autoencoder_omp autoencoder_omp.c -lm
export OMP_NUM_THREADS=4
./autoencoder_omp ./mnist_data


 gcc -O3 -fopenmp -o autoencoder_omp autoencoder_omp.c -lm
 export OMP_NUM_THREADS=16
 ./autoencoder_omp ./mnist_data
*/
// -------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// MNIST parameters
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES  10000
#define IMG_WIDTH        28
#define IMG_HEIGHT       28
#define INPUT_SIZE       (IMG_WIDTH * IMG_HEIGHT)

// Network architecture
#define HIDDEN_SIZE   128
#define BATCH_SIZE    128
#define LEARNING_RATE 0.001f
#define NUM_EPOCHS    5

#define CHECK_ALLOC(ptr) if (!(ptr)) { fprintf(stderr, "Allocation failed\n"); exit(-1); }

// Load raw floats from .bin files
void load_mnist_data(float* train, float* test, const char* path) {
    char train_path[256], test_path[256];
    snprintf(train_path, 256, "%s/mnist_train.bin", path);
    snprintf(test_path, 256, "%s/mnist_test.bin",  path);
    FILE* ftra = fopen(train_path, "rb");
    FILE* ftes = fopen(test_path, "rb");
    if (!ftra || !ftes) { fprintf(stderr, "Cannot open MNIST files\n"); exit(-1); }
    size_t nt = fread(train, sizeof(float), NUM_TRAIN_IMAGES * INPUT_SIZE, ftra);
    size_t ne = fread(test,  sizeof(float), NUM_TEST_IMAGES  * INPUT_SIZE, ftes);
    if (nt != NUM_TRAIN_IMAGES*INPUT_SIZE || ne != NUM_TEST_IMAGES*INPUT_SIZE) {
        fprintf(stderr, "Unexpected MNIST file size\n"); exit(-1);
    }
    fclose(ftra);
    fclose(ftes);
}

// He initialization
void initialize_weights(float* W, int in_dim, int out_dim) {
    float scale = sqrtf(2.0f / in_dim);
    #pragma omp parallel for
    for (int i = 0; i < in_dim * out_dim; i++)
        W[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
}

// Forward: input → hidden
void encoder_forward(const float* input, float* hidden, const float* W_enc, int batch) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            float sum = 0.f;
            for (int i = 0; i < INPUT_SIZE; i++)
                sum += input[b*INPUT_SIZE + i] * W_enc[i*HIDDEN_SIZE + h];
            hidden[b*HIDDEN_SIZE + h] = fmaxf(0.f, sum);
        }
    }
}

// Forward: hidden → output
void decoder_forward(const float* hidden, float* output, const float* W_dec, int batch) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < INPUT_SIZE; o++) {
            float sum = 0.f;
            for (int h = 0; h < HIDDEN_SIZE; h++)
                sum += hidden[b*HIDDEN_SIZE + h] * W_dec[h*INPUT_SIZE + o];
            output[b*INPUT_SIZE + o] = 1.f / (1.f + expf(-sum));
        }
    }
}

// Backprop: compute grads w.r.t weights
void compute_gradients(const float* input,
                       const float* output,
                       const float* hidden,
                       const float* W_dec,
                       float* dW_enc,
                       float* dW_dec,
                       int batch)
{
    memset(dW_enc, 0, INPUT_SIZE*HIDDEN_SIZE*sizeof(float));
    memset(dW_dec, 0, HIDDEN_SIZE*INPUT_SIZE*sizeof(float));

    for (int b = 0; b < batch; b++) {
        float d_hidden[HIDDEN_SIZE];
        memset(d_hidden, 0, sizeof(d_hidden));
        // Decoder layer gradients + accumulate hidden deltas
        for (int i = 0; i < INPUT_SIZE; i++) {
            int idx = b*INPUT_SIZE + i;
            float y = output[idx], t = input[idx];
            float d_out = 2.f * (y - t) / batch * y * (1 - y);
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                dW_dec[h*INPUT_SIZE + i] += hidden[b*HIDDEN_SIZE + h] * d_out;
                d_hidden[h]               += W_dec[h*INPUT_SIZE + i]   * d_out;
            }
        }
        // Encoder layer gradients via ReLU
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            float dh = d_hidden[h] * (hidden[b*HIDDEN_SIZE + h] > 0.f ? 1.f : 0.f);
            for (int i = 0; i < INPUT_SIZE; i++)
                dW_enc[i*HIDDEN_SIZE + h] += input[b*INPUT_SIZE + i] * dh;
        }
    }
}

// SGD update
void update_weights(float* W, const float* dW, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        W[i] -= LEARNING_RATE * dW[i];
}

int main(int argc, char** argv) {
    srand(42);

    // 1) Load data
    float* train_images = malloc(NUM_TRAIN_IMAGES * INPUT_SIZE * sizeof(float));
    float* test_images  = malloc(NUM_TEST_IMAGES  * INPUT_SIZE * sizeof(float));
    CHECK_ALLOC(train_images);
    CHECK_ALLOC(test_images);
    const char* data_path = (argc > 1 ? argv[1] : "./mnist_data/");
    load_mnist_data(train_images, test_images, data_path);

    // 2) Allocate & init weights
    float* W_enc = malloc(INPUT_SIZE*HIDDEN_SIZE * sizeof(float));
    float* W_dec = malloc(HIDDEN_SIZE*INPUT_SIZE * sizeof(float));
    CHECK_ALLOC(W_enc);
    CHECK_ALLOC(W_dec);
    initialize_weights(W_enc, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(W_dec, HIDDEN_SIZE, INPUT_SIZE);

    // 3) Buffers for training
    float* dW_enc = malloc(INPUT_SIZE*HIDDEN_SIZE * sizeof(float));
    float* dW_dec = malloc(HIDDEN_SIZE*INPUT_SIZE * sizeof(float));
    float* hidden = malloc(BATCH_SIZE*HIDDEN_SIZE * sizeof(float));
    float* output = malloc(BATCH_SIZE*INPUT_SIZE  * sizeof(float));
    CHECK_ALLOC(dW_enc); CHECK_ALLOC(dW_dec);
    CHECK_ALLOC(hidden); CHECK_ALLOC(output);

    int num_batches = NUM_TRAIN_IMAGES / BATCH_SIZE;
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        for (int b = 0; b < num_batches; b++) {
            float* batch_ptr = train_images + b * BATCH_SIZE * INPUT_SIZE;
            encoder_forward(batch_ptr, hidden, W_enc, BATCH_SIZE);
            decoder_forward(hidden, output, W_dec, BATCH_SIZE);
            compute_gradients(batch_ptr, output, hidden, W_dec,
                              dW_enc, dW_dec, BATCH_SIZE);
            update_weights(W_enc, dW_enc, INPUT_SIZE*HIDDEN_SIZE);
            update_weights(W_dec, dW_dec, HIDDEN_SIZE*INPUT_SIZE);
        }

        // Evaluate on test set
        float test_loss = 0.f;
        int test_batches = NUM_TEST_IMAGES / BATCH_SIZE;
        for (int tb = 0; tb < test_batches; tb++) {
            float* tb_ptr = test_images + tb * BATCH_SIZE * INPUT_SIZE;
            encoder_forward(tb_ptr, hidden, W_enc, BATCH_SIZE);
            decoder_forward(hidden, output, W_dec, BATCH_SIZE);
            for (int i = 0; i < BATCH_SIZE * INPUT_SIZE; i++) {
                float diff = output[i] - tb_ptr[i];
                test_loss += diff * diff;
            }
        }
        test_loss /= (NUM_TEST_IMAGES * INPUT_SIZE);
        float psnr = 10.f * log10f(1.f / test_loss);
        printf("Epoch %2d — test MSE: %.6f, PSNR: %.2f dB\n",
               epoch, test_loss, psnr);
    }

    // Save final weights
    FILE* fe = fopen("weights_enc.bin", "wb");
    FILE* fd = fopen("weights_dec.bin", "wb");
    fwrite(W_enc, sizeof(float), INPUT_SIZE*HIDDEN_SIZE, fe);
    fwrite(W_dec, sizeof(float), HIDDEN_SIZE*INPUT_SIZE, fd);
    fclose(fe);  fclose(fd);
    printf("Saved final weights.\n");

    // --- generate & save reconstruction grid ---
    const int N = 10;
    const int gw = 2 * IMG_WIDTH;
    const int gh = N * IMG_HEIGHT;
    uint8_t *grid = malloc(gw * gh);
    if (!grid) { fprintf(stderr,"Grid alloc failed\n"); exit(-1); }

    // run the first N test images through the autoencoder
    encoder_forward(test_images, hidden, W_enc, N);
    decoder_forward(hidden, output, W_dec, N);

    for (int r = 0; r < N; r++) {
        // original in col 0
        for (int y = 0; y < IMG_HEIGHT; y++)
        for (int x = 0; x < IMG_WIDTH;  x++) {
            float v = test_images[r*INPUT_SIZE + y*IMG_WIDTH + x];
            grid[(r*IMG_HEIGHT + y)*gw + x] = (uint8_t)(v * 255);
        }
        // reconstruction in col 1
        for (int y = 0; y < IMG_HEIGHT; y++)
        for (int x = 0; x < IMG_WIDTH;  x++) {
            float v = output[r*INPUT_SIZE + y*IMG_WIDTH + x];
            grid[(r*IMG_HEIGHT + y)*gw + (IMG_WIDTH + x)] = (uint8_t)(v * 255);
        }
    }

    stbi_write_png("openmp_mnist_reconstructions.png", gw, gh, 1, grid, gw);
    free(grid);
    printf("Saved reconstructions to openmp_mnist_reconstructions.png\n");

    // Cleanup
    free(train_images);  free(test_images);
    free(W_enc);         free(W_dec);
    free(dW_enc);        free(dW_dec);
    free(hidden);        free(output);
    return 0;
}
