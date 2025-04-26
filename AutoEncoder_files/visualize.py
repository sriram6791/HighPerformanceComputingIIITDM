import numpy as np
import matplotlib.pyplot as plt
import struct
import os

def load_reconstructed_images(filename='reconstructed_images.bin', num_images=10, img_size=28, save_path='mnist_reconstructions.png'):
    """Load and visualize reconstructed images from binary file"""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    # Reshape into pairs of original and reconstructed images
    data = data.reshape(-1, 2, img_size * img_size)

    # Limit to the requested number of images
    data = data[:num_images]

    # Display images
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 2*num_images))

    for i in range(num_images):
        # Original
        axes[i, 0].imshow(data[i, 0].reshape(img_size, img_size), cmap='gray')
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')

        # Reconstructed
        axes[i, 1].imshow(data[i, 1].reshape(img_size, img_size), cmap='gray')
        axes[i, 1].set_title(f"Reconstructed {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

    # Display the figure
    plt.show()

def load_mnist_data():
    """Download and load actual MNIST data"""
    from tensorflow.keras.datasets import mnist

    # Load MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape to vectors
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    print(f"Loaded MNIST: {x_train.shape[0]} training, {x_test.shape[0]} test images")

    # Save to binary files for C program
    with open('mnist_train.bin', 'wb') as f:
        x_train.tofile(f)

    with open('mnist_test.bin', 'wb') as f:
        x_test.tofile(f)

    print("Saved MNIST data to mnist_train.bin and mnist_test.bin")

    return x_train, x_test

# Example usage
if __name__ == "__main__":
    # Option 1: Load and visualize existing reconstructed images
    try:
        load_reconstructed_images(save_path='mnist_reconstructions.png')
    except FileNotFoundError:
        print("No reconstructed images found. Run the autoencoder first.")

    # Option 2: Prepare actual MNIST data for the autoencoder
    try:
        load_mnist_data()
    except Exception as e:
        print(f"Could not load MNIST data: {e}")
        print("Make sure TensorFlow is installed.")
