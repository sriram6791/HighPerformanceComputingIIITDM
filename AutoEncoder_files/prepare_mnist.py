#!/usr/bin/env python3
import os
import numpy as np
from tensorflow.keras.datasets import mnist

# 1) Download & normalize
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# 2) Flatten
x_train = x_train.reshape(-1, 28*28)
x_test  = x_test.reshape(-1, 28*28)

# 3) Save to mnist_data/
os.makedirs('mnist_data', exist_ok=True)
with open('mnist_data/mnist_train.bin', 'wb') as f:
    x_train.tofile(f)
with open('mnist_data/mnist_test.bin', 'wb') as f:
    x_test.tofile(f)

print("Wrote mnist_train.bin and mnist_test.bin into ./mnist_data/")
