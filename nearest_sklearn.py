import numpy as np
import time
import sys
from sklearn.neighbors import KNeighborsClassifier

# Usage check
if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "num_test")
    exit(1)
num_test = int(sys.argv[1])
if num_test > 10000:
    num_test = 10000

# Read MNIST image files
def read_images(filename, num_images):
    with open(filename, 'rb') as f:
        f.read(16)  # skip header
        data = np.frombuffer(f.read(num_images * 28 * 28), dtype=np.uint8)
        return data.reshape(num_images, 784).astype(np.int32)

# Read MNIST label files
def read_labels(filename, num_labels):
    with open(filename, 'rb') as f:
        f.read(8)  # skip header
        return np.frombuffer(f.read(num_labels), dtype=np.uint8)
        
# Load data
train_images = read_images('train-images-idx3-ubyte', 60000)
train_labels = read_labels('train-labels-idx1-ubyte', 60000)
test_images = read_images('t10k-images-idx3-ubyte', 10000)
test_labels = read_labels('t10k-labels-idx1-ubyte', 10000)

# Truncate test set if requested
test_images = test_images[:num_test]
test_labels = test_labels[:num_test]

start = time.time()

# Build and evaluate KNN model
model = KNeighborsClassifier(n_neighbors=1, algorithm='brute', n_jobs=1)
model.fit(train_images, train_labels)
predicted = model.predict(test_images)
elapsed = time.time() - start

# Report results
errors = np.count_nonzero(predicted != test_labels)
accuracy = (num_test - errors) / num_test

print("Time to classify", num_test, "digits =", round(elapsed, 4), "seconds")
print("Number of classification errors:", errors)
print("Classification rate =", round(accuracy, 4))
