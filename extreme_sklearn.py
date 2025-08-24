import numpy as np
import sys
import time
from sklearn.metrics import pairwise_distances

# Usage: python3 sklearn_farthest_pair.py num_points
if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "num_train_subset")
    exit(1)

num_points = int(sys.argv[1])
if num_points > 60000:
    print("Warning: MNIST training set only has 60000 images")
    num_points = 60000

# Load MNIST .ubyte file (uncompressed)
def read_images(filename, num_images):
    with open(filename, 'rb') as f:
        f.read(16)  # Skip header
        data = np.frombuffer(f.read(num_images * 28 * 28), dtype=np.uint8)
        return data.reshape(num_images, 784).astype(np.float32)

# Load full training set
train_images = read_images("train-images-idx3-ubyte", 60000)

# Truncate to selected number of points
subset = train_images[:num_points]

# Time the pairwise distance + argmax
start = time.time()
dists = pairwise_distances(subset, subset, metric='euclidean')
flat_index = np.argmax(dists)
i, j = np.unravel_index(flat_index, dists.shape)
max_dist = dists[i, j]
elapsed = time.time() - start

# Print result
print(f"Used {num_points} images")
print(f"Elapsed time = {elapsed:.4f} seconds")
print(f"Max distance = {max_dist:.2f}")
print(f"Farthest pair: ({i}, {j})")
