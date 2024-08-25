import sys
import numpy as np
import gzip
import time # to time part of the code

# make sure a command line argument for the number of test images is provided
if (len(sys.argv) < 2):
    print ('command usage :',sys.argv[0],'num_test')
    exit(1)
num_test = int(sys.argv[1])
print ('number of digits to classify =',num_test)

# Opens MNIST training image set and stores it as a 60000 x 784 matrix
f = gzip.open('train-images-idx3-ubyte.gz','r')
f.read(16) # skip file header
buf = f.read(60000*28*28)
data = np.frombuffer(buf,dtype=np.uint8)
train = data.reshape(60000,28*28).astype(np.int32)

# Opening and saving the 60000 training labels
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8) #skip header
buf = f.read(60000)
train_labels = np.frombuffer(buf,dtype=np.uint8)

# Opens MNIST test image set and stores it as a 10000 x 784 matrix
f = gzip.open('t10k-images-idx3-ubyte.gz','r')
f.read(16) # skip header
buf = f.read(10000*28*28)
data = np.frombuffer(buf, dtype=np.uint8)
test = data.reshape(10000,28*28).astype(np.int32)

# Opening and saving the 10000 test labels
f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
f.read(8) #skip header
buf = f.read(10000)
test_labels = np.frombuffer(buf,dtype=np.uint8)

# Allocate space to store the nearest neighbor indices
nearest = np.empty(num_test,dtype='int32')

# time just the nearest neighbor code
start = time.process_time()

# find the index of the training image closest to the test image with the given index
# note that we interpret the image data as 32 bit integers to avoid overflow
for test_index in range(num_test):
    min_dist_sq = np.inf
    for train_index in range(len(train)):
        diff = train[train_index]-test[test_index]
        dist_sq = np.dot(diff,diff)
        if (dist_sq < min_dist_sq):
            min_dist_sq = dist_sq
            nearest[test_index] = train_index

# record and print elapsed time
elapsed = time.process_time()-start
print ('Time to find nearest neighbors in Python =',np.round(elapsed,4),'seconds')

# count nearest neighbor classification errors
labels_diff = test_labels[:num_test] - train_labels[nearest]
classify_errors = np.count_nonzero(labels_diff)
print ('number of classification errors =',classify_errors)
print ('classificiation rate =',(num_test-classify_errors)/num_test)
