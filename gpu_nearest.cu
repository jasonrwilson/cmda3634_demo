#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda.h>
#include <cublas_v2.h>

typedef unsigned char byte;

// read byte data from a binary file
void read_bytes_bin (byte* data, int num_bytes, char* filename, int header_size) {
    byte header[header_size];
    FILE* fptr;
    int num_read;
    // open the binary file for reading
    fptr = fopen(filename,"rb");
    // need to check for null
    if (fptr == 0) {
        printf ("Error opening binary data file %s.\n",filename);
        exit(1);
    }
    // read header
    num_read = fread(header, sizeof(byte), header_size, fptr);
    // read data
    num_read = fread(data, sizeof(byte), num_bytes, fptr);
    if (num_read != num_bytes) {
        printf ("Warning : binary data file read error for %s.\n",filename);
    }
    // close the binary file
    fclose(fptr);
}

__global__ void trainDotProds (float *train, int num_train, int dim, float* dot_prods) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < num_train) {
        float result = 0;
        for (int i=0;i<dim;i++) {
            float term = train[i*num_train+t];
            result += term*term;
        }
        dot_prods[t] = result;
    }
}

__global__ void calcNearest(float* train_dot_prods, int num_train,
        float* test_train_dot_prods, int num_test,
        int* nearest) {

    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < num_test) {
        float min_dist_sq = FLT_MAX;
        int nearest_idx;
        for (int i=0;i<num_train;i++) {
            float term1 = train_dot_prods[i];
            float term2 = test_train_dot_prods[i*num_test+t];
            float dist_sq = term1-2*term2;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }
        nearest[t] = nearest_idx;
    }
}

int main (int argc, char* argv[]) {

    int num_train = 60000;
    int num_test = 10000;
    int dim = 784;
    byte* train_bytes = (byte*)malloc(num_train*dim*sizeof(byte));
    byte* test_bytes = (byte*)malloc(num_test*dim*sizeof(byte));
    int* nearest = (int*)malloc(num_test*sizeof(int));
    byte* train_labels = (byte*)malloc(num_train*sizeof(byte));
    byte* test_labels = (byte*)malloc(num_test*sizeof(byte));
    char train_images_file[] = "train-images-idx3-ubyte-c";
    char train_labels_file[] = "train-labels-idx1-ubyte";
    char test_images_file[] = "t10k-images-idx3-ubyte-c";
    char test_labels_file[] = "t10k-labels-idx1-ubyte";
    read_bytes_bin(train_bytes,num_train*dim,train_images_file,16);
    read_bytes_bin(train_labels,num_train,train_labels_file,8);
    read_bytes_bin(test_bytes,num_test*dim,test_images_file,16);
    read_bytes_bin(test_labels,num_test,test_labels_file,8);

    // translate input data from byte to float matrices
    // good to work with floating point data in case we want to further
    // process the data (such as dimensionality reduction, etc.)
    float* test = (float*)malloc(num_test*dim*sizeof(float));
    for (int i=0;i<num_test*dim;i++) {
        test[i] = test_bytes[i];
    }
    float* train = (float*)malloc(num_train*dim*sizeof(float));
    for (int i=0;i<num_train*dim;i++) {
        train[i] = train_bytes[i];
    }

    // allocate device memory
    float* d_test;
    float* d_train;
    float* d_train_dot_prods;
    float* d_test_train_dot_prods;
    int* d_nearest;
    cudaMalloc(&d_test,num_test*dim*sizeof(float));
    cudaMalloc(&d_train,num_train*dim*sizeof(float));
    cudaMalloc(&d_train_dot_prods,num_train*sizeof(float));
    cudaMalloc(&d_test_train_dot_prods,num_test*num_train*sizeof(float));
    cudaMalloc(&d_nearest,num_test*sizeof(int));

    // copy data to device
    cudaMemcpy(d_test,test,num_test*dim*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_train,train,num_train*dim*sizeof(float),cudaMemcpyHostToDevice);

    // setup CUBLAS
    float alpha = 1.0, beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // for timing kernel execution
    cudaEvent_t tic, toc;
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    cudaEventRecord(tic);

    // use CUBLAS to compute test/training dot products
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            num_test, num_train, dim, &alpha,
            d_test, num_test,
            d_train, num_train, &beta,
            d_test_train_dot_prods, num_test);

    // launch kernel to compute training dot products
    int B = 128;
    int G = (num_train+B-1)/B;
    trainDotProds <<< G, B >>> (d_train,num_train,dim,d_train_dot_prods);

    // launch kernel to compute nearest neighbors
    G = (num_test+B-1)/B;
    calcNearest <<< G, B >>>   (d_train_dot_prods, num_train,
            d_test_train_dot_prods, num_test,
            d_nearest);
    cudaEventRecord(toc);

    // copy nearest neighbor indices from device to host
    cudaMemcpy(nearest, d_nearest, num_test*sizeof(int),cudaMemcpyDeviceToHost);

    // calculate elapsed time
    cudaEventSynchronize(toc);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, tic, toc);
    double elapsed_time = milliseconds/1000.0;

    // print results
    int classify_errors = 0;
    for (int i=0;i<num_test;i++) {
        if (test_labels[i] != train_labels[nearest[i]]) {
            classify_errors += 1;
        }
    }
    printf ("number of MNIST training images = %d\n",num_train);
    printf ("number of MNIST test images = %d\n",num_test);
    printf ("elapsed time = %.4f seconds\n",elapsed_time);
    printf ("number of classification errors = %d\n",classify_errors);
    printf ("classificiation rate = %.4f\n",1.0*(num_test-classify_errors)/num_test);

    // free dynamically allocated memory
    free (train_bytes);
    free (test_bytes);
    free (nearest);
    free (train_labels);
    free (test_labels);
    free (train);
    free (test);
    cudaFree(d_test);
    cudaFree(d_train);
    cudaFree(d_train_dot_prods);
    cudaFree(d_test_train_dot_prods);
    cudaFree(d_nearest);
}
