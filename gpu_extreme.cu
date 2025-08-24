#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>

typedef unsigned char byte;
typedef unsigned long long uint64;

// read data from a binary file
void read_bin (byte* data, int num_bytes, char* filename, int header_size) {
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

void extreme_info_expand(uint64 info, int* dist_sq, int* i, int* j) {
    *j = info & 0xFFFF; // apply bit mask to zero out all but lower 16 bits
    info = info >> 16; // shift 16 bits to the right
    *i = info & 0xFFFF; // apply bit mask to zero out all but lower 16 bits
    info = info >> 16; // shift 16 bits to the right
    *dist_sq = info;
}

__device__ uint64 extreme_info_compress(int dist_sq, int i, int j) {
    uint64 info = dist_sq;
    info = info << 16; // shift 16 bits to the left
    info += i;
    info = info << 16; // shift 16 bits to the left
    info += j;
    return info;
}

__global__ void extremeKernel(float* dot_prods, float* i_j_dot_prods, int len,
			      int start, int size, uint64* max_info) {

    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < size) {
	    float term1 = dot_prods[t+start];
        float max_dist_sq = 0;
        int farthest_idx;
	    for (int j=0;j<t+start;j++) {
            float term2 = term1 + dot_prods[j];
            float term3 = i_j_dot_prods[j*size+t];
            float dist_sq = term2 - 2.0*term3;
            if (dist_sq > max_dist_sq) {
                max_dist_sq = dist_sq;
                farthest_idx = j;
            }
        }
	    uint64 info = extreme_info_compress((int)max_dist_sq,t+start,farthest_idx);
	    atomicMax(max_info,info);
    }
}

__global__ void dotprodsKernel (float *data, int len, int dim, float* dot_prods) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < len) {
        float result = 0;
        for (int i=0;i<dim;i++) {
            float term = data[i*len+t];
            result += term*term;
        }
        dot_prods[t] = result;
    }
}

int main (int argc, char** argv) {

    // read in a MNIST image set
    int len = 60000;
    int dim = 784;
    byte* data_bytes = (byte*)malloc(len*dim*sizeof(byte));
    // read the file that thas the dataset stored in column major order
    char images_file[] = "train-images-idx3-ubyte-c";
    read_bin(data_bytes,len*dim,images_file,16);

    // translate input data from byte to float matrices
    // so that we can use CUBLAS
    float* data = (float*)malloc(len*dim*sizeof(float));
    for (int i=0;i<len*dim;i++) {
        data[i] = data_bytes[i];
    }

    // allocate device memory
    int size = 30000;
    float* d_data;
    uint64* d_max_info;
    float* d_dot_prods;
    float* d_i_j_dot_prods;
    cudaMalloc(&d_data,len*dim*sizeof(float));
    if (d_data == NULL) {
	    printf ("cudaMalloc failed\n");
	    return 1;
    }
    cudaMalloc(&d_max_info,sizeof(uint64));
    if (d_max_info == NULL) {
	    printf ("cudaMalloc failed\n");
	    return 1;
    }
    cudaMalloc(&d_dot_prods,len*sizeof(float));
    if (d_dot_prods == NULL) {
	    printf ("cudaMalloc failed\n");
	    return 1;
    }
    cudaMalloc(&d_i_j_dot_prods,(long)size*len*sizeof(float));
    if (d_i_j_dot_prods == NULL) {
	    printf ("cudaMalloc failed\n");
	    return 1;
    }

    // start the timer
    clock_t start = clock();

    // copy data to device
    cudaMemcpy(d_data,data,len*dim*sizeof(float),cudaMemcpyHostToDevice);

    // initialize the device max_info to 0
    cudaMemset(d_max_info,0,sizeof(uint64));

    // launch kernel to compute i-i dot products
    int B = 128;
    int G = (len+B-1)/B;
    dotprodsKernel <<< G, B >>> (d_data,len,dim,d_dot_prods);

    // setup CUBLAS
    float alpha = 1.0, beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // this next command enables the Tensor Cores!
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    G = (size+B-1)/B;
    for (int start = 0; start < len; start += size) {

	    // use CUBLAS to compute a batch of i-j dot products
	    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
		    size, len, dim, &alpha,
		    d_data+start, len,
		    d_data, len, &beta,
		    d_i_j_dot_prods, size);
	    cudaDeviceSynchronize();

	    // launch kernel to compute a batch of extreme distances
	    extremeKernel <<< G, B >>> (d_dot_prods,d_i_j_dot_prods,len,start,size,d_max_info);
    }

    // copy max_info from device to host
    uint64 max_info;
    cudaMemcpy(&max_info,d_max_info,sizeof(uint64),cudaMemcpyDeviceToHost);

    // stop the timer
    clock_t stop = clock();
    double elapsed = (double)(stop-start)/CLOCKS_PER_SEC;

    // expand max_info
    int max_dist_sq, i, j;
    extreme_info_expand(max_info,&max_dist_sq,&i,&j);

    // print results
    printf ("number of images = %d\n",len);
    printf ("elapsed time = %.4f seconds\n",elapsed);
    printf ("extreme distance = %.2f\n",sqrt(max_dist_sq));
    printf ("extreme pair = (%d,%d)\n",i,j);

    // free dynamically allocated memory
    free(data_bytes);
    free(data);
    cudaFree(d_data);
    cudaFree(d_max_info);
    cudaFree(d_dot_prods);
    cudaFree(d_i_j_dot_prods);
}
