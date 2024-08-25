#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

typedef unsigned char byte;

// calculates ||u-v||^2
int vec_dist_sq (byte* u, byte* v, int dim) {
    int dist_sq = 0;
    for (int i=0;i<dim;i++) {
        dist_sq += (u[i]-v[i])*(u[i]-v[i]);
    }
    return dist_sq;
}

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

int main (int argc, char** argv) {

    // read the number of images to test from the command line
    if (argc < 2) {
        printf ("Command usage : %s %s\n",argv[0],"num");
        return 1;
    }
    int num = atoi(argv[1]);
    if (num > 10000) {
        num = 10000;
    }

    // read the data
    int num_train = 60000;
    int num_test = 10000;
    int dim = 784;
    byte* train_bytes = (byte*)malloc(num_train*dim*sizeof(byte));
    byte* test_bytes = (byte*)malloc(num_test*dim*sizeof(byte));
    int* nearest = (int*)malloc(num_test*sizeof(int));
    byte* train_labels = (byte*)malloc(num_train*sizeof(byte));
    byte* test_labels = (byte*)malloc(num_test*sizeof(byte));
    char train_images_file[] = "train-images-idx3-ubyte";
    char train_labels_file[] = "train-labels-idx1-ubyte";
    char test_images_file[] = "t10k-images-idx3-ubyte";
    char test_labels_file[] = "t10k-labels-idx1-ubyte";
    read_bytes_bin(train_bytes,num_train*dim,train_images_file,16);
    read_bytes_bin(train_labels,num_train,train_labels_file,8);
    read_bytes_bin(test_bytes,num_test*dim,test_images_file,16);
    read_bytes_bin(test_labels,num_test,test_labels_file,8);

    // start timer
    clock_t begin = clock();

    // for each test vector find the nearest training vector
    for (int i=0;i<num;i++) {
        int min_dist_sq = INT_MAX;
        for (int j=0;j<num_train;j++) {
            int dist_sq = vec_dist_sq(test_bytes+i*dim,train_bytes+j*dim,dim);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest[i] = j;
            }
        }
    }

    // stop timer
    clock_t end = clock();
    double elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC;

    // print results
    int classify_errors = 0;
    for (int i=0;i<num;i++) {
        if (test_labels[i] != train_labels[nearest[i]]) {
            classify_errors += 1;
        }
    }
    printf ("number of MNIST training images = %d\n",num_train);
    printf ("number of MNIST test images = %d\n",num_test);
    printf ("elapsed time = %.2f seconds\n",elapsed_time);
    printf ("number of classification errors = %d\n",classify_errors);
    printf ("classificiation rate = %.4f\n",1.0*(num_test-classify_errors)/num_test);

    // free dynamically allocated memory
    free (train_bytes);
    free (test_bytes);
    free (nearest);
    free (train_labels);
    free (test_labels);
}
