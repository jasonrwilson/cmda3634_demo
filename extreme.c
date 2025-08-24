#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef unsigned char byte;

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

typedef struct {
    int max_dist_sq;
    int i,j;
} extreme_info;

int vec_dist_sq(byte* u, byte* v, int dim) {
    int dist_sq = 0;
    for (int i=0;i<dim;i++) {
	    dist_sq += (u[i]-v[i])*(u[i]-v[i]);
    }
    return dist_sq;
}

int main (int argc, char** argv) {

	if (argc < 2) {
		printf ("usage: %s num_points\n",argv[0]);
	}
	int num_points = atoi(argv[1]);
	
    // read in a MNIST image set
    int len = 60000;
    int dim = 784;
    byte* data = (byte*)malloc(len*dim*sizeof(byte));
    char images_file[] = "train-images-idx3-ubyte";
    read_bin(data,len*dim,images_file,16);

	if ((num_points < 0) || (num_points > len)) {
		num_points = len;
	}

    // start the timer
    clock_t start = clock();

    // find the extreme pair
    extreme_info info = { 0, -1, -1 };
    for (int i=0;i<num_points-1;i++) {
	    for (int j=i+1;j<num_points;j++) {
	        int dist_sq = vec_dist_sq(data+i*dim,data+j*dim,dim);
	        if (dist_sq > info.max_dist_sq) {
		        info.max_dist_sq = dist_sq;
		        info.i = i;
		        info.j = j;
	        }
	    }
    }

    // stop the timer
    clock_t stop = clock();
    double elapsed = (double)(stop-start)/CLOCKS_PER_SEC;

    // print results
    printf ("number of points = %d\n",num_points);
    printf ("elapsed time = %.4f seconds\n",elapsed);
    printf ("extreme distance = %.2f\n",sqrt(info.max_dist_sq));
    printf ("extreme pair = (%d,%d)\n",info.i,info.j);

    // free dynamically allocated memory
    free(data);
}
