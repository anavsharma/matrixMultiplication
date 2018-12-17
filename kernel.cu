// Matrix multiplication using CUDA
// Based on code from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory

#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#define BLOCK_SIZE 64

//CPU matrix multiplication
__host__ void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m) {
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			float tmp = 0.0;
			for (int h = 0; h < m; ++h)
			{
				tmp += h_a[i * m + h] * h_b[h * m + j];
			}
			h_result[i * m + j] = tmp;
		}
	}
}

// GPU matrix multiplication
// Uses square matrices
// Based on code from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
__global__ void gpu_multiply(float *left, float *right, float *res, int dim) {
	float temp = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < dim; ++e)
		temp += left[row * dim + e]
		* right[e * dim + col];
	res[row * dim + col] = temp;
}


// Function to fill a non-square matrix with zeros to make it a square matrix
__host__ int fill(float **Lmatrix, float **Rmatrix, int LdimX, int LdimY, int RdimX, int RdimY) {

	int sqr_dim_X, sqr_dim_Y, size;

	sqr_dim_X = RdimX;
	if (LdimX > RdimX) {
		sqr_dim_X = LdimX;
	}

	sqr_dim_Y = RdimY;
	if (LdimY > RdimY) {
		sqr_dim_Y = LdimY;
	}

	size = sqr_dim_Y;
	if (sqr_dim_X > sqr_dim_Y) {
		size = sqr_dim_X;
	}

	int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
	size = temp * BLOCK_SIZE;

	size_t pt_size = size * size * sizeof(float);

	*Lmatrix = (float *)malloc(pt_size);
	*Rmatrix = (float *)malloc(pt_size);

	memset(*Lmatrix, 0, pt_size);
	memset(*Rmatrix, 0, pt_size);

	for (int i = 0; i < LdimX; i++) {
		for (int j = 0; j < LdimY; j++) {
			int dummy = size * i + j;
			(*Lmatrix)[dummy] = sinf(dummy);
		}
	}
	for (int i = 0; i < RdimX; i++) {
		for (int j = 0; j < RdimY; j++) {
			int dummy = size * i + j;
			(*Rmatrix)[dummy] = cosf(dummy);
		}
	}
	return size;
}

// GPU matrix multiplication - optimized
// uses shared memory
// based on code from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory

__global__ void multiply(float *left, float *right, float *res, int dim) {

	int i, j;
	float temp = 0;

	__shared__ float Left_shared_t[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

	// Row i of matrix left
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

		// Column j of matrix left
		j = tileNUM * BLOCK_SIZE + threadIdx.x;
		i = tileNUM * BLOCK_SIZE + threadIdx.y;
		// Load left[i][j] to shared mem

		Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
		// Load right[i][j] to shared mem

		Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
		// Synchronize before computation
		__syncthreads();

		// Accumulate one tile of res from tiles of left and right in shared mem
		for (int k = 0; k < BLOCK_SIZE; k++) {

			temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
		}
		// Synchronize
		__syncthreads();
	}
	// Store accumulated value to res
	res[row * dim + col] = temp;
}


int main(void)
{
	// Matrix sizes
	int Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y, Left_vector_size, Right_vector_size;

	// Declaring pointers for arrays
	float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_h, *Res_d, *CPU;  

	printf("Enter m n n k :\n");
	scanf("%d %d %d %d", &Left_matrix_x, &Left_matrix_y, &Right_matrix_x, &Right_matrix_y);

	int dim = fill(&Left_Vector_h, &Right_Vector_h, Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y);

	size_t vector_size;
	vector_size = dim * dim * sizeof(float);

	//Allocate memory on host
	Res_h = (float *)malloc(vector_size); 
	CPU = (float *)malloc(vector_size);
	//Allocate memory on device
	cudaMalloc((void **)&Left_Vector_d, vector_size);     
	cudaMalloc((void **)&Right_Vector_d, vector_size);   
	cudaMalloc((void **)&Res_d, vector_size);

	//Copy values to device
	cudaMemcpy(Left_Vector_d, Left_Vector_h, vector_size, cudaMemcpyHostToDevice);
	//Copy values to device
	cudaMemcpy(Right_Vector_d, Right_Vector_h, vector_size, cudaMemcpyHostToDevice);

	//Block dimension is directly from block_size
	dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
	//Grid dimension is found by dividing matrix dimension to block_size
	dim3 Grid_dim(dim / BLOCK_SIZE, dim / BLOCK_SIZE);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//GPU Kernel call - optimized matrix multiplication
	multiply <<< Grid_dim, Block_dim >>> (Left_Vector_d, Right_Vector_d, Res_d, dim);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float et;
	cudaEventElapsedTime(&et, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//Retrieve result from device and store it in host array
	cudaMemcpy(Res_h, Res_d, vector_size, cudaMemcpyDeviceToHost);

	clock_t begin = clock();
	//CPU Matrix multiplication
	cpu_matrix_mult(Left_Vector_h, Right_Vector_h, CPU, dim); 
	clock_t end = clock();
	double time_spent = (double)1000 * (end - begin) / CLOCKS_PER_SEC;

	//Block dimension is directly from block_size
	dim3 Block_dim_1(BLOCK_SIZE, BLOCK_SIZE);
	//Grid dimension is found by dividing matrix dimension to block_size
	dim3 Grid_dim_1(dim / BLOCK_SIZE, dim / BLOCK_SIZE);
	//commented out the functions which helps to calculate time
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
	//GPU Kernel call - non optimized matrix multiplication
	gpu_multiply <<< Grid_dim_1, Block_dim_1 >>> (Left_Vector_d, Right_Vector_d, Res_d, dim);
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	float mm;
	cudaEventElapsedTime(&mm, start1, stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);


	printf("GPU time= %f ms\n", et);

	printf("CPU time= %lf ms\n", time_spent);

	printf("GPU (not optimized) time = %f ms\n", mm);

	//Cleanup
	free(Left_Vector_h);
	free(Right_Vector_h);
	free(Res_h);
	free(CPU);
	cudaFree(Left_Vector_d);
	cudaFree(Right_Vector_d);
	cudaFree(Res_d);
} 