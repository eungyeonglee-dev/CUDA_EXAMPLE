#include <stdio.h>
#include <cuda_runtime.h>
#include "../Timer/Timer.h"
#define _GPU_ID 0
#define _REPEAT 10


void runMatMulGold(float* A, float* B, float* C, int M, int N, int K)
{
	for(int m=0;m<M;m++) {
		for(int n=0;n<N;n++) {
			int C_ptr = m * N + n;
			C[C_ptr] = 0;
			for(int k=0;k<K;k++) {
				int A_ptr = m * K + k;
				int B_ptr = k * N + n;
				C[C_ptr]+=A[A_ptr]*B[B_ptr];
			}
		}
	}
}

#define BUF_SIZE 64
#define BLOCK_SIZE 4
__global__ void runMatMulGPU(float *A,float *B, float *C, int M, int N, int K)
{
	__shared__ float A_buffer[BLOCK_SIZE][BUF_SIZE];
	__shared__ float B_buffer[BUF_SIZE][BLOCK_SIZE];

	int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

	int m = gid_x;
	int n = gid_y;

	int C_ptr = m * N + n;

	/*for(int k=threadIdx.y;k<K;k+=blockDim.y) {
		int A_ptr = m * K + k;
		A_buffer[threadIdx.x][k] = A[A_ptr];
	}
	for(int k=threadIdx.x;k<K;k+=blockDim.x) {
		int B_ptr = k * N + n;
		B_buffer[k][threadIdx.y] = B[B_ptr];
	}
	__syncthreads();


	for(int k=0;k<K;k++) {
		C[C_ptr]+=A_buffer[threadIdx.x][k]*B_buffer[k][threadIdx.y];
	}*/

	C[C_ptr] = 0;
	for(int k_base=0;k_base<K;k_base+=BUF_SIZE) {
		for(int k=threadIdx.y;k<BUF_SIZE;k+=blockDim.y) {
			int A_ptr = m * K + (k+k_base);
			A_buffer[threadIdx.x][k] = A[A_ptr];
		}
		for(int k=threadIdx.x;k<BUF_SIZE;k+=blockDim.x) {
			int B_ptr = (k+k_base) * N + n;
			B_buffer[k][threadIdx.y] = B[B_ptr];
		}
		__syncthreads();


		for(int k=0;k<BUF_SIZE;k++) {
			C[C_ptr]+=A_buffer[threadIdx.x][k]*B_buffer[k][threadIdx.y];
		}
		__syncthreads();
	}

}

int main(int argc, char* argv[]) {
	int gpu_id = _GPU_ID;
	int repeat = _REPEAT;
	if (argc < 4) {
		printf("arguments: [M] [N] [K]\n\n");
	}
	
	//Get size
	long M = atoi(argv[1]);
	long N = atoi(argv[2]);
	long K = atoi(argv[3]);
	long sz_A = sizeof(float)*M*K;
	long sz_B = sizeof(float)*K*N;
	long sz_C = sizeof(float)*M*N;
	printf("MAT SIZE = %ld(M), %ld(N), %ld(K)\n", M, N, K);
	printf("\n");

	//Define & allocation
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;
	cudaSetDevice(gpu_id);
	h_A = (float*)malloc(sz_A);
	h_B = (float*)malloc(sz_B);
	h_C = (float*)malloc(sz_C);
	cudaMalloc((void**)&d_A,      sz_A);
	cudaMalloc((void**)&d_B,      sz_B);
	cudaMalloc((void**)&d_C,      sz_C);

	//Initialization
	for(int i=0;i<M*K;i++) h_A[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	for(int i=0;i<K*N;i++) h_B[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

	//Memcpy "input" host(CPU) to device(GPU)
	cudaMemcpy(d_A, h_A, sz_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sz_B, cudaMemcpyHostToDevice);

	initTimer();

	//GPU thread
	int threads_x = BLOCK_SIZE, threads_y = BLOCK_SIZE;
	int grid_x = ((M) % threads_x)? (M)/threads_x+1: (M)/threads_x;
	int grid_y = ((N) % threads_y)? (N)/threads_y+1: (N)/threads_y;

	dim3 threads(threads_x, threads_y);
	dim3 grid(grid_x, grid_y);

	startTimer("run_device");
	for(int i=0;i<repeat;i++) {
		runMatMulGPU<<< grid, threads >>>(d_A, d_B, d_C, M, N, K);
	}
	cudaDeviceSynchronize();
	endTimer("run_device");

	//Memcpy "output" host(CPU) to device(GPU)
	cudaMemcpy(h_C, d_C, sz_C, cudaMemcpyDeviceToHost);

	//Gold result
	float* gold_C = (float*)malloc(sz_C);
	startTimer("run_host");
	for(int i=0;i<repeat;i++) {
//		runMatMulGold(h_A, h_B, gold_C, M, N, K);
	}
	endTimer("run_host");

	//check error
	float err = .0;
	for(int i=0;i<N;i++) {
		err += abs(h_C[i]-gold_C[i]);
	}
	printf("GPU compute err: %lf\n", err);

	//print running time
	double run_device_time = getTimer("run_device");
	double run_host_time = getTimer("run_host");
	printf("Average running time(Device)       : %.2lf (ms)\n", run_device_time/repeat);
	printf("Average running time(CPU)          : %.2lf (ms)\n", run_host_time/repeat);
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(gold_C);
	return 0;
}
