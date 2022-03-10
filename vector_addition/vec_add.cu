#include <stdio.h>
#include <cuda_runtime.h>
#include "../Timer/Timer.h"
#define _GPU_ID 0
#define _REPEAT 10

void runVecAddGold(float* A, float* B, float* C, int N)
{
	for(int i=0;i<N;i++)
		C[i]=A[i]+B[i];
}


__global__ void runVecAddGPU(float *a,float *b, float *c, int n)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if( gid < n) {
		c[gid] = a[gid] + b[gid];
	}
}


int main(int argc, char* argv[]) {
	int gpu_id = _GPU_ID;
	int repeat = _REPEAT;
	if (argc < 2) {
		printf("arguments: [VEC SIZE]\n\n");
	}
	
	//Get size
	long N = atoi(argv[1]);
	long sz_N = sizeof(float)*N;
	printf("VEC SIZE = %ld (%ld Bytes)\n", N, sz_N);
	printf("\n");

	//Define & allocation
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;
	cudaSetDevice(gpu_id);
	h_A = (float*)malloc(sz_N);
	h_B = (float*)malloc(sz_N);
	h_C = (float*)malloc(sz_N);
	cudaMalloc((void**)&d_A,      sz_N);
	cudaMalloc((void**)&d_B,      sz_N);
	cudaMalloc((void**)&d_C,      sz_N);

	//Initialization
	for(int i=0;i<N;i++) {
		h_A[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		h_B[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	//Memcpy "input" host(CPU) to device(GPU)
	cudaMemcpy(d_A, h_A, sz_N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sz_N, cudaMemcpyHostToDevice);

	initTimer();

	//GPU thread
	int threads = 1024;
	int grid = (N % threads)? N/threads+1: N/threads;
	startTimer("run_device");
	for(int i=0;i<repeat;i++) {
		runVecAddGPU<<< grid, threads >>>(d_A, d_B, d_C, N);
	}
	cudaDeviceSynchronize();
	endTimer("run_device");

	//Memcpy "output" host(CPU) to device(GPU)
	cudaMemcpy(h_C, d_C, sz_N, cudaMemcpyDeviceToHost);

	//Gold result
	float* gold_C = (float*)malloc(sz_N);
	startTimer("run_host");
	for(int i=0;i<repeat;i++) {
		runVecAddGold(h_A, h_B, gold_C, N);
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
