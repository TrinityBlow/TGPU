#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>

double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}



__global__ void vecMult(double *d_vecA,unsigned long n, unsigned long blockSize){   
    __shared__ double sdata[sizeof(double)*128];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;   
    __syncthreads();
    while (i < n) { sdata[tid] += d_vecA[i] + d_vecA[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) d_vecA[blockIdx.x] = sdata[0];

}





int main(int argc, char *argv[]){


    if (argc != 2){
        printf("Falta argumento: N\n");
        return 0;
    }
	//declaracion de variables
    cudaError_t error;
    unsigned long N = atoi (argv[1]);
    unsigned long CUDA_BLK = 128,GRID_BLK;
    unsigned long numBytes = sizeof(double)*N;
    double *vecA,result,*d_vecA,timetick;
    unsigned long i;


    vecA = (double *)malloc(numBytes);
    result = 1;
    for (i = 0; i < N; i++){
        vecA[i] = 2;
    }
    //comment

    cudaMalloc((void **) &d_vecA, numBytes);

    // Bloque unidimencional de hilos (*cb* hilos)
    dim3 dimBlock(CUDA_BLK);
    //promedio
    timetick = dwalltime();
    cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
    for(i = N ; i > 1; i /= CUDA_BLK){
        printf("%lu %lu\n\n",i,CUDA_BLK);
        GRID_BLK = i / CUDA_BLK ; 
        dim3 dimGrid(GRID_BLK);
        vecMult<<<dimGrid, dimBlock>>>(d_vecA,i,CUDA_BLK);
        cudaThreadSynchronize();
    }
    cudaMemcpy(vecA, d_vecA, sizeof(double)*GRID_BLK, cudaMemcpyDeviceToHost); // GPU -> CPU

  
    for (i = 0; i < GRID_BLK; i++){
        result *= vecA[i];
    }

	printf("Tiempo para la GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n\n",error);
    
/*
    for (i = 0; i < GRID_BLK; i++){
        printf("%f|",vecA[i]);
    }
    printf("\n\n");*/
    printf("%f|",result);
    printf("\n\n");


    cudaFree(d_vecA);
    free(vecA);
    return 0;

}