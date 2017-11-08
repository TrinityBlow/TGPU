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



__global__ void vecPromedio(double *d_vecA,unsigned long dist,unsigned long n,unsigned long tam_tot){    
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n){
        d_vecA[global_id*dist] = d_vecA[global_id*dist] * d_vecA[global_id*dist+dist / 2];
    }
}



int main(int argc, char *argv[]){


    if (argc != 3){
        printf("Falta argumento: N, CUDABLK\n");
        return 0;
    }
	//declaracion de variables
    cudaError_t error;
    unsigned int N = atoi (argv[1]);
    unsigned long CUDA_BLK = atoi (argv[2]);
    unsigned long numBytes = sizeof(double)*N,tam_tot;
    double *vecA,*d_vecA,*d_vecPromedio,*d_parcialA,resultgpu,timetick;
    unsigned int i;


    vecA = (double *)malloc(numBytes);
    for (i = 0; i < N; i++){
        vecA[i] = i;
    }

  tam_tot = N;
  cudaMalloc((void **) &d_vecA, numBytes);
  cudaMalloc((void **) &d_vecPromedio, numBytes);
  cudaMalloc((void **) &d_parcialA, numBytes);

    for (i = 0; i < N; i++){
        vecA[i] = 2;
    }
    cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
    //--------------------------------gpu comienza ------------------------------------

    dim3 dimBlock(CUDA_BLK);  
	timetick = dwalltime();
    cudaMemcpy(d_vecPromedio, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU


    //promedio
    for(i = 2; i <= N ;i *= 2){
        dim3 dimGrid((N / i + dimBlock.x - 1) / dimBlock.x);
        vecPromedio<<<dimGrid, dimBlock>>>(d_vecPromedio,i,N/i,tam_tot);
        cudaThreadSynchronize();
    }
    cudaMemcpy(&resultgpu, d_vecPromedio, sizeof(double), cudaMemcpyDeviceToHost); // GPU -> CPU

	printf("Tiempo para la GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n\n",error);

    //--------------------------------gpu termina ------------------------------------

/*
    printf("----------------------------------------\n\n");
    for(i = 0; i < N; i++){
        printf("%f|",vecA[i]);
    }
	printf("\n\n");
    printf("parcialA: %f||parcialB: %f\n",parcialA,parcialB);*/

    printf("resultadoGPU: %f\n",resultgpu);

    cudaFree(d_vecA);
    free(vecA);
    return 0;
}