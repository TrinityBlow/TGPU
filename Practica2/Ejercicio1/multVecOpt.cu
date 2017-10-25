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



__global__ void vecMult(double *d_vecA,unsigned long dist,unsigned long n){    
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n){
        d_vecA[global_id*dist] = d_vecA[global_id*dist] * d_vecA[global_id*dist+dist / 2];
    }
}





int main(int argc, char *argv[]){


    if (argc != 2){
        printf("Falta argumento: N\n");
        return 0;
    }
	//declaracion de variables
    cudaError_t error;
    unsigned int N = atoi (argv[1]);
    unsigned long CUDA_BLK = 128;
    unsigned long numBytes = sizeof(double)*N;
    double *vecA,result,*d_vecA,timetick;
    unsigned int i;


    vecA = (double *)malloc(numBytes);
    result = 0;
    for (i = 0; i < N; i++){
        vecA[i] = 2;
    }

    cudaMalloc((void **) &d_vecA, numBytes);

    // Bloque unidimencional de hilos (*cb* hilos)
    dim3 dimBlock(CUDA_BLK);
    //promedio
    timetick = dwalltime();
    cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
    for(i = 2; i <= N ;i *= 2){
        dim3 dimGrid((N / i + dimBlock.x - 1) / dimBlock.x);
        vecMult<<<dimGrid, dimBlock>>>(d_vecA,i,N/i);
        cudaThreadSynchronize();
    }
    cudaMemcpy(&result, d_vecA, sizeof(double), cudaMemcpyDeviceToHost); // GPU -> CPU

	printf("Tiempo para la GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n\n",error);
    printf("resultadoGPU: %f\n",result);

    cudaFree(d_vecA);
    free(vecA);
    return 0;

}