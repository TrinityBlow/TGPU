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



__global__ void vecMult(double *d_vecA,unsigned long dist,unsigned long n,unsigned long tam_tot){    
    unsigned long int global_id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
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
    unsigned long CUDA_BLK = 8,gridBlock; //8*8 = 64
    unsigned long numBytes = sizeof(double)*N,tam_tot;
    double *vecA,result,*d_vecA,timetick;
    unsigned int i;


    vecA = (double *)malloc(numBytes);
    result = 0;
    for (i = 0; i < N; i++){
        vecA[i] = 2;
    }


    tam_tot = N;
    cudaMalloc((void **) &d_vecA, numBytes);

    // Bloque unidimencional de hilos (*cb* hilos)
    dim3 dimBlock(CUDA_BLK,CUDA_BLK); // Bloque bidimencional de hilos (*cb* hilos)
    //promedio
    timetick = dwalltime();
    cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
    for(i = 2; i <= N ;i *= 2){
        gridBlock = (unsigned int)sqrt(N*N/CUDA_BLK/CUDA_BLK / i);
        dim3 dimGrid(gridBlock,gridBlock); // Grid bidimencional (*ceil(n/cb)* bloques)
        vecMult<<<dimGrid, dimBlock>>>(d_vecA,i,N/i,tam_tot);
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