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



__global__ void vecMult(double *d_vecA,unsigned long n){      
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double s_vecA[sizeof(double)*32];
    unsigned int i;
 //   int y = 2;


    if (global_id < n){
        
        s_vecA[threadIdx.x]=d_vecA[global_id];
        __syncthreads();



        for( i = 1; i <= 7;  i++) {
            if( threadIdx.x < (int)(128 >>(i))){
                s_vecA[threadIdx.x * (1 << i)] += s_vecA[(threadIdx.x * (1 << i)) + (1 << (i-1))];
            }
 //           y = y * 2;
            __syncthreads();
        }

        if ( threadIdx.x == 0){
            d_vecA[blockIdx.x] = s_vecA[0];
        } 
    }
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
    unsigned long i,j;


    vecA = (double *)malloc(numBytes);
    result = 1;
    for (i = 0; i < N; i++){
        vecA[i] = 2;
    }
    //comment

    cudaMalloc((void **) &d_vecA, numBytes);
    cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

    dim3 dimBlock(CUDA_BLK);
    
    unsigned long int iteraciones = log(N) / log(2);
    timetick = dwalltime();
    cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
    for(i = N ; i > 1; i = ceil(float(i) / CUDA_BLK)){
        GRID_BLK = ceil(float(i) / CUDA_BLK) ; 
        dim3 dimGrid(GRID_BLK);
        vecMult<<<dimGrid, dimBlock>>>(d_vecA,i);
        cudaThreadSynchronize();
    }
    cudaMemcpy(vecA, d_vecA, sizeof(double), cudaMemcpyDeviceToHost); // GPU -> CPU

	printf("Tiempo para la GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n",error);
    
    printf("%f|",vecA[0]);
    printf("\n\n");


    cudaFree(d_vecA);
    free(vecA);
    return 0;

}