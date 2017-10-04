#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>

__global__ void kernel_transpuesta(double *m, int N){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int i = int((1 + sqrtf(1 + 8*tid)) / 2);
	int j = tid - (i*(i-1)/2); int aux;
	if ( (i<N) && (j<N) ){
		aux = m[i*N + j] ;
		m[i*N + j] = m[j*N + i];
		m[j*N + i] = aux;
	}
}


int main(int argc, char *argv[]){

    cudaError_t error;
    unsigned int N = 8;
    unsigned long CUDA_BLK = 2, gridBlock;
    unsigned long numBytes = sizeof(double)*N*N;
    double *matA,*d_matA;
    unsigned int i,j;

	//inicializa variables para cpu
    matA = (double *)malloc(numBytes);
    for (i = 0; i < N*N; i++){
        matA[i] = i;
    }

  //inicializa variables para gpu
  cudaMalloc((void **) &d_matA, numBytes);
  cudaMemcpy(d_matA, matA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  gridBlock = (unsigned int)sqrt(N*N/CUDA_BLK/CUDA_BLK);
  dim3 dimBlock(CUDA_BLK,CUDA_BLK); // Bloque bidimencional de hilos (*cb* hilos)
  dim3 dimGrid(gridBlock,gridBlock); // Grid bidimencional (*ceil(n/cb)* bloques)

    kernel_transpuesta<<<dimGrid, dimBlock>>>(d_matA, N);
    cudaThreadSynchronize();
    kernel_transpuesta<<<dimGrid, dimBlock>>>(d_matA, N);
    cudaThreadSynchronize();
    error = cudaGetLastError();
    printf("error: %d\n\n",error);

    //--------------------------------gpu in-place termina ------------------------------------

    cudaMemcpy(matA, d_matA, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  
  //imprime la matriz matA
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            printf("%f|",matA[i*N+j]);
        }
        printf("\n");
    }
	printf("\n");


    cudaFree(d_matA);
    free(matA);
    return 0;
}