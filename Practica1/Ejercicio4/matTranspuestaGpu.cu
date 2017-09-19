

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>

//134217728

double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

__global__ void transpuesta_out_place (double *mt, double *m,unsigned int N)
{
    int i = blockIdx.y * blockDim.y +  threadIdx.y;
    int j = blockIdx.x * blockDim.x +  threadIdx.x;
    if( (i<N*N) && (j<N*N) ){
        mt [j*N + i] = m[i*N + j];
    }
    
}


void checkparams(unsigned long *n, unsigned int *cb);

int main(int argc, char *argv[]){

        if (argc != 2){
            printf("Falta argumento: N\n");
            return 0;
        }
    cudaError_t error;

    unsigned int N = atoi (argv[1]),tam_tot = N*N;
    unsigned int CUDA_BLK = 2, gridBlock;
    unsigned long numBytes = sizeof(double)*tam_tot;
    double *matA,*matB,*d_matA,*d_matB,timetick;
    unsigned int i,j;


    matA = (double *)malloc(numBytes);
    matB = (double *)malloc(numBytes);

    for (i = 0; i < tam_tot; i++){
        matA[i] = i;
        matB[i] = 0;
    }

  cudaMalloc((void **) &d_matA, numBytes);
  cudaMalloc((void **) &d_matB, numBytes);
  cudaMemcpy(d_matA, matA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  cudaMemcpy(d_matB, matB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  gridBlock = (unsigned int)sqrt(N*N/CUDA_BLK/CUDA_BLK);


  printf("%u||%u||%u||\n",CUDA_BLK,gridBlock,N);
  printf("dimBlockSize:%u\ndimGridSize:%u\ntotalMatriz:%u\n",CUDA_BLK*CUDA_BLK,gridBlock*gridBlock,N*N);
  // Bloque unidimensional de hilos (*cb* hilos)
  dim3 dimBlock(CUDA_BLK,CUDA_BLK);
  // Grid unidimensional (*ceil(n/cb)* bloques)
  dim3 dimGrid(gridBlock,gridBlock);


    

	timetick = dwalltime();
    transpuesta_out_place<<<dimGrid, dimBlock>>>(d_matB, d_matA, N);
    cudaThreadSynchronize();
	printf("Tiempo para sumar las matrices: %f\n",dwalltime() - timetick);

  cudaMemcpy(matB, d_matB, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  /*
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            printf("%f|",matC[i*N+j]);
        }
        printf("\n");
    }
	printf("\n");
    */

printf("%u|||||||\n",CUDA_BLK*(tam_tot + dimBlock.x - 1) / dimBlock.x);
    error = cudaGetLastError();
    printf("error: %d\n",error);
    printf("%.2lf\n",matB[1]);
    printf("%.2lf\n",matB[N*N-2]);

    cudaFree(d_matA);
    cudaFree(d_matB);
    free(matA);
    free(matB);
    return 0;
}

void checkparams(unsigned long *n, unsigned int *cb){
  struct cudaDeviceProp capabilities;

  // Si menos numero total de hilos que tamaño bloque, reducimos bloque
  if (*cb > *n)
    *cb = *n;

  cudaGetDeviceProperties (&capabilities, 0);

  if (*cb > capabilities.maxThreadsDim[0]) {
    *cb = capabilities.maxThreadsDim[0];
    printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n\n", 
	   *cb);
  }

  if (((*n + *cb - 1) / *cb) > capabilities.maxGridSize[0]) {
    *cb = 2 * (*n - 1) / (capabilities.maxGridSize[0] - 1);
    if (*cb > capabilities.maxThreadsDim[0]) {
      *cb = capabilities.maxThreadsDim[0];
      printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n", 
	     *cb);
      if (*n > (capabilities.maxGridSize[0] * *cb)) {
	*n = capabilities.maxGridSize[0] * *cb;
	printf("->Núm. total de hilos cambiado a %lu (máx por grid para \
dev)\n\n", *n);
      } else {
	printf("\n");
      }
    } else {
      printf("->Núm. hilos/bloq cambiado a %d (%d máx. bloq/grid para \
dev)\n\n", 
	     *cb, capabilities.maxGridSize[0]);
    }
  }
}
