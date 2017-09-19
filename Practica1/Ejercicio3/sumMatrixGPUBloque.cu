#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

//134217728

double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

__global__ void sumM_kernel_cuda(double *d_matA,double *d_matB, unsigned long n){    
    int k;
    int distA = blockIdx.y * blockDim.y + threadIdx.y; //i
    int distB = blockIdx.x * blockDim.x + threadIdx.x; //j
    if (distA*n+distB < n*n){
        d_matA[distA*n+distB] += d_matB[distB*n+k];
    }
}


void checkparams(unsigned long *n, unsigned int *cb);

int main(int argc, char *argv[]){

    if (argc != 2){
        printf("Falta argumento: N\n");
        return 0;
    }
    unsigned long N = atoi (argv[1]),tam_tot = N*N;
    unsigned int CUDA_BLK = 8, gridBlock;
    unsigned long numBytes = sizeof(double)*tam_tot;
    double *matA,*matB,*d_matA,*d_matB,timetick;
    unsigned int i,j;
    matA = (double *)malloc(numBytes);
    matB = (double *)malloc(numBytes);

    for (i = 0; i < tam_tot; i++){
        matA[i] = i;
        matB[i] = i;
    }


  cudaMalloc((void **) &d_matA, numBytes);
  cudaMalloc((void **) &d_matB, numBytes);
  cudaMemcpy(d_matA, matA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  cudaMemcpy(d_matB, matB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  // Bloque unidimensional de hilos (*cb* hilos)
  dim3 dimBlock(CUDA_BLK);
  // Grid unidimensional (*ceil(n/cb)* bloques)
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);
    cudaError_t error;

	timetick = dwalltime();
    sumM_kernel_cuda<<<dimGrid, dimBlock>>>(d_matA, d_matB, N);
    cudaThreadSynchronize();
	printf("Tiempo para sumar las matrices: %f\n",dwalltime() - timetick);


error = cudaGetLastError();
printf("error: %d\n",error);
  cudaMemcpy(matA, d_matA, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
    /*
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            printf("%f|",matA[i*N+j]);
        }
        printf("\n");
    }
	printf("\n");*/
    
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