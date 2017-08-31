#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

//134217728

//  Definición de nuestro kernel para función cuadradoV
__global__ void sumV_kernel_cuda(double *arrayA,double *arrayB ,   int n){

  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id < n)
    arrayA[global_id] = arrayA[global_id] + arrayB[global_id];

}

void checkparams(unsigned long *n, unsigned int *cb);
double dwalltime();


__global__ void sumV_kernel_cuda(double *d_vecA,double *d_vecB, long n, unsigned long dist){
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id + dist < n)
        d_vecA[global_id + dist] = d_vecA[global_id + dist] + d_vecB[global_id + dist];
}


int main(int argc, char *argv[]){
/*
    if (argc != 3){
        printf("Falta argumento: N, CUDA_BLK\n");
        return 0;
    }
    unsigned long N = atoi (argv[1]);
    unsigned int CUDA_BLK = atoi(argv[2]);*/
    unsigned long N = 107107840;
    unsigned int CUDA_BLK = 32;
    unsigned long max_N = N;
    checkparams(&max_N,&CUDA_BLK);
    double *vecA,*vecB,*d_vecA,*d_vecB,timetick;
    unsigned int i;
    cudaError_t error;
    unsigned long numBytes =sizeof(double)*N ;
    struct cudaDeviceProp capabilities;
    cudaGetDeviceProperties (&capabilities, 0);

    vecA = (double *)malloc(numBytes);
    vecB = (double *)malloc(numBytes);

    for (i = 0; i < N; i++){
        vecA[i] = i;
        vecB[i] = i;
    }

  cudaMalloc((void **) &d_vecA, numBytes);
  cudaMalloc((void **) &d_vecB, numBytes);
  cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  cudaMemcpy(d_vecB, vecB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  // Bloque unidimensional de hilos (*cb* hilos)
  dim3 dimBlock(32);
  // Grid unidimensional (*ceil(n/cb)* bloques)
  dim3 dimGrid((max_N + dimBlock.x - 1) / dimBlock.x);

     long aux_N = N;
	timetick = dwalltime();
    int rep = 0;
    while(aux_N > 0){
        printf("%lu\n",aux_N);
        sumV_kernel_cuda<<<dimGrid, dimBlock>>>(d_vecA, d_vecB, N, max_N*rep);
        aux_N = aux_N - max_N;
        rep++;
    }
    cudaThreadSynchronize();

  printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
  error = cudaGetLastError();

  // Movemos resultado: GPU -> CPU
  timetick = dwalltime();

  cudaMemcpy(vecA, d_vecA, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);

    for(i= 0; i < 20; i++){
        printf("%f|",vecA[i]);
    }
	printf("\n");
    printf("error code: %d\n",error);
    printf("\n%lu||||%lu\n",(max_N + dimBlock.x - 1) / dimBlock.x,CUDA_BLK);

    cudaFree (vecA);
    cudaFree (vecB);
    free(vecA);
    free(vecB);
    return 0;
}

double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
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

