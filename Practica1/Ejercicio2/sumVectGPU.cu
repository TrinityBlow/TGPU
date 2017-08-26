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


int main(int argc, char *argv[]){

    if (argc == 2){
        printf("Falta argumento: N, CUDA_BLK\n");
        return 0;
    }
    unsigned long N = atoi (argv[1]);
    unsigned int CUDA_BLK = atoi(argv[2]);
    checkparams(&N,&CUDA_BLK);
    double *vecA,*vecB,timetick;
    unsigned int i;
    cudaError_t error;
    
    vecA = (double *)malloc(sizeof(double)*N);
    vecB = (double *)malloc(sizeof(double)*N);

    for (i = 0; i < N; i++){
        vecA[i] = i;
        vecB[i] = i;
    }

	timetick = dwalltime();



	printf("Tiempo para sumar los vectoresGPU: %f\n",dwalltime() - timetick);

    for(i= 0; i < 20; i++){
        printf("%f|",vecA[i]);
    }
	printf("\n");

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

