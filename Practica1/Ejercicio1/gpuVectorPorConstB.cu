#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <driver_types.h>


void checkparams(unsigned long *n, unsigned int *cb);
double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

#define _INT_

#ifdef _INT_
typedef int basetype;     // Tipo para elementos: int 
#define labelelem    "ints"
#elif _DOUBLE_
typedef double basetype;  // Tipo para elementos: double
#define labelelem    "doubles"
#else
typedef float basetype;   // Tipo para elementos: float PREDETERMINADO
#define labelelem    "floats"
#endif


__constant__ unsigned long d_n ;
__constant__ basetype d_C = 10;


const unsigned long N = 134217728 * 4 ;    // Número predeterminado de elementos en los vectores

const int CUDA_BLK = 64;  // Tamaño predeterminado de bloque de hilos CUDA


//  Definición de nuestro kernel para función cuadradoV
__global__ void constV_kernel_cuda(basetype *const cV){

  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id < d_n)
    cV[global_id] = cV[global_id]*d_C;

}



int main(int argc, char *argv[]){
 
  // Número de elementos del vector (predeterminado: N 1048576)
  unsigned long n = (argc > 1)?atoi (argv[1]):N;
  // Número de hilos en cada bloque CUDA (predeterminado: CUDA_BLK 64)
  unsigned int cb = (argc > 2)?atoi (argv[2]):CUDA_BLK;
  checkparams(&n, &cb);
  unsigned int numBytes = n * sizeof(basetype);
  unsigned int i;
  basetype *vectorV = (basetype *) malloc(numBytes);
  // Reservamos memoria global del device (GPU) para el array y lo copiamos
  double timetick;
  cudaError_t error;
	basetype *cV;

  for(i = 0; i < n; i++) {
    vectorV[i] = (basetype)i;
  }

  cudaMalloc((void **) &cV, numBytes);
  cudaMemcpy(cV, vectorV, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  cudaMemcpyToSymbol(d_n, &n, sizeof(unsigned long) ); // CPU -> GPU

  // Bloque unidimensional de hilos (*cb* hilos)
  dim3 dimBlock(cb);

  // Grid unidimensional (*ceil(n/cb)* bloques)
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

  timetick = dwalltime();
  constV_kernel_cuda<<<dimGrid, dimBlock>>>(cV);
  cudaThreadSynchronize();
  printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
  error = cudaGetLastError();

  // Movemos resultado: GPU -> CPU
  timetick = dwalltime();
  cudaMemcpy(vectorV, cV, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);

  for (i = 0; i < 20; i++){
    printf("%d|",vectorV[i]);
  }
 printf("\n");
 unsigned long temp = n-1;
  for (i = 0; i < 20; i++){
    printf("%d|",vectorV[temp]);
	temp--;
  }
  printf("\n%lu",n);
  printf("\n");
  printf("%d\n",error);

  // Liberamos memoria global del device utilizada
  cudaFree (cV);
  free(vectorV);
}


//  Función que ajusta el número de hilos, de bloques, y de bloques por hilo 
//  de acuerdo a las restricciones de la GPU
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
