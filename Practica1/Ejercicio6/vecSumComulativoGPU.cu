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


__global__ void vecSum_kernel_cuda(double *d_vecA,double *d_result,unsigned long dist,unsigned long n){    
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n){
        d_vecA[global_id*dist] = d_vecA[global_id*dist] + d_vecA[global_id*dist+dist / 2];
    }
}

void checkparams(unsigned long *n, unsigned int *cb);
void checkparamsB(unsigned long *n, unsigned int *cb);


int main(int argc, char *argv[]){

    if (argc != 2){
        printf("Falta argumento: N\n");
        return 0;
    }


    unsigned long N = atoi (argv[1]);
    double *vecA,*result,timetick;
    unsigned int i;

    cudaError_t error;
    unsigned int CUDA_BLK = 32;
   // checkparamsB(&N,&CUDA_BLK);
    unsigned long numBytes = sizeof(double)*N;
    double *d_vecA,*d_result;


    vecA = (double *)malloc(numBytes); 
    result = (double *)malloc(sizeof(double)); 
    *result = 0;   
    for (i = 0; i < N; i++){
        vecA[i] = i;
    }
  
    cudaMalloc((void **) &d_vecA, numBytes);  
    cudaMalloc((void **) &d_result, sizeof(double));  
    cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice);


    // Bloque unidimensional de hilos (*cb* hilos)
    dim3 dimBlock(CUDA_BLK);
    // Grid unidimensional (*ceil(n/cb)* bloques)



    timetick = dwalltime();
    for(i = 2; i <= N ;i *= 2){
        dim3 dimGrid((N / i + dimBlock.x - 1) / dimBlock.x);
        vecSum_kernel_cuda<<<dimGrid, dimBlock>>>(d_vecA,d_result,i,N/i);
        cudaThreadSynchronize();
    }
    printf("Tiempo para sumar las matrices: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n",error);

    cudaMemcpy(result, d_vecA, sizeof(double), cudaMemcpyDeviceToHost); // GPU -> CPU



    printf("%f\n",*result);

    cudaFree(d_vecA);
    cudaFree(d_result);
    free(vecA);
    free(result);

    return 0;
}


void checkparamsB(unsigned long *n, unsigned int *cb){

    struct cudaDeviceProp capabilities;
    cudaGetDeviceProperties (&capabilities, 0);
    *cb = capabilities.maxThreadsDim[0];
    printf("%d\n",*cb);
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