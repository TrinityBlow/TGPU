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

__global__ void ecuacion_kernel_outplace_p1(double *d_matA,double *d_matAT,double *d_matB,double *d_matBT, unsigned int n){   
    int distA = blockIdx.y * blockDim.y + threadIdx.y; //i
    int distB = blockIdx.x * blockDim.x + threadIdx.x; //j
    
    //transpuesta out-place A y B
    if( (distA<n*n) && (distB<n*n) ){
        d_matAT [distB*n + distA] = d_matA[distA*n + distB];
        d_matBT [distB*n + distA] = d_matB[distA*n + distB];
    }

    
}

__global__ void ecuacion_kernel_outplace_p2(double *d_matA,double *d_matB,double *d_matC,double *d_matAT,double *d_matBT, unsigned int n){    
    int distA = blockIdx.y * blockDim.y + threadIdx.y; //i
    int distB = blockIdx.x * blockDim.x + threadIdx.x; //j
    int k;
    if (distA*n+distB < (n*n - 1)){
        //multiplicacion 
        for(k = 0; k < n ;k++){
            d_matC[distA*n+distB] += d_matA[distA*n+k] * d_matBT[distB+k*n];
        }
        //suma
        d_matC[distA*n+distB] += d_matB[distA*n+distB] + d_matAT[distA*n+distB];
    }
}
__global__ void ecuacion_kernel_inplace_suma (double *d_matA,double *d_matB,double *d_matC, unsigned int n){
    int distA = blockIdx.y * blockDim.y + threadIdx.y; //i
    int distB = blockIdx.x * blockDim.x + threadIdx.x; //j
    int k;
    //multiplicacion 
    if (distA*n+distB < (n*n - 1)){
        d_matC[distA*n+distB] += d_matB[distA*n+distB] + d_matA[distA+distB*n]; 
        for(k = 0; k < n ;k++){
            d_matC[distA*n+distB] += d_matA[distA*n+k] * d_matB[distB*n+k]; 
        }
    }

}

__global__ void transpuesta_in_place(double *m, int N){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int i = int((1 + sqrtf(1 + 8*tid)) / 2);
	int j = tid - (i*(i-1)/2); int aux;
	if ( (i<N) && (j<N) ){
		aux = m[i*N + j] ;
		m[i*N + j] = m[j*N + i];
		m[j*N + i] = aux;
	}
}


__global__ void ecuacion_kernel_inplace_multiplicacion_suma (double *d_matA,double *d_matB,double *d_matC, unsigned int n){
    int distA = blockIdx.y * blockDim.y + threadIdx.y; //i
    int distB = blockIdx.x * blockDim.x + threadIdx.x; //j
    int k;
    //multiplicacion 
    if (distA*n+distB < (n*n - 1)){
        d_matC[distA*n+distB] += d_matB[distA*n+distB] + d_matA[distA+distB*n]; //d_matA busca por columna para simular la suma con la transpuesta
        for(k = 0; k < n ;k++){
            d_matC[distA*n+distB] += d_matA[distA*n+k] * d_matB[distB*n+k]; //multiplica d_matB por fila para simular la transpuesta
        }
	//suma
    }

}


void checkparams(unsigned long *n, unsigned int *cb);

int main(int argc, char *argv[]){

        if (argc != 3){
            printf("Falta argumento: N, CUDABLK\n");
            return 0;
        }
	//declaracion de variables
    cudaError_t error;
    unsigned int N = atoi (argv[1]);
    unsigned long CUDA_BLK = atoi (argv[2]), gridBlock;
    unsigned long numBytes = sizeof(double)*N*N;
    double *matA,*matB,*matC,*d_matA,*d_matB,*d_matC,*d_matAT,*d_matBT,timetick;
    unsigned int i,j,k;

	//inicializa variables para cpu
    matA = (double *)malloc(numBytes);
    matB = (double *)malloc(numBytes);
    matC = (double *)malloc(numBytes);
    for (i = 0; i < N*N; i++){
        matA[i] = 2;
        matB[i] = 3;
        matC[i] = 0;
    }

  //inicializa variables para gpu
  cudaMalloc((void **) &d_matA, numBytes);
  cudaMalloc((void **) &d_matAT, numBytes);
  cudaMalloc((void **) &d_matB, numBytes);
  cudaMalloc((void **) &d_matBT, numBytes);
  cudaMalloc((void **) &d_matC, numBytes);
  cudaMemcpy(d_matA, matA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  cudaMemcpy(d_matB, matB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  cudaMemcpy(d_matC, matC, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  gridBlock = (unsigned int)sqrt(N*N/CUDA_BLK/CUDA_BLK);
  dim3 dimBlock(CUDA_BLK,CUDA_BLK); // Bloque bidimencional de hilos (*cb* hilos)
  dim3 dimGrid(gridBlock,gridBlock); // Grid bidimencional (*ceil(n/cb)* bloques)

    //--------------------------------cpu comienza ------------------------------------

    //secuencial
	timetick = dwalltime();

    //multiplicacion
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            for(k = 0; k < N ;k++){
                matC[i*N+j] += matA[i*N+k] * matB[j*N+k]; //multiplica a matB por fila, eso simula la matB transpuesta
            }
        }
    }
    //suma
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            matC[i*N+j] += matB[i*N+j] + matA[i+j*N];
        }
    }

	printf("Tiempo para la ecuacion CPU: %f\n\n",dwalltime() - timetick);

    //--------------------------------cpu termina ------------------------------------



    for (i = 0; i < N*N; i++){
        matA[i] = i;
        matB[i] = i;
        matC[i] = 0;
    }

    //--------------------------------gpu out-place comienza ------------------------------------

	timetick = dwalltime();
    ecuacion_kernel_outplace_p1<<<dimGrid, dimBlock>>>(d_matA, d_matAT,d_matB,d_matBT, N);
    cudaThreadSynchronize();
    ecuacion_kernel_outplace_p2<<<dimGrid, dimBlock>>>(d_matA, d_matB,d_matC,d_matAT,d_matBT, N);
    cudaThreadSynchronize();
	printf("Tiempo para la ecuacion out-place GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n\n",error);
    
    //--------------------------------gpu out-place termina ------------------------------------

    cudaMemcpy(d_matA, matA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
    cudaMemcpy(d_matB, matB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
    cudaMemcpy(d_matC, matC, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU


    //--------------------------------gpu in-place comienza ------------------------------------

	timetick = dwalltime();
    ecuacion_kernel_inplace<<<dimGrid, dimBlock>>>(d_matA, d_matB,d_matC, N);
    cudaThreadSynchronize();
	printf("Tiempo para la ecuacion in-place GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n\n",error);

    //--------------------------------gpu in-place termina ------------------------------------

    cudaMemcpy(matC, d_matC, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  
/*  //imprime la matriz matC
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            printf("%f|",matC[i*N+j]);
        }
        printf("\n");
    }
	printf("\n");
*/

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
    cudaFree(d_matAT);
    cudaFree(d_matBT);
    free(matA);
    free(matB);
    free(matC);
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