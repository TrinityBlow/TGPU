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



__global__ void vecPromedio(double *d_vecA,unsigned long dist,unsigned long n,unsigned long tam_tot){    
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n){
        d_vecA[global_id*dist] = d_vecA[global_id*dist] + d_vecA[global_id*dist+dist / 2];
        if(dist == tam_tot) {
            d_vecA[global_id*dist] /= tam_tot;
        }
    }
}

__global__ void acomulativo(double *d_parcialA,double *d_parcialB,unsigned long dist,unsigned long n,unsigned long tam_tot){    
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n){
        d_parcialA[global_id*dist] = d_parcialA[global_id*dist] + d_parcialA[global_id*dist+dist / 2];
        d_parcialB[global_id*dist] = d_parcialB[global_id*dist] + d_parcialB[global_id*dist+dist / 2];
        if(dist == tam_tot) {
            d_parcialB[0] += 1;
            d_parcialB[0] = sqrt(d_parcialA[0] / d_parcialB[0]); 
        }
    }
}

__global__ void sumatoria(double *d_parcialA,double *d_parcialB,double *d_vecPromedio, unsigned long n){    
    unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n){
        d_parcialA[global_id] = (d_parcialB[global_id] - d_vecPromedio[0]) * (d_parcialB[global_id] - d_vecPromedio[0]);
        d_parcialB[global_id] = (d_parcialB[global_id] + d_vecPromedio[0]) * (d_parcialB[global_id] + d_vecPromedio[0]);
    }
}




int main(int argc, char *argv[]){


    if (argc != 3){
        printf("Falta argumento: N, CUDABLK\n");
        return 0;
    }
	//declaracion de variables
    cudaError_t error;
    unsigned int N = atoi (argv[1]);
    unsigned long CUDA_BLK = atoi (argv[2]);
    unsigned long numBytes = sizeof(double)*N,tam_tot;
    double *vecA,promedio,result,parcialA,parcialB,*d_vecA,*d_vecPromedio,*d_parcialA,resultgpu,timetick;
    unsigned int i;


    vecA = (double *)malloc(numBytes);
    promedio = 0;
    result = 0;
    parcialA = 0;
    parcialB = 0;
    for (i = 0; i < N; i++){
        vecA[i] = i;
    }

  cudaMalloc((void **) &d_vecA, numBytes);
  cudaMalloc((void **) &d_vecPromedio, numBytes);
  cudaMalloc((void **) &d_parcialA, numBytes);
  cudaMemcpy(d_vecA, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  cudaMemcpy(d_vecPromedio, vecA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  tam_tot = N;

  // Bloque unidimencional de hilos (*cb* hilos)
  dim3 dimBlock(CUDA_BLK);

    //--------------------------------cpu comienza ------------------------------------

    //secuencial
	timetick = dwalltime();
    for (i = 0; i < N; i++){
        promedio += vecA[i];
    }
    promedio /= N;

    for (i = 0; i < N; i++){
        parcialA += (vecA[i] - promedio) * (vecA[i] - promedio);
        parcialB += (vecA[i] + promedio) * (vecA[i] + promedio);
    }
    parcialB += 1;

    result = sqrt(parcialA / parcialB);
	printf("Tiempo para la ecuacion CPU: %f\n\n",dwalltime() - timetick);

    //--------------------------------cpu termina ------------------------------------

    for (i = 0; i < N; i++){
        vecA[i] = i;
    }
    //--------------------------------gpu comienza ------------------------------------

	timetick = dwalltime();

    //promedio
    for(i = 2; i <= N ;i *= 2){
        dim3 dimGrid((N / i + dimBlock.x - 1) / dimBlock.x);
        vecPromedio<<<dimGrid, dimBlock>>>(d_vecPromedio,i,N/i,tam_tot);
        cudaThreadSynchronize();
    }

    // Grid unidimencional (*ceil(n/cb)* bloques)
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    //sumatoria
    sumatoria<<<dimGrid, dimBlock>>>(d_parcialA,d_vecA,d_vecPromedio,N);
    cudaThreadSynchronize();

    //sumatoria acumulativo
    for(i = 2; i <= N ;i *= 2){
        dim3 dimGrid((N / i + dimBlock.x - 1) / dimBlock.x);
        acomulativo<<<dimGrid, dimBlock>>>(d_parcialA,d_vecA,i,N/i,tam_tot);
        cudaThreadSynchronize();
    }

	printf("Tiempo para la ecuacion in-place GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n\n",error);
    cudaMemcpy(&resultgpu, d_vecA, sizeof(double), cudaMemcpyDeviceToHost); // GPU -> CPU

    //--------------------------------gpu termina ------------------------------------

    cudaMemcpy(vecA, d_vecA, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
/*
    printf("----------------------------------------\n\n");
    for(i = 0; i < N; i++){
        printf("%f|",vecA[i]);
    }
	printf("\n\n");
    printf("promedio: %f\n",promedio);
    printf("parcialA: %f||parcialB: %f\n",parcialA,parcialB);*/
    printf("resultadoCPU: %f\n",result);
    printf("resultadoGPU: %f\n",resultgpu);

    cudaFree(d_vecA);
    free(vecA);
    return 0;
}