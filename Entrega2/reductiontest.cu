#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>

#define BASETYPE float

double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

__global__ void matDet(BASETYPE *d_matA, BASETYPE *detM, int desp){ 
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ BASETYPE datos[];
    BASETYPE *s_mat = &datos[0];
    BASETYPE *s_detAux = &datos[desp];
    int offset = (threadIdx.x)*16; 
    unsigned int i;

    for(i = 0; i < 16; i++){
        s_mat[(threadIdx.x) * 16 + i]=d_matA[global_id * 16 + i];

    }  
    __syncthreads();  

    for(i = 0; i < 4; i++){
        s_detAux[(threadIdx.x) * 4+i]=0;
    }
    __syncthreads();

    //  printf("globalId:%d|%d|%d|%d|%d\n",global_id,(threadIdx.x)*4,(threadIdx.x)*4+1,(threadIdx.x)*4+2,(threadIdx.x)*4+3);
    s_detAux[(threadIdx.x)*4] +=  s_mat[offset] * ( (s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+13])+(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+13]))   + (-1*(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+15])) );

    s_detAux[(threadIdx.x)*4+1] +=  (-1*s_mat[offset+1]) * ( (s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+15])) );    

    s_detAux[(threadIdx.x)*4+2] +=  s_mat[offset+2] * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+15])+(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+15])) );        

    s_detAux[(threadIdx.x)*4+3] +=  (-1*s_mat[offset+3]) * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+14])+(s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+12])+(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+14])) );        
    detM[blockIdx.x * blockDim.x + (threadIdx.x)] = s_detAux[(threadIdx.x)*4] + s_detAux[(threadIdx.x)*4+1] + s_detAux[(threadIdx.x)*4+2] + s_detAux[(threadIdx.x)*4+3]; 
    __syncthreads();
        
}

__global__ void vecMult(BASETYPE *d_matA,unsigned long n, int iteraciones){     
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ BASETYPE s_mat[];
    unsigned int i,j;


    for(i = 0; i < 16; i++){
        s_mat[threadIdx.x * 16 + i]=d_matA[global_id * 16 + i];
    }
    __syncthreads();

    for( j = 1; j < blockDim.x; j *= 2 ){
        if( threadIdx.x < blockDim.x / (j * 2)){
            for( i = 0; i < 16;  i++) {
                s_mat[(threadIdx.x) * 16 + i] += s_mat[((threadIdx.x) * 16 + i) + (blockDim.x / (j * 2)) * 16]; // 2 * 16 = 32
            }
        }
        __syncthreads();
    }


    if ((threadIdx.x) == 0){
        for (i = 0; i < 16; i++){
            d_matA[(blockIdx.x * 16) + i] = s_mat[i];
        }
    }
}

int main(int argc, char *argv[]){


    if (argc != 3){
        printf("Falta argumento: N\n");
        printf("Falta argumento: CUDA_BLK \n");
        return 0;
    }
	//declaracion de variables
    cudaError_t error;
    unsigned long N = atoi (argv[1]);
    unsigned long CUDA_BLK = atoi(argv[2]),GRID_BLK,cant_blk;
    unsigned long numBytes = sizeof(BASETYPE)*4*4;
    BASETYPE *matrices,*d_matrices,*d_detM,*detM;
	double timetick;
    unsigned long i,j;
    int iteraciones,datos_matDet,datos_vecMult,matDet_desp;


    matrices = (BASETYPE *)malloc(numBytes*N);
    detM = (BASETYPE *)malloc(sizeof(BASETYPE)*N);
    for (i = 0; i < 4*4*N; i++){
        matrices[i] = 1;
    }

    for (i = 0; i < N; i++){
        detM[i] = 0;
    }
    matrices[2] = 220;
    matrices[13] = 220;
    matrices[7] = 6;
    matrices[14] = 6;
    //comment

    cudaMalloc((void **) &d_matrices, numBytes*N);
    cudaMalloc((void **) &d_detM, sizeof(BASETYPE)*N);

    datos_matDet = numBytes * CUDA_BLK + sizeof(BASETYPE) * 4 * CUDA_BLK;

    datos_vecMult = numBytes * CUDA_BLK;
    matDet_desp = CUDA_BLK * 16;

	cant_blk = N / CUDA_BLK;

    dim3 dimBlock(CUDA_BLK);
    dim3 dimGrid(cant_blk);
    
    timetick = dwalltime();

    iteraciones = log(CUDA_BLK) / log(2);

    cudaMemcpy(d_matrices, matrices, numBytes*N, cudaMemcpyHostToDevice); // CPU -> GPU
    cudaMemcpy(d_detM, detM, sizeof(BASETYPE)*N, cudaMemcpyHostToDevice); // CPU -> GPU
    matDet<<<dimGrid, dimBlock,datos_matDet>>>(d_matrices,d_detM,matDet_desp);
    cudaThreadSynchronize();
    for(i = CUDA_BLK ; i <= N; i *= CUDA_BLK){
        GRID_BLK = N / (i / (CUDA_BLK)) / (CUDA_BLK); 
        dim3 dimGrid(GRID_BLK);
        vecMult<<<dimGrid, dimBlock,datos_vecMult>>>(d_matrices,(4*4*N) / (i / CUDA_BLK),iteraciones);
        cudaThreadSynchronize();
    }
    cudaMemcpy(matrices, d_matrices, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
    cudaMemcpy(detM, d_detM, sizeof(BASETYPE)*N, cudaMemcpyDeviceToHost); // GPU -> CPU

    for(i = 1; i < N ; i++){
        detM[0] += detM[i]; 
    }
    detM[0] = detM[0] / N;

    for (i = 0; i < 4*4; i++){
        matrices[i] *= detM[0];
    }

	printf("Tiempo para la GPU: %f\n",dwalltime() - timetick);
    error = cudaGetLastError();
    printf("error: %d\n",error);
    
    printf("%.2lf|\n",detM[0]);

    for(i=0; i < 4; i++){
        for(j=0; j < 4; j++){
            printf("%.2lf|",matrices[i*4+j]);
        }
        printf("\n");
    }


    cudaFree(d_matrices);
    cudaFree(d_detM);
    free(matrices);
    free(detM);
    return 0;

}
