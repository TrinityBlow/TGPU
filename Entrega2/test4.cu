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
//	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int distA_id = blockIdx.x * blockDim.x * 16;
    extern __shared__ BASETYPE datos[];
    BASETYPE *s_mat = &datos[0];
    BASETYPE *s_detAux = &datos[desp];
    int offset = (threadIdx.x)*16; 

	s_mat[threadIdx.x]=d_matA[distA_id + threadIdx.x];
	s_mat[threadIdx.x + blockDim.x]=d_matA[distA_id + threadIdx.x + blockDim.x];
	s_mat[threadIdx.x + blockDim.x * 2]=d_matA[distA_id + threadIdx.x + blockDim.x * 2];
	s_mat[threadIdx.x + blockDim.x * 3]=d_matA[distA_id + threadIdx.x + blockDim.x * 3];
	s_mat[threadIdx.x + blockDim.x * 4]=d_matA[distA_id + threadIdx.x + blockDim.x * 4];
	s_mat[threadIdx.x + blockDim.x * 5]=d_matA[distA_id + threadIdx.x + blockDim.x * 5];
	s_mat[threadIdx.x + blockDim.x * 6]=d_matA[distA_id + threadIdx.x + blockDim.x * 6];
	s_mat[threadIdx.x + blockDim.x * 7]=d_matA[distA_id + threadIdx.x + blockDim.x * 7];
	s_mat[threadIdx.x + blockDim.x * 8]=d_matA[distA_id + threadIdx.x + blockDim.x * 8];
	s_mat[threadIdx.x + blockDim.x * 9]=d_matA[distA_id + threadIdx.x + blockDim.x * 9];
	s_mat[threadIdx.x + blockDim.x * 10]=d_matA[distA_id + threadIdx.x + blockDim.x * 10];
	s_mat[threadIdx.x + blockDim.x * 11]=d_matA[distA_id + threadIdx.x + blockDim.x * 11];
	s_mat[threadIdx.x + blockDim.x * 12]=d_matA[distA_id + threadIdx.x + blockDim.x * 12];
	s_mat[threadIdx.x + blockDim.x * 13]=d_matA[distA_id + threadIdx.x + blockDim.x * 13];
	s_mat[threadIdx.x + blockDim.x * 14]=d_matA[distA_id + threadIdx.x + blockDim.x * 14];
	s_mat[threadIdx.x + blockDim.x * 15]=d_matA[distA_id + threadIdx.x + blockDim.x * 15];
	__syncthreads();

    s_detAux[(threadIdx.x) * 4]=0;
    s_detAux[(threadIdx.x) * 4 + 1]=0;
    s_detAux[(threadIdx.x) * 4 + 2]=0;
    s_detAux[(threadIdx.x) * 4 + 3]=0;
    __syncthreads();

    //  printf("globalId:%d|%d|%d|%d|%d\n",global_id,(threadIdx.x)*4,(threadIdx.x)*4+1,(threadIdx.x)*4+2,(threadIdx.x)*4+3);
    s_detAux[(threadIdx.x)*4] +=  s_mat[offset] * ( (s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+13])+(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+13]))   + (-1*(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+15])) );

    s_detAux[(threadIdx.x)*4+1] +=  (-1*s_mat[offset+1]) * ( (s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+15])) );    

    s_detAux[(threadIdx.x)*4+2] +=  s_mat[offset+2] * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+15])+(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+15])) );        

    s_detAux[(threadIdx.x)*4+3] +=  (-1*s_mat[offset+3]) * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+14])+(s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+12])+(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+14])) );        
    detM[blockIdx.x * blockDim.x + (threadIdx.x)] = s_detAux[(threadIdx.x)*4] + s_detAux[(threadIdx.x)*4+1] + s_detAux[(threadIdx.x)*4+2] + s_detAux[(threadIdx.x)*4+3]; 
    __syncthreads();
        
}

__global__ void vecMult(BASETYPE *d_matA){     
//	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int distA_id = blockIdx.x * blockDim.x * 16;
    int distB_id = threadIdx.x * 16;
    int distC_id;
    extern __shared__ BASETYPE s_mat[];
    unsigned int j;

	s_mat[threadIdx.x]=d_matA[distA_id + threadIdx.x];
	s_mat[threadIdx.x + blockDim.x]=d_matA[distA_id + threadIdx.x + blockDim.x];
	s_mat[threadIdx.x + blockDim.x * 2]=d_matA[distA_id + threadIdx.x + blockDim.x * 2];
	s_mat[threadIdx.x + blockDim.x * 3]=d_matA[distA_id + threadIdx.x + blockDim.x * 3];
	s_mat[threadIdx.x + blockDim.x * 4]=d_matA[distA_id + threadIdx.x + blockDim.x * 4];
	s_mat[threadIdx.x + blockDim.x * 5]=d_matA[distA_id + threadIdx.x + blockDim.x * 5];
	s_mat[threadIdx.x + blockDim.x * 6]=d_matA[distA_id + threadIdx.x + blockDim.x * 6];
	s_mat[threadIdx.x + blockDim.x * 7]=d_matA[distA_id + threadIdx.x + blockDim.x * 7];
	s_mat[threadIdx.x + blockDim.x * 8]=d_matA[distA_id + threadIdx.x + blockDim.x * 8];
	s_mat[threadIdx.x + blockDim.x * 9]=d_matA[distA_id + threadIdx.x + blockDim.x * 9];
	s_mat[threadIdx.x + blockDim.x * 10]=d_matA[distA_id + threadIdx.x + blockDim.x * 10];
	s_mat[threadIdx.x + blockDim.x * 11]=d_matA[distA_id + threadIdx.x + blockDim.x * 11];
	s_mat[threadIdx.x + blockDim.x * 12]=d_matA[distA_id + threadIdx.x + blockDim.x * 12];
	s_mat[threadIdx.x + blockDim.x * 13]=d_matA[distA_id + threadIdx.x + blockDim.x * 13];
	s_mat[threadIdx.x + blockDim.x * 14]=d_matA[distA_id + threadIdx.x + blockDim.x * 14];
	s_mat[threadIdx.x + blockDim.x * 15]=d_matA[distA_id + threadIdx.x + blockDim.x * 15];
	__syncthreads();

	for( j = 1; j < blockDim.x; j *= 2 ){
	    if( threadIdx.x < blockDim.x / (j * 2)){
            distC_id = (blockDim.x / (j * 2)) * 16;
            s_mat[distB_id] += s_mat[(distB_id) + distC_id];
            s_mat[distB_id + 1] += s_mat[(distB_id + 1) + distC_id];
            s_mat[distB_id + 2] += s_mat[(distB_id + 2) + distC_id];
            s_mat[distB_id + 3] += s_mat[(distB_id + 3) + distC_id];
            s_mat[distB_id + 4] += s_mat[(distB_id + 4) + distC_id];
            s_mat[distB_id + 5] += s_mat[(distB_id + 5) + distC_id];
            s_mat[distB_id + 6] += s_mat[(distB_id + 6) + distC_id];
            s_mat[distB_id + 7] += s_mat[(distB_id + 7) + distC_id];
            s_mat[distB_id + 8] += s_mat[(distB_id + 8) + distC_id];
            s_mat[distB_id + 9] += s_mat[(distB_id + 9) + distC_id];
            s_mat[distB_id + 10] += s_mat[(distB_id + 10) + distC_id];
            s_mat[distB_id + 11] += s_mat[(distB_id + 11) + distC_id];
            s_mat[distB_id + 12] += s_mat[(distB_id + 12) + distC_id];
            s_mat[distB_id + 13] += s_mat[(distB_id + 13) + distC_id];
            s_mat[distB_id + 14] += s_mat[(distB_id + 14) + distC_id];
            s_mat[distB_id + 15] += s_mat[(distB_id + 15) + distC_id];
	        
	    }
	    __syncthreads();
	}


	if ((threadIdx.x) == 0){
        d_matA[(blockIdx.x * 16)] = s_mat[0];
        d_matA[(blockIdx.x * 16) + 1] = s_mat[1];
        d_matA[(blockIdx.x * 16) + 2] = s_mat[2];
        d_matA[(blockIdx.x * 16) + 3] = s_mat[3];
        d_matA[(blockIdx.x * 16) + 4] = s_mat[4];
        d_matA[(blockIdx.x * 16) + 5] = s_mat[5];
        d_matA[(blockIdx.x * 16) + 6] = s_mat[6];
        d_matA[(blockIdx.x * 16) + 7] = s_mat[7];
        d_matA[(blockIdx.x * 16) + 8] = s_mat[8];
        d_matA[(blockIdx.x * 16) + 9] = s_mat[9];
        d_matA[(blockIdx.x * 16) + 10] = s_mat[10];
        d_matA[(blockIdx.x * 16) + 11] = s_mat[11];
        d_matA[(blockIdx.x * 16) + 12] = s_mat[12];
        d_matA[(blockIdx.x * 16) + 13] = s_mat[13];
        d_matA[(blockIdx.x * 16) + 14] = s_mat[14];
        d_matA[(blockIdx.x * 16) + 15] = s_mat[15];
	}
}

__global__ void vecMult2(BASETYPE *d_matA,unsigned long n,int offset_m,int cant_m ){     
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int distB_id = threadIdx.x * 16;
    extern __shared__ BASETYPE s_mat[];
    unsigned int j;

    if( global_id < n){

        s_mat[threadIdx.x]=d_matA[(offset_m * 16) + threadIdx.x];
        s_mat[threadIdx.x + n]=d_matA[(offset_m * 16) + threadIdx.x + n];
        s_mat[threadIdx.x + n * 2]=d_matA[(offset_m * 16) + threadIdx.x + n * 2];
        s_mat[threadIdx.x + n * 3]=d_matA[(offset_m * 16) + threadIdx.x + n * 3];
        s_mat[threadIdx.x + n * 4]=d_matA[(offset_m * 16) + threadIdx.x + n * 4];
        s_mat[threadIdx.x + n * 5]=d_matA[(offset_m * 16) + threadIdx.x + n * 5];
        s_mat[threadIdx.x + n * 6]=d_matA[(offset_m * 16) + threadIdx.x + n * 6];
        s_mat[threadIdx.x + n * 7]=d_matA[(offset_m * 16) + threadIdx.x + n * 7];
        s_mat[threadIdx.x + n * 8]=d_matA[(offset_m * 16) + threadIdx.x + n * 8];
        s_mat[threadIdx.x + n * 9]=d_matA[(offset_m * 16) + threadIdx.x + n * 9];
        s_mat[threadIdx.x + n * 10]=d_matA[(offset_m * 16) + threadIdx.x + n * 10];
        s_mat[threadIdx.x + n * 11]=d_matA[(offset_m * 16) + threadIdx.x + n * 11];
        s_mat[threadIdx.x + n * 12]=d_matA[(offset_m * 16) + threadIdx.x + n * 12];
        s_mat[threadIdx.x + n * 13]=d_matA[(offset_m * 16) + threadIdx.x + n * 13];
        s_mat[threadIdx.x + n * 14]=d_matA[(offset_m * 16) + threadIdx.x + n * 14];
        s_mat[threadIdx.x + n * 15]=d_matA[(offset_m * 16) + threadIdx.x + n * 15];
        __syncthreads();

        for( j = 1; j < cant_m; j *= 2 ){
            if( threadIdx.x < cant_m / (j * 2)){
                s_mat[distB_id] += s_mat[(distB_id) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 1] += s_mat[(distB_id + 1) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 2] += s_mat[(distB_id + 2) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 3] += s_mat[(distB_id + 3) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 4] += s_mat[(distB_id + 4) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 5] += s_mat[(distB_id + 5) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 6] += s_mat[(distB_id + 6) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 7] += s_mat[(distB_id + 7) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 8] += s_mat[(distB_id + 8) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 9] += s_mat[(distB_id + 9) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 10] += s_mat[(distB_id + 10) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 11] += s_mat[(distB_id + 11) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 12] += s_mat[(distB_id + 12) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 13] += s_mat[(distB_id + 13) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 14] += s_mat[(distB_id + 14) + (cant_m / (j * 2)) * 16];
                s_mat[distB_id + 15] += s_mat[(distB_id + 15) + (cant_m / (j * 2)) * 16];
                
            }
            __syncthreads();
        }


        if ((threadIdx.x) == 0){
            d_matA[(offset_m / blockDim.x) * 16 + (blockIdx.x * 16)] = s_mat[0];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 1] = s_mat[1];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 2] = s_mat[2];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 3] = s_mat[3];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 4] = s_mat[4];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 5] = s_mat[5];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 6] = s_mat[6];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 7] = s_mat[7];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 8] = s_mat[8];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 9] = s_mat[9];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 10] = s_mat[10];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 11] = s_mat[11];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 12] = s_mat[12];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 13] = s_mat[13];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 14] = s_mat[14];
            d_matA[((offset_m / blockDim.x) * 16 + (blockIdx.x * 16)) + 15] = s_mat[15];
        }
    }
}



__global__ void sumDet(BASETYPE *detM ){   
	//int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ BASETYPE s_vec[];
    
    unsigned int j;

	s_vec[threadIdx.x]=detM[ blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();

	for( j = 1; j < blockDim.x; j *= 2 ){
	    if( threadIdx.x < blockDim.x / (j * 2)){
            s_vec[threadIdx.x] += s_vec[threadIdx.x + (blockDim.x / (j * 2))];
	    }
	    __syncthreads();
	}


	if ((threadIdx.x) == 0){
        detM[blockIdx.x] = s_vec[0];
	}


}

__global__ void sumDet2(BASETYPE *detM,unsigned long n,int offset_m,int cant_m ){     
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ BASETYPE s_vec[];
    unsigned int j;

    if( global_id < n){

        s_vec[threadIdx.x]=detM[offset_m  + threadIdx.x];
        __syncthreads();

        for( j = 1; j < cant_m; j *= 2 ){
            if( threadIdx.x < cant_m / (j * 2)){
                s_vec[threadIdx.x] += s_vec[threadIdx.x + (cant_m / (j * 2))];
            }
            __syncthreads();
        }


        if ((threadIdx.x) == 0){
            detM[(offset_m / blockDim.x) + blockIdx.x] = s_vec[0];
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
    int datos_matDet,datos_vecMult,matDet_desp;


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


    cudaMemcpy(d_matrices, matrices, numBytes*N, cudaMemcpyHostToDevice); // CPU -> GPU
    cudaMemcpy(d_detM, detM, sizeof(BASETYPE)*N, cudaMemcpyHostToDevice); // CPU -> GPU
    matDet<<<dimGrid, dimBlock,datos_matDet>>>(d_matrices,d_detM,matDet_desp);
    cudaThreadSynchronize();
    for(i = N ; i > 1; i = i / CUDA_BLK){
        GRID_BLK = i / CUDA_BLK; 
        if ((i % CUDA_BLK) == 0){
            dim3 dimGrid(GRID_BLK);
            vecMult<<<dimGrid, dimBlock,datos_vecMult>>>(d_matrices);
            cudaThreadSynchronize();
        } else{
            if(GRID_BLK != 0){
                vecMult<<<dimGrid, dimBlock,datos_vecMult>>>(d_matrices);  
                cudaThreadSynchronize(); 
            }
            dim3 dimGrid2(1);
            vecMult2<<<dimGrid2, dimBlock,datos_vecMult>>>(d_matrices,(i % CUDA_BLK),GRID_BLK * CUDA_BLK,(i % CUDA_BLK));  
            cudaThreadSynchronize();
            i = i + (i % CUDA_BLK);
        }
    }

    for(i = N ; i > 1; i = i / CUDA_BLK){
        GRID_BLK = i / CUDA_BLK; 
        if ((i % CUDA_BLK) == 0){
            dim3 dimGrid(GRID_BLK);
            sumDet<<<dimGrid, dimBlock,sizeof(BASETYPE) * 4 * CUDA_BLK>>>(d_detM);
            cudaThreadSynchronize();
        } else{
            if(GRID_BLK != 0){
                sumDet<<<dimGrid, dimBlock,sizeof(BASETYPE) * 4 * CUDA_BLK>>>(d_detM); 
                cudaThreadSynchronize(); 
            }
            dim3 dimGrid2(1);
            sumDet2<<<dimGrid, dimBlock,sizeof(BASETYPE) * 4 * CUDA_BLK>>>(d_detM,(i % CUDA_BLK),GRID_BLK * CUDA_BLK,(i % CUDA_BLK));
            cudaThreadSynchronize();
            i = i + (i % CUDA_BLK);
        }
    }

    cudaThreadSynchronize();


    cudaMemcpy(matrices, d_matrices, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
    cudaMemcpy(detM, d_detM, sizeof(BASETYPE), cudaMemcpyDeviceToHost); // GPU -> CPU

    detM[0] = detM[0] / N;

    for (i = 0; i < 16; i++){
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
