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
__global__ void matDet(double *d_matA, double *detM){ 
    
	int global_id =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ double s_mat[sizeof(double)*64];
    __shared__ double s_detAux[16];
    int offset = (threadIdx.y * blockDim.x + threadIdx.x)*16; 

    if ((threadIdx.y * blockDim.x + threadIdx.x) < 64){
        s_mat[(threadIdx.y * blockDim.x + threadIdx.x)]=d_matA[global_id];
        if(threadIdx.y * blockDim.x + threadIdx.x < 16){
            s_detAux[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }
        __syncthreads();
 
        if(threadIdx.y * blockDim.x + threadIdx.x < 4){
          //  printf("globalId:%d|%d|%d|%d|%d\n",global_id,(threadIdx.y * blockDim.x + threadIdx.x)*4,(threadIdx.y * blockDim.x + threadIdx.x)*4+1,(threadIdx.y * blockDim.x + threadIdx.x)*4+2,(threadIdx.y * blockDim.x + threadIdx.x)*4+3);
            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4] +=  s_mat[offset] * ( (s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+13])+(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+13]))   + (-1*(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+15])) );
    //        __syncthreads();       
           
            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+1] +=  (-1*s_mat[offset+1]) * ( (s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+15])) );    
     //       __syncthreads();

            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+2] +=  s_mat[offset+2] * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+15])+(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+15])) );        
     //       __syncthreads();
      
            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+3] +=  (-1*s_mat[offset+3]) * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+14])+(s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+12])+(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+14])) );            
       //     __syncthreads();
          
            detM[blockIdx.x*4 + (threadIdx.y * blockDim.x + threadIdx.x)] = s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4] + s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+1] + s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+2] + s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+3]; 
  //          __syncthreads();
          
        }
    }
}

__global__ void vecMult(double *d_matA,unsigned long n){      
	int global_id =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ double s_mat[sizeof(double)*64];
    unsigned int i;

    if ((threadIdx.y * blockDim.x + threadIdx.x) < 64){
        s_mat[(threadIdx.y * blockDim.x + threadIdx.x)]=d_matA[global_id];
        __syncthreads();



        for( i = 1; i <= 2;  i++) {
            if(( threadIdx.y * blockDim.x + threadIdx.x )< (int)(64 >> i)){
                s_mat[(threadIdx.y * blockDim.x + threadIdx.x)] += s_mat[((threadIdx.y * blockDim.x + threadIdx.x ) + (64 >> i))];
            }
            __syncthreads();
        }

        if ( (threadIdx.y * blockDim.x + threadIdx.x) < 16){
            d_matA[blockIdx.x * 16 + (threadIdx.y * blockDim.x + threadIdx.x)] = s_mat[(threadIdx.y * blockDim.x + threadIdx.x)];
        }
    }
}



__global__ void vecMult3(double *d_matA,unsigned long n,int iteraciones){      
	int global_id =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ double s_mat[sizeof(double)*128];
    unsigned int i;
        /*
        for( i = 1; i <= 2;  i++) {
            if(( threadIdx.y * blockDim.x + threadIdx.x )< (int)(64 >> i)){
                s_mat[(threadIdx.y * blockDim.x + threadIdx.x)] += s_mat[((threadIdx.y * blockDim.x + threadIdx.x ) + (64 >> i))];
            }
            __syncthreads();
        }
        */
    for(i = 1 ; i <= iteraciones ; i++){
        if ( global_id < ( n / (1 << (i) ))){
            s_mat[(threadIdx.y * blockDim.x + threadIdx.x)]=d_matA[(((int)(global_id / 16))*32 + (global_id % 16))];
            s_mat[(threadIdx.y * blockDim.x + threadIdx.x)+64]=d_matA[((int)(global_id / 16)*32 + (global_id % 16)+ 16)];
         //   printf("global:%d||%.2lf||%.2lf||\n",global_id,s_mat[(threadIdx.y * blockDim.x + threadIdx.x)],s_mat[(threadIdx.y * blockDim.x + threadIdx.x)+64]);
            printf("global:%d||%d||\n",global_id,(((int)(global_id / 16))*32 + (global_id % 16)));
            __syncthreads();

            
            s_mat[(threadIdx.y * blockDim.x + threadIdx.x)] += s_mat[(threadIdx.y * blockDim.x + threadIdx.x) + 64];
           __syncthreads();

            d_matA[global_id] = s_mat[(threadIdx.y * blockDim.x + threadIdx.x)];
            __syncthreads();
            if(global_id == 0){
                printf("-------------------------------------------\n");

            }
        }
    }
}

__global__ void vecMult2(double *d_matA,unsigned long n,int iteraciones){      
	int global_id =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ double s_mat[128];
    unsigned int i;
        /*
        for( i = 1; i <= 2;  i++) {
            if(( threadIdx.y * blockDim.x + threadIdx.x )< (int)(64 >> i)){
                s_mat[(threadIdx.y * blockDim.x + threadIdx.x)] += s_mat[((threadIdx.y * blockDim.x + threadIdx.x ) + (64 >> i))];
            }
            __syncthreads();
        }
        */
    for(i = 1 ; i <= iteraciones ; i++){
        if ( global_id < ( n / (1 << (i) ))){
            s_mat[(threadIdx.y * blockDim.x + threadIdx.x)]=d_matA[global_id];
            __syncthreads();
            s_mat[(threadIdx.y * blockDim.x + threadIdx.x)+64]=d_matA[(int)(global_id + ( n / (1 << (i) )))];
            __syncthreads();
            printf("global:%d||%.2lf||%.2lf||\n",global_id,s_mat[(threadIdx.y * blockDim.x + threadIdx.x)],s_mat[(threadIdx.y * blockDim.x + threadIdx.x)+64]);

            __syncthreads();

         /*   if(global_id == 0){
                printf("global:%d||%d||suma:%.2lf v1:%.2lf v2:%.2lf\n",global_id,(int)(global_id + ( n / (1 << (i) ))),s_mat[(threadIdx.y * blockDim.x + threadIdx.x)]+s_mat[(threadIdx.y * blockDim.x + threadIdx.x) + 64],s_mat[(threadIdx.y * blockDim.x + threadIdx.x)],s_mat[(threadIdx.y * blockDim.x + threadIdx.x) + 64]);
            }*/
            s_mat[(threadIdx.y * blockDim.x + threadIdx.x)] += s_mat[(threadIdx.y * blockDim.x + threadIdx.x) + 64];           

           __syncthreads();

            d_matA[global_id] = s_mat[(threadIdx.y * blockDim.x + threadIdx.x)];
         __syncthreads();
        /*    if(global_id == 0){
                printf("-------------------------------------------\n");

            }*/
        }
         __syncthreads();
    }
}





int main(int argc, char *argv[]){


    if (argc != 2){
        printf("Falta argumento: N\n");
        return 0;
    }
	//declaracion de variables
    cudaError_t error;
    unsigned long N = atoi (argv[1]);
    unsigned long CUDA_BLK = 8,GRID_BLK;
    unsigned long numBytes = sizeof(double)*4*4;
    double *matrices,*d_matrices,*d_detM,*detM,timetick;
    unsigned long i,j;
    int iteraciones;


    matrices = (double *)malloc(numBytes*N);
    detM = (double *)malloc(sizeof(double)*N);
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
    cudaMalloc((void **) &d_detM, sizeof(double)*N);

    dim3 dimBlock(CUDA_BLK,CUDA_BLK);
    dim3 dimGrid(N/4);
    
    timetick = dwalltime();

    iteraciones = log(N) / log(2);

    cudaMemcpy(d_matrices, matrices, numBytes*N, cudaMemcpyHostToDevice); // CPU -> GPU
    cudaMemcpy(d_detM, detM, sizeof(double)*N, cudaMemcpyHostToDevice); // CPU -> GPU
    matDet<<<dimGrid, dimBlock>>>(d_matrices,d_detM);
    cudaThreadSynchronize();
   /* for(i = 4 ; i <= N; i *= 4){
        GRID_BLK = N / i; 
        dim3 dimGrid(GRID_BLK);
        vecMult<<<dimGrid, dimBlock>>>(d_matrices,(4*4*N) / (i / 2));
        cudaThreadSynchronize();
    }*/
    dim3 dimGrid2(4*4*N/CUDA_BLK/CUDA_BLK / 2);
    vecMult2<<<dimGrid2, dimBlock>>>(d_matrices,(4*4*N),iteraciones);
    cudaThreadSynchronize();
    cudaMemcpy(matrices, d_matrices, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
    cudaMemcpy(detM, d_detM, sizeof(double)*N, cudaMemcpyDeviceToHost); // GPU -> CPU

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