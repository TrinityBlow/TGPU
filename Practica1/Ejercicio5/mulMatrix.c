
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

//Para calcular tiempo
double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

int N = 128;


int main(int argc, char *argv[]){
	if(argc != 2){
		return 1;
	}
	N = atoi(argv[1]);
	double timetick,*A,*B,*C;
	int i,j,k,tam_matrix = (N*N);
	A=(double*)malloc(sizeof(double)*N*N);
	B=(double*)malloc(sizeof(double)*N*N);
	C=(double*)malloc(sizeof(double)*N*N);
	for(i = 0; i < tam_matrix; i++){
		A[i] = 2.0;
		B[i] = 3.0;
		C[i] = 0.0;
	}
	

	timetick = dwalltime();
	//printf("%d %d %d\n",id,base,fin);
	for(i = 0;i < N; i++ ){
		for(j = 0;j < N; j++){
			for(k = 0; k < N; k++){
				C[i*N+j] += A[i*N+k] * B[j*N+k];
			}
		}
	}
	
	/*for(i = 0; i < N; i++){
	    for(j = 0; j < N; j++){
		printf("|%.2lf",C[i*N+j]);
    	}
	printf("\n");
	}*/
	printf("%.2lf\n",C[0]);
    
	 printf("Tiempo en segundos %f\n", dwalltime() - timetick);
	free(A);
	free(B);
	free(C);
	
	return 0;
}