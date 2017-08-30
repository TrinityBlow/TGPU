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

int main(int argc, char *argv[]){

    if (argc != 2){
        printf("Falta argumento: N\n");
        return 0;
    }
    unsigned int N = atoi (argv[1]);
    double *matA,*matB,timetick;
    unsigned int i,j;
    matA = (double *)malloc(sizeof(double)*N*N);
    matB = (double *)malloc(sizeof(double)*N*N);

    for (i = 0; i < N*N; i++){
        matA[i] = i;
        matB[i] = i;
    }

	timetick = dwalltime();
    for (i = 0; i < N; i++){
        for(j = 0; j < N;j++){
        matA[i*N+j] = matA[i*N+j] + matB[i*N+j];
        }
	}
	printf("Tiempo para sumar los vectores: %f\n",dwalltime() - timetick);
/*
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            printf("%f|",matA[i*N+j]);
        }
        printf("\n");
    }
	printf("\n");*/

    return 0;
}