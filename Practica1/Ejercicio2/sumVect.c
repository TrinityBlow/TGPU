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

    if (argc == 1){
        printf("Falta argumento: N\n");
        return 0;
    }
    unsigned int N = atoi (argv[1]);
    double *vecA,*vecB,timetick;
    unsigned int i;

    vecA = (double *)malloc(sizeof(double)*N);
    vecB = (double *)malloc(sizeof(double)*N);

    for (i = 0; i < N; i++){
        vecA[i] = i;
        vecB[i] = i;
    }

	timetick = dwalltime();
    for (i = 0; i < N; i++){
		vecA[i] = vecA[i] + vecB[i];
	}
	printf("Tiempo para sumar los vectores: %f\n",dwalltime() - timetick);

    for(i= 0; i < 20; i++){
        printf("%f|",vecA[i]);
    }
	printf("\n");

    return 0;
}