#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

//524288000 4GB ram con double
//134217728 1GB ram con double

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
    double *vecA,min,max,timetick;
    unsigned int i;

    vecA = (double *)malloc(sizeof(double)*N);

    for (i = 0; i < N; i++){
        vecA[i] = i;
    }
    min = vecA[0];
    max = vecA[0];

	timetick = dwalltime();
    for (i = 1; i < N; i++){
		if (vecA[i] > max) {
            max = vecA[i];
        } else{
            min = vecA[i];
        }
	}
	printf("Tiempo de busqueda: %f\n",dwalltime() - timetick);
    printf("Max:%f\nMin:%f\n",max,min);

    return 0;
}