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
    double *vecA,result,timetick;
    unsigned int i;

    vecA = (double *)malloc(sizeof(double)*N);
    result = 0;

    for (i = 0; i < N; i++){
        vecA[i] = i;
    }

	timetick = dwalltime();
    for (i = 0; i < N; i++){
		result = vecA[i] + result ;
	}
	printf("Tiempo para sumar los elementos: %f\n",dwalltime() - timetick);
    printf("%f\n",result);

    return 0;
}