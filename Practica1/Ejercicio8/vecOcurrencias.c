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

    if (argc != 3){
        printf("Falta argumento: N y X\n");
        return 0;
    }
    unsigned int N;
    double *vecA,X,timetick;
    unsigned int i,ocu;
    

    N = atoi (argv[1]);
    X = atoi (argv[2]);

    vecA = (double *)malloc(sizeof(double)*N);

    for (i = 0; i < N; i++){
        vecA[i] = i;
    }
    ocu = 0;

	timetick = dwalltime();
    for (i = 1; i < N; i++){
		if (vecA[i] == X){
            ocu++;
        }
	}
	printf("Tiempo de busqueda: %f\n",dwalltime() - timetick);
    printf("Ocurrencia de %f: %d\n",X,ocu);

    return 0;
}