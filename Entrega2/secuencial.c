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


int main(int argc, char *argv[]){


    if (argc != 2){
        printf("Falta argumento: N\n");
        return 0;
    }

	//declaracion de variables
    unsigned long N = atoi (argv[1]);
    unsigned long numBytes = sizeof(double)*4*4;
    double *matrices,*result,detAux,detP,timetick;
    unsigned long i,j;


    matrices = (double *)malloc(numBytes*N);
    result = (double *)malloc(numBytes);
    detP = 0;

    for (i = 0; i < N; i++){
        for (j = 0; j < 4*4; j++){
            matrices[i*4*4+j] = 1;
        }
    }
    matrices[2] = 220;
    matrices[13] = 220;
    matrices[7] = 6;
    matrices[14] = 6;

    for (i = 0; i < 4*4; i++){
        result[i] = 0;
    }
    
    timetick = dwalltime();


    for (i = 0; i < N; i++){
        detAux = 0;

        if(matrices[i*4*4] != 0){
            detAux +=  matrices[i*4*4] * ( (matrices[i*4*4+5]*matrices[i*4*4+10]*matrices[i*4*4+15])+(matrices[i*4*4+6]*matrices[i*4*4+11]*matrices[i*4*4+13])+(matrices[i*4*4+7]*matrices[i*4*4+9]*matrices[i*4*4+14])   +  (-1*(matrices[i*4*4+7]*matrices[i*4*4+10]*matrices[i*4*4+13]))   + (-1*(matrices[i*4*4+5]*matrices[i*4*4+11]*matrices[i*4*4+14]))  + (-1*(matrices[i*4*4+6]*matrices[i*4*4+9]*matrices[i*4*4+15])) );
        }

        if(matrices[i*4*4+1] != 0){
            detAux +=  (-1*matrices[i*4*4+1]) * ( (matrices[i*4*4+4]*matrices[i*4*4+10]*matrices[i*4*4+15])+(matrices[i*4*4+6]*matrices[i*4*4+11]*matrices[i*4*4+12])+(matrices[i*4*4+7]*matrices[i*4*4+8]*matrices[i*4*4+14])   +  (-1*(matrices[i*4*4+7]*matrices[i*4*4+10]*matrices[i*4*4+12]))   + (-1*(matrices[i*4*4+4]*matrices[i*4*4+11]*matrices[i*4*4+14]))  + (-1*(matrices[i*4*4+6]*matrices[i*4*4+8]*matrices[i*4*4+15])) );
        }

        if(matrices[i*4*4+2] != 0){
            detAux +=  matrices[i*4*4+2] * ( (matrices[i*4*4+4]*matrices[i*4*4+9]*matrices[i*4*4+15])+(matrices[i*4*4+5]*matrices[i*4*4+11]*matrices[i*4*4+12])+(matrices[i*4*4+7]*matrices[i*4*4+8]*matrices[i*4*4+13])   +  (-1*(matrices[i*4*4+7]*matrices[i*4*4+9]*matrices[i*4*4+12]))   + (-1*(matrices[i*4*4+4]*matrices[i*4*4+11]*matrices[i*4*4+13]))  + (-1*(matrices[i*4*4+5]*matrices[i*4*4+8]*matrices[i*4*4+15])) );
        }

        if(matrices[i*4*4+3] != 0){
            detAux +=  (-1*matrices[i*4*4+3]) * ( (matrices[i*4*4+4]*matrices[i*4*4+9]*matrices[i*4*4+14])+(matrices[i*4*4+5]*matrices[i*4*4+10]*matrices[i*4*4+12])+(matrices[i*4*4+6]*matrices[i*4*4+8]*matrices[i*4*4+13])   +  (-1*(matrices[i*4*4+6]*matrices[i*4*4+9]*matrices[i*4*4+12]))   + (-1*(matrices[i*4*4+4]*matrices[i*4*4+10]*matrices[i*4*4+13]))  + (-1*(matrices[i*4*4+5]*matrices[i*4*4+8]*matrices[i*4*4+14])) );
        }

        detP += detAux;
        for (j = 0; j < 4*4; j++){
            result[j] += matrices[i*4*4+j];
        }
    }

    detP = detP / N;

    for (i = 0; i < 4*4; i++){
        result[i] *= detP;
    }


    printf("Tiempo para la CPU: %f\n",dwalltime() - timetick);
    
    printf("%.2lf|\n",detP);
    for (i = 0; i < 4; i++){
        for (j = 0; j < 4; j++){
            printf("%.2lf|",result[i*4+j]);
        }
        printf("\n");
    }

    free(matrices);
    free(result);
    return 0;

}

/*
        matrices[i*4*4];
        
        matrices[i*4*4+5] | matrices[i*4*4+6] | matrices[i*4*4+7];
        matrices[i*4*4+9] | matrices[i*4*4+10] | matrices[i*4*4+11];
        matrices[i*4*4+13] | matrices[i*4*4+14] | matrices[i*4*4+15];


        matrices[i*4*4+1];
        
        matrices[i*4*4+4] | matrices[i*4*4+6] | matrices[i*4*4+7];
        matrices[i*4*4+8] | matrices[i*4*4+10] | matrices[i*4*4+11];
        matrices[i*4*4+12] | matrices[i*4*4+14] | matrices[i*4*4+15];


        matrices[i*4*4+2];
        
        matrices[i*4*4+4] | matrices[i*4*4+5] | matrices[i*4*4+7];
        matrices[i*4*4+8] | matrices[i*4*4+9] | matrices[i*4*4+11];
        matrices[i*4*4+12] | matrices[i*4*4+13] | matrices[i*4*4+15];


        matrices[i*4*4+3];
        
        matrices[i*4*4+4] | matrices[i*4*4+5] | matrices[i*4*4+6];
        matrices[i*4*4+8] | matrices[i*4*4+9] | matrices[i*4*4+10];
        matrices[i*4*4+12] | matrices[i*4*4+13] | matrices[i*4*4+14];
*/