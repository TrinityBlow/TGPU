#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>

double dwalltime();


int main(int argc,char*argv[]){

  if (argc != 2){
   printf("Faltan argumentos: N  \n");
   return 0;
  }

  double *A, aux,timetick;
  int i,j, N;
 
  N=atoi(argv[1]);
  A=(double*)malloc(sizeof(double)*N*N);

  for(i=0;i<N;i++){
   for(j=0;j<N;j++){
		if (i<=j){
			A[i*N+j]= 1.0;
		}
		else{
			A[i*N+j]= 0.0;
		}

   }
  }
  /*
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
        printf("%2lf|",A[i*N+j]);
    }
    printf("\n");
}
printf("\n");
printf("\n");
  */
  
 timetick = dwalltime();
  for(i=0;i<N;i++){
   for(j=i+1;j<N;j++){
		aux = A[i*N+j];
		A[i*N+j]= A[j*N+i];
		A[j*N+i]= aux;

    }
  }
  /*
  for(i = 0; i < N; i++){
      for(j = 0; j < N; j++){
          printf("%2lf|",A[i*N+j]);
      }
      printf("\n");
  }
  prinf("\n");
  */
    printf("Tiempo en segundos: %f \n",dwalltime() - timetick);

 free(A);
 return(0);
}

double dwalltime()
{
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}