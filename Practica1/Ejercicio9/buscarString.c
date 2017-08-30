#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>

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
        printf("Falta argumento: \"cadena\" y tamaño de string a buscar\n");
        return 0;
    }

    double timetick;
    unsigned int i;
    int encontrado,pos,tam,pos_aux,tam_aux;
    char *cadena, buscador;

    cadena = argv[1];
    tam = atoi(argv[2]);
    encontrado = 0;    
    pos_aux = 0;
    tam_aux = 1;
    buscador = cadena[0];
    
    printf("%lu\n",strlen(cadena));
    
    timetick = dwalltime();
    if (tam > 0){
        while ((buscador != '\0') && !encontrado){
            while(buscador == ' '){
                pos_aux++;
                buscador = cadena[pos_aux];
            }
            tam_aux = 0;
            pos = pos_aux;
            while((buscador != ' ') && (buscador != '\0') ){
                tam_aux++;
                pos_aux++;
                buscador = cadena[pos_aux];
            }
            if (tam_aux == tam){
                encontrado = 1;
            }
        }
    }
	printf("Tiempo de busqueda: %f\n",dwalltime() - timetick);
    if (encontrado){
        printf("El string se encuentra en: %d\n",pos);
    }else{
        printf("String no encontrado \n");
    }

    return 0;
}