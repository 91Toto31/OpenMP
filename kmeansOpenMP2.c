#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define N 100000 // N is the number of patterns
#define Nc 100 // Nc is the number of classes or centers
#define Nv 1000 // Nv is the length of each pattern (vector)
#define Maxiters 15 // Maxiters is the maximum number of iterations
#define Threshold 0.000001

double *mallocArray( double ***array, int n, int m, int initialize ) ;
void freeArray( double ***array, double *arrayData ) ;

void kMeans( double patterns[][Nv], double centers[][Nv] ) ;
void initialCenters( double patterns[][Nv], double centers[][Nv] ) ;
double findClosestCenters( double patterns[][Nv], double centers[][Nv], int classes[], double ***distances ) ;
void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z);

double distEucl( double pattern[], double center[] ) ;
int argMin( double array[], int length ) ;

void createRandomVectors( double patterns[][Nv] ) ;

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];

    createRandomVectors(patterns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < Nc; i++) {
            initialCenters(patterns, centers);
        }

        #pragma omp for nowait
        for (int i = 0; i < N; i++) {
            kMeans(patterns, centers);
        }
    }

    return EXIT_SUCCESS;
}

void createRandomVectors( double patterns[][Nv] ) {

	srand( 1059364 ) ;

	size_t i, j ;
	for ( i = 0; i < N; i++ ) {
		for ( j = 0; j < Nv; j++ ) {
			patterns[i][j] = (double) (random()%100) - 0.1059364*(i+j) ;
		}
	}

	return ;
}

void kMeans(double patterns[][Nv], double centers[][Nv]) {
    double error = INFINITY;
    double errorBefore;
    int step = 0;

    int *classes = (int *)malloc(N * sizeof(int));
    double **distances;
    double *distanceData = mallocArray(&distances, N, Nc, 0);
    double **y, **z;
    double *yData = mallocArray(&y, Nc, Nv, 1);
    double *zData = mallocArray(&z, Nc, Nv, 1);

    initialCenters(patterns, centers);

    do {
        errorBefore = error;
        error = findClosestCenters(patterns, centers, classes, &distances);

        #pragma omp parallel for
        for (int i = 0; i < Nc; i++) {
            recalculateCenters(patterns, centers, classes, &y, &z);
        }

        printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
        step++;
    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold));

    free(classes);
    freeArray(&distances, distanceData);
    freeArray(&y, yData);
    freeArray(&z, zData);
}

double *mallocArray( double ***array, int n, int m, int initialize ) {

    * array = (double **)malloc( n * sizeof(double *) ) ;
    // avoid to fill heap with small memory allocations.
    double *arrayData = malloc( n*m * sizeof(double) ) ;

    if ( initialize != 0)
        memset( arrayData, 0, n*m ) ;
    
    size_t i ;
    for( i = 0; i < n; i++ )
        (* array)[i] = arrayData + i*m ;
    
    return arrayData;
}

void initialCenters( double patterns[][Nv], double centers[][Nv] ) {

    int centerIndex ;
    size_t i, j ;
    for ( i = 0; i < Nc; i++ ) {
        // split patterns in Nc blocks of N/Nc length
        // use rand and % to pick a random number of each block.
        centerIndex = rand()%( N/Nc*(i+1) - N/Nc*i + 1 ) + N/Nc*i ;
        for ( j = 0; j < Nv; j ++ ) {
            centers[i][j] = patterns[centerIndex][j] ;
        }
    }

    return ;
}

double findClosestCenters( double patterns[][Nv], double centers[][Nv], int classes[], double ***distances ) {

    double error = 0.0 ;
    size_t i, j ;
    for ( i = 0; i < N; i++ ) {
        for ( j = 0; j < Nc; j++ )
            (* distances)[i][j] = distEucl( patterns[i], centers[j] ) ;
        classes[i] = argMin( (* distances)[i], Nc ) ;
        error += (* distances)[i][classes[i]] ;
    }

    return error;
}


void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z) {
    double error = 0.0;
    size_t i, j;

    // calculate tmp arrays
    #pragma omp parallel for private(j) reduction(+:error)
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
            #pragma omp atomic
            (*y)[classes[i]][j] += patterns[i][j];
            #pragma omp atomic
            (*z)[classes[i]][j]++;
        }
    }

    // update step of centers
    #pragma omp parallel for private(j)
    for (i = 0; i < Nc; i++) {
        for (j = 0; j < Nv; j++) {
            #pragma omp critical
            centers[i][j] = (*z)[i][j] != 0 ? (*y)[i][j] / (*z)[i][j] : centers[i][j];
            #pragma omp atomic
            (*y)[i][j] = 0.0;
            #pragma omp atomic
            (*z)[i][j] = 0.0;
        }
    }
    return;
}


double distEucl( double pattern[], double center[] ) {

    double distance = 0.0 ;

    for ( int i = 0; i < Nv; i++ )
        distance += ( pattern[i]-center[i] )*( pattern[i]-center[i] ) ;
    
    return sqrt(distance) ;
}

int argMin( double array[], int length ) {

    int index = 0 ;
    double min = array[0] ;

    for ( int i = 1; i < length; i++ ) {
        if ( min > array[i] ) {
            index = i ;
            min = array[i] ;
        }
    }

    return index ;
}

void freeArray( double ***array, double *arrayData ) {
    
    free( arrayData ) ;
    free( * array ) ;

    return ;
}



