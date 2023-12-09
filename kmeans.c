/*
 *
 * kMeans.c
 * 
 * Created on 28/9/22
 * Author Chris Kaldis
 * Version 0.0.4
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 100000 // N is the number of patterns
#define Nc 100 // Nc is the number of classes or centers
#define Nv 1000 // Nv is the length of each pattern (vector)
#define Maxiters 15 // Maxiters is the maximum number of iterations
#define Threshold 0.000001

double *mallocArray(double ***array, int n, int m, int initialize);
void freeArray(double ***array, double *arrayData);
void kMeans(double patterns[][Nv], double centers[][Nv]);
void initialCenters(double patterns[][Nv], double centers[][Nv]);
double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances);
void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z);
double distEucl(double pattern[], double center[]);
int argMin(double array[], int length);
void createRandomVectors(double patterns[][Nv]);
void printPatterns(double patterns[][Nv]);

void selectRandomInitialCenters(double patterns[][Nv], double centers[][Nv]);
void calculateDistanceAndAssignCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances);
void updateMeans(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z);

double calculateEuclideanDistance(double pattern[], double center[]);

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];

    createRandomVectors(patterns);
    printPatterns(patterns);

    selectRandomInitialCenters(patterns, centers);

    simpleLloydsAlgorithm(patterns, centers);

    // Libération de la mémoire allouée
    freeArray((double ***)&patterns, patternData);
    freeArray((double ***)&centers, centerData);

    return EXIT_SUCCESS;
}

void printPatterns(double patterns[][Nv]) {
    size_t i, j;
    printf("Random Patterns:\n");
    for (i = 0; i < N; i++) {
        printf("[ ");
        for (j = 0; j < Nv; j++) {
            printf("%lf ", patterns[i][j]);
        }
        printf("]\n");
    }
    printf("\n");
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

/*** Simple Implementation of Lloyd's Algorithm ***/
void simpleLloydsAlgorithm(double patterns[][Nv], double centers[][Nv]) {
    double error = INFINITY;
    double errorBefore;
    int step = 0;

    // class or category of each pattern
    int *classes = (int *)malloc(N * sizeof(int));
    // distances between patterns and centers
    double **distances;
    double *distanceData = mallocArray(&distances, N, Nc, 0);

    initialCenters(patterns, centers); // Step 1

    do {
        errorBefore = error;
        error = findClosestCenters(patterns, centers, classes, &distances); // Step 2
        printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
        step++;
    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold)); // Step 3

    free(classes);
    freeArray(&distances, distanceData);
}

void kMeans( double patterns[][Nv], double centers[][Nv] ) {

    double error = INFINITY ;
    double errorBefore ;
    int step = 0 ;
    
    // class or category of each pattern
    int *classes = (int *)malloc( N*sizeof(int) ) ;
    // distances between patterns and centers
    double **distances ;
    double *distanceData = mallocArray( &distances, N, Nc, 0 ) ;
    // tmp data for recalculating centers
    double **y, **z ;
    double *yData = mallocArray( &y, Nc, Nv, 1 ) ;
    double *zData = mallocArray( &z, Nc, Nv, 1 ) ;

    initialCenters( patterns, centers ) ; //step 1
    do {
        errorBefore = error ;
        error = findClosestCenters( patterns, centers, classes, &distances ) ; // step 2
        recalculateCenters( patterns, centers, classes, &y, &z ) ; // step 3
        printf( "Step:%d||Error:%lf,\n",step, (errorBefore-error)/error ) ;
        step ++ ;
    } while ( (step < Maxiters) && ((errorBefore-error)/error > Threshold) ) ; // step 4

    free( classes ) ;
    freeArray( &distances, distanceData ) ;
    freeArray( &y, yData ) ;
    freeArray( &z, zData ) ;

    return ;
}

double *mallocArray(double ***array, int n, int m, int initialize) {
    *array = (double **)malloc(n * sizeof(double *));
    // Pour éviter de remplir le tas avec de petites allocations de mémoire.
    double *arrayData = malloc(n * m * sizeof(double));

    if (initialize != 0)
        memset(arrayData, 0, n * m * sizeof(double));

    size_t i;
    for (i = 0; i < n; i++)
        (*array)[i] = arrayData + i * m;

    return arrayData;

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


oid selectRandomInitialCenters(double patterns[][Nv], double centers[][Nv]) {
    size_t i, j;

    for (i = 0; i < Nc; i++) {
        // Sélectionne un motif aléatoire comme centre initial
        size_t centerIndex = rand() % N;
        for (j = 0; j < Nv; j++) {
            centers[i][j] = patterns[centerIndex][j];
        }
    }
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

void calculateDistanceAndAssignCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances) {
    double error = 0.0;
    size_t i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < Nc; j++)
            (*distances)[i][j] = distEucl(patterns[i], centers[j]);

        classes[i] = argMin((*distances)[i], Nc);
        error += (*distances)[i][classes[i]];
    }

    printf("Total Error: %lf\n", error);
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

void updateMeans(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z) {
    size_t i, j;

    // Calcul des nouvelles moyennes pour chaque classe
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
            (*y)[classes[i]][j] += patterns[i][j];
            (*z)[classes[i]][j]++;
        }
    }

    // Mise à jour des centres
    for (i = 0; i < Nc; i++) {
        for (j = 0; j < Nv; j++) {
            centers[i][j] = (*y)[i][j] / (*z)[i][j];
            (*y)[i][j] = 0.0;
            (*z)[i][j] = 0.0;
        }
    }
}

void recalculateCenters( double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z ) {

    double error = 0.0 ;

    size_t i, j;
    // calculate tmp arrays
    for ( i = 0; i < N; i++ ) {
        for ( j = 0; j < Nv; j++ ) {
            (* y)[classes[i]][j] += patterns[i][j] ;
            (* z)[classes[i]][j] ++ ;
        }
    }

    // update step of centers
    for ( i = 0; i < Nc; i++ ) {
        for ( j = 0; j < Nv; j++ ) {
            centers[i][j] = (* y)[i][j]/(* z)[i][j] ;
            (* y)[i][j] = 0.0 ;
            (* z)[i][j] = 0.0 ;
        }
    }
    
    return ;
}

double calculateEuclideanDistance(double pattern[], double center[]) {
    double distance = 0.0;

    for (int i = 0; i < Nv; i++) {
        distance += (pattern[i] - center[i]) * (pattern[i] - center[i]);
    }

    return sqrt(distance);
}

double distEucl( double pattern[], double center[] ) {

    double distance = 0.0 ;

    for ( int i = 0; i < Nv; i++ )
        distance += ( pattern[i]-center[i] )*( pattern[i]-center[i] ) ;
    
    return sqrt(distance) ;
}

int findMinIndex(double array[], int length) {
    int index = 0;
    double min = array[0];

    for (int i = 1; i < length; i++) {
        if (min > array[i]) {
            index = i;
            min = array[i];
        }
    }

    return index;
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

/*
 *
 *
 * Free memory of a 2D array of double type.
 * Memory allocated with the function mallocArray().
 *
 *
 */

void freeArray( double ***array, double *arrayData ) {
    
    free( arrayData ) ;
    free( * array ) ;

    return ;
}
