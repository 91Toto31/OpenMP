#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define N 100000 // N is the number of patterns
#define Nc 100    // Nc is the number of classes or centers
#define Nv 1000   // Nv is the length of each pattern (vector)
#define Maxiters 15    // Maxiters is the maximum number of iterations
#define Threshold 0.000001

double *mallocArray(double ***array, int n, int m, int initialize);
void freeArray(double ***array, double *arrayData);

void kMeans(double **patterns, double **centers);
void initialCenters(double **patterns, double **centers);
double findClosestCenters(double **patterns, double **centers, int *classes, double ***distances);
void recalculateCenters(double **patterns, double **centers, int *classes, double ***y, double ***z);

double distEucl(double *pattern, double *center);
int argMin(double *array, int length);

void createRandomVectors(double **patterns);

int main(int argc, char *argv[]) {
    double **patterns = (double **)malloc(N * sizeof(double *));
    double **centers = (double **)malloc(Nc * sizeof(double *));
    
    createRandomVectors(patterns);
    
    for (int i = 0; i < Nc; i++) {
        centers[i] = (double *)malloc(Nv * sizeof(double));
    }

    kMeans(patterns, centers);

    // Free memory
    freeArray(&patterns, N);
    freeArray(&centers, Nc);

    return EXIT_SUCCESS;
}

void createRandomVectors(double **patterns) {
    srand(1059364);

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        patterns[i] = (double *)malloc(Nv * sizeof(double));
        for (int j = 0; j < Nv; j++) {
            patterns[i][j] = (double)(rand() % 100) - 0.1059364 * (i + j);
        }
    }
}

double *mallocArray(double ***array, int n, int m, int initialize) {
    *array = (double **)malloc(n * sizeof(double *));

    double *arrayData = (double *)malloc(n * m * sizeof(double));

    if (initialize != 0)
        memset(arrayData, 0, n * m * sizeof(double));

    for (int i = 0; i < n; i++)
        (*array)[i] = arrayData + i * m;

    return arrayData;
}

void freeArray(double ***array, int n) {
    for (int i = 0; i < n; i++) {
        free((*array)[i]);
    }
    free(*array);
}

void kMeans(double **patterns, double **centers) {
    double error = INFINITY;
    double errorBefore;
    int step = 0;

    int *classes = (int *)malloc(N * sizeof(int));
    double **distances;  // Change to double **
    double *distanceData = mallocArray(&distances, N, Nc, 0);
    double **y;         // Change to double **
    double *yData = mallocArray(&y, Nc, Nv, 1);
    double **z;         // Change to double **
    double *zData = mallocArray(&z, Nc, Nv, 1);

    size_t j;  // Declare j at this level

    initialCenters(patterns, centers);

    do {
        errorBefore = error;

#pragma omp parallel for private(j) shared(N, Nc, patterns, centers, distances, classes) reduction(+:error) firstprivate(error)
        for (int i = 0; i < N; i++) {
            for (j = 0; j < Nc; j++) {
                distances[i][j] = distEucl(patterns[i], centers[j]);
            }
            classes[i] = argMin(distances[i], Nc);
            #pragma omp atomic
            error += distances[i][classes[i]];
        }

        error = findClosestCenters(patterns, centers, classes, &distances);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < Nc; i++) {
            for (j = 0; j < Nv; j++) {
                double tmp_y = 0.0;
                double tmp_z = 0.0;
                for (int k = 0; k < N; k++) {
                    if (classes[k] == i) {
                        tmp_y += patterns[k][j];
                        tmp_z += 1.0;
                    }
                }
                #pragma omp atomic
                y[i][j] += tmp_y;
                #pragma omp atomic
                z[i][j] += tmp_z;
            }
        }

#pragma omp parallel for collapse(2)
        for (int i = 0; i < Nc; i++) {
            for (j = 0; j < Nv; j++) {
                centers[i][j] = y[i][j] / z[i][j];
                y[i][j] = 0.0;
                z[i][j] = 0.0;
            }
        }

        printf("Step:%d||Error:%lf,\n", step, error);
        step++;
    } while ((step < Maxiters) && (errorBefore - error > Threshold));

    free(classes);
    freeArray(&distances, N);
    freeArray(&y, Nc);
    freeArray(&z, Nc);
}

void initialCenters(double **patterns, double **centers) {
    int centerIndex;
    size_t i, j;
    for (i = 0; i < Nc; i++) {
        centerIndex = rand() % (N / Nc * (i + 1) - N / Nc * i + 1) + N / Nc * i;
        for (j = 0; j < Nv; j++) {
            centers[i][j] = patterns[centerIndex][j];
        }
    }
}

double findClosestCenters(double **patterns, double **centers, int *classes, double ***distances) {
    double error = 0.0;
    size_t i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nc; j++)
            (*distances)[i][j] = distEucl(patterns[i], centers[j]);
        classes[i] = argMin((*distances)[i], Nc);
        error += (*distances)[i][classes[i]];
    }
    return error;
}

void recalculateCenters(double **patterns, double **centers, int *classes, double ***y, double ***z) {
    size_t i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
            (*y)[classes[i]][j] += patterns[i][j];
            (*z)[classes[i]][j]++;
        }
    }
    for (i = 0; i < Nc; i++) {
        for (j = 0; j < Nv; j++) {
            centers[i][j] = (*y)[i][j] / (*z)[i][j];
            (*y)[i][j] = 0.0;
            (*z)[i][j] = 0.0;
        }
    }
}

double distEucl(double *pattern, double *center) {
    double distance = 0.0;
    for (int i = 0; i < Nv; i++)
        distance += (pattern[i] - center[i]) * (pattern[i] - center[i]);
    return sqrt(distance);
}

int argMin(double *array, int length) {
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



