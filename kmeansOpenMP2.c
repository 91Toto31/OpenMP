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

// Structure pour stocker les arguments de kMeans
struct KMeansArgs {
    double patterns[N][Nv];
    double centers[Nc][Nv];
};

double *mallocArray(double ***array, int n, int m, int initialize);
void freeArray(double ***array, double *arrayData);

void kMeans(double patterns[][Nv], double centers[][Nv]);
void initialCenters(double patterns[][Nv], double centers[][Nv]);
double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances);
void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z);

double distEucl(double pattern[], double center[]);
int argMin(double array[], int length);

void createRandomVectors(double patterns[][Nv]);

void kMeansWrapper(void *args) {
    struct KMeansArgs *kmeansArgs = (struct KMeansArgs *)args;
    kMeans(kmeansArgs->patterns, kmeansArgs->centers);
}

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];
    int *classes = (int *)malloc(N * sizeof(int));
    double **y;
    double *yData = mallocArray(&y, Nc, Nv, 1);
    double **z;
    double *zData = mallocArray(&z, Nc, Nv, 1);


    createRandomVectors(patterns);

    double error = INFINITY;
    double errorBefore;
    int step = 0;

    // Structure pour stocker les arguments de kMeans
    struct KMeansArgs kmeansArgs;
    memcpy(kmeansArgs.patterns, patterns, sizeof(patterns));
    memcpy(kmeansArgs.centers, centers, sizeof(centers));

    do {
        errorBefore = error;

        // Appel parallélisé à la fonction kMeans
#pragma omp parallel sections
        {
#pragma omp section
            kMeansWrapper((void *)&kmeansArgs);

#pragma omp section
            recalculateCenters(patterns, centers, classes, &y, &z);
        }

        // Affichage et vérification de la condition de boucle
#pragma omp master
        {
            printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
            step++;
        }

    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold)); // step 4

    free(classes);
    freeArray(&distances, distanceData);
    freeArray(&y, yData);
    freeArray(&z, zData);

    return EXIT_SUCCESS;
}

void createRandomVectors(double patterns[][Nv]) {
    srand(1059364);

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < Nv; j++) {
            patterns[i][j] = (double)(rand() % 100) - 0.1059364 * (i + j);
        }
    }
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

    initialCenters(patterns, centers); // step 1

    do {
        errorBefore = error;

#pragma omp parallel sections
        {
#pragma omp section
            error = findClosestCenters(patterns, centers, classes, &distances); // step 2

#pragma omp section
            recalculateCenters(patterns, centers, classes, &y, &z); // step 3
        }

#pragma omp barrier // Synchronisation des sections

        // Affichage et vérification de la condition de boucle
#pragma omp master
        {
            printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
            step++;
        }

    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold)); // step 4

    free(classes);
    freeArray(&distances, distanceData);
    freeArray(&y, yData);
    freeArray(&z, zData);
}

double *mallocArray(double ***array, int n, int m, int initialize) {
    *array = (double **)malloc(n * sizeof(double *));
    double *arrayData = (double *)malloc(n * m * sizeof(double));

    if (initialize != 0) {
#pragma omp parallel for
        for (int i = 0; i < n * m; i++) {
            arrayData[i] = 0.0;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        (*array)[i] = arrayData + i * m;
    }

    return arrayData;
}

void initialCenters(double patterns[][Nv], double centers[][Nv]) {
    srand(time(NULL));
#pragma omp parallel for
    for (size_t i = 0; i < Nc; i++) {
        int centerIndex = rand() % (N / Nc * (i + 1) - N / Nc * i + 1) + N / Nc * i;
        for (size_t j = 0; j < Nv; j++) {
            centers[i][j] = patterns[centerIndex][j];
        }
    }
}

double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances) {
    double error = 0.0;

#pragma omp parallel for reduction(+ \
                                   : error)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < Nc; j++) {
            (*distances)[i][j] = distEucl(patterns[i], centers[j]);
        }
        classes[i] = argMin((*distances)[i], Nc);
        error += (*distances)[i][classes[i]];
    }

    return error;
}


oid recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z) {
    size_t i, j;

#pragma omp parallel for collapse(2)
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
#pragma omp atomic
            (*y)[classes[i]][j] += patterns[i][j];
#pragma omp atomic
            (*z)[classes[i]][j]++;
        }
    }

#pragma omp parallel for collapse(2)
    for (i = 0; i < Nc; i++) {
        for (j = 0; j < Nv; j++) {
            centers[i][j] = (*y)[i][j] / (*z)[i][j];

#pragma omp critical
            {
                (*y)[i][j] = 0.0;
                (*z)[i][j] = 0.0;
            }
        }
    }
}

double distEucl(double pattern[], double center[]) {
    double distance = 0.0;

#pragma omp parallel for reduction(+:distance)
    for (int i = 0; i < Nv; i++) {
        distance += (pattern[i] - center[i]) * (pattern[i] - center[i]);
    }

    return sqrt(distance);
}

int argMin(double array[], int length) {
    int global_index = 0;
    double global_min = array[0];

#pragma omp parallel
    {
        int private_index = 0;
        double private_min = array[0];

#pragma omp for
        for (int i = 1; i < length; i++) {
            if (private_min > array[i]) {
                private_index = i;
                private_min = array[i];
            }
        }

#pragma omp critical
        {
            if (global_min > private_min) {
                global_index = private_index;
                global_min = private_min;
            }
        }
    }

    return global_index;
}

void freeArray(double ***array, double *arrayData) {
    free(arrayData);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
#pragma omp critical
        {
            free((*array)[i]);
        }
    }

    free(*array);

    return;
}
