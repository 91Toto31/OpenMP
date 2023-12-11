#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define N 100000
#define Nc 100
#define Nv 1000
#define Maxiters 15
#define Threshold 0.000001

struct KMeansArgs {
    double (*patterns)[Nv];
    double (*centers)[Nv];
};

void freeArray(double ***array, double *data) {
    free(data);
    *array = NULL;
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

void createRandomVectors(double patterns[][Nv]) {
    srand(1059364);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < Nv; j++) {
            // Modification pour éviter le même nombre aléatoire dans chaque itération
            patterns[i][j] = (double)(rand() % 100) - 0.1059364 * (i + j + 1);
        }
    }
}


double distEucl(double pattern[], double center[]) {
    double distance = 0.0;

#pragma omp parallel for reduction(+:distance) schedule(static)
    for (int i = 0; i < Nv; i++) {
        double diff = pattern[i] - center[i];
        distance += diff * diff;
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

#pragma omp for schedule(static)
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

void recalculateCenters(int Np, double patterns[][Nv], double centers[][Nv], int *classes, double (*y)[Nv], double (*z)[Nv]) {
    #pragma omp parallel for
    for (int i = 0; i < Nc; i++) {
        for (int j = 0; j < Nv; j++) {
            y[i][j] = 0.0;
            z[i][j] = 0.0;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < Np; i++) {
        int cluster = classes[i];
        #pragma omp parallel for
        for (int j = 0; j < Nv; j++) {
            #pragma omp atomic
            y[cluster][j] += patterns[i][j];
            #pragma omp atomic
            z[cluster][j] += 1.0;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < Nc; i++) {
        for (int j = 0; j < Nv; j++) {
            if (z[i][j] != 0) {
                centers[i][j] = y[i][j] / z[i][j];
            }
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

    initialCenters(patterns, centers);

    do {
        errorBefore = error;

#pragma omp parallel sections private(errorBefore, error)
        {
#pragma omp section
            error = findClosestCenters(patterns, centers, classes, &distances);

#pragma omp section
            recalculateCenters(patterns, centers, classes, &y, &z);
        }

#pragma omp barrier

#pragma omp single
        {
            printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
            step++;
        }

    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold));

    free(classes);
    freeArray(&distances, distanceData);
    freeArray(&y, yData);
    freeArray(&z, zData);
}

void kMeansWrapper(void *args) {
    struct KMeansArgs *kmeansArgs = (struct KMeansArgs *)args;
    kMeans(kmeansArgs->patterns, kmeansArgs->centers);
}

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];
    int *classes = (int *)malloc(N * sizeof(int));
    double **y, **z;
    double *yData = mallocArray(&y, Nc, Nv, 1);
    double *zData = mallocArray(&z, Nc, Nv, 1);
    double **distances;
    double *distanceData = mallocArray(&distances, N, Nc, 0);

    // Allocation dynamique des tableaux dans la structure KMeansArgs
    struct KMeansArgs kmeansArgs;
    kmeansArgs.patterns = malloc(N * sizeof(double[Nv]));
    kmeansArgs.centers = malloc(Nc * sizeof(double[Nv]));

    // Vérifiez si l'allocation a réussi
    if (kmeansArgs.patterns == NULL || kmeansArgs.centers == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        exit(EXIT_FAILURE);
    }

    createRandomVectors(patterns);

    double error = INFINITY;
    double errorBefore;
    int step = 0;

    // Copie des données dans la structure
    memcpy(kmeansArgs.patterns, patterns, N * sizeof(double[Nv]));
    memcpy(kmeansArgs.centers, centers, Nc * sizeof(double[Nv]));

    do {
        errorBefore = error;

#pragma omp parallel sections
        {
#pragma omp section
            kMeansWrapper((void *)&kmeansArgs);

#pragma omp section
            recalculateCenters(patterns, centers, classes, &y, &z);
        }

#pragma omp master
{
    if (error != 0.0) {
        printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
    } else {
        // Cas où error est égal à zéro, évitons la division par zéro
        printf("Step:%d||Error: N/A (error is zero),\n", step);
    }
    step++;
}

} while ((step < Maxiters) && ((errorBefore - error) / error > Threshold));

// Libération de la mémoire allouée dynamiquement
free(classes);
freeArray(&distances, distanceData);
freeArray(&y, yData);
freeArray(&z, zData);

// Libération de la mémoire allouée pour les tableaux dans la structure KMeansArgs
free(kmeansArgs.patterns);
free(kmeansArgs.centers);

return EXIT_SUCCESS;
}

