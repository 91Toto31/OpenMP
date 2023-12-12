#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h> // Ajout du fichier d'en-tête pour DBL_MAX
#include <omp.h>
#include <time.h>

#define N 100000
#define Nc 100
#define Nv 1000
#define Maxiters 15
#define Threshold 0.000001

typedef struct {
    double (*patterns)[Nv];
    double (*centers)[Nv];
} KMeansArgs;

void freeArray(double (*array)[Nv], int n) {
    free(array);
}

double (*mallocArray(int n, int m))[Nv] {
    double (*array)[Nv] = malloc(n * sizeof(*array));
    if (array == NULL) {
        perror("mallocArray");
        exit(EXIT_FAILURE);
    }

    return array;
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

double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double distances[][Nc]) {
    double error = 0.0;

#pragma omp parallel for reduction(+ \
                                   : error)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < Nc; j++) {
            distances[i][j] = distEucl(patterns[i], centers[j]);
        }
        classes[i] = argMin(distances[i], Nc);
        error += distances[i][classes[i]];
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
    int *classes = (int *)malloc(N * sizeof(int));
    if (classes == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    double (*y)[Nv] = mallocArray(Nc, Nv);
    double (*z)[Nv] = mallocArray(Nc, Nv);

    int step = 1;
    double errorBefore = DBL_MAX;
    double error = 0.0;

    // Ajout de la déclaration de distances
    double distances[N][Nc];

    while (step <= Maxiters && (errorBefore - error) / error > Threshold) {
        recalculateCenters(N, patterns, centers, classes, y, z);

        // Calcul de l'erreur
        errorBefore = error;
        error = findClosestCenters(patterns, centers, classes, distances);

        // Affichage des résultats
        printf("Step:%d || Error:%lf\n", step, (errorBefore - error) / error);
        step++;
    }

    freeArray(y, Nc);
    freeArray(z, Nc);
    free(classes);
}

void kMeansWrapper(void *args) {
    KMeansArgs *kmeansArgs = (KMeansArgs *)args;
    kMeans(kmeansArgs->patterns, kmeansArgs->centers);
}

int main() {
    KMeansArgs kmeansArgs;
    kmeansArgs.patterns = mallocArray(N, Nv);
    kmeansArgs.centers = mallocArray(Nc, Nv);

    initialCenters(kmeansArgs.patterns, kmeansArgs.centers);
    kMeans(kmeansArgs.patterns, kmeansArgs.centers);

    freeArray(kmeansArgs.patterns, N);
    freeArray(kmeansArgs.centers, Nc);

    return 0;
}



