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

void freeArray(double (*array)[Nv]) {
    free(*array);
    *array = NULL;
}

double *mallocArray(int n, int m, int initialize) {
    double *array = (double *)malloc(n * m * sizeof(double));

    if (initialize != 0) {
#pragma omp parallel for
        for (int i = 0; i < n * m; i++) {
            array[i] = 0.0;
        }
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
    double error = INFINITY;
    double errorBefore;
    int step = 0;

    int *classes = (int *)malloc(N * sizeof(int));
    double distances[N][Nc];
    double (*y)[Nv] = (double (*)[Nv])mallocArray(Nc, Nv, 1);
    double (*z)[Nv] = (double (*)[Nv])mallocArray(Nc, Nv, 1);

    initialCenters(patterns, centers);

    do {
        errorBefore = error;

#pragma omp parallel sections private(errorBefore, error)
        {
#pragma omp section
            error = findClosestCenters(patterns, centers, classes, distances);

#pragma omp section
            recalculateCenters(N, patterns, centers, classes, y, z);
        }

#pragma omp barrier

#pragma omp single
        {
            printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
            step++;
        }

    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold));

    free(classes);
    freeArray(y);
    freeArray(z);
}

void kMeansWrapper(void *args) {
    struct KMeansArgs *kmeansArgs = (struct KMeansArgs *)args;
    kMeans(kmeansArgs->patterns, kmeansArgs->centers);
}

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];

    struct KMeansArgs kmeansArgs;
    kmeansArgs.patterns = (double (*)[Nv])mallocArray(N, Nv, 0);
    kmeansArgs.centers = (double (*)[Nv])mallocArray(Nc, Nv, 0);

    if (kmeansArgs.patterns == NULL || kmeansArgs.centers == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        exit(EXIT_FAILURE);
    }

    createRandomVectors(patterns);

    kMeansWrapper(&kmeansArgs);

    freeArray(kmeansArgs.patterns);
    freeArray(kmeansArgs.centers);

    return EXIT_SUCCESS;
}



