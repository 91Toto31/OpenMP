#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define N 100000
#define Nc 100
#define Nv 1000
#define Maxiters 15
#define Threshold 0.000001
#define DEBUG 0

int **mallocIntArray(int n, int m);
double **mallocArray(int n, int m);
void freeArray(double **array);
void freeIntArray(int **array);
void createRandomVectors(double patterns[][Nv]);
void initialCenters(double patterns[][Nv], double centers[][Nv]);
double distEuclSquare(double pattern[], double center[]);
int argMin(double array[], int length);
void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, int ***z);
void kMeans(double patterns[][Nv], double centers[][Nv]);

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];

    createRandomVectors(patterns);
    initialCenters(patterns, centers);

    #pragma omp parallel
    {
        #pragma omp single
        kMeans(patterns, centers);
    }

    return EXIT_SUCCESS;
}

void createRandomVectors(double patterns[][Nv]) {
    srand(1059364);

    size_t i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
            patterns[i][j] = (double)(rand() % 100) - 0.1059364 * (i + j);
        }
    }
}

void initialCenters(double patterns[][Nv], double centers[][Nv]) {
    int centerIndex;
    size_t i, j;
    for (i = 0; i < Nc; i++) {
        centerIndex = rand() % (N / Nc * (i + 1) - N / Nc * i + 1) + N / Nc * i;
        for (j = 0; j < Nv; j++) {
            centers[i][j] = patterns[centerIndex][j];
        }
    }
}

void kMeans(double patterns[][Nv], double centers[][Nv]) {
    double error = INFINITY;
    double errorBefore;
    int step = 0;

    int classes[N];
    int local_classes[N];

    double **distances = mallocArray(N, Nc);

    double **y = mallocArray(Nc, Nv);
    int **z = mallocIntArray(Nc, Nv);

    do {
        errorBefore = error;
        double local_error = 0.0;  // Déplacer la variable en dehors de la région parallèle

        #pragma omp parallel
        {
            #pragma omp for reduction(+:local_error)
            for (size_t i = 0; i < N; i++) {
                local_error += distEuclSquare(patterns[i], centers[classes[i]]);
            }

            #pragma omp for
            for (size_t i = 0; i < N; i++) {
                local_classes[i] = argMin(distances[i], Nc);
            }
        }

        #pragma omp atomic update
        error += local_error;

        recalculateCenters(patterns, centers, local_classes, &y, &z);

        #if DEBUG
        printf("Step:%d || Error:%lf\n", step, (errorBefore - error) / error);
        #endif

        step++;

    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold));

    freeArray(distances);
    freeIntArray(z);
    freeArray(y);
}

void freeArray(double **array) {
    free(array[0]);
    free(array);
}

double **mallocArray(int n, int m) {
    double **array = (double **)malloc(n * sizeof(double *));
    double *arrayData = (double *)malloc(n * m * sizeof(double));

    if (array == NULL || arrayData == NULL) {
        printf("Erreur d'allocation de mémoire.\n");
        exit(EXIT_FAILURE);
    }

    memset(arrayData, 0, n * m);

    size_t i;
    for (i = 0; i < n; i++)
        array[i] = arrayData + i * m;

    return array;
}

int **mallocIntArray(int n, int m) {
    int **array = (int **)malloc(n * sizeof(int *));
    int *arrayData = (int *)malloc(n * m * sizeof(int));

    if (array == NULL || arrayData == NULL) {
        printf("Erreur d'allocation de mémoire.\n");
        exit(EXIT_FAILURE);
    }

    memset(arrayData, 0, n * m);

    size_t i;
    for (i = 0; i < n; i++)
        array[i] = arrayData + i * m;

    return array;
}

void freeIntArray(int **array) {
    free(array[0]);  // Libération des données
    free(array);     // Libération des pointeurs de lignes
}

double distEuclSquare(double pattern[], double center[]) {
    double distance = 0.0;
    #pragma omp parallel for reduction(+:distance)
    for (int i = 0; i < Nv; i++) {
        double diff = pattern[i] - center[i];
        distance += diff * diff;
    }
    return distance;
}

int argMin(double array[], int length) {
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

void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, int ***z) {
    double error = 0.0;
    size_t i, j;

    double *local_y = (double *)malloc(Nc * Nv * sizeof(double));
    int *local_z = (int *)malloc(Nc * Nv * sizeof(int));

    if (local_y == NULL || local_z == NULL) {
        printf("Erreur d'allocation de mémoire.\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (i = 0; i < Nc * Nv; i++) {
        local_y[i] = 0.0;
        local_z[i] = 0;
    }

    #pragma omp parallel for private(i, j) reduction(+:error)
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
            int index = classes[i] * Nv + j;
            if (index >= 0 && index < Nc * Nv) {
                #pragma omp atomic update
                local_y[index] += patterns[i][j];
                #pragma omp atomic update
                local_z[index]++;
            } else {
                #pragma omp critical
                {
                    printf("Erreur : Accès hors limites pour local_y et local_z (1).\n");
                    printf("i = %zu, j = %zu, classes[i] = %d\n", i, j, classes[i]);
                    printf("index = %d, Nc = %d, Nv = %d\n", index, Nc, Nv);
                }
                exit(EXIT_FAILURE);
            }
        }
    }

    #pragma omp parallel for private(i, j)
    for (i = 0; i < Nc * Nv; i++) {
        int row = i / Nv;
        int col = i % Nv;
        int index = row * Nv + col;
        if (local_z[index] != 0) {
            (*y)[row][col] = local_y[index] / local_z[index];
        } else {
            // Réinitialiser au centre actuel pour éviter la division par zéro
            (*y)[row][col] = centers[row][col];
        }
        printf("Center[%zu][%zu]: %lf\n", row, col, (*y)[row][col]); // Ajout d'une sortie de débogage
    }

    free(local_y);
    free(local_z);

    printf("Recalculate Centers Successful\n"); // Ajout d'une sortie de débogage
}


