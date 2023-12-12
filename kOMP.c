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

double *mallocArray(double ***array, int n, int m, int initialize);
void freeArray(double ***array, double *arrayData);

void kMeans(double patterns[][Nv], double centers[][Nv]);
void initialCenters(double patterns[][Nv], double centers[][Nv]);
double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances);
void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z);

double distEuclSquare(double pattern[], double center[]);
int argMin(double array[], int length);

void createRandomVectors(double patterns[][Nv]);

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];

    createRandomVectors(patterns);

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < Nc; i++) {
            initialCenters(patterns, centers);
        }

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
            patterns[i][j] = (double)(random() % 100) - 0.1059364 * (i + j);
        }
    }
}

void kMeans(double patterns[][Nv], double centers[][Nv]) {
    double error = INFINITY;
    double errorBefore;
    int step = 0;

    int classes[N];  // Ajout de la déclaration de la variable classes
    int local_classes[N];
    
    double **distances;
    double *distanceData = mallocArray(&distances, N, Nc, 0);

    double **y, **z;
    double *yData = mallocArray(&y, Nc, Nv, 1);
    double *zData = mallocArray(&z, Nc, Nv, 1);

    initialCenters(patterns, centers);

    do {
        errorBefore = error;
        #pragma omp parallel
        {
            double local_error = 0.0;

            #pragma omp for
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < Nc; j++)
                    distances[i][j] = distEuclSquare(patterns[i], centers[j]);
                local_classes[i] = argMin(distances[i], Nc);
                local_error += distances[i][local_classes[i]];
            }

            #pragma omp atomic update
            error += local_error;

            #pragma omp for
            for (size_t i = 0; i < N; i++) {
                classes[i] = local_classes[i];
            }
        }

        recalculateCenters(patterns, centers, classes, &y, &z);
        printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
        step++;

    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold));

    freeArray(&distances, distanceData);
    freeArray(&y, yData);
    freeArray(&z, zData);
    return;
}

double *mallocArray(double ***array, int n, int m, int initialize) {
    *array = (double **)malloc(n * sizeof(double *));
    double *arrayData = malloc(n * m * sizeof(double));

    if (initialize != 0)
        memset(arrayData, 0, n * m);

    size_t i;
    for (i = 0; i < n; i++)
        (*array)[i] = arrayData + i * m;

    return arrayData;
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
    return;
}

double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances) {
    double error = 0.0;

    #pragma omp parallel for reduction(+:error)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < Nc; j++)
            (*distances)[i][j] = distEuclSquare(patterns[i], centers[j]);
        classes[i] = argMin((*distances)[i], Nc);
        error += (*distances)[i][classes[i]];
    }

    return error;
}

void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z) {
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
            centers[row][col] = local_y[index] / local_z[index];
        } else {
            // Réinitialiser au centre actuel pour éviter la division par zéro
            centers[row][col] = centers[row][col];
        }
        printf("Center[%zu][%zu]: %lf\n", row, col, centers[row][col]); // Ajout d'une sortie de débogage
    }

    free(local_y);
    free(local_z);

    printf("Recalculate Centers Successful\n"); // Ajout d'une sortie de débogage

    return;
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

void freeArray(double ***array, double *arrayData) {
    free(arrayData);
    free(*array);
    return;
}

