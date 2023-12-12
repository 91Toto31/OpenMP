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
#define DEBUG 0  // Mettez à 1 pour activer les sorties de débogage

int **mallocIntArray(int n, int m);
double **mallocArray(int n, int m);
void freeIntArray(int **array);
void freeArray(double **array);
void createRandomVectors(double patterns[][Nv]);
void initialCenters(double patterns[][Nv], double centers[][Nv]);
double distEuclSquare(double pattern[], double center[]);
int argMin(double array[], int length);

void kMeans(double patterns[][Nv], double centers[][Nv]);
double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances);
void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, int ***z);

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];

    createRandomVectors(patterns);

    // Initialisation des centres en dehors de la boucle parallèle
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
        #pragma omp parallel
        {
            double local_error = findClosestCenters(patterns, centers, local_classes, &distances);

            #pragma omp atomic update
            error += local_error;

            #pragma omp for
            for (size_t i = 0; i < N; i++) {
                classes[i] = local_classes[i];
            }
        }

        recalculateCenters(patterns, centers, classes, &y, &z);

        #if DEBUG
        printf("Step:%d || Error:%lf\n", step, (errorBefore - error) / error);
        #endif

        step++;

    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold));

    freeArray(distances);
    freeArray(y);
    freeIntArray(z);
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

void freeArray(double **array) {
    free(array[0]);  // Libération des données
    free(array);     // Libération des pointeurs de lignes
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
    size_t i, j;

    double **local_y = mallocArray(Nc, Nv);
    int **local_z = mallocArray(Nc, Nv);

    #pragma omp parallel for collapse(2)
    for (i = 0; i < Nc; i++) {
        for (j = 0; j < Nv; j++) {
            local_y[i][j] = 0.0;
            local_z[i][j] = 0;
        }
    }

    #pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
            int index = classes[i] * Nv + j;
            if (index >= 0 && index < Nc * Nv) {
                #pragma omp atomic update
                local_y[classes[i]][j] += patterns[i][j];
                #pragma omp atomic update
                local_z[classes[i]][j]++;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (i = 0; i < Nc; i++) {
        for (j = 0; j < Nv; j++) {
            if (local_z[i][j] != 0) {
                centers[i][j] = local_y[i][j] / local_z[i][j];
            } else {
                // Réinitialiser au centre actuel pour éviter la division par zéro
                centers[i][j] = centers[i][j];
            }

            #if DEBUG
            printf("Center[%zu][%zu]: %lf\n", i, j, centers[i][j]);
            #endif
        }
    }

    freeArray(local_y);
    freeArray(local_z);
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

/*void freeArray(double ***array, double *arrayData) {
    free(arrayData);
    free(*array);
    return;
}*/

