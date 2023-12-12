#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define N 100000 // N is the number of patterns
#define Nc 100    // Nc is the number of classes or centers
#define Nv 1000   // Nv is the length of each pattern (vector)
#define Maxiters 15   // Maxiters is the maximum number of iterations
#define Threshold 0.000001

// ... (autres déclarations et définitions)

double *mallocArray(double ***array, int n, int m, int initialize);
void freeArray(double ***array, double *arrayData);

void kMeans(double patterns[][Nv], double centers[][Nv]);
void initialCenters(double patterns[][Nv], double centers[][Nv]);
double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances);
void recalculateCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z);

double distEucl(double pattern[], double center[]);
int argMin(double array[], int length);

void createRandomVectors(double patterns[][Nv]);

int main(int argc, char *argv[]) {
    static double patterns[N][Nv];
    static double centers[Nc][Nv];

    createRandomVectors(patterns);

    // Utilisation de la directive OpenMP pour paralléliser kMeans
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

    // class or category of each pattern
    int *classes = (int *)malloc(N * sizeof(int));
    // distances between patterns and centers
    double **distances;
    double *distanceData = mallocArray(&distances, N, Nc, 0);
    // tmp data for recalculating centers
    double **y, **z;
    double *yData = mallocArray(&y, Nc, Nv, 1);
    double *zData = mallocArray(&z, Nc, Nv, 1);

    initialCenters(patterns, centers); //step 1
    do {
        errorBefore = error;
        error = findClosestCenters(patterns, centers, classes, &distances); // step 2
        recalculateCenters(patterns, centers, classes, &y, &z); // step 3
        printf("Step:%d||Error:%lf,\n", step, (errorBefore - error) / error);
        step++;
    } while ((step < Maxiters) && ((errorBefore - error) / error > Threshold)); // step 4

    free(classes);
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
        // split patterns in Nc blocks of N/Nc length
        // use rand and % to pick a random number of each block.
        centerIndex = rand() % (N / Nc * (i + 1) - N / Nc * i + 1) + N / Nc * i;
        for (j = 0; j < Nv; j++) {
            centers[i][j] = patterns[centerIndex][j];
        }
    }
    return;
}

double findClosestCenters(double patterns[][Nv], double centers[][Nv], int classes[], double ***distances) {
    double error = 0.0;
    size_t i, j;
    #pragma omp parallel for private(j) reduction(+:error)
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nc; j++)
            (*distances)[i][j] = distEucl(patterns[i], centers[j]);
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

    // Initialize local_y and local_z
    #pragma omp parallel for
    for (i = 0; i < Nc * Nv; i++) {
        local_y[i] = 0.0;
        local_z[i] = 0;
    }

    // Calculate tmp arrays
    #pragma omp parallel for private(j) reduction(+:error)
    for (i = 0; i < N; i++) {
        for (j = 0; j < Nv; j++) {
            #pragma omp atomic update
            local_y[classes[i] * Nv + j] += patterns[i][j];
            #pragma omp atomic update
            local_z[classes[i] * Nv + j]++;
        }
    }

    // Update step of centers
    #pragma omp parallel for private(j)
    for (i = 0; i < Nc; i++) {
        for (j = 0; j < Nv; j++) {
            // Check if divisor is zero to avoid division by zero
            if (local_z[i * Nv + j] != 0) {
                centers[i][j] = local_y[i * Nv + j] / local_z[i * Nv + j];
            } else {
                // Avoid division by zero, keep the previous value of centers
                centers[i][j] = centers[i][j];
            }

            // Reset local_y and local_z
            local_y[i * Nv + j] = 0.0;
            local_z[i * Nv + j] = 0;
        }
    }

    free(local_y);
    free(local_z);

    return;
}


double distEucl(double pattern[], double center[]) {
    double distance = 0.0;

    #pragma omp parallel for reduction(+:distance)
    for (int i = 0; i < Nv; i++) {
        double diff = pattern[i] - center[i];
        distance += diff * diff;
    }

    return sqrt(distance);
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
