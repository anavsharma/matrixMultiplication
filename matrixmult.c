#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

const xSize = 1000;
const ySize = 100;

void matrixMult(int **m1, int **m2, int **result, int x, int y){
    result[x][y] = 0;
    int i;
    for(i = 0; i < ySize; i++){
        result[x][y] += m1[x][i]*m2[i][y];
    }
}

void printMatrix(int x, int y, int **mat){
    int i, j;
    for(i = 0; i < x; i++){
        for(j = 0; j < y; j++){
            printf("%d\t", mat[i][j]);
        }
        printf("\n");
    }

    printf("\n\n");
}

int** initMatrix(int x, int y){
    //int len = sizeof(int *) * xSize + sizeof(int) * xSize * ySize;
    int **mat = (int **)malloc(x * sizeof(int *));
    int i;
    for(i = 0; i < x; i++)
        mat[i] = (int *)malloc(y * sizeof(int));

    int j;
    for(i = 0; i < x; i++){
        for(j = 0; j < y; j++){
            mat[i][j] = rand() % 50;
        }
    }

    return mat;
}

int main(int argc, char **argv){
    //Init random number gen
    int seed = 1;//time(0);
    srand(seed);

    printf("init rng, now init matrices\n");

    //init random array of size based on args
    int **matrix1 = initMatrix(xSize, ySize);  //int matrix1[xSize][ySize];
    int **matrix2 = initMatrix(ySize, xSize);  //[ySize][xSize];
    int **result = initMatrix(xSize, xSize);   //[xSize][xSize];

    //int lenm1 = sizeof(int *) * xSize + sizeof(int) * xSize * ySize;
    //matrix1 = (int **)malloc(lenm1);
    //printMatrix(xSize, ySize, matrix1);
    //printMatrix(ySize, xSize, matrix2);
    //printMatrix(xSize, xSize, result);

    //Configure row pointers of each matrix
    

    printf("initialized matrices, filling them now\n");

    printf("About to calculate multiplication!\n\n");

    struct timeval start, end;

    gettimeofday(&start, NULL);

    //Call multiplication on each index in matrix
    int i, j;
    #pragma omp parallel for
    for(i = 0; i < xSize; i++){
        //#pragma omp parallel for
        for(j = 0; j < xSize; j++){
            matrixMult(matrix1, matrix2, result, i, j);
        }
    }

    gettimeofday(&end, NULL);
    printf("Microseconds taken: %lu\n", end.tv_usec - start.tv_usec);

    //Print everything
    //printMatrix(xSize, ySize, matrix1);

    printf("\n\n");

    //printMatrix(ySize, xSize, matrix2);

    printf("\n\n");

    //printMatrix(xSize, xSize, result);

    return 0;

}