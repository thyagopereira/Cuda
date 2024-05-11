#include "cuda_runtime.h"
#include <cuda.h>
#include <stdio.h>
#include<unistd.h>


// O ponto desse exercicio é forçar a manipulação de um problema utilizando duas
// duas dimensões do grid; 
// Lembre-se que o numero de threads, deve ser sempre uma constante multipla de 32
// aredonda-da para cima via ceil, de modo a evitar indices não mapeados.

const int N = 32 ;// Numero de linhas da matriz
const int M = 32 ; // Numero de colunas da matriz


// R = M X N 
__global__ void multiplica(float* ma, float* mb, float* mc, int width) {


    printf("CUDA IS RUNNING"); //  Nao printa pq ? --- tem tempo sleep(20);
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    printf("%d, %d", row, col);

    // Isso torna possível trabalhar apenas matrizes quadradas? 
    if((row < width) && (col <  width)){
        float Pvalue = 0;

        for(int k = 0; k < width; k++){
            Pvalue += ma[row*width + k] * mb[k*width+col];
        }

        mc[row*width + col] = Pvalue;
        printf("%f \n", mc[row*width + col]);
    }
}

int main() {

    cudaDeviceReset();
    float *dma, *dmb, *dmc;
    float *ma, *mb, *mc;
    int size = N * M * sizeof(float); // Size of a NxM matriz


    // Declarando matriz in memory do host -- Isso me permite setar valores nela  
    ma = (float*)malloc(size);
    mb = (float*)malloc(size);
    mc = (float*)malloc(size);

    // Declarando matrizes na memoria do device -- APENAS DECLARADO 
    cudaMalloc((void**)&dma, size); 
    cudaMalloc((void**)&dmb, size);
    cudaMalloc((void**)&dmc, size);

    // Preenchendo matriz com valores  
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++){
            ma[i * M + j] = i + j;
            mb[i * M + j] = i + j;
        }
    }

    cudaMemcpy(dma, ma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dmb, mb, size, cudaMemcpyHostToDevice);

    multiplica<<<32, 32>>>(dma, dmb, dmc, N);


    cudaMemcpy(mc, dmc, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dma); cudaFree(dmb); cudaFree(dmc);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++){
            printf("R[%d][%d] = %f \n", i,j, mc[i * M + j]);
        }
    }
    
    return 0;
}
