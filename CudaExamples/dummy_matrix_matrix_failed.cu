#include "cuda_runtime.h"
#include <stdio.h>
#include<unistd.h>


// O ponto desse exercicio é forçar a manipulação de um problema utilizando duas
// duas dimensões do grid; 
// Lembre-se que o numero de threads, deve ser sempre uma constante multipla de 32
// aredonda-da para cima via ceil, de modo a evitar indices não mapeados.


// Usando sempre matriz quadradas para evitar mapear numero de colunas e de linhas 
// de cada matriz. 
const int N = 2 ;// Numero de linhas da matriz
const int M = 2 ; // Numero de colunas da matriz


// R = M X N 
__global__ void multiplica(float* ma, float* mb, float* mc, int width) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Isso torna possível trabalhar apenas matrizes quadradas N x N 
    if(row < 2 && col < 2){
        // Numero de colunas de Ma deve ser igual o numero de colunas de MB (4x4)
        // Multiplicação de vetores.
        float value = 0;
        for(int k = 0; k < width; k++){
            value += ma[row * width + k] * mb[k * width + col];
        }

        printf("Value %f \n", value);
        mc[row*width + col] = value;
        printf("mc[%d] \n", row*width + col);
    }
}


// Só vai entender no papel.
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
            ma[i * M + j] = 2;
            mb[i * M + j] = 2;
        }
    }

    // printf("Inited matrix A \n");
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0; j < M; j++){
    //         printf("R[%d][%d] = %f \n", i,j, ma[i * M + j]);
    //     }
    // }

    // printf("Inited matrix B \n");
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0; j < M; j++){
    //         printf("R[%d][%d] = %f \n", i,j, mb[i * M + j]);
    //     }
    // }

    cudaMemcpy(dma, ma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dmb, mb, size, cudaMemcpyHostToDevice);

    multiplica<<<2,2>>>(dma, dmb, dmc, N*M);
    cudaDeviceSynchronize(); // Always before cudaMemcpy  device -> host (u can get crap if not this way)
    cudaMemcpy(mc, dmc, size, cudaMemcpyDeviceToHost);
    

    // Cuda kernels does not throw exceptions, ALWAYS CHECK FOR ERROR
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    cudaFree(dma); cudaFree(dmb); cudaFree(dmc);

    printf("Result Matrix C \n");    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++){
            printf("R[%d][%d] = %f \n", i,j, mc[i * M + j]);
        }
    }
    
    return 0;
}
