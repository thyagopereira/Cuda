#include "cuda_runtime.h"
#include <stdio.h>
#include<unistd.h>


// O ponto desse exercicio é forçar a manipulação de um problema utilizando duas
// duas dimensões do grid; 
// Lembre-se que o numero de threads, deve ser sempre uma constante multipla de 32
// aredonda-da para cima via ceil, de modo a evitar indices não mapeados.

const int N = 2 ;// Numero de linhas da matriz ; 
// Numero de colunas da matriz


// R = A X B  
__global__ void multiplica(float* ma, float* mb, float* mc, int width) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0;
    // Isso torna possível trabalhar apenas matrizes quadradas N x N 
    if(row < width && col < width){
        // Numero de colunas de Ma deve ser igual o numero de colunas de MB (4x4)
        // Multiplicação de vetores.
        for(int i = 0; i < N; i++){
            sum += ma[row * N + i] * mb[i * N + col];
        }
    }
    mc[row * N + col] = sum;
}


// Só vai entender no papel.
int main() {

    cudaDeviceReset();
    float *dma, *dmb, *dmc;
    float *ma, *mb, *mc;
    int size = N * N * sizeof(float); // Size of a NxN matriz


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
        for(int j = 0; j < N; j++){
            ma[i * N + j] = 2;
            mb[i * N + j] = 2;
        }
    }

    // Copy from host to device
    cudaMemcpy(dma, ma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dmb, mb, size, cudaMemcpyHostToDevice);


    // Setando dimensões do grid. 
    dim3 threadsPerBlock(N,N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512){
        threadsPerBlock.x = 512; 
        threadsPerBlock.y = 512; 
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }


    multiplica<<<blocksPerGrid, threadsPerBlock>>>(dma, dmb, dmc, N);
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
        for(int j = 0; j < N; j++){
            printf("R[%d][%d] = %f \n", i,j, mc[i * N + j]);
        }
    }
    
    return 0;
}


// Dúvida final, esse problema pode ser resolvido mais eficientemente se diminuirmos o dominio ?
// - Se em vez de cada threads computar a multiplicação de vetores, fizemos mais threads, e cada
// uma delas agora vai multiplicar valores, e acumular ? Removendo a execução do laço
