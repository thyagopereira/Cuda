// This is a paralell version of a vector sum
#include "cuda_runtime.h"
#include <cuda.h>
#include <stdio.h>

int *a, *b, *c; // Host data

// Execute vector sum in paralell
__global__ void vecAdd(int* a, int* b, int* c){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Code to run in cpu
int main(){
    cudaDeviceReset();
    int *d_a, *d_b, *d_c;
    int N = 256; 
    int size = N * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size); //Espaço de memoriade um array
    c = (int*)malloc(size); 

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    for(int i = 0; i < N; i++){
        a[i] = i; b[i] = i;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    vecAdd<<<1,N>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Resultado da soma: \n");
    for(int i = 0; i < N; i++){
        printf("%d \n", c[i]);
    }
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}