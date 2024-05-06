#include "cuda_runtime.h"
#include  <cuda.h> 
#include <stdio.h>
#include<unistd.h>


const int N = 256;

__global__  void produtoEscalar(int* a, int b, int* c){
    int i = threadIdx.x ;
    c[i] = a[i] * b;
}

int main(){
    cudaDeviceReset();
    int *a, b, *c, *d_a, *d_c;

    int aSize = N * sizeof(int);
    a = (int*)malloc(aSize);
    c = (int*)malloc(aSize);

    cudaMalloc((void**)&d_a,aSize);
    cudaMalloc((void**)&d_c,aSize);

    b = 5;
    for(int i = 0; i < N; i++){
        a[i] = i;
    }

    cudaMemcpy(d_a, a, aSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, aSize, cudaMemcpyHostToDevice);

    produtoEscalar<<<1, N>>>(d_a, b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, aSize, cudaMemcpyDeviceToHost);

    printf("Resultado escalar: \n");
    for(int i = 0; i < N; i++){
        printf("%d \n", c[i]);
    }

    cudaFree(d_a); cudaFree(d_c);
    return 0;
}