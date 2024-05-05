// Este exercicio Necessita passar dados CPU -> GPU
// IMPORTANTE --- A passagem de parametros da função do kernel é 
// sempre feita por referencia.

// Malloc() -> cudaMalloc() | Alocação de memoria cpu e gpu
// Free() -> cudaFree() | Liberação de memoria cpu e gpu
// memcpy() -> cudaMemcpy() | Copia de dados ram -> gpu

// PONTOS DE ATENÇÃO
//  ----- A importancia do cudaFree() é altíssima, a gpu
// não possui sistemas operacionais que entendam o conceito
// de garbage collector. A responsabilidade é toda do
// programador. 

// cudaDeviceReset(); | È uma função que tem como objetivo
// resetar todos os espaços da GPU (caso vc nao tenha certeza)
// que cudaFree() foi utilizado corretamente. 

#include "cuda_runtime.h"
#include "stdio.h"

// GPU Kernel for simple aritimetics | NO Paralelism
__global__ void add(int *a, int *b, int *c){
    *c = *a + *b;
}

int main(void){
    int a, b, c; // CPU
    int *d_a, *d_b, *d_c; // GPU
    int size = sizeof(int); // 32 bits? 64? (depende da arch)

    // Allocate space on device GPU
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    // Input values 
    a = 10;
    b = 20; 

    // CPU -> GPU data transference
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
        // THIS IS EXPANSIVE TOO MANY CLOCK CYCLES

    //Kernel execution single thread on GPU
    add<<<1,1>>>(d_a, d_b, d_c);

    //GPU -> CPU collecting results
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost); 
        // ALSO EXPANSIVE TOO MANY CLOCK CYCLES

    // Free memory from GPU
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 
    
    printf("%d", c);
    return 0;
}