#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

// O ponto desse codigo é verificar o tempo de execução do kernel cuda.
// quando trazemos um kernel que recebe todos os dados necessarios, para
// a manipulação no mesmo warp, vs um kernel que so recebe os dados nece-
// ssários para a alocação em varios warps diferentes.


// Suponha uma struct que represente dados fisicos de uma particula, tridimensionalmente.
struct st_particle{
    // ---- Verifique que aqui temos 3 * sizeof(float) representando posição ----
    float3 p; // Posição no espaço; 
    // ---- Verifique que aqui temos 3 * sizeof(float) representando velocidade ----
    float3 v; // Componentes do vetor velocidade 
    // ---- Verifique que aqui temos 3 * sizeof(float) representando aceleração ----
    float3 a; // Componentes do vetor de aceleração

    // ------ Vamos encontrar a velocidade apenas após 9 alocações de memoria
    // Uma demanda que acrescenta mais warps, na medida que carrega menos dados da
    // memoria global para a da thread por warp. 
};

const int N = 10;
// Assume um movimento retilineo uniformemente variado MRUV
__global__ void k_Particle_non_coalescent(st_particle *d_vet, int size){

    int i = blockDim.x * blockIdx.x + threadIdx.x; // Posição da thread no bloco
    
    if(i < size){
        d_vet[i].p.x = d_vet[i].p.x + d_vet[i].v.x * d_vet[i].a.x;
        d_vet[i].p.y = d_vet[i].p.y + d_vet[i].v.y * d_vet[i].a.y;
        d_vet[i].p.z = d_vet[i].p.z + d_vet[i].v.z * d_vet[i].a.z;
    }

}

void non_coalescent_data(){
    int stSize = sizeof(struct st_particle);
    struct st_particle *d_vet = (struct st_particle *)malloc(stSize * N);
    struct st_particle *h_vet = (struct st_particle *)malloc(stSize * N);
    
    // Preenchendo o vetor;
    for(int i = 0; i < N; i++){
        h_vet[i].p.x = 0; h_vet[i].p.y = 0; h_vet[i].p.z = 0; 
        h_vet[i].v.x = 10; h_vet[i].v.y = 10; h_vet[i].a.y = 5;  
        h_vet[i].a.x = 10; h_vet[i].a.y = 10; h_vet[i].a.z = 2;
    }

    cudaMalloc((void**)&d_vet, stSize * N);
    cudaMemcpy(d_vet, h_vet, stSize * N, cudaMemcpyHostToDevice);
    
    // Setando as dimensões do grid
    dim3 threadsPerBlock(N, 1);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512){
        threadsPerBlock.x = 512; 
        threadsPerBlock.y = 512; 
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }

    cudaEvent_t start, stop; cudaEventCreate (&start); cudaEventCreate (&stop);

    cudaEventRecord(start, 0);
    k_Particle_non_coalescent<<<blocksPerGrid, threadsPerBlock>>>(d_vet, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total GPU Time no coalescent data: %3.3f ms \n", elapsedTime);
    cudaEventDestroy(start); cudaEventDestroy(start); 

    cudaMemcpy(h_vet, d_vet, stSize, cudaMemcpyDeviceToHost);
    cudaFree(d_vet);
    for(int i = 0; i < N ; i++){
        printf("p.x = %f , p.y = %f , p.z = %f", h_vet[i].p.x, h_vet[i].p.y, h_vet[i].p.z);
    }
}

int main(){
    non_coalescent_data();
    return 0;
}

