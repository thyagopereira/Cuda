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


__global__ void k_Particle_non_coalescent(st_particle *d_vet, int size){

    int i = blockDim.x * blockIdx.x + threadIdx.x; // Posição da thread no bloco
    
    if(i < size){
        d_vet[i].p.x = d_vet[i].p.x + d_vet[i].v.x * d_vet[i].a.x;
        d_vet[i].p.y = d_vet[i].p.y + d_vet[i].v.y * d_vet[i].a.y;
        d_vet[i].p.z = d_vet[i].p.z + d_vet[i].v.z * d_vet[i].a.z;
    }
    // Obs Seria ainda mais rápido se as dimensões fossem tratadas separadamente.
    // bloco de threads;

}

__global__ void k_Particle_coalescent_data( float* d_px, float* d_py, float* d_pz,
                                            float* d_vx, float* d_vy, float* d_vz,
                                            float* d_ax, float* d_ay, float* d_az){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        d_px[i] = d_px[i] + d_vx[i] * d_ax[i];
        d_py[i] = d_py[i] + d_vx[i] * d_ax[i];
        d_pz[i] = d_pz[i] + d_vx[i] * d_ax[i];
    } 

}

void coalescent_data(){
    cudaDeviceReset();

    int size = sizeof(float) * N;
    float *m_px, *m_py, *m_pz; 
    float *m_vx, *m_vy, *m_vz;
    float *m_ax, *m_ay, *m_az;

    float *d_px, *d_py, *d_pz; 
    float *d_vx, *d_vy, *d_vz;
    float *d_ax, *d_ay, *d_az;

    m_px = (float *)malloc(size); m_py = (float *)malloc(size); m_pz =(float *)malloc(size);
    m_vx = (float *)malloc(size); m_vy = (float *)malloc(size); m_vz =(float *)malloc(size);
    m_ax = (float *)malloc(size); m_ay = (float *)malloc(size); m_az =(float *)malloc(size);

    for(int i = 0; i < N; i++){
        m_px[i] = 0; m_py[i] = 0; m_pz[i] = 0; 
        m_vx[i] = 10; m_vy[i] = 10; m_vz[i] = 5;  
        m_ax[i] = 10; m_ay[i] = 5; m_az[i] = 2;
    }

    cudaMalloc((void**)&d_px, size); cudaMalloc((void**)&d_py, size); cudaMalloc((void**)&d_pz, size);
    cudaMalloc((void**)&d_vx, size); cudaMalloc((void**)&d_vy, size); cudaMalloc((void**)&d_vz, size);
    cudaMalloc((void**)&d_ax, size); cudaMalloc((void**)&d_ay, size); cudaMalloc((void**)&d_az, size);

    cudaMemcpy(d_px, m_px, size, cudaMemcpyHostToDevice); cudaMemcpy(d_py, m_py, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz, m_pz, size, cudaMemcpyHostToDevice); cudaMemcpy(d_vx, m_vx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, m_vy, size, cudaMemcpyHostToDevice); cudaMemcpy(d_vz, m_vz, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ax, m_ax, size, cudaMemcpyHostToDevice); cudaMemcpy(d_ay, m_ay, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_az, m_az, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1, 1);
    blocksPerGrid.x = 1;
    if(N > 512) {
        threadsPerBlock.x = ceil(double(N)/ (512));
       
    }else{
        threadsPerBlock.x = N;
    }

    cudaEvent_t start, stop; cudaEventCreate (&start); cudaEventCreate (&stop);
    cudaEventRecord(start, 0);
    k_Particle_coalescent_data<<<blocksPerGrid, threadsPerBlock>>>( m_px, m_py, m_pz,
                                                                    m_vx, m_vy, m_vz,
                                                                    m_ax, m_ay, m_az );
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
    cudaEventDestroy(start); cudaEventDestroy(stop); 

    cudaMemcpy(m_px, d_px, size, cudaMemcpyDeviceToHost); cudaMemcpy(m_py, d_py,size, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_pz, d_pz, size, cudaMemcpyDeviceToHost);

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);

    printf("NON COALESCENT DATA");
    printf("Total GPU Time no coalescent data: %3.3f ms \n", elapsedTime);
    printf("------------------- COALESCENT DATA -------------------- \n");

    for(int i = 0; i < N ; i++){
        printf("px = %f , py = %f , pz = %f \n", m_px[i], m_py[i], m_pz[i]);
    }

    printf("---------------------------------------------------------- \n");
}

void non_coalescent_data(){
    cudaDeviceReset();

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
    dim3 threadsPerBlock(N,1);
    dim3 blocksPerGrid(1, 1);
    blocksPerGrid.x = 1;
    if(N > 512) {
        threadsPerBlock.x = ceil(double(N)/ (512));
       
    }else{
        threadsPerBlock.x = N;
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
    cudaEventDestroy(start); cudaEventDestroy(stop); 

    cudaMemcpy(h_vet, d_vet, stSize, cudaMemcpyDeviceToHost);
    cudaFree(d_vet);

    printf("NON COALESCENT DATA \n");
    printf("Total GPU Time no coalescent data: %3.3f ms \n", elapsedTime);
    printf("-------------------NON COALESCENT DATA -------------------- \n");

    for(int i = 0; i < N ; i++){
        printf("p.x = %f , p.y = %f , p.z = %f \n", h_vet[i].p.x, h_vet[i].p.y, h_vet[i].p.z);
    }
    printf("---------------------------------------------------------- \n");
}

int main(){
    non_coalescent_data();
    // coalescent_data();
    return 0;
}

