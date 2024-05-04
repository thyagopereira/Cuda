#include "cuda_runtime.h"
#include "stdio.h"
#include<unistd.h>

// Kernel of GPU
__global__ void meuKernel(){
    printf("Hello from GPU \n"); 
}

// CPU CODE
int main(){

    meuKernel<<<3,10>>>();
    printf("Hello World");
    sleep(4);
    return 0;
}