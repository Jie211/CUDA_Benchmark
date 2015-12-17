#include "cuda.cuh"

__global__ void
kernel(void){
}

void run(){
  kernel<<<1,1>>>();
}
