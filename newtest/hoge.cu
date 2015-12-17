// includes, system
#include <stdio.h>

// includes, project
/* #include <cuda.h> */

// some defines
#define VECTOR_SIZE 1024
#define BLOCK_SIZE 256

///////////////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void vectorInc(unsigned int* A)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int a;

  a = A[index];

  a++;

  A[index] = a;

  return;
}

////////////////////////////////////////////////////////////////////////////////
// Program Main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  // デバイスにフラグをセット
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // Host MemoryをMapped Memoryとして確保
  unsigned int* h_A;
  cudaHostAlloc((void**)&h_A, VECTOR_SIZE*sizeof(unsigned int), cudaHostAllocMapped);

  // 配列の初期化
  for( unsigned int i = 0; i < VECTOR_SIZE; i++) {
    h_A[i] = (unsigned int) i;
  }

  // Host MemoryをDevice Memoryにマップする
  unsigned int* d_A;
  cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);

  // Kernelを実行する
  vectorInc<<<VECTOR_SIZE/BLOCK_SIZE, BLOCK_SIZE>>>(d_A);

  // 結果の表示
  for( unsigned int i = 0; i < VECTOR_SIZE; i++) {
    printf("h_A[%d]:%d\n", i, h_A[i]);
  }

  // 領域の開放
  cudaFreeHost(h_A);

  cudaThreadExit();

  return 0;

}

