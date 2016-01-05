#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "MT.h"

__global__ void dot_kernel(long int n, int ThreadPerBlock, double *a, double *b, double *c) {
  extern __shared__ double share[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;

    if(i < n)
      share[threadIdx.x] = a[i] * b[i];
    else
      share[threadIdx.x] = 0.0;
    __syncthreads();

    for(j=ThreadPerBlock/2; j>31; j>>=1){
      if(threadIdx.x < j)
        share[threadIdx.x] += share[threadIdx.x + j];
      __syncthreads();
    }
    if(threadIdx.x < 16){
      share[threadIdx.x] += share[threadIdx.x + 16];
      __syncthreads();
      share[threadIdx.x] += share[threadIdx.x + 8];
      __syncthreads();
      share[threadIdx.x] += share[threadIdx.x + 4];
      __syncthreads();
      share[threadIdx.x] += share[threadIdx.x + 2];
      __syncthreads();
      share[threadIdx.x] += share[threadIdx.x + 1];
    }
    __syncthreads();

    if(threadIdx.x == 0)
      c[blockIdx.x] = share[0];
}

__global__ void dot_kernel2(long int n, int ThreadPerBlock, double *a, double *b, double *c){
  extern __shared__ double tmp[];
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int loc_t = threadIdx.x;

  if(t<n)tmp[loc_t]=a[t]*b[t];
  __syncthreads();

  for(int stride=blockDim.x/2; stride>0; stride/=2){
    if(loc_t<stride)
      tmp[loc_t]+=tmp[loc_t+stride];
    __syncthreads();
  }

  if(threadIdx.x==0){
    c[blockIdx.x] = tmp[0];
  }
}
double gettimeofday_sec()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

double dot(long int N, double *a, double *b, int BlockPerGrid, int ThreadPerBlock){
  int i;
  double sum=0.0;

  double *tmp_h, *tmp_d;

  tmp_h=(double *)malloc(sizeof(double)*BlockPerGrid);
  checkCudaErrors( cudaMalloc((void **)&tmp_d, sizeof(double)*BlockPerGrid) );

  /* dot_kernel<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*ThreadPerBlock+16>>>(N, ThreadPerBlock, a, b, tmp_d); */
  dot_kernel2<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*ThreadPerBlock>>>(N, ThreadPerBlock, a, b, tmp_d);

  checkCudaErrors( cudaMemcpy(tmp_h, tmp_d, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost) );

  for(i=0;i<BlockPerGrid;i++){
    sum+=tmp_h[i];
  }

  free(tmp_h);
  checkCudaErrors( cudaFree(tmp_d) );

  return sum;
}

double dot2(long int N, double *a, double *b, int BlockPerGrid, int ThreadPerBlock){
  int i;
  double sum=0.0;

  double *tmp_h, *tmp_d;

  /* tmp_h=(double *)malloc(sizeof(double)*BlockPerGrid); */
  checkCudaErrors( cudaHostAlloc((void **)&tmp_h, sizeof(double)*BlockPerGrid, cudaHostAllocWriteCombined|cudaHostAllocMapped) );
  /* cudaMalloc((void **)&tmp_d, sizeof(double)*BlockPerGrid); */
  checkCudaErrors( cudaHostGetDevicePointer((void **)&tmp_d, (void *)tmp_h, 0) );

  dot_kernel<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*ThreadPerBlock+16>>>(N, ThreadPerBlock, a, b, tmp_d);
  /* dot_kernel2<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*ThreadPerBlock>>>(N, ThreadPerBlock, a, b, tmp_d); */

  /* cudaMemcpy(tmp_h, tmp_d, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost); */
  checkCudaErrors( cudaThreadSynchronize() );

  for(i=0;i<BlockPerGrid;i++){
    sum+=tmp_h[i];
  }

  checkCudaErrors( cudaFreeHost(tmp_h) );
  /* cudaFree(tmp_d); */

  return sum;
}

int main(int argc, char const* argv[])
{
  long int i,  N=10000000;


  checkCudaErrors( cudaSetDeviceFlags(cudaDeviceMapHost) );

  double *a, *b, sum=0.0;
  double sum2=0.0;
  double sum3=0.0;

  double st1, et1, t1=0.0;
  double st2, et2, t2=0.0;
  double st3, et3, t3=0.0;

  omp_set_num_threads(8);

  /* a=(double *)malloc(sizeof(double)*N); */
  /* b=(double *)malloc(sizeof(double)*N); */

  checkCudaErrors( cudaHostAlloc((void **)&a, sizeof(double)*N, cudaHostAllocWriteCombined|cudaHostAllocMapped) );
  checkCudaErrors( cudaHostAlloc((void **)&b, sizeof(double)*N, cudaHostAllocWriteCombined|cudaHostAllocMapped) );

  for(i=0;i<N;i++){
    b[i]=genrand_real3();
    a[i]=genrand_real3();
  }
//-------------------------------------------
  st1=gettimeofday_sec();
  for(i=0;i<N;i++){
    sum+=a[i]*b[i];
  }
  et1=gettimeofday_sec();
  t1=et1-st1;


//-------------------------------------------
  st2=gettimeofday_sec();
#pragma omp parallel for shared(N, a, b) private(i) reduction(+:sum2)
  for(i=0;i<N;i++){
    sum2=sum2+a[i]*b[i];
  }
  et2=gettimeofday_sec();
  t2=et2-st2;


//-------------------------------------------

  double *d_a, *d_b;
  /* cudaMalloc((void **)&d_a, sizeof(double)*N); */
  /* cudaMalloc((void **)&d_b, sizeof(double)*N); */

  st3=gettimeofday_sec();
  /* cudaMemcpy(d_a, a, sizeof(double)*N, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(d_b, b, sizeof(double)*N, cudaMemcpyHostToDevice); */

  checkCudaErrors( cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0) );
  checkCudaErrors( cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0) );

  int ThreadPerBlock=128;
  int BlockPerGrid=(N-1)/(ThreadPerBlock)+1;

  /* sum3=dot(N, d_a, d_b, BlockPerGrid, ThreadPerBlock); */
  sum3=dot2(N, d_a, d_b, BlockPerGrid, ThreadPerBlock);

  et3=gettimeofday_sec();
  t3=et3-st3;

  /* cudaFree(d_a); */
  /* cudaFree(d_b); */

//=========================================================
  printf("sum1=%.12e, t=%.12es\n", sum,t1);
  printf("sum2=%.12e, t=%.12es\n", sum2,t2);
  printf("sum3=%.12e, t=%.12es\n", sum3,t3);

  /* free(b); */
  /* free(a); */
  checkCudaErrors( cudaFreeHost(b) );
  checkCudaErrors( cudaFreeHost(a) );
  checkCudaErrors( cudaThreadExit() );
  return 0;
}
