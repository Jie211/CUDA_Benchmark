#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

double gettimeofday_sec()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

__global__ void
dot1(int n, double *output, double *x, double *y){
  int i;
  for(i=0;i<n;i++){
    output[i] = x[i] * y[i];
  }
}

__global__ void
dot2(int n, double *output, double *x, double *y){
  int i = blockIdx.x;
  output[i] = x[i] * y[i];
}

__global__ void
dot3(int n, double *output, double *x, double *y){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<n)
    output[i] = x[i] * y[i];
}

__global__ void dot5(double *x, double *y, double *z, int N)
{
  extern __shared__ double partials[];

  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int col_stride = blockDim.x * gridDim.x;

  unsigned int j, k;
  float sum;

  float A1, A2, A3, A4;
  float B1, B2, B3, B4;

  sum = 0;

  for (j = col; j < N; j += (col_stride * 4))
  {
    A1 = x[j];
    B1 = y[j];
    if ((j + col_stride) < N)
    {
      A2 = x[j + col_stride];
      B2 = y[j + col_stride];
    }
    else
    {
      A2 = 0;
      B2 = 0;
    }
    if ((j + col_stride + col_stride) < N)
    {
      A3 = x[j + col_stride + col_stride];
      B3 = y[j + col_stride + col_stride];
    }
    else
    {
      A3 = 0;
      B3 = 0;
    }
    if ((j + col_stride + col_stride + col_stride) < N)
    {
      A4 = x[j + col_stride + col_stride + col_stride];
      B4 = y[j + col_stride + col_stride + col_stride];
    }
    else
    {
      A4 = 0;
      B4 = 0;
    }

    sum += A1 * B1;
    sum += A2 * B2;
    sum += A3 * B3;
    sum += A4 * B4;
  }

  partials[threadIdx.x] = sum;

  for (k = 1; (k << 1) <= blockDim.x; k <<= 1)
  {
    __syncthreads();
    if (threadIdx.x + k < blockDim.x)
      partials[threadIdx.x] += partials[threadIdx.x + k];
  }

  if (threadIdx.x == 0)
  {
    z[blockIdx.x] = partials[0];
  }
}


int main(void){
  int i;
  /* int N = 50000000; */
  int N = 50000000;
  double *x, *y, *z;
  double sum1=0.0, sum2=0.0, sum3=0.0, sum4=0.0, sum5=0.0, sum7=0.0;
  double *d_x, *d_y, *d_z;
  double start1, end1, time1=0.0;
  double start2, end2, time2=0.0;
  double start3, end3, time3=0.0;
  double start4, end4, time4=0.0;
  double start5, end5, time5=0.0;
  double start7, end7, time7=0.0;

  omp_set_num_threads(8);
  srand((unsigned)time(NULL));

  x=(double *)malloc(sizeof(double) * N);
  y=(double *)malloc(sizeof(double) * N);
  z=(double *)malloc(sizeof(double) * N);

  cudaMalloc((void **)&d_x, sizeof(double) * N);
  cudaMalloc((void **)&d_y, sizeof(double) * N);
  cudaMalloc((void **)&d_z, sizeof(double) * N);

  //---------------------- initialization
  for(i=0;i<N;i++){
    double num = ((double)rand()/(double)RAND_MAX);
    x[i] = num;
    num = ((double)rand()/(double)RAND_MAX);
    y[i] = num;
  }
  memset(z, 0, sizeof(double)*N);

  //---------------------- normal
  start1 = gettimeofday_sec();
  for(i=0;i<N;i++){
    z[i] = x[i] * y[i];
  }
  for(i=0;i<N;i++){
    sum1 += z[i]; 
  }
  end1 = gettimeofday_sec();
  time1 = end1 - start1;

  //---------------------- openmp
  memset(z, 0, sizeof(double)*N);
  start2 = gettimeofday_sec();
#pragma omp parallel for shared(N, x, y) private(i) reduction(+:sum2)
  for(i=0;i<N;i++){
    sum2 = sum2 + x[i] * y[i];
  }
  end2 = gettimeofday_sec();
  time2 = end2 - start2;

  //---------------------- cuda 1thread
  start3 = gettimeofday_sec();
  cudaMemcpy(d_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(double) * N, cudaMemcpyHostToDevice);

  dot1<<<1, 1>>>(N, d_z, d_x, d_y);

  cudaMemcpy(z, d_z, sizeof(double) * N, cudaMemcpyDeviceToHost);
#pragma omp parallel for shared(N, z) private(i) reduction(+:sum3)
  for(i=0;i<N;i++){
    sum3 = sum3 + z[i];
  }
  end3 = gettimeofday_sec();
  time3 = end3 - start3;
  //---------------------- cuda 1 thread in N blocks
  start4 = gettimeofday_sec();
  cudaMemcpy(d_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(double) * N, cudaMemcpyHostToDevice);

  dot2<<<N, 1>>>(N, d_z, d_x, d_y);

  cudaMemcpy(z, d_z, sizeof(double) * N, cudaMemcpyDeviceToHost);
#pragma omp parallel for shared(N, z) private(i) reduction(+:sum4)
  for(i=0;i<N;i++){
    sum4 = sum4 + z[i];
  }
  end4 = gettimeofday_sec();
  time4 = end4 - start4;
  //---------------------- cuda 32 thread in N/32 blocks
  int block = (N-1)/(32+1);
  int thread = 32;

  start5 = gettimeofday_sec();
  cudaMemcpy(d_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(double) * N, cudaMemcpyHostToDevice);

  dot3<<<block, thread>>>(N, d_z, d_x, d_y);

  cudaMemcpy(z, d_z, sizeof(double) * N, cudaMemcpyDeviceToHost);
#pragma omp parallel for shared(N, z) private(i) reduction(+:sum5)
  for(i=0;i<N;i++){
    sum5 = sum5 + z[i];
  }
  end5 = gettimeofday_sec();
  time5 = end5 - start5;

  //---------------------- cuda reduction2
  thread = 16;
  block = 3*(1536/16);

  start7 = gettimeofday_sec();
  cudaMemcpy(d_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(double) * N, cudaMemcpyHostToDevice);

  dot5<<<block, thread, sizeof(double)*thread>>>(d_x, d_y, d_z, N);
  cudaThreadSynchronize();

  cudaMemcpy(z, d_z, sizeof(double)*block, cudaMemcpyDeviceToHost);
  for(i=0;i<block;i++){
    sum7 += z[i];
  }
  end7 = gettimeofday_sec();
  time7 = end7 - start7;


  //---------------------- 
  printf("sum1 = %.12f\n", sum1);
  printf("time1 = %.12f\n", time1);
  printf("----------------------\n");
  printf("sum2 = %.12f\n", sum2);
  printf("time2 = %.12f\n", time2);
  printf("----------------------\n");
  printf("sum3 = %.12f\n", sum3);
  printf("time3 = %.12f\n", time3);
  printf("----------------------\n");
  printf("sum4 = %.12f\n", sum4);
  printf("time4 = %.12f\n", time4);
  printf("----------------------\n");
  printf("sum5 = %.12f\n", sum5);
  printf("time5 = %.12f\n", time5);
  printf("----------------------\n");
  printf("sum7 = %.12f\n", sum7);
  printf("time7 = %.12f\n", time7);


  free(x);
  free(y);
  free(z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return 0;
}
