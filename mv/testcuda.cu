#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "MT.h"
#include <papi.h>

#define N 40000
/* #define RAND */
/* #define N 50000 */

double gettimeofday_sec(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec*1e-6;
}
__global__ void
mv1(int n, double *a, double *b, double *c){
  int i, j;
  for(i=0;i<n;i++){
    double tmp=0.0;
    for(j=0;j<n;j++){
      tmp += a[i*n+j]*b[j];
    }
    c[i]=tmp;
  }
}
__global__ void
mv2(int n, double *a, double *b, double *c){
  int row=blockDim.x * blockIdx.x + threadIdx.x;
  int i, j;
  if(row<n){
    for(i=0;i<n;i++){
      double tmp=0.0;
      for(j=0;j<n;j++){
        tmp += a[i*n+j]*b[j];
      }
      c[i]=tmp;
    }
  }
}
int main(int argc, char const* argv[])
{
  int i, j;
  double *a, *b, *c;
  double gamma=2.5;
  double st, et;
  double t1, t2, t3;
  double checksum1=0.0, checksum2=0.0, checksum3=0.0;

  a=(double *)malloc(sizeof(double)*N*N);
  b=(double *)malloc(sizeof(double)*N);
  c=(double *)malloc(sizeof(double)*N);

  double *da, *db, *dc;
  cudaMalloc((void **)&da, sizeof(double)*N*N);
  cudaMalloc((void **)&db, sizeof(double)*N);
  cudaMalloc((void **)&dc, sizeof(double)*N);

  init_genrand((unsigned)time(NULL));
  int ThreadPerBlock=1024;
  int BlockPerGrid=(N-1)/(ThreadPerBlock)+1;
#ifdef RAND
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      /* a[i][j]=genrand_real3(); */
      a[i*N+j]=genrand_real3();
    }
  }

  for(i=0;i<N;i++){
    b[i]=genrand_real3();
    c[i]=0.0;
  }
#else
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if(i==j){
        /* a[i][j]=gamma; */
        a[i*N+j]=gamma;
      }else if((j==i+1) || (j==i-1)){
        /* a[i][j]=-1.0; */
        a[i*N+j]=-1.0;
      }else{
        /* a[i][j]=0.0; */
        a[i*N+j]=0.0;
      }
    }
  }

  for(i=0;i<N;i++){
    b[i]=genrand_real3();
    c[i]=0.0;
  }

#endif
  /* for(i=0;i<N;i++){ */
  /*   for(j=0;j<N;j++){ */
  /*     printf("%.3f ", a[i*N+j]); */
  /*   } */
  /*   putchar('\n'); */
  /* } */

  /*-------------------------------------------------*/
  st=gettimeofday_sec();

  for(i=0;i<N;i++){
    double tmp=0.0;
    for(j=0;j<N;j++){
      /* tmp += a[i][j]*b[j]; */
      tmp += a[i*N+j]*b[j];
    }
    c[i]=tmp;
  }

  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum1+=c[i];
  }
  t1=et-st;
  /*-------------------------------------------------*/
  st=gettimeofday_sec();

  cudaMemcpy(da, a, sizeof(double)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice);
  mv2<<<BlockPerGrid, ThreadPerBlock>>>(N, da, db, dc);
  cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost);
  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum2+=c[i];
  }
  t2=et-st;

  /*-------------------------------------------------*/
st=gettimeofday_sec();

  cudaMemcpy(da, a, sizeof(double)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice);
  mv2<<<BlockPerGrid, ThreadPerBlock>>>(N, da, db, dc);
  cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost);
  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum3+=c[i];
  }
  t3=et-st;


  printf("c1=%f\nc2=%f\n",checksum1,checksum2);
  printf("c3=%f\n",checksum3);
  printf("CPU time=%f \n", t1);
  printf("CUDA1 time=%f \n", t2);
  printf("CUDA2 time=%f \n", t3);
  /* for(i=0;i<N;i++){ */
  /*   free(a[i]); */
  /* } */
  free(a);
  free(b);
  free(c);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}
