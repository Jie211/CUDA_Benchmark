#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "MT.h"

void GetHead(const char *bx, const char *col, const char *ptr, int *n, int *nnz)
{
  FILE *in1, *in2, *in3;

  if((in1 = fopen(bx, "r")) == NULL)
  {
    printf("** error in head %s file open **\n", bx);
    exit(-1);
  }

  if((in2 = fopen(col, "r")) == NULL)
  {
    printf("** error head %s file open **\n", col);
    exit(-1);
  }

  if((in3 = fopen(ptr, "r")) == NULL)
  {
    printf("** error head %s file open **\n", ptr);
    exit(-1);
  }
  int N11, N12, N21, N22, N31, N32;
  int NZ1, NZ2, NZ3;

  fscanf(in1, "%d %d %d\n", &N11, &N12, &NZ1);
  fscanf(in2, "%d %d %d\n", &N21, &N22, &NZ2);
  fscanf(in3, "%d %d %d\n", &N31, &N32, &NZ3);

  if(N11!=N12)
  {
    printf("** error in %s N!=M **\n", bx);
    exit(-1);
  }
  if(N21!=N22)
  {
    printf("** error in %s N!=M **\n", col);
    exit(-1);
  }
  if(N31!=N32)
  {
    printf("** error in %s N!=M **\n", ptr);
    exit(-1);
  }

  if(N11 != N21 || N21!=N31 || N31!=N11)
  {
    printf("** error N was not same in 3files **\n");
    exit(-1);
  }

  if(NZ1 != NZ2 || NZ2!=NZ3 || NZ3!=NZ1)
  {
    printf("** error NNZ was not same in 3files **\n");
    exit(-1);
  }
  *n = N11;
  *nnz = NZ1;

  fclose(in1);
  fclose(in2);
  fclose(in3);
}
void GetData(const char *file1, const char *file2, const char *file3, int *col, int *ptr, double *val, double *b, double *x, int N, int NZ)
{
  FILE *in1,*in2,*in3;
  if((in1 = fopen(file1, "r")) == NULL)
  {
    printf("** error %s file open **", file1);
    exit(0);
  }

  if((in2 = fopen(file2, "r")) == NULL)
  {
    printf("** error %s file open **", file2);
    exit(0);
  }

  if((in3 = fopen(file3, "r")) == NULL)
  {
    printf("** error %s file open **", file3);
    exit(0);
  }
  int getint;
  double getdouble, getdouble2;
  int skip1, skip2, skip3;

  fscanf(in1, "%d %d %d\n", &skip1, &skip2, &skip3);
  fscanf(in2, "%d %d %d\n", &skip1, &skip2, &skip3);
  fscanf(in3, "%d %d %d\n", &skip1, &skip2, &skip3);
  for(int i=0;i<NZ;i++)
  {
    fscanf(in1,"%d %le\n",&getint,&getdouble);
    col[i] = getint;
    val[i] = getdouble;
  }

  for(int i=0;i<N+1;i++)
  {
    fscanf(in2,"%d\n",&getint);
    ptr[i] = getint;
  }

  for(int i=0;i<N;i++)
  {
    fscanf(in3,"%le %le\n",&getdouble,&getdouble2);
    b[i] = getdouble;
    x[i] = getdouble2;
  }


  fclose(in1);
  fclose(in2);
  fclose(in3);
}
double gettimeofday_sec()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec*1e-6;
}
__global__ void
mv1(int n, double *val, int *col, int *ptr, double *b, double *c)
{
  long int i, j;
  for(i=0;i<n;i++){
    double tmp=0.0;
    for(j=ptr[i];j<ptr[i+1];j++){
      tmp+=val[j]*b[col[j]];
    }
    c[i]=tmp;
  }
  /* __syncthreads(); */
}
__global__ void
mv2(int n, double *val, int *col, int *ptr, double *b, double *c)
{
  long row=blockDim.x * blockIdx.x + threadIdx.x;
  long int i;
  if(row<n){
    double tmp=0.0;
    long int row_start=ptr[row];
    long int row_end=ptr[row+1];
    for(i=row_start;i<row_end;i++){
      tmp+=val[i]*b[col[i]];
    }
    /* __syncthreads(); */
    c[row]=tmp;
    /* printf("%d %.12e\n", row, c[row]); */
  }
  /* __syncthreads(); */
}
__global__ void
mv3(int n, double *val, int *col, int *ptr, double *b, double *c){
  /* extern __shared__ volatile double vals[]; */
  extern __shared__ double vals[];

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id/32;
  int lane = thread_id & (32 - 1);

  int row = warp_id;
  if(row<n)
  {
    int row_start = ptr[row];
    int row_end = ptr[row+1];

    vals[threadIdx.x] = 0.0;

    for(int jj = row_start+lane; jj<row_end; jj+=32)
    { 
      vals[threadIdx.x]+=val[jj] * b[col[jj]];
    }

    if(lane <16)
      vals[threadIdx.x] += vals[threadIdx.x +16];
    if(lane<8)
      vals[threadIdx.x] += vals[threadIdx.x + 8];
    if(lane<4)
      vals[threadIdx.x] += vals[threadIdx.x + 4];
    if(lane<2)
      vals[threadIdx.x] += vals[threadIdx.x + 2];
    if(lane<1)
      vals[threadIdx.x] += vals[threadIdx.x + 1];

    if(lane == 0){
      c[row] += vals[threadIdx.x];
    }
  }
}
int main(int argc, char const* argv[])
{
  int i, j, N, NNZ;

  cudaSetDeviceFlags(cudaDeviceMapHost);

  GetHead(argv[1], argv[2], argv[3], &N, &NNZ);

  printf("n=%d, nnz=%d\n", N, NNZ);
  printf("----------------------------------------------\n");
  double *val, *b, *c;
  int *col, *ptr;

  val=(double *)malloc(sizeof(double)*NNZ);
  col=(int *)malloc(sizeof(int)*NNZ);
  ptr=(int *)malloc(sizeof(int)*(N+1));
  b=(double *)malloc(sizeof(double)*N);
  c=(double *)malloc(sizeof(double)*N);

  double *dval, *db, *dc;
  int *dcol, *dptr;

  GetData(argv[1], argv[2], argv[3], col, ptr, val, b, c, N, NNZ);

  for(i=0;i<N;i++){
    b[i]=genrand_real3();
    c[i]=0.0;
  }
  //------------------------------------
  double st1, et1, t1, sum1=0.0;
  st1=gettimeofday_sec();
  for(i=0;i<N;i++){
    double tmp=0.0;
    for(j=ptr[i];j<ptr[i+1];j++){
      tmp+=val[j] * b[col[j]];
    }
    c[i]=tmp;
  }
  et1=gettimeofday_sec();
  t1=et1-st1;
  for(i=0;i<N;i++){
    sum1+=c[i];
    c[i]=0.0;
  }
  printf("sum1=%f,t1=%.12e\n",sum1,t1);

  //------------------------------------
  omp_set_num_threads(8);

  double st2, et2, t2, sum2=0.0;
  st2=gettimeofday_sec();
  double tmp_omp=0.0;
  
 #pragma omp parallel for private(j) reduction(+:tmp_omp) schedule(static) firstprivate(c, val, b) lastprivate(c)
  for(i=0;i<N;i++){
    tmp_omp=0.0;
    for(j=ptr[i];j<ptr[i+1];j++){
      tmp_omp+=val[j] * b[col[j]];
    }
    c[i]=tmp_omp;
  }
  et2=gettimeofday_sec();
  t2=et2-st2;
  for(i=0;i<N;i++){
    sum2+=c[i];
    c[i]=0.0;
  }
  printf("sum2=%f,t2=%.12e\n",sum2,t2);
  //------------------------------------
  checkCudaErrors( cudaMalloc((void **)&dval, sizeof(double)*NNZ) );
  checkCudaErrors( cudaMalloc((void **)&dcol, sizeof(int)*NNZ) );
  checkCudaErrors( cudaMalloc((void **)&dptr, sizeof(int)*(N+1)) );
  checkCudaErrors( cudaMalloc((void **)&db, sizeof(double)*N) );
  checkCudaErrors( cudaMalloc((void **)&dc, sizeof(double)*N) );

  double st3, et3, t3, sum3=0.0;
  st3=gettimeofday_sec();
  checkCudaErrors( cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dc, 0, sizeof(double)*N) );

  mv1<<<1, 1>>>(N, dval, dcol, dptr, db, dc);

  checkCudaErrors(cudaPeekAtLastError());


  checkCudaErrors( cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost) );

  et3=gettimeofday_sec();

  t3=et3-st3;
  for(i=0;i<N;i++){
    sum3+=c[i];
    c[i]=0.0;
  }
  printf("sum3=%f,t3=%.12e\n",sum3,t3);


   //------------------------------------
 
  double st4, et4, t4, sum4=0.0;
  st4=gettimeofday_sec();
  checkCudaErrors( cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dc, 0, sizeof(double)*N) );
  
  int ThreadPerBlock=960;
  int BlockPerGrid=ceil((double)N/(double)ThreadPerBlock);

  mv2<<<BlockPerGrid, ThreadPerBlock>>>(N, dval, dcol, dptr, db, dc);

  checkCudaErrors(cudaPeekAtLastError());


  checkCudaErrors( cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost) );

  et4=gettimeofday_sec();

  t4=et4-st4;
  for(i=0;i<N;i++){
    sum4+=c[i];
    c[i]=0.0;
  }
  printf("sum4=%f,t4=%.12e\n",sum4,t4);

 
  //--------------------------------------
  
  double st5, et5, t5, sum5=0.0;
  st5=gettimeofday_sec();
  checkCudaErrors( cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dc, 0, sizeof(double)*N) );
  
  ThreadPerBlock=960;
  /* BlockPerGrid=ceil((double)N/(double)ThreadPerBlock/32); */
  BlockPerGrid=(N-1)/(ThreadPerBlock/32)+1;

  mv3<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(N, dval, dcol, dptr, db, dc);

  checkCudaErrors(cudaPeekAtLastError());


  checkCudaErrors( cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost) );

  et5=gettimeofday_sec();

  t5=et5-st5;
  for(i=0;i<N;i++){
    sum5+=c[i];
    c[i]=0.0;
  }
  printf("sum5=%f,t5=%.12e\n",sum5,t5);

  //--------------------------------------------
  checkCudaErrors( cudaHostRegister(val, sizeof(double)*NNZ, cudaHostRegisterDefault) );
  checkCudaErrors( cudaHostRegister(col, sizeof(int)*NNZ, cudaHostRegisterDefault) );
  checkCudaErrors( cudaHostRegister(ptr, sizeof(int)*(N+1), cudaHostRegisterDefault) );
  checkCudaErrors( cudaHostRegister(b, sizeof(double)*N, cudaHostRegisterDefault) );
  checkCudaErrors( cudaHostRegister(c, sizeof(double)*N, cudaHostRegisterDefault) );

  double st6, et6, t6, sum6=0.0;
  st6=gettimeofday_sec();

  checkCudaErrors( cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dc, 0, sizeof(double)*N) );
  
  ThreadPerBlock=960;
  BlockPerGrid=(N-1)/(ThreadPerBlock/32)+1;

  mv3<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(N, dval, dcol, dptr, db, dc);

  checkCudaErrors(cudaPeekAtLastError());


  checkCudaErrors( cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost) );

  et6=gettimeofday_sec();

  t6=et6-st6;
  for(i=0;i<N;i++){
    sum6+=c[i];
    c[i]=0.0;
  }
  printf("sum6=%f,t6=%.12e\n",sum6,t6);
  
  checkCudaErrors( cudaHostUnregister(val) );
  checkCudaErrors( cudaHostUnregister(col) );
  checkCudaErrors( cudaHostUnregister(ptr) );
  checkCudaErrors( cudaHostUnregister(b) );
  checkCudaErrors( cudaHostUnregister(c) );

  //------------------------------------------
  double *val2, *b2, *c2;
  int *col2, *ptr2;


  checkCudaErrors( cudaHostAlloc((void **)&val2, sizeof(double)*NNZ, cudaHostAllocMapped)  );
  checkCudaErrors( cudaHostAlloc((void **)&col2, sizeof(int)*NNZ, cudaHostAllocMapped)  );
  checkCudaErrors( cudaHostAlloc((void **)&ptr2, sizeof(int)*(N+1), cudaHostAllocMapped)  );
  checkCudaErrors( cudaHostAlloc((void **)&b2, sizeof(double)*N, cudaHostAllocMapped)  );
  checkCudaErrors( cudaHostAlloc((void **)&c2, sizeof(double)*N, cudaHostAllocMapped)  );

  double *dval2, *db2, *dc2;
  int *dcol2, *dptr2;

  GetData(argv[1], argv[2], argv[3], col2, ptr2, val2, b2, c2, N, NNZ);

  for(i=0;i<N;i++){
    b2[i]=b[i];
    c2[i]=1.0;
  }

  checkCudaErrors( cudaHostGetDevicePointer( (void **)&dval2, (void *)val2, 0) );
  checkCudaErrors( cudaHostGetDevicePointer( (void **)&dcol2, (void *)col2, 0) );
  checkCudaErrors( cudaHostGetDevicePointer( (void **)&dptr2, (void *)ptr2, 0) );
  checkCudaErrors( cudaHostGetDevicePointer( (void **)&db2, (void *)b2, 0) );
  checkCudaErrors( cudaHostGetDevicePointer( (void **)&dc2, (void *)c2, 0) );

  double st7, et7, t7, sum7=0.0;
  st7=gettimeofday_sec();

  /* checkCudaErrors( cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice) ); */
  /* checkCudaErrors( cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice) ); */
  /* checkCudaErrors( cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice) ); */
  /* checkCudaErrors( cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice) ); */
  /* checkCudaErrors( cudaMemset(dc, 0, sizeof(double)*N) ); */
  
  ThreadPerBlock=960;
  BlockPerGrid=(N-1)/(ThreadPerBlock/32)+1;

  mv3<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(N, dval2, dcol2, dptr2, db2, dc2);

  checkCudaErrors(cudaPeekAtLastError());


  /* checkCudaErrors( cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost) ); */

  et7=gettimeofday_sec();

  t7=et7-st7;
  for(i=0;i<N;i++){
    sum7+=c2[i];
    c[i]=0.0;
  }
  printf("sum7=%f,t7=%.12e\n",sum7,t7);
  
  checkCudaErrors( cudaFreeHost(val2) );
  checkCudaErrors( cudaFreeHost(col2) );
  checkCudaErrors( cudaFreeHost(ptr2) );
  checkCudaErrors( cudaFreeHost(b2) );
  checkCudaErrors( cudaFreeHost(c2) );

  free(dval2);
  free(dcol2);
  free(dptr2);
  free(db2);
  free(dc2);




  checkCudaErrors( cudaFree(dval) );
  checkCudaErrors( cudaFree(dcol) );
  checkCudaErrors( cudaFree(dptr) );
  checkCudaErrors( cudaFree(db) );
  checkCudaErrors( cudaFree(dc) );


  free(dval);
  free(dcol);
  free(dptr);
  free(b);
  free(c);


  /* checkCudaErrors( cudaFree(dval) ); */
  /* checkCudaErrors( cudaFree(dcol) ); */
  /* checkCudaErrors( cudaFree(dptr) ); */
  /* checkCudaErrors( cudaFree(db) ); */
  /* checkCudaErrors( cudaFree(dc) ); */



  free(val);
  free(col);
  free(ptr);
  free(b);
  free(c);
  return 0;
}
