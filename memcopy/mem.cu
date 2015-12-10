#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
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
mv1(int n, double *val, int *col, int *ptr, double *b, double *c){
  int i, j;
 
  for(i=0;i<n;i++){
    double tmp=0.0;
    for(j=ptr[i];j<ptr[i+1];j++){
      tmp+=val[j] * b[col[j]];
    }
    c[i]=tmp;
  }
}
__global__ void
mv2(int n, double *val, int *col, int *ptr, double *b, double *c){
  int row=blockDim.x * blockIdx.x + threadIdx.x;
  int i, j;
  if(row<n){
    for(i=0;i<n;i++){
      double tmp=0.0;
      for(j=ptr[i];j<ptr[i+1];j++){
        tmp+=val[j] * b[col[j]];
      }
      c[i]=tmp;
    }
  }
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

  GetHead(argv[1], argv[2], argv[3], &N, &NNZ);

  printf("n=%d, nnz=%d\n", N, NNZ);
  printf("----------------------------------------------\n");
  double *val, *b, *c;
  int *col, *ptr;
  col=(int *)malloc(sizeof(int)*NNZ);
  val=(double *)malloc(sizeof(double)*NNZ);
  ptr=(int *)malloc(sizeof(int)*(N+1));
  b=(double *)malloc(sizeof(double)*N);
  c=(double *)malloc(sizeof(double)*N);

  double *dval, *db, *dc;
  int *dcol, *dptr;
  cudaMalloc((void **)&dcol, sizeof(int)*NNZ);
  cudaMalloc((void **)&dval, sizeof(double)*NNZ);
  cudaMalloc((void **)&dptr, sizeof(int)*(N+1));
  cudaMalloc((void **)&db, sizeof(double)*N);
  cudaMalloc((void **)&dc, sizeof(double)*N);


  init_genrand((unsigned)time(NULL));
  double st, et;
  double t1, t11, t2, t3, t4, t5, t6, t7;
  double checksum1=0.0, checksum11=0.0, checksum2=0.0, checksum3=0.0, checksum4=0.0, checksum5=0.0, checksum6=0.0, checksum7=0.0;
  int ThreadPerBlock=1024;
  int BlockPerGrid=(N-1)/(ThreadPerBlock)+1;


  GetData(argv[1], argv[2], argv[3], col, ptr, val, b, c, N, NNZ);

  for(i=0;i<N;i++){
    b[i]=genrand_real3();
    c[i]=0.0;
  }
/*-----------------------------------------------------------------*/
  st=gettimeofday_sec();
  for(i=0;i<N;i++){
    double tmp=0.0;
    for(j=ptr[i];j<ptr[i+1];j++){
      tmp+=val[j] * b[col[j]];
    }
    c[i]=tmp;
  }
  et=gettimeofday_sec();
  t1=et-st;

  for(i=0;i<N;i++){
    checksum1+=c[i];
  }
/*-----------------------------------------------------------------*/
  omp_set_num_threads(8);
  st=gettimeofday_sec();
  double tmp_omp=0.0;
#pragma omp parallel for private(j) reduction(+:tmp_omp) schedule(static) firstprivate(c, val, b) lastprivate(c)
  for(i=0;i<N;i++){
    tmp_omp=0.0;
    for(j=ptr[i];j<ptr[i+1];j++){
      tmp_omp+=val[j] * b[col[j]];
    }
    c[i]=tmp_omp;
  }
  et=gettimeofday_sec();
  t11=et-st;

  for(i=0;i<N;i++){
    checksum11+=c[i];
  }

/*-----------------------------------------------------------------*/

  st=gettimeofday_sec();
  cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice);

  mv1<<<1, 1>>>(N, dval, dcol, dptr, db, dc);

  cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost);
  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum2+=c[i];
  }
  t2=et-st;

/*-----------------------------------------------------------------*/

  st=gettimeofday_sec();
  cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice);

  mv2<<<BlockPerGrid, ThreadPerBlock>>>(N, dval, dcol, dptr, db, dc);

  cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost);
  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum3+=c[i];
  }
  t3=et-st;

/*-----------------------------------------------------------------*/

  st=gettimeofday_sec();
  cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice);

  mv3<<<BlockPerGrid, ThreadPerBlock>>>(N, dval, dcol, dptr, db, dc);

  cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost);
  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum4+=c[i];
  }
  t4=et-st;

/*-----------------------------------------------------------------*/
  cudaHostRegister(col, sizeof(int)*NNZ, cudaHostRegisterDefault);
  cudaHostRegister(val, sizeof(double)*NNZ, cudaHostRegisterDefault);
  cudaHostRegister(ptr, sizeof(int)*(N+1), cudaHostRegisterDefault);
  cudaHostRegister(b, sizeof(double)*N, cudaHostRegisterDefault);
  cudaHostRegister(c, sizeof(double)*NNZ, cudaHostRegisterDefault);

  st=gettimeofday_sec();
  cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice);

  mv3<<<BlockPerGrid, ThreadPerBlock>>>(N, dval, dcol, dptr, db, dc);

  cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost);
  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum5+=c[i];
  }
  t5=et-st;
  cudaHostUnregister(col);
  cudaHostUnregister(val);
  cudaHostUnregister(ptr);
  cudaHostUnregister(b);
  cudaHostUnregister(c);
/*-----------------------------------------------------------------*/
  cudaHostRegister(col, sizeof(int)*NNZ, cudaHostRegisterMapped);
  cudaHostRegister(val, sizeof(double)*NNZ, cudaHostRegisterMapped);
  cudaHostRegister(ptr, sizeof(int)*(N+1), cudaHostRegisterMapped);
  cudaHostRegister(b, sizeof(double)*N, cudaHostRegisterMapped);
  cudaHostRegister(c, sizeof(double)*N, cudaHostRegisterMapped);

  st=gettimeofday_sec();

  cudaHostGetDevicePointer((void **)&dcol, col, 0);
  cudaHostGetDevicePointer((void **)&dval, val, 0);
  cudaHostGetDevicePointer((void **)&dptr, ptr, 0);
  cudaHostGetDevicePointer((void **)&db, b, 0);
  cudaHostGetDevicePointer((void **)&dc, c, 0);


  mv3<<<BlockPerGrid, ThreadPerBlock>>>(N, dval, dcol, dptr, db, dc);

  /* cudamemcpy(c, dc, sizeof(double)*n, cudamemcpydevicetohost); */
  et=gettimeofday_sec();
  for(i=0;i<N;i++){
    checksum6+=c[i];
  }
  t6=et-st;
  cudaHostUnregister(col);
  cudaHostUnregister(val);
  cudaHostUnregister(ptr);
  cudaHostUnregister(b);
  cudaHostUnregister(c);

/*-----------------------------------------------------------------*/

  double *val2, *b2, *c2;
  int *col2, *ptr2;
  cudaMallocManaged(&col2, sizeof(int)*NNZ);
  cudaMallocManaged(&val2, sizeof(double)*NNZ);
  cudaMallocManaged(&ptr2, sizeof(int)*(N+1));
  cudaMallocManaged(&b2, sizeof(double)*N);
  cudaMallocManaged(&c2, sizeof(double)*N);

  st=gettimeofday_sec();

  mv3<<<BlockPerGrid, ThreadPerBlock>>>(N, dval, dcol, dptr, db, dc);
  cudaDeviceSynchronize();

  et=gettimeofday_sec();

  for(i=0;i<N;i++){
    checksum7+=c[i];
  }
  t7=et-st;

  cudaFree(col2);
  cudaFree(val2);
  cudaFree(ptr2);
  cudaFree(b2);
  cudaFree(c2);


  printf("checksum1  =\t%f\n",checksum1);
  printf("checksum1.1=\t%f\n",checksum11);
  printf("checksum2  =\t%f\n",checksum2);
  printf("checksum3  =\t%f\n",checksum3);
  printf("checksum4  =\t%f\n",checksum4);
  printf("checksum5  =\t%f\n",checksum5);
  printf("checksum6  =\t%f\n",checksum6);
  printf("checksum7  =\t%f\n",checksum7);
  printf("----------------------------------------------\n");
  printf("CPU\ttime=\t\t\t%.12e\n",t1);
  printf("CPU\t+OpenMP time=\t\t%.12e\n",t11);
  printf("----------------------------------------------\n");
  printf("CUDA\t+origin time=\t\t%.12e\n",t2);
  printf("CUDA\t+block  time=\t\t%.12e\n",t3);
  printf("CUDA\t++shared memory time=\t%.12e\n",t4);
  printf("CUDA\t++pinned memory time=\t%.12e\n",t5);
  printf("CUDA\t++mapped memory time=\t%.12e\n",t6);
  printf("CUDA\t++unified memory time=\t%.12e\n",t7);


  free(col);
  free(val);
  free(ptr);
  free(b);
  free(c);
  cudaFree(dcol);
  cudaFree(dval);
  cudaFree(dptr);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}
