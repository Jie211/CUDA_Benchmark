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
  long int row=blockDim.x * blockIdx.x + threadIdx.x;
  long int i, j;
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
__global__ void
mv4(double *c, double *val, int *col, int *ptr, double *b, int n){
  extern __shared__ double vals[];

  int thread_id=blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id=thread_id/32;
  int lane=thread_id&(32-1);

  int row=warp_id;

  if(row < n){
    int row_start=ptr[row];
    int row_end=ptr[row+1];

    double sum=0.0;
    for(int jj=row_start+lane; jj<row_end ;jj+=32){
      sum+=val[jj]*b[col[jj]];
    }
    vals[threadIdx.x] = sum; 
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x +   8];
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x +   4];
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x +   2];
    sum = sum + vals[threadIdx.x+ 1];

    if(lane==0){
      c[row] += vals[threadIdx.x];
    }
  }

}
__global__ void
spmv_crs_kernel(const int num_rows, 
                        const int *ptr, 
                        const int *indices, 
                        const double *data, 
                        const double *x, 
                        double *y)
{
  extern __shared__ double vals[];

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id/32;
  int lane = thread_id & (32 - 1);

  int row = warp_id;
  if(row<num_rows)
  {
    int row_start = ptr[row];
    int row_end = ptr[row+1];

    vals[threadIdx.x] = 0.0;

    for(int jj = row_start+lane; jj<row_end; jj+=32)
    { 
      /* int2 const v = tex1Dfetch(texture_, indices[jj]); */
      vals[threadIdx.x]+=data[jj] * x[indices[jj]];
      /* vals[threadIdx.x]+=data[jj] * __hiloint2double(v.y, v.x); */
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
      y[row] += vals[threadIdx.x];
    }
  }

}

texture<int2, 1, cudaReadModeElementType> vec_tex;//Texture

__global__ void crs_gemv(int n, double *out,          
    double *mat, int *col, int *row){
  int i, j;//for
  int2 v;//vector
  int THREAD=32;
  __shared__ double share[32+16];

  int rn = row[blockIdx.x + 1] - row[blockIdx.x];
  int ro = row[blockIdx.x];

  //init shared memory 
  share[threadIdx.x] = 0.0;

  //mat * vec
  for(i=0; i<rn/THREAD; i++){
    j = i * THREAD + threadIdx.x;

    v = tex1Dfetch(vec_tex, col[ro+j]);

    share[threadIdx.x] += mat[ro+j] * __hiloint2double(v.y, v.x);
  }
  __syncthreads();

  //remaindar
  if(threadIdx.x < rn%THREAD){
    j = rn - (rn % THREAD) + threadIdx.x;

    v = tex1Dfetch(vec_tex, col[ro+j]);

    share[threadIdx.x] += mat[ro+j] * __hiloint2double(v.y, v.x);
  }
  __syncthreads();

  //reduce
  for(j=THREAD/2; j>31; j>>=1){
    if(threadIdx.x < j)
      share[threadIdx.x] += share[threadIdx.x + j];
    __syncthreads();
  }
  if(threadIdx.x < 16){//(32/2)
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

  //return
  if(threadIdx.x == 0)
    out[blockIdx.x] = share[0];     
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
  double t1, t11, t2, t3, t4, t5;
  double checksum1=0.0, checksum11=0.0, checksum2=0.0, checksum3=0.0, checksum4=0.0, checksum5=0.0;
  int ThreadPerBlock=1024;
  int BlockPerGrid=(N-1)/(ThreadPerBlock)+1;


  GetData(argv[1], argv[2], argv[3], col, ptr, val, b, c, N, NNZ);

  for(i=0;i<N;i++){
    /* b[i]=genrand_real3(); */
    b[i]=1.0;
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
    c[i]=0.0;
  }
  printf("checksum1  =%f\n",checksum1);
  printf("CPU\ttime=%.12e\n",t1);
/*-----------------------------------------------------------------*/
  omp_set_num_threads(4);
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
    c[i]=0.0;
  }
  printf("checksum1.1=%f\n",checksum11);
  printf("CPU\t+OpenMP time=%.12e\n",t11);
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
    c[i]=0.0;
  }
  t2=et-st;
  printf("checksum2  =%f\n",checksum2);
  printf("CUDA\t+origin time=%.12e\n",t2);

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
    c[i]=0.0;
  }
  t3=et-st;

  printf("checksum3  =%f\n",checksum3);
  printf("CUDA\t+block  time=%.12e\n",t3);
/*-----------------------------------------------------------------*/
  /* ThreadPerBlock = 128;  */
  /* BlockPerGrid=ceil((double)N/(double)ThreadPerBlock/32);  */
  /*  */
  /*  */
  /* st=gettimeofday_sec(); */
  /* cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice); */
  /* cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice); */
  /*  */
  /* cudaBindTexture(NULL, vec_tex, db, sizeof(double) * N); */
  /* crs_gemv<<<N, 32>>>(N, dc, dval, dcol, dptr); */
  /* cudaUnbindTexture(vec_tex); */
  /*  */
  /* cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost); */
  /*  */
  /* et=gettimeofday_sec(); */
  /* for(i=0;i<N;i++){ */
  /*   checksum4+=c[i]; */
  /*   c[i]=0.0; */
  /* } */
  /* t4=et-st; */
  /* printf("checksum4  =\t%f\n",checksum4); */
  /* printf("CUDA\t++shared memory time=\t%.12e\n",t4); */

/*-----------------------------------------------------------------*/
  /* cudaHostRegister(col, sizeof(int)*NNZ, cudaHostRegisterDefault); */
  /* cudaHostRegister(val, sizeof(double)*NNZ, cudaHostRegisterDefault); */
  /* cudaHostRegister(ptr, sizeof(int)*(N+1), cudaHostRegisterDefault); */
  /* cudaHostRegister(b, sizeof(double)*N, cudaHostRegisterDefault); */
  /* cudaHostRegister(c, sizeof(double)*NNZ, cudaHostRegisterDefault); */
  /*  */
  /* st=gettimeofday_sec(); */
  /* cudaMemcpy(dcol, col, sizeof(int)*NNZ, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(dval, val, sizeof(double)*NNZ, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(dptr, ptr, sizeof(int)*(N+1), cudaMemcpyHostToDevice); */
  /* cudaMemcpy(db, b, sizeof(double)*N, cudaMemcpyHostToDevice); */
  /*  */
  //mv3<<<BlockPerGrid, ThreadPerBlock>>>(N, dval, dcol, dptr, db, dc);
  /* cudaBindTexture(NULL, vec_tex, db, sizeof(double) * N); */
  /* crs_gemv<<<N, 128>>>(N, dc, dval, dcol, dptr); */
  /* cudaUnbindTexture(vec_tex); */
  /*  */
  /* cudaMemcpy(c, dc, sizeof(double)*N, cudaMemcpyDeviceToHost); */
  /* et=gettimeofday_sec(); */
  /*  */
  /* for(i=0;i<N;i++){ */
  /*   checksum5+=c[i]; */
  /*   c[i]=0.0; */
  /* } */
  /*  */
  /* t5=et-st; */
  /* cudaHostUnregister(col); */
  /* cudaHostUnregister(val); */
  /* cudaHostUnregister(ptr); */
  /* cudaHostUnregister(b); */
  /* cudaHostUnregister(c); */
  /* printf("checksum5  =\t%f\n",checksum5); */
  /* printf("CUDA\t++pinned memory time=\t%.12e\n",t5); */

  printf("----------------------------------------------\n");
  printf("----------------------------------------------\n");


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
