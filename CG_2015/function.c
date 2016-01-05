#include "header.h"

void
display_mat(const double *a, 
            const int n_data)
{
  int i, j;
  for(i = 0;i<n_data;i++)
  {
    for(j = 0;j<n_data;j++){
      printf("%4g\t", a[i*n_data+j]);
    }
  putchar('\n');
  }
}

void
display_vec(const double *a, 
            const int n_data)
{
  int i;
  for(i = 0;i<n_data;i++)
  {
    printf("[%d] = %12.6e\n",i , a[i]);
  }
}

void
list_vec_i(const int *a, 
            const int n_data)
{
  int i;
  for(i = 0;i<n_data;i++)
  {
    printf("%d ", a[i]);
  }
  putchar('\n');
}

void
list_vec_d(const double *a, 
            const int n_data)
{
  int i;
  for(i = 0;i<n_data;i++)
  {
    printf("%3.1f ", a[i]);
  }
  putchar('\n');
}

void
bxcreate(const double *a,
			  double *b,
			  double *x,
			  const int n_data)
{
	double *x_setup;
	double vxm;
	int i,j;
	x_setup=(double *)malloc(sizeof(double)*n_data);
	#ifdef EBUG
	printf("=start to create vec b vec x=\n");
	#endif
  for(i = 0;i<n_data;i++)
  {
    x_setup[i] = 1.0;
    x[i] = 1.0;
  }
  for(i = 0;i<n_data;i++)
  {
    vxm = 0.0;
    for(j = 0;j<n_data;j++)
    {
      vxm+=a[i*n_data+j]*x_setup[j];
    }
    b[i]=vxm;
  }
  #ifdef EBUG
  printf("vec x\n");
  list_vec_d(x,n_data);
  printf("vec b\n");
  list_vec_d(b,n_data);
  putchar('\n');
  printf("=end to create vec b vec x=\n");
  #endif

  free(x_setup);
}

void
CSR(const double *a,
    double *val,
    int *col,
    int *ptr,
    const int n_data,
    const int m)
{
  int i, j;
  int val_count = 0;
  int ptr_flag = 0;
  int ptr_count = 0;
#ifdef EBUG
	printf("\n=start to transfor CSR=\n");
#endif
  for(i = 0;i<n_data;i++)
  {
    for(j = 0;j<n_data;j++)
    {
      if(a[i*n_data+j]!=0.0){
        val[val_count] = a[i*n_data+j];
        col[val_count] = j;
        if(ptr_flag == 0)
        {
          ptr[ptr_count] = val_count;
          ptr_count++;
          ptr_flag = 1;
        }
        val_count++;
      }
    }
    ptr_flag = 0;
  }
  ptr[n_data] = m;

#ifdef EBUG
  printf("val = \n");
  list_vec_d(val,m);
  printf("col = \n");
  list_vec_i(col,m);
  printf("ptr = \n");
  list_vec_i(ptr,n_data+1);
  printf("=end to transfor CSR=\n");
#endif
}

void 
usage(const char *a,const char *b,const char *x, int *ndata, int *m)
{
  FILE *matrix;
  FILE *bvector;
  FILE *xvector;
  
  if((matrix = fopen(a, "r")) == NULL)
  {
    printf("%s file open error\n", a);
  }
  if((bvector = fopen(b, "r")) == NULL)
  {
    printf("%s file open error\n", b);
  }
  if((xvector = fopen(x, "r")) == NULL)
  {
    printf("%s file open error\n", x);
  }
  
  int a_n1, a_n2, b_n, x_n;
  int a_m;
  int b_col, x_col;

  fscanf(matrix, "%d %d %d", &a_n1, &a_n2, &a_m);
  fscanf(bvector, "%d %d", &b_n, &b_col);
  fscanf(xvector, "%d %d", &x_n, &x_col);

  if(a_n1 == a_n2)
    printf("col = row ~~~~ pass\n");
  else{
    printf("col != row error\n");
    exit(0);
  }
  if(a_n1 == b_n && b_n == x_n )
    printf("data N ~~~~ pass\n");
  else{
    printf("N not same error\n");
    exit(0);
  }
  if(b_col  == 1)
    printf("b col ~~~~ pass\n");
  else{
    printf("b col != 1 error\n");
    exit(0);
  }
  if(x_col == 1)
    printf("x col ~~~~ pass\n");
  else{
    printf("x col != 1 error\n");
    exit(0);
  }

  printf("all pass\n");
  *m = a_m;
  *ndata = a_n1;

  fclose(matrix);
  fclose(bvector);
  fclose(xvector);

}

void getmatrix(const char *file, double *a, int ndata, int m)
{
  FILE *matrix;
  int i, j;
  int Indx, Indy;
  double val;

  if((matrix = fopen(file, "r")) == NULL){
    printf("%s file open error\n", file);
    exit(0);
  }

  int hoge1, hoge2, hoge3;
  fscanf(matrix, "%d %d %d", &hoge1, &hoge2, &hoge3);

  for(i = 0;i<ndata;i++)
    for(j = 0;j<ndata;j++)
      a[i*ndata+j] = 0.0;
  for(i = 0;i<m;i++)
  {
    fscanf(matrix, "%d %d %le", &Indx, &Indy, &val);
    a[(Indx - 1)*ndata+(Indy - 1)] = val;
  }

  fclose(matrix);
}

void getvector(const char *file, double *v, int ndata)
{
  FILE *vector;
  int i;
  double val;

  if((vector = fopen(file, "r")) == NULL){
    printf("%s file open error\n", file);
    exit(0);
  }

  for(i = 0;i<ndata;i++)
    v[i] = 0.0
      ;
  int hoge1, hoge2;
  fscanf(vector, "%d %d", &hoge1, &hoge2);

  for(i = 0;i<ndata;i++)
  {
    fscanf(vector, "%le", &val);
    v[i] = val;
  }

  fclose(vector);
}

void gethead(const char *file1, const char *file2, const char *file3, int *N, int *NZ)
{
  FILE *in1, *in2, *in3;

  if((in1 = fopen(file1, "r")) == NULL)
  {
    printf("%s file open error", file1);
    exit(0);
  }
  
  if((in2 = fopen(file2, "r")) == NULL)
  {
    printf("%s file open error", file2);
    exit(0);
  }
  
  if((in3 = fopen(file3, "r")) == NULL)
  {
    printf("%s file open error", file3);
    exit(0);
  }

  int N11, N12, N21, N22, N31, N32;
  int NZ1, NZ2, NZ3;

  fscanf(in1, "%d %d %d\n", &N11, &N12, &NZ1);
  fscanf(in2, "%d %d %d\n", &N21, &N22, &NZ2);
  fscanf(in3, "%d %d %d\n", &N31, &N32, &NZ3);

  if(N11!=N12)
  {
    printf("in %s N!=M \n", file1);
    exit(0);
  }
  if(N21!=N22)
  {
    printf("in %s N!=M \n", file2);
    exit(0);
  }
  if(N31!=N32)
  {
    printf("in %s N!=M \n", file3);
    exit(0);
  }

  if(N11 != N21 || N21!=N31 || N31!=N11)
  {
    printf("N was not same in 3files\n");
    exit(0);
  }

  if(NZ1 != NZ2 || NZ2!=NZ3 || NZ3!=NZ1)
  {
    printf("NZ was not same in 3files\n");
    exit(0);
  }

  *N = N11;
  *NZ = NZ1;

  fclose(in1);
  fclose(in2);
  fclose(in3);
}
void getdata(const char *file1, const char *file2, const char *file3,int *col,int *ptr,double *val,double *b,double *x, int N, int NZ)
{
  FILE *in1,*in2,*in3;
  
  if((in1 = fopen(file1, "r")) == NULL)
  {
    printf("%s file open error", file1);
    exit(0);
  }
  
  if((in2 = fopen(file2, "r")) == NULL)
  {
    printf("%s file open error", file2);
    exit(0);
  }
  
  if((in3 = fopen(file3, "r")) == NULL)
  {
    printf("%s file open error", file3);
    exit(0);
  }
  int getint;
  double getdouble, getdouble2;
  int skip1, skip2, skip3;

  fscanf(in1, "%d %d %d\n", &skip1, &skip2, &skip3);
  printf("%d %d %d\n", skip1, skip2, skip3);
  fscanf(in2, "%d %d %d\n", &skip1, &skip2, &skip3);
  printf("%d %d %d\n", skip1, skip2, skip3);
  fscanf(in3, "%d %d %d\n", &skip1, &skip2, &skip3);
  printf("%d %d %d\n", skip1, skip2, skip3);
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
  return tv.tv_sec+(double)tv.tv_usec*1e-6;
}
