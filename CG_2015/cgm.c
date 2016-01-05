#include "header.h"

void
cgm_CRS(double *val,
    const int *col,
    const int *ptr,
    double *bvec,
    double *xvec,
    const int ndata,
    const double eps,
    const int i_max,
    const int m)
{
  int i, n;
  int j;
#ifdef EBUG
  FILE *p_x;
  FILE *p_his;
  double baxk_norm = 0;
  double bax0_norm = 0;
  double true_r = 0;
  double true_r2 = 0;
#endif
  double *pvec, *apvec, *axvec, *rvec;
  double alpha, beta, r_norm_old, r_norm_new, pap_norm, b_norm;

  double tmp=0.0;


  // /*探索方向ベクトル*/
   pvec  = (double *)malloc(sizeof(double)*ndata);
   apvec = (double *)malloc(sizeof(double)*ndata);
  // ベクトル行列積
   axvec = (double *)malloc(sizeof(double)*ndata);
  // /*k回目の近似値xの残差*/
   rvec  = (double *)malloc(sizeof(double)*ndata);

#ifdef EBUG
if((p_x = fopen("./output/x.txt", "w")) == NULL)
{
  printf("x.txt open error\n");
  exit(0);
}
if((p_his = fopen("./output/his.txt", "w")) == NULL)
{
  printf("his.txt open error\n");
  exit(0);
}
#endif
#ifdef EBUG
  double st, et;
  st = gettimeofday_sec();
#endif
  //
  //Setting initial values
  //
  for (i=0; i<ndata; i++){
    xvec[i] = 0.0;
    axvec[i] = 0.0;
    pvec[i] = 0.0;
  }

  r_norm_old  = 0.0;
  r_norm_new = 0.0;
  b_norm = 0.0;
#pragma omp parallel for private(j) reduction(+:tmp) schedule(static) firstprivate(axvec, val, xvec) lastprivate(axvec) 
   for(i = 0;i<ndata;i++){
     tmp=0.0;
     for(j = ptr[i];j<ptr[i+1];j++)
     {
       tmp+=val[j]*xvec[col[j]];
       axvec[i]=tmp;
     }
   }

  for (i=0; i<ndata; i++){
    rvec[i] = bvec[i] - axvec[i];
    b_norm += bvec[i] * bvec[i];
    r_norm_old  += rvec[i]*rvec[i];
  }
  beta = 0.0;
  //
  // Main loop
  //

  n = 0;
  while(n<i_max){
    for (i=0; i<ndata; i++){
      pvec[i] = rvec[i] + beta * pvec[i];
      apvec[i] = 0.0;
    }
    pap_norm = 0.0;
#pragma omp parallel for private(j) reduction(+:tmp) schedule(static) firstprivate(apvec, val, pvec) lastprivate(apvec) 
     for (i=0; i<ndata; i++){
       tmp=0.0;
       for(j = ptr[i];j<ptr[i+1];j++)
       {
        tmp+=val[j]*pvec[col[j]];
        apvec[i]=tmp;
       }
     }

    for(i = 0;i<ndata;i++){
      pap_norm += pvec[i]*apvec[i];
    }
    alpha = r_norm_old/pap_norm;
    r_norm_new = 0.0;
    for (i=0; i<ndata; i++){
      xvec[i] += alpha*pvec[i];
      rvec[i] -= alpha*apvec[i];
      r_norm_new += rvec[i]*rvec[i];
    }
    beta = r_norm_new/r_norm_old;
    r_norm_old = r_norm_new;
#ifdef EBUG
    fprintf(p_his, "%d %.12e\n", n,sqrt(r_norm_new)/sqrt(b_norm));
#endif
    if(sqrt(r_norm_new)/sqrt(b_norm)<eps){
      break;
    }
    n++;
  }
#ifdef EBUG
  et = gettimeofday_sec();
#endif
  if(sqrt(r_norm_new)/sqrt(b_norm)<eps){
     printf("good\n");
   }else{
     printf("bad\n");
   }
  printf("requset loop = %d\n",i_max);
  printf("run loop  = %d\n", n);

#ifdef EBUG
  printf("ElapsedTime = %.6f ms = %.6f s\n", (et-st)*1000, (et-st));
#endif
#ifdef EBUG
  for(i = 0;i<ndata;i++)
  {
    fprintf(p_x, "%d %.12e\n", i,  xvec[i]);
    axvec[0] = 0.0;
  }
    for(i = 0;i<ndata;i++){
     for(j = ptr[i];j<ptr[i+1];j++)
     {
       axvec[i]+=val[j]*xvec[col[j]];
     }
   }
  for(i = 0;i<ndata;i++)
  {
    baxk_norm+=(bvec[i] - axvec[i])*(bvec[i] - axvec[i]);
    xvec[i] = 0.0;
    axvec[i] = 0.0;
  }
    for(i = 0;i<ndata;i++){
     for(j = ptr[i];j<ptr[i+1];j++)
     {
       axvec[i]+=val[j]*xvec[col[j]];
     }
   }
      for(i = 0;i<ndata;i++)
  {
    bax0_norm+=(bvec[i] - axvec[i])*(bvec[i] - axvec[i]);
  }
  true_r = log10(sqrt(baxk_norm)/sqrt(bax0_norm));
  true_r2 = sqrt(baxk_norm);
  printf("r_norm=%.12e\n", sqrt(r_norm_new)/sqrt(b_norm));
  printf("||b-Ax_k||_2=%.12e\n", true_r2);
  printf("log_10(||b-Ax_k||_2/||b-Ax_0||_2)=%.12e\n", true_r);
  fclose(p_x);
  fclose(p_his);
#endif

   free(pvec);
   free(apvec);
   free(axvec);
   free(rvec);
  
  return;
}
