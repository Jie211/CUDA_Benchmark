#include <stdio.h>
#include "cuda.cuh"

int main(int argc, char const* argv[])
{
  
  printf("hoge!\n");
  /* kernel<<<1,1>>>(); */
  run();
  return 0;
}
