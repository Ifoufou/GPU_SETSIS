#include <stdio.h>
/* a simple CUDA kernel */
__global__ void add ( int a , int b , int *c ) {
  *c = a + b; 
}

int main ( void ) 
{
  int c;
  int * dev_c;

  // GPU device memory allocation
  cudaMalloc ( ( void **) & dev_c , sizeof ( int ) ) ;

  // perform computation on GPU
  add<<<1 ,1>>>(2, 7, dev_c);

  // get back computation result into host CPU memory
  cudaMemcpy ( &c , dev_c , sizeof (int) , cudaMemcpyDeviceToHost ) ;

  // output result on screen
  printf ( "2 + 7 = %d \n" , c );

  // de - allocate GPU device memory
  cudaFree ( dev_c ) ;

  return 0;
}
