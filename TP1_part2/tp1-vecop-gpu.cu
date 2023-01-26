#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void cuda_error(cudaError_t err,const char *file,int line) 
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n" , cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define cuda_error_check(err) (cuda_error( err, __FILE__, __LINE__ ))


// TODO : write kernel
__global__ void vecop(float * a, float * b, float * c, unsigned int N)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
  // float const b1 = atanf(b[i]);
  // c[i] = (sinf(b1)/cosf(b1) - b[i]) + (1.0f - a[i] / 2.0f - b[i]);
	if (i < N)
		c[i] = a[i] + b[i];
}


void check(float * v, unsigned long N, float e)
{
  unsigned long count = 0;
  for(unsigned long i = 0; i < N; i++)
  {
    if(fabs(v[i] - 9.0f) > e)
      count++;
  }
  printf("Number of errors : %lu\n", count);
}


int main(int argc, char ** argv)
{
  float const ERROR_CHECK = 0.1f;
  unsigned long const N = 100000000;
  float * a_cpu = (float*)malloc(N*sizeof(float));
  float * b_cpu = (float*)malloc(N*sizeof(float));
  float * c_cpu = (float*)malloc(N*sizeof(float));

  for(unsigned long i = 0; i < N; i++)
  {
    // a_cpu[i] = (float)(2.0 - (double)i * 2.0 / (double)N);
		a_cpu[i] = 4.0f;
		b_cpu[i] = 5.0f;
    //b_cpu[i] = (float)((double)i / (double)N);
    c_cpu[i] = 42.0f;
  }

  float * a_gpu = NULL;
  float * b_gpu = NULL;
  float * c_gpu = NULL;

	clock_t t = clock();
  
  // Allocate and copy into GPU memory
  cuda_error_check(cudaMalloc(&a_gpu, N*sizeof(float)));
	cuda_error_check(cudaMalloc(&b_gpu, N*sizeof(float)));
	cuda_error_check(cudaMalloc(&c_gpu, N*sizeof(float)));
	cuda_error_check(cudaMemcpy(a_gpu, a_cpu, N*sizeof(float), cudaMemcpyHostToDevice));
	cuda_error_check(cudaMemcpy(b_gpu, b_cpu, N*sizeof(float), cudaMemcpyHostToDevice));
	cuda_error_check(cudaMemcpy(c_gpu, c_cpu, N*sizeof(float), cudaMemcpyHostToDevice));

  unsigned int const nbThreads = 256;
  dim3 const blockSize(nbThreads, 1, 1);
  dim3 const gridSize((N-1)/nbThreads + 1, 1, 1);
  //clock_t t = clock();
  // Call vecop kernel
  vecop<<<390625,256>>>(a_gpu, b_gpu, c_gpu, N);
  cudaDeviceSynchronize();
  //t = clock() - t;

  // Get results back from GPU memory
  cuda_error_check(cudaMemcpy(c_cpu, c_gpu, N*sizeof(float), cudaMemcpyDeviceToHost));
	t = clock() - t;

  double const elapsed_time_seconds = ((double)t)/CLOCKS_PER_SEC;
  printf("Elapsed time : %lf seconds\n", elapsed_time_seconds);
  check(c_cpu, N, ERROR_CHECK);

  cuda_error_check(cudaFree(a_gpu));
  cuda_error_check(cudaFree(b_gpu));
  cuda_error_check(cudaFree(c_gpu));

  free(a_cpu);
  free(b_cpu);
  free(c_cpu);
  return 0;
}

// nvcc -O3 -o tp1-vecop-gpu tp1-vecop-gpu.cu

