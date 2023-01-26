#include <stdio.h>
#include <time.h>

void cuda_error(cudaError_t err,const char *file,int line) 
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n" , cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define cuda_error_check(err) (cuda_error( err, __FILE__, __LINE__ ))


int main(int argc, char ** argv)
{
  unsigned long const N = 1000000000;
  float * v_cpu = (float*)malloc(N * sizeof(float));
  float * v_gpu = NULL;
  cuda_error_check(cudaMalloc((void **)&v_gpu, N*sizeof(float)));
  clock_t t = clock();
  cuda_error_check(cudaMemcpy(v_gpu, v_cpu, N*sizeof(float), cudaMemcpyHostToDevice));
  cuda_error_check(cudaDeviceSynchronize());
  t = clock() - t;
  double const elapsed_time_seconds = ((double)t)/CLOCKS_PER_SEC;
  cuda_error_check(cudaFree(v_gpu));
  free(v_cpu);
  printf("Elapsed time to send %lu bytes : %lf seconds (bandwidth : %lf GB/s)\n",
         N*sizeof(float),
         elapsed_time_seconds,
         (((double)N)*sizeof(float))/1024.0/1024.0/1024.0/elapsed_time_seconds);
  return 0;
}

// nvcc -O3 -o tp1-bandwidth tp1-bandwidth.cu

