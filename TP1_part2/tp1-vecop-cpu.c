#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void vecop(float * a, float * b, float * c, unsigned long N)
{
  for(unsigned long i = 0; i < N; i++)
  {
    //float const b1 = atanf(b[i]);
    //c[i] = (sinf(b1)/cosf(b1) - b[i]) + (1.0f - a[i] / 2.0f - b[i]);
		c[i] = a[i] + b[i];
  }
}


void check(float * v, unsigned long N, float e)
{
  unsigned long count = 0;
  for(unsigned long i = 0; i < N; i++)
  {
    if(v[i] != 9.0f)
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
    //a_cpu[i] = (float)(2.0 - (double)i * 2.0 / (double)N);
		a_cpu[i] = 4.0f;
    // b_cpu[i] = (float)((double)i / (double)N);
		b_cpu[i] = 5.0f;
    c_cpu[i] = 42.0f;
  }

  clock_t t = clock();
  vecop(a_cpu, b_cpu, c_cpu, N);
  t = clock() - t;
  double const elapsed_time_seconds = ((double)t)/CLOCKS_PER_SEC;
  printf("Elapsed time : %lf seconds\n", elapsed_time_seconds);
  check(c_cpu, N, ERROR_CHECK);

  free(a_cpu);
  free(b_cpu);
  free(c_cpu);
  return 0;
}

// gcc -O3 -o tp1-vecop-cpu tp1-vecop-cpu.c -lm

