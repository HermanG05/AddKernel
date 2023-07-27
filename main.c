#include <stdio.h>

#define N 1e6
#define BLOCK_SIZE 1024

__global__ void Assigntest(float* A, float* B, float* R, size_t arr_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < arr_size) 
    {
        R[i] = A[i] + B[i];
    }
}

void Assigntest_cpu(float* A, float* B, float* R, size_t arr_size)
{
    #pragma unroll
    for (int i = 0; i < arr_size; i++)
    {
        R[i] = A[i] + B[i];
    }
}

double computeError(float *a, float *b, int n)
{
    double error = 0.0;
    for(int i=0;i<n;++i)
    {
        error += fabs(a[i]-b[i]);
    }
    return error/n;
}

int main() 
{
   float *A, *B, *R, *D;
   float *d_A, *d_B, *d_R;
   size_t size = N*sizeof(float);
   
   A = (float*)malloc(size);
   B = (float*)malloc(size);
   R = (float*)malloc(size);
   
   for(int i = 0; i < N; i++)
   {
      A[i] = rand()/(float)RAND_MAX;
      B[i] = rand()/(float)RAND_MAX;
   }
   
   Assigntest_cpu(A, B, R, N);
   
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, size);
   cudaMalloc(&d_R, size);
   
   cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
   
   Assigntest<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_R, N);
   
   D = (float*)malloc(size);
   cudaMemcpy(D, d_R, size, cudaMemcpyDeviceToHost);
   
   double error = computeError(R, D, N);
   printf("Average error between CUDA and CPU results: %f\n", error);
   
   free(A); free(B); free(R); free(D);
   cudaFree(d_A); cudaFree(d_B); cudaFree(d_R);
   
   return 0;
}
