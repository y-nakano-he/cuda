/*cudaでGPU実装：
  unified memory(cudaMallocManaged)
  block = 1, thread = 1 */

#include <iostream>
#include <math.h>
__global__ 
// function to add the elements of two arrays
void saxpy(int n, float a, float *x, float *y)
{
 for (int i = 0; i < n; i++)
     y[i] = a * x[i] + y[i];
}
 
int main(void)
{
 int N = 1<<20; // 1M elements
 float coef = 2.0;

 float *x, *y;
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));
 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) 
 {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }
 
 // Run kernel on 1M elements on the CPU
 saxpy<<<1, 1>>>(N, coef, x, y);
 
 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();
 
 // Check for errors (all values should be 4.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
 {
   maxError = fmax(maxError, fabs(y[i]-4.0f));
 } 
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 cudaFree (x);
 cudaFree (y);
 
 return 0;
}
