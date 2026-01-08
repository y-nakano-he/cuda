/* saxpy.cuから初期化、検算部分もGPUに変更 */
#include <stdio.h>

__global__
 void init(int n, float* x, float* y) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { x[i] = 1.0f; y[i] = 2.0f; }
}

__global__
void saxpy(int n, float a, float *x, float *y)
{
  /* GPUで並列計算をするために担当スレッドを一意にするインデックスを作成
    blockIdx: グリッド内のブロック番号
    blockDim: ブロック内のスレッド数
    threadIdx: ブロック内のスレッド番号                 */

  int i = ((blockIdx.x) * (blockDim.x)) + (threadIdx.x);
  if (i < n) y[i] = a*x[i] + y[i];
}

__global__
void check_err(int n, float* y, float ref, int* d_err_count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    if(y[i] != ref)
    {
      //*d_err_count += 1;
      atomicAdd(d_err_count, 1);   
    }
}  

int main(void)
{
  int N = 1<<20;
  int thread = 256;
  int grid0 = (N + thread - 1) / thread;
  float coef = 2.0;
  float correct = 4.0f;
  int *d_err_count;
  cudaMalloc(&d_err_count, sizeof(int));
  cudaMemset(d_err_count, 0, sizeof(int));

  /* GPU側(device)のメモリ領域を確保 */
  float *d_x, *d_y;
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  /* GPU(device)メモリを初期化 */
  init<<<grid0, thread>>>(N, d_x, d_y);

  /* GPUで計算 
   func <<< (num_blocks), (num_threads/1block) >>> (arg);
  */
  saxpy<<<grid0, thread>>>(N, coef, d_x, d_y);

  /* GPU側で検算 */
  check_err<<<grid0, thread>>>(N, d_y, correct, d_err_count);
  int err = 0; 
  cudaMemcpy(&err, d_err_count, sizeof(int), cudaMemcpyDeviceToHost);
  printf("error count = %d\n", err);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_err_count);
}

