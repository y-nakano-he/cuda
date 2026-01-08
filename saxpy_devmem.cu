/* cudaMalloc
   →cudaMemcpy(host->dev)
   →cudaMemcpy(dev->host) で明示的にメモリ管理 */
  
#include <stdio.h>
#include <math.h>

__global__ //宣言指定子によるカーネル定義
void saxpy(int n, float a, float *x, float *y)
{
  /* GPU担当スレッドを一意にするインデックスを作成
    blockIdx: グリッド内のブロック番号
    blockDim: ブロック内のスレッド数
    threadIdx: ブロック内のスレッド番号          */

  int i = ((blockIdx.x) * (blockDim.x)) + (threadIdx.x);
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float coef = 2.0;
  /* CPU側(host)のメモリ領域を確保 */
  float *x, *y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  /* GPU側(device)のメモリ領域を確保 */
  float *d_x, *d_y;
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  /* hostメモリを初期化 */
  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  /* 初期化したCPU(host)メモリデータをGPU(device)メモリデータへ転写 */
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  /* GPUカーネルを指定して稼働開始 
   func <<< (num_blocks), (num_threads/1block) >>> (arg);
  */
  saxpy<<<(N+255)/256, 256>>>(N, coef, d_x, d_y);

  /* GPU(device)側のメモリデータをCPU(host)メモリへ返却 */
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}

