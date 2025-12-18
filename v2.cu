#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define ITERATIONS 10
#define TILE_WIDTH 32

using namespace std;

__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * 16 + tx;

  float c[2][2] = {0};

  for (int m = 0; m < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
  #pragma unroll
  for(int i=0; i<4; i++) {
    int load_idx = tid + i * 256;
    int row_local = load_idx / TILE_WIDTH;
    int col_local = load_idx % TILE_WIDTH;

    int row_global = by * TILE_WIDTH + row_local;
    int col_global = m * TILE_WIDTH + col_local;

    if (row_global < n && col_global < n)
      As[row_local][col_local] = A[row_global * n + col_global];
    else
      As[row_local][col_local] = 0.0;
  }

  #pragma unroll
  for(int i=0; i<4; i++) {
    int load_idx = tid + i * 256;
    int row_local = load_idx / TILE_WIDTH;
    int col_local = load_idx % TILE_WIDTH;

    int row_global = m * TILE_WIDTH + row_local;
    int col_global = bx * TILE_WIDTH + col_local;

    if (row_global < n && col_global < n)
      Bs[row_local][col_local] = B[row_global * n + col_global];
    else
      Bs[row_local][col_local] = 0.0;
  }

  __syncthreads();

  #pragma unroll
  for (int k = 0; k < TILE_WIDTH; ++k) {
    float b_val0 = Bs[k][tx];
    float b_val1 = Bs[k][tx + 16];
    float a_val0 = As[ty][k];
    float a_val1 = As[ty + 16][k];

    c[0][0] += a_val0 * b_val0;
    c[0][1] += a_val0 * b_val1;
    c[1][0] += a_val1 * b_val0;
    c[1][1] += a_val1 * b_val1;
  }

  __syncthreads();
  }

  int row_base = by * TILE_WIDTH;
  int col_base = bx * TILE_WIDTH;

  int rows[2] = {ty, ty + 16};
  int cols[2] = {tx, tx + 16};

  for(int i=0; i<2; i++) {
    for(int j=0; j<2; j++) {
      int r = row_base + rows[i];
      int c_idx = col_base + cols[j];
      if (r < n && c_idx < n) {
        C[r * n + c_idx] = b * C[r * n + c_idx] + a * c[i][j];
      }
    }
  }
}

void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else 
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else 
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    if(i<10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>0.0005)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(){
  float *A = (float*)malloc(MY_N*MY_N*sizeof(float));
  float *B = (float*)malloc(MY_N*MY_N*sizeof(float));
  float *C_cpu = (float*)malloc(MY_N*MY_N*sizeof(float));
  float *C_gpu_final = (float*)malloc(MY_N*MY_N*sizeof(float));
  //float A[MY_N][MY_N], B[MY_N][MY_N], C_cpu[MY_N][MY_N], C_gpu_final[MY_N][MY_N];
  float a=0.5, b=0.3;
  for(int i=0; i<MY_N; i++){
    for(int j=0; j<MY_N; j++){
      A[i*MY_N+j]=(float)rand()/(float)(RAND_MAX/a);
      B[i*MY_N+j]=(float)rand()/(float)(RAND_MAX/a);
      C_cpu[i*MY_N+j]=0;
      C_gpu_final[i*MY_N+j]=0;
    }
  }

  for(int j=0; j<MY_N; j++){
    for(int i=0; i<MY_N; i++){
      C_cpu[i*MY_N+j]+=b*C_cpu[i*MY_N+j];
      for(int k=0; k<MY_N; k++){
        C_cpu[i*MY_N+j] += a*A[i*MY_N+k]*B[k*MY_N+j];
      }
    }
  }

  float *A_gpu;
  float *B_gpu;
  float *C_gpu;
  cudaMalloc((void **)&A_gpu, sizeof(float)*MY_N*MY_N);
  cudaMalloc((void **)&B_gpu, sizeof(float)*MY_N*MY_N);
  cudaMalloc((void **)&C_gpu, sizeof(float)*MY_N*MY_N);
  cudaMemcpy(A_gpu, A, sizeof(float)*MY_N*MY_N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*MY_N*MY_N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*MY_N*MY_N, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((MY_N + TILE_WIDTH - 1) / TILE_WIDTH, (MY_N + TILE_WIDTH - 1) / TILE_WIDTH);

  sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, MY_N, a, b);
  cudaThreadSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*MY_N*MY_N, cudaMemcpyDeviceToHost);
  compare(C_cpu, C_gpu_final, MY_N*MY_N);

  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
    sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, MY_N, a, b);
  }
  cudaThreadSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double flops = 2.0*MY_N*MY_N*MY_N;
  double gflopsPerSecond = flops/(1000000000)/time;
  // Effective Bandwidth: Read A, Read B, Read C, Write C (4 matrices)
  double GB = 4.0*(double)(MY_N)*MY_N*4/1000000000;
  double GBpS = 4.0*(double)(MY_N)*MY_N*4/1000000000/time;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",GBpS);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("GB=%lf\n",GB);
  printf("time(s)=%lf\n",time);

  
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  return 0;
}
