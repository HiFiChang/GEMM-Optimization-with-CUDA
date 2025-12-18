#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define ITERATIONS 10
#define TILE_WIDTH 64

using namespace std;

__global__ void sgemm(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int n, float a, float b) {
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * 16 + tx;

  int load_a_row[4], load_a_col[4];
  int load_b_row[4], load_b_col[4];
  int global_a_r[4], global_a_c[4];
  int global_b_r[4], global_b_c[4];
  const float* src_A_ptr[4];
  const float* src_B_ptr[4];

  #pragma unroll
  for(int i=0; i<4; i++) {
    int load_idx = tid + i * 256;
    int r = load_idx / 16; int c = (load_idx % 16) * 4;
    load_a_row[i] = r; load_a_col[i] = c;
    global_a_r[i] = by * TILE_WIDTH + r;
    global_a_c[i] = 0 * TILE_WIDTH + c;
    src_A_ptr[i] = A + (global_a_r[i] * n + global_a_c[i]);
    load_b_row[i] = r; load_b_col[i] = c;
    global_b_r[i] = 0 * TILE_WIDTH + r;
    global_b_c[i] = bx * TILE_WIDTH + c;
    src_B_ptr[i] = B + (global_b_r[i] * n + global_b_c[i]);
  }

  float4 ldg_a_reg[4];
  float4 ldg_b_reg[4];
  float c_reg[4][4] = {0};
  float a_vals[4];
  float b_vals[4];
  float next_a_vals[4];
  float next_b_vals[4];

  #pragma unroll
  for(int i=0; i<4; i++) {
    if (global_a_r[i] < n && global_a_c[i] < n)
      ldg_a_reg[i] = reinterpret_cast<const float4*>(src_A_ptr[i])[0];
    else
      ldg_a_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (global_b_r[i] < n && global_b_c[i] < n)
      ldg_b_reg[i] = reinterpret_cast<const float4*>(src_B_ptr[i])[0];
    else
      ldg_b_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  int num_tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int m = 0; m < num_tiles; ++m) {

    #pragma unroll
    for(int i=0; i<4; i++) {
      As[load_a_row[i]][load_a_col[i]]   = ldg_a_reg[i].x;
      As[load_a_row[i]][load_a_col[i]+1] = ldg_a_reg[i].y;
      As[load_a_row[i]][load_a_col[i]+2] = ldg_a_reg[i].z;
      As[load_a_row[i]][load_a_col[i]+3] = ldg_a_reg[i].w;

      Bs[load_b_row[i]][load_b_col[i]]   = ldg_b_reg[i].x;
      Bs[load_b_row[i]][load_b_col[i]+1] = ldg_b_reg[i].y;
      Bs[load_b_row[i]][load_b_col[i]+2] = ldg_b_reg[i].z;
      Bs[load_b_row[i]][load_b_col[i]+3] = ldg_b_reg[i].w;
    }
    __syncthreads();

    #pragma unroll
    for(int i=0; i<4; i++) {
      src_A_ptr[i] += TILE_WIDTH;
      global_a_c[i] += TILE_WIDTH;
      src_B_ptr[i] += TILE_WIDTH * n;
      global_b_r[i] += TILE_WIDTH;
    }
    if (m < num_tiles - 1) {
      #pragma unroll
      for(int i=0; i<4; i++) {
        if (global_a_r[i] < n && global_a_c[i] < n)
          ldg_a_reg[i] = reinterpret_cast<const float4*>(src_A_ptr[i])[0];
        else
          ldg_a_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (global_b_r[i] < n && global_b_c[i] < n)
          ldg_b_reg[i] = reinterpret_cast<const float4*>(src_B_ptr[i])[0];
        else
          ldg_b_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }

    #pragma unroll
    for(int i=0; i<4; i++) { a_vals[i] = As[ty + i*16][0]; }
    #pragma unroll
    for(int j=0; j<4; j++) { b_vals[j] = Bs[0][tx + j*16]; }

    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
      if (k < TILE_WIDTH - 1) {
        #pragma unroll
        for(int i=0; i<4; i++) { next_a_vals[i] = As[ty + i*16][k+1]; }
        #pragma unroll
        for(int j=0; j<4; j++) { next_b_vals[j] = Bs[k+1][tx + j*16]; }
      }

      #pragma unroll
      for(int i=0; i<4; i++) {
        #pragma unroll
        for(int j=0; j<4; j++) {
          c_reg[i][j] += a_vals[i] * b_vals[j];
        }
      }

      if (k < TILE_WIDTH - 1) {
        #pragma unroll
        for(int i=0; i<4; i++) { a_vals[i] = next_a_vals[i]; }
        #pragma unroll
        for(int j=0; j<4; j++) { b_vals[j] = next_b_vals[j]; }
      }
    }

    __syncthreads();
  }

  int row_base = by * TILE_WIDTH;
  int col_base = bx * TILE_WIDTH;
  for(int i=0; i<4; i++) {
    for(int j=0; j<4; j++) {
      int r = row_base + ty + i*16;
      int c_idx = col_base + tx + j*16;
      if (r < n && c_idx < n) {
        C[r * n + c_idx] = b * C[r * n + c_idx] + a * c_reg[i][j];
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