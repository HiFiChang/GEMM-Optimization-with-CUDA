#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define ITERATIONS 10

using namespace std;

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

const int NUM_THREADS = 256;

__global__ void sgemm(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int N, float alpha, float beta) {
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    int ty = tid / 16;
    int tx = tid % 16;

    int row_c = ty * TM;
    int col_c = tx * TN;

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 4;

    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    const float* src_A = A + (by * BM * N);
    const float* src_B = B + (bx * BN);

    float c_reg[TM][TN] = {0.0f};

    float4 ldg_a_reg;
    float4 ldg_b_reg;

    float4 frag_a[2][TM/4];
    float4 frag_b[2][TN/4];

    int global_r = load_a_row;
    int global_c = 0 + load_a_col;
    if (by * BM + global_r < N && global_c < N)
        ldg_a_reg = reinterpret_cast<const float4*>(&src_A[global_r * N + global_c])[0];
    else
        ldg_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    global_r = 0 + load_b_row;
    global_c = load_b_col;
    if (global_r < N && bx * BN + global_c < N)
        ldg_b_reg = reinterpret_cast<const float4*>(&src_B[global_r * N + global_c])[0];
    else
        ldg_b_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    int num_tiles = (N + BK - 1) / BK;

    for (int k_tile = 0; k_tile < num_tiles; ++k_tile) {
        As[load_a_col + 0][load_a_row] = ldg_a_reg.x;
        As[load_a_col + 1][load_a_row] = ldg_a_reg.y;
        As[load_a_col + 2][load_a_row] = ldg_a_reg.z;
        As[load_a_col + 3][load_a_row] = ldg_a_reg.w;

        reinterpret_cast<float4*>(&Bs[load_b_row][load_b_col])[0] = ldg_b_reg;

        __syncthreads();

        int next_k = k_tile + 1;
        if (next_k < num_tiles) {
            int global_r = load_a_row;
            int global_c = next_k * BK + load_a_col;
            if (by * BM + global_r < N && global_c < N)
                ldg_a_reg = reinterpret_cast<const float4*>(&src_A[global_r * N + global_c])[0];
            else
                ldg_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            global_r = next_k * BK + load_b_row;
            global_c = load_b_col;
            if (global_r < N && bx * BN + global_c < N)
                ldg_b_reg = reinterpret_cast<const float4*>(&src_B[global_r * N + global_c])[0];
            else
                ldg_b_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        frag_a[0][0] = reinterpret_cast<float4*>(&As[0][row_c])[0];
        frag_a[0][1] = reinterpret_cast<float4*>(&As[0][row_c+4])[0];
        frag_b[0][0] = reinterpret_cast<float4*>(&Bs[0][col_c])[0];
        frag_b[0][1] = reinterpret_cast<float4*>(&Bs[0][col_c+4])[0];

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            if (k < BK - 1) {
                int next_buf = (k + 1) % 2;
                frag_a[next_buf][0] = reinterpret_cast<float4*>(&As[k+1][row_c])[0];
                frag_a[next_buf][1] = reinterpret_cast<float4*>(&As[k+1][row_c+4])[0];
                frag_b[next_buf][0] = reinterpret_cast<float4*>(&Bs[k+1][col_c])[0];
                frag_b[next_buf][1] = reinterpret_cast<float4*>(&Bs[k+1][col_c+4])[0];
            }

            int cur_buf = k % 2;

            float a_ary[8] = {frag_a[cur_buf][0].x, frag_a[cur_buf][0].y, frag_a[cur_buf][0].z, frag_a[cur_buf][0].w,
                              frag_a[cur_buf][1].x, frag_a[cur_buf][1].y, frag_a[cur_buf][1].z, frag_a[cur_buf][1].w};
            float b_ary[8] = {frag_b[cur_buf][0].x, frag_b[cur_buf][0].y, frag_b[cur_buf][0].z, frag_b[cur_buf][0].w,
                              frag_b[cur_buf][1].x, frag_b[cur_buf][1].y, frag_b[cur_buf][1].z, frag_b[cur_buf][1].w};

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    c_reg[i][j] += a_ary[i] * b_ary[j];
                }
            }
        }
        __syncthreads();
    }

    int global_row_base = by * BM + row_c;
    int global_col_base = bx * BN + col_c;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = global_row_base + i;
        if (r < N) {
             if (global_col_base + 3 < N) {
                 float4 val = make_float4(c_reg[i][0], c_reg[i][1], c_reg[i][2], c_reg[i][3]);
                 if (beta != 0.0f) {
                     float4 old = reinterpret_cast<float4*>(&C[r*N + global_col_base])[0];
                     val.x = alpha * val.x + beta * old.x;
                     val.y = alpha * val.y + beta * old.y;
                     val.z = alpha * val.z + beta * old.z;
                     val.w = alpha * val.w + beta * old.w;
                 } else {
                     val.x *= alpha; val.y *= alpha; val.z *= alpha; val.w *= alpha;
                 }
                 reinterpret_cast<float4*>(&C[r*N + global_col_base])[0] = val;
             }
             if (global_col_base + 7 < N) {
                 float4 val = make_float4(c_reg[i][4], c_reg[i][5], c_reg[i][6], c_reg[i][7]);
                 if (beta != 0.0f) {
                     float4 old = reinterpret_cast<float4*>(&C[r*N + global_col_base + 4])[0];
                     val.x = alpha * val.x + beta * old.x;
                     val.y = alpha * val.y + beta * old.y;
                     val.z = alpha * val.z + beta * old.z;
                     val.w = alpha * val.w + beta * old.w;
                 } else {
                     val.x *= alpha; val.y *= alpha; val.z *= alpha; val.w *= alpha;
                 }
                 reinterpret_cast<float4*>(&C[r*N + global_col_base + 4])[0] = val;
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

  dim3 block(NUM_THREADS);
  dim3 grid((MY_N + BN - 1) / BN, (MY_N + BM - 1) / BM);

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
