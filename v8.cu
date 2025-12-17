#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define ITERATIONS 10
#define TILE_WIDTH 64

using namespace std;

__global__ void sgemm(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int n, float a, float b) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH + 4];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH + 4];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    int row_c = ty * 4;
    int col_c = tx * 4;

    float c_reg[4][4] = {0.0f};
    float4 frag_a[2];
    float4 frag_b[2];
    float4 ldg_a_reg[4];
    float4 ldg_b_reg[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * 256;
        int r = idx / 16;
        int c = (idx % 16) * 4;

        int global_r = by * TILE_WIDTH + r;
        int global_c = c;
        if (global_r < n && global_c + 3 < n)
            ldg_a_reg[i] = reinterpret_cast<const float4*>(&A[global_r * n + global_c])[0];
        else if (global_r < n && global_c < n) {
            ldg_a_reg[i] = make_float4(
                A[global_r * n + global_c],
                (global_c + 1 < n) ? A[global_r * n + global_c + 1] : 0.0f,
                (global_c + 2 < n) ? A[global_r * n + global_c + 2] : 0.0f,
                (global_c + 3 < n) ? A[global_r * n + global_c + 3] : 0.0f
            );
        } else {
            ldg_a_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        global_r = r;
        global_c = bx * TILE_WIDTH + c;
        if (global_r < n && global_c + 3 < n)
            ldg_b_reg[i] = reinterpret_cast<const float4*>(&B[global_r * n + global_c])[0];
        else if (global_r < n && global_c < n) {
            ldg_b_reg[i] = make_float4(
                B[global_r * n + global_c],
                (global_c + 1 < n) ? B[global_r * n + global_c + 1] : 0.0f,
                (global_c + 2 < n) ? B[global_r * n + global_c + 2] : 0.0f,
                (global_c + 3 < n) ? B[global_r * n + global_c + 3] : 0.0f
            );
        } else {
            ldg_b_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    int num_tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < num_tiles; ++m) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 256;
            int r = idx / 16;
            int c = (idx % 16) * 4;

            As[c+0][r] = ldg_a_reg[i].x;
            As[c+1][r] = ldg_a_reg[i].y;
            As[c+2][r] = ldg_a_reg[i].z;
            As[c+3][r] = ldg_a_reg[i].w;

            reinterpret_cast<float4*>(&Bs[r][c])[0] = ldg_b_reg[i];
        }

        __syncthreads();

        int next_m = m + 1;
        if (next_m < num_tiles) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int idx = tid + i * 256;
                int r = idx / 16;
                int c = (idx % 16) * 4;

                int global_r = by * TILE_WIDTH + r;
                int global_c = next_m * TILE_WIDTH + c;
                if (global_r < n && global_c + 3 < n)
                    ldg_a_reg[i] = reinterpret_cast<const float4*>(&A[global_r * n + global_c])[0];
                else if (global_r < n && global_c < n) {
                    ldg_a_reg[i] = make_float4(
                        A[global_r * n + global_c],
                        (global_c + 1 < n) ? A[global_r * n + global_c + 1] : 0.0f,
                        (global_c + 2 < n) ? A[global_r * n + global_c + 2] : 0.0f,
                        (global_c + 3 < n) ? A[global_r * n + global_c + 3] : 0.0f
                    );
                } else {
                    ldg_a_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }

                global_r = next_m * TILE_WIDTH + r;
                global_c = bx * TILE_WIDTH + c;
                if (global_r < n && global_c + 3 < n)
                    ldg_b_reg[i] = reinterpret_cast<const float4*>(&B[global_r * n + global_c])[0];
                else if (global_r < n && global_c < n) {
                    ldg_b_reg[i] = make_float4(
                        B[global_r * n + global_c],
                        (global_c + 1 < n) ? B[global_r * n + global_c + 1] : 0.0f,
                        (global_c + 2 < n) ? B[global_r * n + global_c + 2] : 0.0f,
                        (global_c + 3 < n) ? B[global_r * n + global_c + 3] : 0.0f
                    );
                } else {
                    ldg_b_reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }

        frag_a[0] = reinterpret_cast<float4*>(&As[0][row_c])[0];
        frag_b[0] = reinterpret_cast<float4*>(&Bs[0][col_c])[0];

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            if (k < TILE_WIDTH - 1) {
                frag_a[(k+1)%2] = reinterpret_cast<float4*>(&As[k+1][row_c])[0];
                frag_b[(k+1)%2] = reinterpret_cast<float4*>(&Bs[k+1][col_c])[0];
            }

            float4 cur_a = frag_a[k%2];
            float4 cur_b = frag_b[k%2];

            c_reg[0][0] += cur_a.x * cur_b.x;
            c_reg[0][1] += cur_a.x * cur_b.y;
            c_reg[0][2] += cur_a.x * cur_b.z;
            c_reg[0][3] += cur_a.x * cur_b.w;

            c_reg[1][0] += cur_a.y * cur_b.x;
            c_reg[1][1] += cur_a.y * cur_b.y;
            c_reg[1][2] += cur_a.y * cur_b.z;
            c_reg[1][3] += cur_a.y * cur_b.w;

            c_reg[2][0] += cur_a.z * cur_b.x;
            c_reg[2][1] += cur_a.z * cur_b.y;
            c_reg[2][2] += cur_a.z * cur_b.z;
            c_reg[2][3] += cur_a.z * cur_b.w;

            c_reg[3][0] += cur_a.w * cur_b.x;
            c_reg[3][1] += cur_a.w * cur_b.y;
            c_reg[3][2] += cur_a.w * cur_b.z;
            c_reg[3][3] += cur_a.w * cur_b.w;
        }

        __syncthreads();
    }

    int global_row_base = by * TILE_WIDTH + row_c;
    int global_col_base = bx * TILE_WIDTH + col_c;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = global_row_base + i;
        int c = global_col_base;

        if (r < n && c < n) {
             if (c + 3 < n) {
                 float4 c_val = make_float4(c_reg[i][0], c_reg[i][1], c_reg[i][2], c_reg[i][3]);
                 float4 old_c = reinterpret_cast<float4*>(&C[r * n + c])[0];
                 c_val.x = a * c_val.x + b * old_c.x;
                 c_val.y = a * c_val.y + b * old_c.y;
                 c_val.z = a * c_val.z + b * old_c.z;
                 c_val.w = a * c_val.w + b * old_c.w;
                 reinterpret_cast<float4*>(&C[r * n + c])[0] = c_val;
             } else {
                 for(int j=0; j<4; j++) {
                     if (c+j < n) C[r*n + c+j] = a * c_reg[i][j] + b * C[r*n + c+j];
                 }
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

  if (MY_N <= 1024) {
      for(int j=0; j<MY_N; j++){
        for(int i=0; i<MY_N; i++){
          C_cpu[i*MY_N+j]+=b*C_cpu[i*MY_N+j];
          for(int k=0; k<MY_N; k++){
            C_cpu[i*MY_N+j] += a*A[i*MY_N+k]*B[k*MY_N+j];
          }
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
  cudaDeviceSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*MY_N*MY_N, cudaMemcpyDeviceToHost);
  
  if (MY_N <= 1024) {
      compare(C_cpu, C_gpu_final, MY_N*MY_N);
  }

  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
    sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, MY_N, a, b);
  }
  cudaDeviceSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double gflops = (2.0*MY_N*MY_N*MY_N*1e-9)/time;
  double bandwidth = (3.0*MY_N*MY_N*sizeof(float)*1e-9)/time;
  printf("GFLOPS/s=%lf\n",gflops);
  printf("GB/s=%lf\n",bandwidth);
  printf("time(s)=%lf\n",time);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  free(A); free(B); free(C_cpu); free(C_gpu_final);
  return 0;
}