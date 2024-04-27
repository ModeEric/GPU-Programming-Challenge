#include <iostream>
#include <cuda_runtime.h>

__global__ void matmulOptimized(float* A, float* B, float* C, int N){
    const int TILE_SIZE = 16;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    for(int m=0;m<N/TILE_SIZE;m++){
        int ax = row*N+(m*TILE_SIZE+tx);
        int bx = (m*TILE_SIZE+ty)*N+col;
        As[ty][tx] = A[ax];
        Bs[ty][tx] = B[bx];

        __syncthreads();

        for(int i=0;i<TILE_SIZE;i++){
            sum+= As[i][tx]*Bs[ty][i];
        }
        __syncthreads();
    }
    C[row*N + col] = sum;
}

int main(void){
    //remaking from scratch for practice
    int N = 1024;
    size_t size = N*N*sizeof(float);
    float *Ac, *Bc, *Cc, *Ad, *Bd, *Cd;
    Ac = (float*) malloc(size);
    Bc = (float*) malloc(size);
    Cc = (float*) malloc(size);

    cudaMalloc((void**)&Ad,size);
    cudaMalloc((void**)&Bd,size);
    cudaMalloc((void**)&Cd,size);

    for(int i=0;i<N*N;i++){
        Ac[i] = 2.0;
        Bc[i] = 4.0;
    }

    cudaMemcpy(Ad,Ac,size,cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,Bc,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((N-1+threadsPerBlock.x)/threadsPerBlock.x,(N-1+threadsPerBlock.y)/threadsPerBlock.y);


    matmulOptimized<<<numBlocks, threadsPerBlock>>>(Ad,Bd,Cd,N);


    cudaMemcpy(Cc, Cd, size, cudaMemcpyDeviceToHost);

    //for (int i=0;i<N;i++){
        //for(int j=0;j<N;j++){
          //  std::cout<<Cc[i*N+j] << " ";
        //}
      //  std::cout<<std::endl;
    //}

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    free(Ad);
    free(Bd);
    free(Cd);
    return 0;

}