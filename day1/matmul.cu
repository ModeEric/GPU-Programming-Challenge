#include <iostream>
#include <cuda_runtime.h>
__global__ void matMulKernel(float* A, float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<N && col < N){
        float i = 0.0;
        for(int k=0;k<N;k++){
            i+=A[row*N+k]*B[k*N+col];
        }
        C[row*N+col] = i;
    }
}
int main(void){
    int N = 16;
    size_t size = N * N * sizeof(float);

    float *Ac, *Bc, *Cc, *Ad, *Bd, *Cd;

    Ac = (float*)malloc(size);
    Bc = (float*)malloc(size);
    Cc = (float*)malloc(size);

    for(int i=0;i<N*N;i++){
        Ac[i] = 1.0;
        Bc[i] = 3.0;
    }
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);
    cudaMalloc((void**)&Cd, size);

    cudaMemcpy(Ad, Ac, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, Bc, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((N-1+threadsPerBlock.x)/threadsPerBlock.x, (N-1+threadsPerBlock.y)/threadsPerBlock.y);

    matMulKernel<<<numBlocks,threadsPerBlock>>>(Ad,Bd,Cd,N);

    cudaMemcpy(Cc,Cd, size, cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            std::cout<<Cc[i*N+j] << " ";
        }
        std::cout<<std::endl;
    }

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    free(Ac);
    free(Bc);
    free(Cc);

    return 0;

}