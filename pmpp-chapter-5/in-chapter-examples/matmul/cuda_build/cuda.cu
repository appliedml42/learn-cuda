#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#define TILE_WIDTH 8

inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

__global__
void matmul_kernel(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float Pvalue = 0.0f;
    for(int i = 0; i < ceil(M_cols/(float)TILE_WIDTH); i++){
        if (row < M_rows && (i * TILE_WIDTH + threadIdx.x) < M_cols)
            Mds[threadIdx.y][threadIdx.x] = M[row * M_cols + (i * TILE_WIDTH + threadIdx.x)];
        else 
            Mds[threadIdx.y][threadIdx.x] = 0.0f;
        
        if ((i * TILE_WIDTH + threadIdx.y) < N_rows && col < N_cols)
            Nds[threadIdx.y][threadIdx.x] = N[(i * TILE_WIDTH + threadIdx.y) * N_cols + col];
        else
            Nds[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++){
            Pvalue += Mds[threadIdx.y][j] * Nds[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M_rows && col < N_cols){
        P[row * N_cols + col] = Pvalue;
    }
}

torch::Tensor matmul(torch::Tensor M, torch::Tensor N){
    assert(M.device().type() == torch::kCUDA);
    assert(M.dtype() == torch::kFloat32);

    assert(N.device().type() == torch::kCUDA);
    assert(N.dtype() == torch::kFloat32);

    const auto M_rows = M.size(0);
    const auto M_cols = M.size(1);
    const auto N_rows = N.size(0);
    const auto N_cols = N.size(1);

    auto P = torch::empty({M_rows, N_cols}, M.options());

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_of_blocks(cdiv(N_cols, threads_per_block.x), cdiv(M_rows, threads_per_block.y));

    printf("Threads per block: (%d, %d)\n", threads_per_block.x, threads_per_block.y);
    printf("Number of blocks: (%d, %d)\n", num_of_blocks.x, num_of_blocks.y);

    matmul_kernel<<<num_of_blocks, threads_per_block>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), M_rows, M_cols, N_rows, N_cols);

    return P;
}