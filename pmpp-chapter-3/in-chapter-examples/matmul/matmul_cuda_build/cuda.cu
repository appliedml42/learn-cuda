#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

__global__
void matmul_kernel(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int P_rows = M_rows;
    const int P_cols = N_cols;

    if (row < P_rows && col < P_cols) {
        float sum = 0.0f;
        for (int i=0 ; i < M_cols ; i++){
            sum += M[row * M_cols + i] * N[i * N_cols + col];
        }
        P[row * P_cols + col] = sum;
    }

}

inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
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

    dim3 threads_per_block(16, 16);
    dim3 num_of_blocks(cdiv(M_rows, threads_per_block.y), cdiv(N_cols, threads_per_block.x));

    matmul_kernel<<<num_of_blocks, threads_per_block>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), M_rows, M_cols, N_rows, N_cols);

    return P;
}