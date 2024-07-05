#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

__global__
void color_to_grayscale_kernel(unsigned char * in, unsigned char *out, int w, int h) {
    const int channels = 3;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < h && col < w) {
        int colorOffset = channels * (row * w + col);
        int greyOffset = row * w + col;

        unsigned char r = in[colorOffset];
        unsigned char g = in[colorOffset + 1];
        unsigned char b = in[colorOffset + 2];
        out[greyOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor color_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kUInt8);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto out = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    dim3 threads_per_block(16, 16);
    dim3 num_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));

    color_to_grayscale_kernel<<<num_of_blocks, threads_per_block>>>(image.data_ptr<unsigned char>(), out.data_ptr<unsigned char>(), width, height);

    return out;
}