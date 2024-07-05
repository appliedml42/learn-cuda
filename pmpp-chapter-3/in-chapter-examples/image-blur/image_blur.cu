#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

__global__ void image_blur_kernel(unsigned char *in, unsigned char *out, int w, int h, int blur_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = threadIdx.z;
    int baseOffset = channel * w * h;

    if (row < h && col < w)
    {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -blur_size; blurRow <= blur_size; blurRow++)
        {
            for (int blurCol = -blur_size; blurCol <= blur_size; blurCol++)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
                {
                    pixVal += in[baseOffset + curRow * w + curCol];
                    pixels++;
                }
            }
        }

        out[baseOffset + row * w + col] = (unsigned char)(pixVal / pixels);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b)
{
    return (a + b - 1) / b;
}

torch::Tensor image_blur(torch::Tensor image, int blur_size)
{
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kUInt8);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto out = torch::empty_like(image);

    dim3 threads_per_block(16, 16, channels);
    dim3 num_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));

    image_blur_kernel<<<num_of_blocks, threads_per_block>>>(image.data_ptr<unsigned char>(), out.data_ptr<unsigned char>(), width, height, blur_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}