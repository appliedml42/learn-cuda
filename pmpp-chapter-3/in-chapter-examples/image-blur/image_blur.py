import os
from pathlib import Path

from torch.utils.cpp_extension import load_inline
from torchvision.io import read_image, write_jpeg


def compile_extension():
    cuda_source = Path("image_blur.cu").read_text()
    cpp_source = "torch::Tensor image_blur(torch::Tensor input, int blur_size);"

    os.makedirs("./image_blur_cuda_build", exist_ok=True)

    image_blur_extension = load_inline(
        name="image_blur_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["image_blur"],
        with_cuda=True,
        extra_cflags=["-O2"],
        build_directory="./image_blur_cuda_build",
    )

    return image_blur_extension


def main():
    ext = compile_extension()

    input_image = read_image("input.jpg").contiguous().cuda()
    print(input_image)

    output_image = ext.image_blur(input_image, 3)  # type: ignore
    print(output_image)
    write_jpeg(output_image.cpu(), "./output.jpg")


if __name__ == "__main__":
    main()
