import os
from pathlib import Path

import torch
from torch.testing import assert_close
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("matmul_kernel.cu").read_text()
    cpp_source = "torch::Tensor matmul(torch::Tensor M, torch::Tensor N);"

    os.makedirs("./cuda_build", exist_ok=True)

    matmul_extension = load_inline(
        name="matmul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul"],
        with_cuda=True,
        extra_cflags=["-O2"],
        build_directory="./cuda_build",
    )

    return matmul_extension


def main():
    ext = compile_extension()

    M = torch.randn(16, 3).cuda()
    N = torch.randn(3, 4).cuda()

    P_real = ext.matmul(M, N)  # type: ignore
    P_expected = torch.matmul(M, N)

    assert_close(P_real, P_expected)


if __name__ == "__main__":
    main()
