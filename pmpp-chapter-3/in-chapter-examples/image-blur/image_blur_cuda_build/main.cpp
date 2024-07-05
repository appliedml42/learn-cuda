#include <torch/extension.h>
torch::Tensor image_blur(torch::Tensor input, int blur_size);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("image_blur", torch::wrap_pybind_function(image_blur), "image_blur");
}