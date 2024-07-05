#include <torch/extension.h>
torch::Tensor color_to_grayscale(torch::Tensor image);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("color_to_grayscale", torch::wrap_pybind_function(color_to_grayscale), "color_to_grayscale");
}