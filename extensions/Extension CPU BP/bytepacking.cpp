#include <iostream>
#include <stdint.h>
#include <torch/extension.h>


torch::Tensor packing(torch::Tensor src)
{
    src = src.to(torch::kInt64);
    auto src_len = src.numel();
    std::vector<torch::Tensor> packed_vector;

    int64_t ind = 0;
    int64_t code_count = 0;
    int i = 0, j = 0;

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(src.device());

    while (ind < src_len)
    {
        torch::Tensor code = torch::zeros(1, opts);
        for (i = 7, j = 0; i >= 0 && (ind + j) < src_len; i--, j++)
        {
            code = code | (src[ind + j] & 0xff).__lshift__(8 * i);
        }

        packed_vector.push_back(code);
        code_count++;
        ind += 8;
    }

    torch::Tensor packed_tensor = torch::cat(packed_vector);

    return packed_tensor;
}


torch::Tensor unpacking(torch::Tensor src)
{
    auto src_len = src.numel();
    std::vector<torch::Tensor> unpacked_vector;

    int64_t ind = 0;
    int64_t element_count = 0;
    int mode = -1;
    int i = 0, j = 0;

    while (ind < src_len)
    {
        auto code = src[ind];

        for (i = 7, j = 0; i >= 0 ; i--)
        {
            auto element = (code.__rshift__(8 * i) & 0xff).to(torch::kInt8);
            unpacked_vector.push_back(element);
            element_count++;
        }
        ind += 1;
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(src.device());
    torch::Tensor unpacked_tensor = torch::stack(unpacked_vector);

    return unpacked_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packing", &packing, "packing");
    m.def("unpacking", &unpacking, "unpacking");
}