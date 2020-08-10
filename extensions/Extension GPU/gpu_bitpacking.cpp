#include <iostream>
#include <torch/extension.h>


torch::Tensor packing(torch::Tensor src)
{
    auto src_len = src.numel();
    std::vector<torch::Tensor> packed_vector;

    int64_t ind = 0;
    int64_t code_count = 0;
    int mode = -1;
    int i = 0, j = 0;

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(src.device());
    torch::Tensor code = torch::zeros(1, opts);

    while (ind < src_len)
    {
        code = code & 0x0;
        if (src.slice(0, ind, ind + 15).max().item<int>() < 4)
        {
            mode = 0;
            for (i = 28, j = 0; (i >= 0) && (j < 15) && (ind + j) < src_len; i -= 2, j += 1)
                code = code | (src[ind + j].__lshift__(i));

            ind += 15;
        }
        else if (src.slice(0, ind, ind + 7).max().item<int>() < 16)
        {
            mode = 1;
            for (i = 26, j = 0; (i >= 0) && (j < 7) && (ind + j) < src_len; i -= 4, j += 1)
                code = code | (src[ind + j].__lshift__(i));

            ind += 7;
        }
        else if (src.slice(0, ind, ind + 4).max().item<int>() < 128)
        {
            mode = 2;
            for (i = 23, j = 0; (i >= 0) && (j < 4) && (ind + j) < src_len; i -= 7, j += 1)
                code = code | (src[ind + j].__lshift__(i));

            ind += 4;
        }
        else if (src.slice(0, ind, ind + 3).max().item<int>() < 256)
        {
            mode = 3;
            for (i = 22, j = 0; (i >= 0) && (j < 3) && (ind + j) < src_len; i -= 8, j += 1)
                code = code | (src[ind + j].__lshift__(i));

            ind += 3;
        }

        code = code | (mode << 30);
        packed_vector.push_back(code);
        code_count++;
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

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(src.device());
    torch::Tensor code = torch::zeros(1, opts);

    while (ind < src_len)
    {
        code = src[ind];
        mode = (code.__rshift__(30) & 0x3).item<int>();

        if (mode == 0)
        {
            for (i = 28, j = 0; (i >= 0) && (j < 15); i -= 2, j += 1)
            {
                auto element = (code.__rshift__(i)) & 0x3;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        else if (mode == 1)
        {
            for (i = 26, j = 0; (i >= 0) && (j < 7); i -= 4, j += 1)
            {
                auto element = (code.__rshift__(i)) & 0xf;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        else if (mode == 2)
        {
            for (i = 23, j = 0; (i >= 0) && (j < 4); i -= 7, j += 1)
            {
                auto element = (code.__rshift__(i)) & 0x7f;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        else if (mode == 3)
        {
            for (i = 22, j = 0; (i >= 0) && (j < 3); i -= 8, j += 1)
            {
                auto element = (code.__rshift__(i)) & 0xff;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        ind++;
    }

    torch::Tensor unpacked_tensor = torch::stack(unpacked_vector);

    return unpacked_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packing", &packing, "packing");
    m.def("unpacking", &unpacking, "unpacking");
}