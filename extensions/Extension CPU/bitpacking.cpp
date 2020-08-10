#include <iostream>
#include <torch/extension.h>


torch::Tensor packing(torch::Tensor src)
{
    auto src_len = src.numel();
    auto src_accessor = src.accessor<int, 1>();
    std::vector<long> packed_vector;

    int64_t ind = 0;
    int64_t code_count = 0;
    int mode = -1;
    int i = 0, j = 0;

    while (ind < src_len)
    {
        long code = 0;
        if (src.slice(0, ind, ind + 15).max().item<int>() < 4)
        {
            mode = 0;
            for (i = 28, j = 0; (i >= 0) && (j < 15) && (ind + j) < src_len; i -= 2, j += 1)
                code = code | (src_accessor[ind + j] << i);

            ind += 15;
        }
        else if (src.slice(0, ind, ind + 7).max().item<int>() < 16)
        {
            mode = 1;
            for (i = 26, j = 0; (i >= 0) && (j < 7) && (ind + j) < src_len; i -= 4, j += 1)
                code = code | (src_accessor[ind + j] << i);

            ind += 7;
        }
        else if (src.slice(0, ind, ind + 4).max().item<int>() < 128)
        {
            mode = 2;
            for (i = 23, j = 0; (i >= 0) && (j < 4) && (ind + j) < src_len; i -= 7, j += 1)
                code = code | (src_accessor[ind + j] << i);

            ind += 4;
        }
        else if (src.slice(0, ind, ind + 3).max().item<int>() < 256)
        {
            mode = 3;
            for (i = 22, j = 0; (i >= 0) && (j < 3) && (ind + j) < src_len; i -= 8, j += 1)
                code = code | (src_accessor[ind + j] << i);

            ind += 3;
        }

        code = code | (mode << 30);
        packed_vector.push_back(code);
        code_count++;
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(src.device());
    torch::Tensor packed_tensor = torch::from_blob(packed_vector.data(), {code_count}, opts).to(torch::kInt32);

    return packed_tensor;
}


torch::Tensor unpacking(torch::Tensor src)
{
    auto src_len = src.numel();
    auto src_accessor = src.accessor<int, 1>();
    std::vector<long> unpacked_vector;

    int64_t ind = 0;
    int64_t element_count = 0;
    int mode = -1;
    int i = 0, j = 0;

    while (ind < src_len)
    {
        long code = src_accessor[ind];
        mode = (code & 0xc0000000) >> 30;
        int element;

        if (mode == 0)
        {
            for (i = 28, j = 0; (i >= 0) && (j < 15); i -= 2, j += 1)
            {
                element = (code >> i) & 0x3;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        else if (mode == 1)
        {
            for (i = 26, j = 0; (i >= 0) && (j < 7); i -= 4, j += 1)
            {
                element = (code >> i) & 0xf;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        else if (mode == 2)
        {
            for (i = 23, j = 0; (i >= 0) && (j < 4); i -= 7, j += 1)
            {
                element = (code >> i) & 0x7f;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        else if (mode == 3)
        {
            for (i = 22, j = 0; (i >= 0) && (j < 3); i -= 8, j += 1)
            {
                element = (code >> i) & 0xff;
                unpacked_vector.push_back(element);
                element_count++;
            }
        }
        ind++;
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(src.device());
    torch::Tensor unpacked_tensor = torch::from_blob(unpacked_vector.data(), {element_count}, opts).to(torch::kInt32);

    return unpacked_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packing", &packing, "packing");
    m.def("unpacking", &unpacking, "unpacking");
}