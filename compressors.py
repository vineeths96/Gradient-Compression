import torch
import struct
import numpy as np


class NoneCompressor:
    def __init__(self, device):
        self._device = device

    def compress(self, tensor):
        compressed_tensor = tensor
        compressed_tensor_size = torch.tensor(compressed_tensor.size(), device=self._device)

        return compressed_tensor, compressed_tensor_size

    def decompress(self, compressed_tensor, compressed_tensor_size):
        unpadded_compressed_tensor = compressed_tensor[:compressed_tensor_size]

        return unpadded_compressed_tensor


class QSGDCompressor:
    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level
        self._sign_int_bit = 62
        self._encode_dict = self.elias_dict()

    def elias_dict(self):
        s = (1 << self._quantization_level) - 1
        keys = set(np.arange(0, s))
        encode_dict = dict.fromkeys(keys)

        for key in encode_dict:
            encode_dict[key] = self.elias_encode(key)

        return encode_dict

    def compress(self, tensor):
        s = (1 << self._quantization_level) - 1

        norm = torch.norm(tensor)

        sign_array = torch.sign(tensor)
        sign_array *= -1
        sign_array[sign_array == -1] = 0
        sign_array = sign_array.to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array).to(torch.int)
        xi_array = l_array_floored + mask

        code = ""
        code += self.float_to_bin(norm)

        for sign, xi in zip(sign_array, xi_array):
            code += str(sign.item())
            # code += self.elias_encode(xi.item())
            code += self._encode_dict[xi.item()]

        code_int_list = []
        for i in range(len(code) // self._sign_int_bit + 1):
            code_chunk = '1' + code[i * self._sign_int_bit: (i + 1) * self._sign_int_bit]
            code_int_list.append(int(code_chunk, 2))

        compressed_tensor = torch.tensor(code_int_list, dtype=torch.int64, device=self._device)
        compressed_tensor_size = torch.tensor(compressed_tensor.size(), device=self._device)

        # import sys
        # print("Org", sys.getsizeof(tensor.storage()), "Comp", sys.getsizeof(compressed_tensor.storage()))
        # print("Size", tensor.size(), tensor.dtype, compressed_tensor.size())

        return compressed_tensor, compressed_tensor_size

    def decompress(self, compressed_tensor, compressed_tensor_size):
        s = (1 << self._quantization_level) - 1

        unpadded_compressed_tensor = compressed_tensor[:compressed_tensor_size]
        code_int_list = unpadded_compressed_tensor.tolist()

        code = ""
        for ind, code_int in enumerate(code_int_list):
            if ind == len(code_int_list) - 1:
                code += bin(code_int)[3:]
                continue
            code += bin(code_int)[3:].zfill(self._sign_int_bit)

        norm = self.bin_to_float(code[:32])
        code = code[32:]

        xi_list = []
        sign_list = []

        while code != "":
            sign = int(code[0])

            xi, code = self.elias_decode(code[1:])
            sign_list.append(sign)
            xi_list.append(xi)

        norm = torch.tensor(norm) / s
        sign_array = torch.tensor(sign_list)
        xi_array = torch.tensor(xi_list)

        sign_array[sign_array == 1] = -1
        sign_array[sign_array == 0] = 1

        return norm * sign_array * xi_array

    def float_to_bin(self, num):
        return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

    def bin_to_float(self, binary):
        return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]

    def elias_encode(self, n):
        elias_code = "0"

        while n > 1:
            binary = bin(n)[2:]
            elias_code = binary + elias_code
            n = len(binary) - 1

        return elias_code

    def elias_decode(self, elias_code):
        n = 1

        while elias_code[0] != "0":
            m = int(elias_code[:n + 1], 2)
            elias_code = elias_code[n + 1:]
            n = m

        elias_code = elias_code[1:]

        return n, elias_code


class QSGDWECCompressor:
    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

    def compress(self, tensor):
        s = (1 << self._quantization_level) - 1

        # norm = torch.norm(tensor)
        norm = tensor.abs().max()

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int8)

        return norm, sign_array, xi_array

    def decompress(self, norm, sign_array, xi_array):
        return norm * sign_array * xi_array


class QSGDWECModCompressor:
    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

    def compress(self, tensor):
        s = (1 << self._quantization_level) - 1

        # norm = torch.norm(tensor)
        norm = tensor.abs().max()

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int8)

        sign_xi_array = sign_array * xi_array

        return norm, sign_xi_array

    def decompress(self, norm, sign_xi_array):
        return norm * sign_xi_array


class TernGradCompressor:
    def __init__(self, device):
        self._device = device

    def compress(self, tensor):
        scaler = tensor.abs().max()

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        prob_array = torch.abs(tensor) / scaler
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)
        b_array = torch.bernoulli(prob_array).to(torch.int8)

        return scaler, sign_array, b_array

    def decompress(self, scaler, sign_array, b_array):
        return scaler * sign_array * b_array


class TernGradModCompressor:
    def __init__(self, device):
        self._device = device

    def compress(self, tensor):
        scaler = tensor.abs().max()

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        prob_array = torch.abs(tensor) / scaler
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)
        b_array = torch.bernoulli(prob_array).to(torch.int8)

        sign_b_array = sign_array * b_array

        return scaler, sign_b_array

    def decompress(self, scaler, sign_b_array):
        return scaler * sign_b_array