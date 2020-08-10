import torch
import struct
import numpy as np


class NoneCompressor:
    """
    No compression.
    """
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
    """
    QSGD Compressor with Elias coding.
    Code: Elias coded string is represented in 64 bit integers.
    """
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

        norm = norm / s
        code = ""
        code += self.float_to_bin(norm)

        for sign, xi in zip(sign_array, xi_array):
            code += str(sign.item())
            code += self._encode_dict[xi.item()]

        code_int_list = []
        for i in range(len(code) // self._sign_int_bit + 1):
            code_chunk = '1' + code[i * self._sign_int_bit: (i + 1) * self._sign_int_bit]
            code_int_list.append(int(code_chunk, 2))

        compressed_tensor = torch.tensor(code_int_list, dtype=torch.int64, device=self._device)
        compressed_tensor_size = torch.tensor(compressed_tensor.size(), device=self._device)

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
    """
    QSGD Compressor without Elias coding.
    Code: norm, sign array, xi array.
    """
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
        xi_array = xi_array.to(dtype=torch.int32)

        norm = norm / s

        return norm, sign_array, xi_array

    def decompress(self, norm, sign_array, xi_array):
        return norm * sign_array * xi_array


class QSGDWECModCompressor:
    """
    Modified QSGD Compressor without Elias coding.
    Code: norm, sign array * xi array.
    """
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
        xi_array = xi_array.to(dtype=torch.int32)

        sign_xi_array = sign_array * xi_array

        norm = norm / s
        
        return norm, sign_xi_array

    def decompress(self, norm, sign_xi_array):
        return norm * sign_xi_array


class TernGradCompressor:
    """
    TernGrad Compressor.
    Code: norm, sign array, b array.
    """
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
    """
    TernGrad Compressor.
    Code: norm, sign array * b array.
    """
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



# Allreduce norm
class QSGDWECMod2Compressor:
    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

    def compress(self, norm, tensor):
        s = (1 << self._quantization_level) - 1

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int32)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int32)

        sign_xi_array = sign_array * xi_array

        return sign_xi_array

    def decompress(self, norm, sign_xi_array):
        return norm * sign_xi_array


# All gather norm max norm
class QSGDWECMod3Compressor:
    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

    def compress(self, norm, tensor):
        s = (1 << self._quantization_level) - 1

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int32)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int32)

        sign_xi_array = sign_array * xi_array

        return sign_xi_array

    def decompress(self, norm, sign_xi_array):
        return norm * sign_xi_array


import torch
import bitpacking
import gpu_bitpacking
class QSGDBPCompressor:
    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

    def compress(self, tensor):
        s = (1 << self._quantization_level) - 1

        # norm = torch.norm(tensor)
        norm = tensor.abs().max()

        sign_array = torch.sign(tensor).to(dtype=torch.int32)
        sign_array *= -1
        sign_array[sign_array == -1] = 0

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int32)

        sign_packed = bitpacking.packing(sign_array.to('cpu')).to(device=self._device)
        xi_packed = bitpacking.packing(xi_array.to('cpu')).to(device=self._device)
        xi_size = torch.tensor(xi_packed.size(), device=self._device)

        # sign_packed = gpu_bitpacking.packing(sign_array)
        # xi_packed = gpu_bitpacking.packing(xi_array)
        # xi_size = torch.tensor(xi_packed.size(), device=self._device)

        norm = norm / s

        return norm, sign_packed, xi_packed, xi_size

    def decompress(self, norm, sign_packed, xi_packed, tensor_size):
        sign_array = bitpacking.unpacking(sign_packed.to('cpu')).to(device=self._device)
        sign_array = sign_array[:tensor_size]
        xi_array = bitpacking.unpacking(xi_packed.to('cpu')).to(device=self._device)
        xi_array = xi_array[:tensor_size]

        sign_array[sign_array == 1] = -1
        sign_array[sign_array == 0] = 1

        return norm * sign_array * xi_array
