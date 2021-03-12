import torch
import struct

# import bitpacking
# import gpu_bitpacking
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
            code_chunk = "1" + code[i * self._sign_int_bit : (i + 1) * self._sign_int_bit]
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
        return format(struct.unpack("!I", struct.pack("!f", num))[0], "032b")

    def bin_to_float(self, binary):
        return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

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
            m = int(elias_code[: n + 1], 2)
            elias_code = elias_code[n + 1 :]
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

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

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
        xi_array = xi_array.to(dtype=self._dtype)

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

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

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

        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)
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


class QSGDMaxNormCompressor:
    """
    Modified QSGD Compressor without Elias coding.
    Normalizing with max norm among thw workers.
    Code: sign array * xi array.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

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

        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)

        return sign_xi_array

    def decompress(self, norm, sign_xi_array):
        s = (1 << self._quantization_level) - 1

        return norm / s * sign_xi_array


# class QSGDBPCompressor:
#     """
#     Modified QSGD Compressor without Elias coding.
#     Bit packing greedily in four modes.
#     Code: norm, sign_packed, xi_packed, xi_size
#     """
#
#     def __init__(self, device, quantization_level=8):
#         self._device = device
#         self._quantization_level = quantization_level
#
#     def compress(self, tensor):
#         s = (1 << self._quantization_level) - 1
#
#         # norm = torch.norm(tensor)
#         norm = tensor.abs().max()
#
#         sign_array = torch.sign(tensor).to(dtype=torch.int32)
#         sign_array *= -1
#         sign_array[sign_array == -1] = 0
#
#         l_array = torch.abs(tensor) / norm * s
#         l_array_floored = l_array.to(dtype=torch.int)
#         prob_array = l_array - l_array_floored
#         prob_array = torch.clamp(prob_array, min=0.0, max=1.0)
#
#         mask = torch.bernoulli(prob_array)
#         xi_array = l_array_floored + mask
#         xi_array = xi_array.to(dtype=torch.int32)
#
#         sign_packed = bitpacking.packing(sign_array.to("cpu")).to(device=self._device)
#         xi_packed = bitpacking.packing(xi_array.to("cpu")).to(device=self._device)
#         xi_size = torch.tensor(xi_packed.size(), device=self._device)
#
#         # sign_packed = gpu_bitpacking.packing(sign_array)
#         # xi_packed = gpu_bitpacking.packing(xi_array)
#         # xi_size = torch.tensor(xi_packed.size(), device=self._device)
#
#         norm = norm / s
#
#         return norm, sign_packed, xi_packed, xi_size
#
#     def decompress(self, norm, sign_packed, xi_packed, tensor_size):
#         sign_array = bitpacking.unpacking(sign_packed.to("cpu")).to(device=self._device)
#         sign_array = sign_array[:tensor_size]
#         xi_array = bitpacking.unpacking(xi_packed.to("cpu")).to(device=self._device)
#         xi_array = xi_array[:tensor_size]
#
#         sign_array[sign_array == 1] = -1
#         sign_array[sign_array == 0] = 1
#
#         return norm * sign_array * xi_array
#
#
# class QSGDBPAllReduceCompressor:
#     """
#     Modified QSGD Compressor without Elias coding.
#     Normalizing with max norm among thw workers.
#     Bit packing eight ints in 64bits to allreduce.
#     Code: sign_xi_packed
#     """
#
#     def __init__(self, device, quantization_level=8):
#         self._device = device
#         self._quantization_level = quantization_level
#
#     def compress(self, norm, tensor):
#         s = (1 << self._quantization_level) - 1
#
#         sign_array = torch.sign(tensor).to(dtype=torch.int8)
#
#         l_array = torch.abs(tensor) / norm * s
#         l_array_floored = l_array.to(dtype=torch.int32)
#         prob_array = l_array - l_array_floored
#         prob_array = torch.clamp(prob_array, min=0.0, max=1.0)
#
#         mask = torch.bernoulli(prob_array)
#         xi_array = l_array_floored + mask
#         xi_array = xi_array.to(dtype=torch.int32)
#
#         sign_xi_array = sign_array * xi_array
#         sign_xi_packed = bitpacking.packing(sign_xi_array.to("cpu")).to(device=self._device)
#
#         return sign_xi_packed
#
#     def decompress(self, norm, sign_xi_array):
#         s = (1 << self._quantization_level) - 1
#         sign_xi_unpacked = bitpacking.unpacking(sign_xi_array.to("cpu")).to(device=self._device)
#
#         return norm / s * sign_xi_unpacked


class GlobalRandKMaxNormCompressor:
    """
    RandK compressor with max norm.
    Normalizing with max norm among thw workers.
    Code: sign array * xi array.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

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

        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)

        return sign_xi_array

    def decompress(self, norm, sign_xi_array):
        s = (1 << self._quantization_level) - 1

        return norm / s * sign_xi_array


class MaxNormGlobalRandKCompressor:
    """
    Compressor with max norm.
    Normalizing with max norm among thw workers.
    Code: sign array * xi array.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

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

        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)

        return sign_xi_array

    def decompress(self, norm, sign_xi_array):
        s = (1 << self._quantization_level) - 1

        return norm / s * sign_xi_array


class NUQSGDModCompressor:
    """
    Non uniform QSGD Compressor without encoding.
    Code: norm, sign array * xi array.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

    def compress(self, tensor):
        s = 1 << self._quantization_level

        norm = torch.norm(tensor)
        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        r_array = torch.abs(tensor) / norm * s
        floored_log2 = torch.floor(torch.log2(r_array))
        floored_log2[floored_log2 < 0] = -float("inf")
        lsr = torch.pow(2, floored_log2)
        lsr_1 = torch.pow(2, floored_log2 + 1)
        lsr_1[lsr_1 == 0] = 1
        prob_array = (r_array - lsr) / (lsr_1 - lsr)
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        h_array = (1 - mask) * lsr + mask * lsr_1
        h_array = h_array.to(dtype=torch.int32)

        sign_h_array = (sign_array * h_array).to(dtype=self._dtype, device=self._device)
        norm = norm / s

        return norm, sign_h_array

    def decompress(self, norm, sign_h_array):
        return norm * sign_h_array


class NUQSGDMaxNormCompressor:
    """
    Modified Non uniform QSGD Compressor without encoding.
    Normalizing with max norm among thw workers.
    Code: sign array * xi array.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

    def compress(self, norm, tensor):
        s = 1 << self._quantization_level

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        r_array = torch.abs(tensor) / norm * s
        floored_log2 = torch.floor(torch.log2(r_array))
        floored_log2[floored_log2 < 0] = -float("inf")
        lsr = torch.pow(2, floored_log2)
        lsr_1 = torch.pow(2, floored_log2 + 1)
        lsr_1[lsr_1 == 0] = 1
        prob_array = (r_array - lsr) / (lsr_1 - lsr)
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        h_array = (1 - mask) * lsr + mask * lsr_1
        h_array = h_array.to(dtype=torch.int32)

        sign_h_array = (sign_array * h_array).to(dtype=self._dtype, device=self._device)

        return sign_h_array

    def decompress(self, norm, sign_h_array):
        s = 1 << self._quantization_level

        return norm / s * sign_h_array


class QSGDMaxNormBiasedCompressor:
    """
    Modified QSGD Compressor without Elias coding and randomized rounding.
    Normalizing with max norm among thw workers.
    Code: sign array * xi array.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

    def compress(self, norm, tensor):
        s = (1 << self._quantization_level) - 1

        l_array = tensor / norm * s
        l_array_floored = l_array.to(dtype=self._dtype, device=self._device)

        return l_array_floored

    def decompress(self, norm, l_array_floored):
        s = (1 << self._quantization_level) - 1

        return norm / s * l_array_floored


class NUQSGDMaxNormBiasedCompressor:
    """
    Modified Non uniform QSGD Compressor without encoding.
    Normalizing with max norm among thw workers.
    Code: sign array * xi array.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level

        if quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

    def compress(self, norm, tensor):
        s = 1 << self._quantization_level

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        r_array = torch.abs(tensor) / norm * s
        floored_log2 = torch.floor(torch.log2(r_array))
        floored_log2[floored_log2 < 0] = -float("inf")
        lsr = torch.pow(2, floored_log2)

        l_array_floored = (sign_array * lsr).to(dtype=self._dtype, device=self._device)

        return l_array_floored

    def decompress(self, norm, l_array_floored):
        s = 1 << self._quantization_level

        return norm / s * l_array_floored


class QSGDMaxNormTwoScaleCompressor:
    """
    QSGD MaxNorm Compressor with two scale compression.
    Normalizing with max norm among thw workers.
    Calculates common low resolution masks, and returns two scale vector
    Code: sign array * xi array.
    """

    def __init__(self, device, lower_quantization_level=6, higher_quantization_level=10):
        self._device = device
        self._lower_quantization_level = lower_quantization_level
        self._higher_quantization_level = higher_quantization_level

        if lower_quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

    def compress_lower(self, norm, tensor):
        s = (1 << self._lower_quantization_level) - 1

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int32)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int32)

        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)

        return sign_xi_array

    def compress_higher(self, norm, tensor):
        s_lower = (1 << self._lower_quantization_level) - 1
        s_higher = (1 << self._higher_quantization_level) - 1

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s_higher
        l_array_floored = l_array.to(dtype=torch.int32)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int32)

        higher_resolution_mask = (xi_array <= s_lower).to(torch.int8)
        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)

        return sign_xi_array, higher_resolution_mask

    def decompress(self, norm, sign_xi_array, higher_resolution_mask):
        s_lower = (1 << self._lower_quantization_level) - 1
        s_higher = (1 << self._higher_quantization_level) - 1

        decompressed_lower_scale = norm / s_lower * sign_xi_array
        decompressed_higher_scale = norm / s_higher * sign_xi_array

        decompressed_tensor = (
            higher_resolution_mask * decompressed_higher_scale
            + (1 - higher_resolution_mask) * decompressed_lower_scale
        )

        return decompressed_tensor


class GlobalRandKMaxNormTwoScaleCompressor:
    """
    Global RandK MaxNorm Compressor with two scale compression.
    Normalizing with max norm among thw workers.
    Calculates common low resolution masks, and returns two scale vector
    Code: sign array * xi array.
    """

    def __init__(self, device, lower_quantization_level=6, higher_quantization_level=10):
        self._device = device
        self._lower_quantization_level = lower_quantization_level
        self._higher_quantization_level = higher_quantization_level

        if lower_quantization_level < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

    def compress_lower(self, norm, tensor):
        s = (1 << self._lower_quantization_level) - 1

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int32)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int32)

        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)

        return sign_xi_array

    def compress_higher(self, norm, tensor):
        s_lower = (1 << self._lower_quantization_level) - 1
        s_higher = (1 << self._higher_quantization_level) - 1

        sign_array = torch.sign(tensor).to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s_higher
        l_array_floored = l_array.to(dtype=torch.int32)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array)
        xi_array = l_array_floored + mask
        xi_array = xi_array.to(dtype=torch.int32)

        higher_resolution_mask = (xi_array <= s_lower).to(torch.int8)
        sign_xi_array = (sign_array * xi_array).to(dtype=self._dtype, device=self._device)

        return sign_xi_array, higher_resolution_mask

    def decompress(self, norm, sign_xi_array, higher_resolution_mask):
        s_lower = (1 << self._lower_quantization_level) - 1
        s_higher = (1 << self._higher_quantization_level) - 1

        decompressed_lower_scale = norm / s_lower * sign_xi_array
        decompressed_higher_scale = norm / s_higher * sign_xi_array

        decompressed_tensor = (
            higher_resolution_mask * decompressed_higher_scale
            + (1 - higher_resolution_mask) * decompressed_lower_scale
        )

        return decompressed_tensor


class QSGDMaxNormMultiScaleCompressor:
    """
    QSGD MaxNorm Compressor with Multi scale compression.
    Normalizing with max norm among thw workers.
    Calculates common low resolution masks, and returns two scale vector
    Code: sign array * xi array.
    """

    def __init__(self, device, quantization_levels=None):
        self._device = device

        if not quantization_levels:
            quantization_levels = [6, 10]

        quantization_levels.sort()
        self._quantization_levels = quantization_levels

        if quantization_levels[0] < 8:
            self._dtype = torch.int8
        else:
            self._dtype = torch.int32

        self._cache = None

    def compress_cache(self, norm, tensor):
        if not self._cache:
            self._cache = torch.zeros(len(self._quantization_levels), tensor.size(0), device=self._device)

        for ind, quantization_level in enumerate(self._quantization_levels):
            s = (1 << quantization_level) - 1

            sign_array = torch.sign(tensor).to(dtype=torch.int8)

            l_array = torch.abs(tensor) / norm * s
            l_array_floored = l_array.to(dtype=torch.int32)
            prob_array = l_array - l_array_floored
            prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

            mask = torch.bernoulli(prob_array)
            xi_array = l_array_floored + mask
            xi_array = xi_array.to(dtype=torch.int32)

            sign_xi_array = sign_array * xi_array  # ).to(device=self._device)
            self._cache[ind] = sign_xi_array

    def compress_mask(self, norm, tensor):
        # TODO: Magic number 8
        MAX_VAL = 2 ** (7 - 1) - 1
        self.compress_cache(norm, tensor)

        resolution_mask = torch.zeros_like(tensor, dtype=torch.int8)
        for ind in range(len(self._quantization_levels)):
            resolution_mask[self._cache[ind].abs() <= MAX_VAL] = ind

        return resolution_mask

    def compress(self, resolution_mask):
        sign_xi_array = torch.zeros_like(resolution_mask, dtype=torch.float32)

        for ind in range(len(self._quantization_levels)):
            sign_xi_array[resolution_mask == ind] = self._cache[ind][resolution_mask == ind]

        sign_xi_array = sign_xi_array.to(dtype=self._dtype, device=self._device)

        return sign_xi_array

    def decompress(self, norm, sign_xi_array, resolution_mask):
        decompressed_tensor = torch.zeros_like(self._cache[0], dtype=torch.float32)

        for ind, quantization_level in enumerate(self._quantization_levels):
            s = (1 << quantization_level) - 1
            decompressed_tensor[resolution_mask == ind] = sign_xi_array[resolution_mask == ind] * norm / s

        return decompressed_tensor
