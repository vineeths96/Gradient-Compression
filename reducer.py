import torch
import torch.distributed

from compressors import (
    NoneCompressor,
    QSGDCompressor,
    QSGDWECCompressor,
    QSGDWECModCompressor,
    TernGradCompressor,
    TernGradModCompressor,
    QSGDMaxNormCompressor,
    # QSGDBPAllReduceCompressor,
    # QSGDBPCompressor,
    GlobalRandKMaxNormCompressor,
    MaxNormGlobalRandKCompressor,
    NUQSGDModCompressor,
    NUQSGDMaxNormCompressor,
    QSGDMaxNormBiasedCompressor,
    NUQSGDMaxNormBiasedCompressor,
    QSGDMaxNormTwoScaleCompressor,
    GlobalRandKMaxNormTwoScaleCompressor,
    QSGDMaxNormMultiScaleCompressor,
    # GlobalRandKMultiScaleCompressor,
)


class Reducer:
    """
    Base class for Custom Reducers. All reducers derive from this class.
    """

    def __init__(self, device, timer):
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0

        self._device = device
        self._timer = timer

    def reduce(self, grad_in, grad_out):
        raise NotImplementedError()


class TensorBuffer:
    """
    Class to flatten and deflatten the gradient vector.
    """

    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._len_tensors = len(tensors)
        self._tensor_shapes = [tensor.size() for tensor in tensors]

        self.buffer = torch.cat([tensor.view(-1) for tensor in tensors])

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(self._tensor_shapes[index])

    def __len__(self):
        return self._len_tensors


class NoneReducer(Reducer):
    """
    All gather reducer without any compressing.
    """

    def __init__(self, device, timer):
        super(NoneReducer, self).__init__(device, timer)

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = NoneCompressor(self._device)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.compress", verbosity=2):
            compressed_tensor, compressed_tensor_size = compressor.compress(flat_grad.buffer)

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_tensor_sizes = [torch.empty_like(compressed_tensor_size) for _ in range(self.n_workers)]
                size_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_tensor_sizes,
                    tensor=compressed_tensor_size,
                    async_op=True,
                )
                size_gather_op.wait()

                max_size = max(collected_tensor_sizes).item()
                padded_compressed_tensors = torch.zeros(max_size, dtype=torch.int64, device=self._device)
                padded_compressed_tensors[:compressed_tensor_size] = compressed_tensor

                collected_tensors = [
                    torch.zeros(max_size, dtype=torch.int64, device=self._device) for _ in range(self.n_workers)
                ]
                tensor_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_tensors,
                    tensor=padded_compressed_tensors,
                    async_op=True,
                )
                tensor_gather_op.wait()
            else:
                collected_tensors = [compressed_tensor]
                collected_tensor_sizes = [compressed_tensor_size]

        bits_communicated += self.n_bits(compressed_tensor) + self.n_bits(compressed_tensor_size)

        with self._timer("reduce.decompress", verbosity=2):
            decompressed_tensors = []
            for comp_tensor, comp_tensor_size in zip(collected_tensors, collected_tensor_sizes):
                decomp_tensor = compressor.decompress(comp_tensor, comp_tensor_size)
                decompressed_tensors.append(decomp_tensor)

        with self._timer("reduce.average", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for decompressed_tensor in decompressed_tensors:
                flat_grad.buffer = decompressed_tensor
                for grad, out in zip(flat_grad, grad_out):
                    # TODO Average or Sum
                    grad = grad.to(self._device)
                    out.add_(other=grad, alpha=1 / self.n_workers)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class NoneAllReducer(Reducer):
    """
    All reduce reducer without any compressing.
    """

    def __init__(self, device, timer):
        super(NoneAllReducer, self).__init__(device, timer)

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.allreduce", verbosity=2):
            if self.n_workers > 1:
                tensor_reduce_op = torch.distributed.all_reduce(tensor=flat_grad.buffer, async_op=True)
                tensor_reduce_op.wait()
            else:
                flat_grad = flat_grad

            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1 / self.n_workers)

            bits_communicated += self.n_bits(flat_grad.buffer)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDReducer(Reducer):
    """
    All gather reducer with QSGD compression and Elias encoding.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(QSGDReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.compress", verbosity=2):
            compressed_tensor, compressed_tensor_size = compressor.compress(flat_grad.buffer)

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_tensor_sizes = [torch.empty_like(compressed_tensor_size) for _ in range(self.n_workers)]
                size_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_tensor_sizes,
                    tensor=compressed_tensor_size,
                    async_op=True,
                )
                size_gather_op.wait()

                max_size = max(collected_tensor_sizes).item()
                padded_compressed_tensors = torch.zeros(max_size, dtype=torch.int64, device=self._device)
                padded_compressed_tensors[:compressed_tensor_size] = compressed_tensor

                collected_tensors = [
                    torch.zeros(max_size, dtype=torch.int64, device=self._device) for _ in range(self.n_workers)
                ]
                tensor_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_tensors,
                    tensor=padded_compressed_tensors,
                    async_op=True,
                )
                tensor_gather_op.wait()
            else:
                collected_tensors = [compressed_tensor]
                collected_tensor_sizes = [compressed_tensor_size]

        bits_communicated += self.n_bits(compressed_tensor) + self.n_bits(compressed_tensor_size)

        with self._timer("reduce.decompress", verbosity=2):
            decompressed_tensors = []
            for comp_tensor, comp_tensor_size in zip(collected_tensors, collected_tensor_sizes):
                decomp_tensor = compressor.decompress(comp_tensor, comp_tensor_size)
                decompressed_tensors.append(decomp_tensor)

        with self._timer("reduce.average", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for decompressed_tensor in decompressed_tensors:
                flat_grad.buffer = decompressed_tensor
                for grad, out in zip(flat_grad, grad_out):
                    # TODO Average or Sum
                    grad = grad.to(self._device)
                    out.add_(other=grad, alpha=1 / self.n_workers)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDWECReducer(Reducer):
    """
    All gather reducer with QSGD compression and without Elias encoding.
    All gathers norms, sign array and xi vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(QSGDWECReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDWECCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.compress", verbosity=2):
            norm, sign_array, xi_array = compressor.compress(flat_grad.buffer)

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                collected_signs = [torch.empty_like(sign_array) for _ in range(self.n_workers)]
                signs_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_signs, tensor=sign_array, async_op=True
                )

                collected_xis = [torch.empty_like(xi_array) for _ in range(self.n_workers)]
                xi_gather_op = torch.distributed.all_gather(tensor_list=collected_xis, tensor=xi_array, async_op=True)

                norms_gather_op.wait()
                signs_gather_op.wait()
                xi_gather_op.wait()
            else:
                collected_norms = [norm]
                collected_signs = [sign_array]
                collected_xis = [xi_array]

        bits_communicated += self.n_bits(norm) + self.n_bits(sign_array) + self.n_bits(xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            decompressed_tensors = []
            for norm, sign_array, xi_array in zip(collected_norms, collected_signs, collected_xis):
                decomp_tensor = compressor.decompress(norm, sign_array, xi_array)
                decompressed_tensors.append(decomp_tensor)

        with self._timer("reduce.average", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for decompressed_tensor in decompressed_tensors:
                flat_grad.buffer = decompressed_tensor
                for grad, out in zip(flat_grad, grad_out):
                    # TODO Average or Sum
                    grad = grad.to(self._device)
                    out.add_(other=grad, alpha=1 / self.n_workers)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDWECModReducer(Reducer):
    """
    All gather reducer with QSGD compression and without Elias encoding.
    All gathers norms, sign array * xi vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(QSGDWECModReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDWECModCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.compress", verbosity=2):
            norm, sign_xi_array = compressor.compress(flat_grad.buffer)

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                collected_sign_xis = [torch.empty_like(sign_xi_array) for _ in range(self.n_workers)]
                sign_xis_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_sign_xis, tensor=sign_xi_array, async_op=True
                )

                norms_gather_op.wait()
                sign_xis_gather_op.wait()
            else:
                collected_norms = [norm]
                collected_sign_xis = [sign_xi_array]

        bits_communicated += self.n_bits(norm) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            decompressed_tensors = []
            for norm, sign_xi_array in zip(collected_norms, collected_sign_xis):
                decomp_tensor = compressor.decompress(norm, sign_xi_array)
                decompressed_tensors.append(decomp_tensor)

        with self._timer("reduce.average", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for decompressed_tensor in decompressed_tensors:
                flat_grad.buffer = decompressed_tensor
                for grad, out in zip(flat_grad, grad_out):
                    # TODO Average or Sum
                    grad = grad.to(self._device)
                    out.add_(other=grad, alpha=1 / self.n_workers)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class TernGradReducer(Reducer):
    """
    All gather reducer with TernGrad compression.
    All gathers norms, sign array and b vector.
    """

    def __init__(self, device, timer):
        super(TernGradReducer, self).__init__(device, timer)

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = TernGradCompressor(self._device)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.compress", verbosity=2):
            scaler, sign_array, b_array = compressor.compress(flat_grad.buffer)

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_scalers = [torch.empty_like(scaler) for _ in range(self.n_workers)]
                scaler_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_scalers, tensor=scaler, async_op=True
                )

                collected_signs = [torch.empty_like(sign_array) for _ in range(self.n_workers)]
                signs_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_signs, tensor=sign_array, async_op=True
                )

                collected_bs = [torch.empty_like(b_array) for _ in range(self.n_workers)]
                b_gather_op = torch.distributed.all_gather(tensor_list=collected_bs, tensor=b_array, async_op=True)

                scaler_gather_op.wait()
                signs_gather_op.wait()
                b_gather_op.wait()
            else:
                collected_scalers = [scaler]
                collected_signs = [sign_array]
                collected_bs = [b_array]

        bits_communicated += self.n_bits(scaler) + self.n_bits(sign_array) + self.n_bits(b_array)

        with self._timer("reduce.decompress", verbosity=2):
            decompressed_tensors = []
            for scaler, sign_array, b_array in zip(collected_scalers, collected_signs, collected_bs):
                decomp_tensor = compressor.decompress(scaler, sign_array, b_array)
                decompressed_tensors.append(decomp_tensor)

        with self._timer("reduce.average", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for decompressed_tensor in decompressed_tensors:
                flat_grad.buffer = decompressed_tensor
                for grad, out in zip(flat_grad, grad_out):
                    # TODO Average or Sum
                    grad = grad.to(self._device)
                    out.add_(other=grad, alpha=1 / self.n_workers)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class TernGradModReducer(Reducer):
    """
    All gather reducer with TernGrad compression.
    All gathers norms, sign array * xi vector.
    """

    def __init__(self, device, timer):
        super(TernGradModReducer, self).__init__(device, timer)

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = TernGradModCompressor(self._device)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.compress", verbosity=2):
            scaler, sign_b_array = compressor.compress(flat_grad.buffer)

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_scalers = [torch.empty_like(scaler) for _ in range(self.n_workers)]
                scaler_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_scalers, tensor=scaler, async_op=True
                )

                collected_sign_bs = [torch.empty_like(sign_b_array) for _ in range(self.n_workers)]
                sign_bs_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_sign_bs, tensor=sign_b_array, async_op=True
                )

                scaler_gather_op.wait()
                sign_bs_gather_op.wait()
            else:
                collected_scalers = [scaler]
                collected_sign_bs = [sign_b_array]

        bits_communicated += self.n_bits(scaler) + self.n_bits(sign_b_array)

        with self._timer("reduce.decompress", verbosity=2):
            decompressed_tensors = []
            for scaler, sign_b_array in zip(collected_scalers, collected_sign_bs):
                decomp_tensor = compressor.decompress(scaler, sign_b_array)
                decompressed_tensors.append(decomp_tensor)

        with self._timer("reduce.average", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for decompressed_tensor in decompressed_tensors:
                flat_grad.buffer = decompressed_tensor
                for grad, out in zip(flat_grad, grad_out):
                    # TODO Average or Sum
                    grad = grad.to(self._device)
                    out.add_(other=grad, alpha=1 / self.n_workers)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDMaxNormReducer(Reducer):
    """
    All reduce reducer with QSGD compression and without Elias encoding.
    All gathers norms, normalizing with max norm, all reduces sign array * xi vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(QSGDMaxNormReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDMaxNormCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            sign_xi_array = compressor.compress(max_norm, flat_grad.buffer)

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
                sign_xi_reduce_op.wait()
                sign_xi_array.true_divide(self.n_workers)
            else:
                sign_xi_array = sign_xi_array

        bits_communicated += self.n_bits(norm) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, sign_xi_array)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


# class QSGDBPReducer(Reducer):
#     """
#     All gather reducer with QSGD compression and without Elias encoding.
#     All gathers norms, bit packed sign vector, bit packed xi vector.
#     """
#
#     def __init__(self, device, timer, quantization_level=8):
#         super(QSGDBPReducer, self).__init__(device, timer)
#         self._quantization_level = quantization_level
#
#     def reduce(self, grad_in, grad_out):
#         bits_communicated = 0
#         compressor = QSGDBPCompressor(self._device, self._quantization_level)
#
#         with self._timer("reduce.flat_pack"):
#             flat_grad = TensorBuffer(grad_in)
#             tensor_size = flat_grad.buffer.shape[0]
#
#         with self._timer("reduce.compress", verbosity=2):
#             norm, sign_packed, xi_packed, xi_size = compressor.compress(flat_grad.buffer)
#
#         with self._timer("reduce.gather", verbosity=2):
#             if self.n_workers > 1:
#                 collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
#                 norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)
#
#                 collected_signs = [torch.empty_like(sign_packed) for _ in range(self.n_workers)]
#                 signs_gather_op = torch.distributed.all_gather(
#                     tensor_list=collected_signs, tensor=sign_packed, async_op=True
#                 )
#
#                 collected_xi_sizes = [torch.empty_like(xi_size) for _ in range(self.n_workers)]
#                 size_gather_op = torch.distributed.all_gather(
#                     tensor_list=collected_xi_sizes, tensor=xi_size, async_op=True
#                 )
#                 size_gather_op.wait()
#
#                 max_size = max(collected_xi_sizes).item()
#                 padded_xi_tensor = torch.zeros(max_size, dtype=torch.int32, device=self._device)
#                 padded_xi_tensor[:xi_size] = xi_packed
#
#                 collected_xis = [torch.empty_like(padded_xi_tensor) for _ in range(self.n_workers)]
#                 xi_gather_op = torch.distributed.all_gather(
#                     tensor_list=collected_xis, tensor=padded_xi_tensor, async_op=True
#                 )
#
#                 norms_gather_op.wait()
#                 signs_gather_op.wait()
#                 xi_gather_op.wait()
#             else:
#                 collected_norms = [norm]
#                 collected_signs = [sign_packed]
#                 collected_xis = [xi_packed]
#
#         bits_communicated += (
#             self.n_bits(norm) + self.n_bits(sign_packed) + self.n_bits(xi_packed) + self.n_bits(xi_size)
#         )
#
#         with self._timer("reduce.decompress", verbosity=2):
#             decompressed_tensors = []
#             for norm, sign_packed, xi_packed in zip(collected_norms, collected_signs, collected_xis):
#                 decomp_tensor = compressor.decompress(norm, sign_packed, xi_packed, tensor_size)
#                 decompressed_tensors.append(decomp_tensor)
#
#         with self._timer("reduce.average", verbosity=2):
#             for out in grad_out:
#                 out[:] = 0.0
#
#             for decompressed_tensor in decompressed_tensors:
#                 flat_grad.buffer = decompressed_tensor
#                 for grad, out in zip(flat_grad, grad_out):
#                     # TODO Average or Sum
#                     grad = grad.to(self._device)
#                     out.add_(other=grad, alpha=1 / self.n_workers)
#
#         return bits_communicated
#
#     def n_bits(self, tensor):
#         return 8 * tensor.nelement() * tensor.element_size()
#
#
# class QSGDBPAllReducer(Reducer):
#     """
#     All reduce reducer with QSGD compression and without Elias encoding.
#     All gathers norms, normalizing with max norm, all reduces packed sign array * xi vector.
#     """
#
#     def __init__(self, device, timer, quantization_level=8):
#         super(QSGDBPAllReducer, self).__init__(device, timer)
#         self._quantization_level = quantization_level
#
#     def reduce(self, grad_in, grad_out):
#         bits_communicated = 0
#         compressor = QSGDBPAllReduceCompressor(self._device, self._quantization_level)
#
#         with self._timer("reduce.flat_pack"):
#             flat_grad = TensorBuffer(grad_in)
#
#         with self._timer("reduce.reduce.norm", verbosity=2):
#             norm = flat_grad.buffer.abs().max()
#
#             if self.n_workers > 1:
#                 collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
#                 norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)
#
#                 norms_gather_op.wait()
#                 max_norm = max(collected_norms)
#             else:
#                 max_norm = norm
#
#         with self._timer("reduce.compress", verbosity=2):
#             sign_xi_array = compressor.compress(max_norm, flat_grad.buffer)
#
#         with self._timer("reduce.reduce.vector", verbosity=2):
#             if self.n_workers > 1:
#                 sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
#                 sign_xi_reduce_op.wait()
#                 sign_xi_array.true_divide(self.n_workers)
#             else:
#                 sign_xi_array = sign_xi_array
#
#         bits_communicated += self.n_bits(norm) + self.n_bits(sign_xi_array)
#
#         with self._timer("reduce.decompress", verbosity=2):
#             flat_grad.buffer = compressor.decompress(max_norm, sign_xi_array)
#
#         with self._timer("reduce.setgrad", verbosity=2):
#             for out in grad_out:
#                 out[:] = 0.0
#
#             for grad, out in zip(flat_grad, grad_out):
#                 # TODO Average or Sum
#                 grad = grad.to(self._device)
#                 out.add_(other=grad, alpha=1)
#
#         return bits_communicated
#
#     def n_bits(self, tensor):
#         return 8 * tensor.nelement() * tensor.element_size()


class GlobalRandKMaxNormReducer(Reducer):
    """
    All reduce reducer with max norm compression of random K indices.
    All gathers norms, normalizing with max norm, all reduces sign array * xi vector.
    """

    def __init__(self, device, timer, K=10000, quantization_level=8):
        super(GlobalRandKMaxNormReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level
        self._K = K
        self._indices_queue = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = GlobalRandKMaxNormCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._indices_queue:
            self._indices_queue = torch.randperm(len(flat_grad.buffer)).split(self._K)
            self._indices_queue = list(self._indices_queue)

        RandK_indices = self._indices_queue.pop().numpy()
        RandK_flat_grad = flat_grad.buffer[RandK_indices]

        with self._timer("reduce.norm", verbosity=2):
            norm = RandK_flat_grad.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            sign_xi_array = compressor.compress(max_norm, RandK_flat_grad)

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
                sign_xi_reduce_op.wait()
                sign_xi_array.true_divide(self.n_workers)
            else:
                sign_xi_array = sign_xi_array

        bits_communicated += self.n_bits(norm) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            RandK_decompressed = compressor.decompress(max_norm, sign_xi_array)

        with self._timer("reduce.setgrad", verbosity=2):
            flat_grad.buffer[RandK_indices] = RandK_decompressed

            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class MaxNormGlobalRandKReducer(Reducer):
    """
    All reduce reducer of random K indices with max norm compression.
    All gathers norms, normalizing with max norm, all reduces sign array * xi vector.
    """

    def __init__(self, device, timer, K=10000, quantization_level=8):
        super(MaxNormGlobalRandKReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level
        self._K = K
        self._indices_queue = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = MaxNormGlobalRandKCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._indices_queue:
            self._indices_queue = torch.randperm(len(flat_grad.buffer)).split(self._K)
            self._indices_queue = list(self._indices_queue)

        RandK_indices = self._indices_queue.pop().numpy()

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            sign_xi_array = compressor.compress(max_norm, flat_grad.buffer)
            sign_xi_array = sign_xi_array[RandK_indices]

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
                sign_xi_reduce_op.wait()
                sign_xi_array.true_divide(self.n_workers)
            else:
                sign_xi_array = sign_xi_array

        bits_communicated += self.n_bits(norm) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            RandK_decompressed = compressor.decompress(max_norm, sign_xi_array)

        with self._timer("reduce.setgrad", verbosity=2):
            flat_grad.buffer[RandK_indices] = RandK_decompressed

            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class NUQSGDModReducer(Reducer):
    """
    All gather reducer with NUQSGD compression and without encoding.
    All gathers norms, sign array * xi vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(NUQSGDModReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = NUQSGDModCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.compress", verbosity=2):
            norm, sign_xi_array = compressor.compress(flat_grad.buffer)

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                collected_sign_xis = [torch.empty_like(sign_xi_array) for _ in range(self.n_workers)]
                sign_xis_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_sign_xis, tensor=sign_xi_array, async_op=True
                )

                norms_gather_op.wait()
                sign_xis_gather_op.wait()
            else:
                collected_norms = [norm]
                collected_sign_xis = [sign_xi_array]

        bits_communicated += self.n_bits(norm) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            decompressed_tensors = []
            for norm, sign_xi_array in zip(collected_norms, collected_sign_xis):
                decomp_tensor = compressor.decompress(norm, sign_xi_array)
                decompressed_tensors.append(decomp_tensor)

        with self._timer("reduce.average", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for decompressed_tensor in decompressed_tensors:
                flat_grad.buffer = decompressed_tensor
                for grad, out in zip(flat_grad, grad_out):
                    # TODO Average or Sum
                    grad = grad.to(self._device)
                    out.add_(other=grad, alpha=1 / self.n_workers)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class NUQSGDMaxNormReducer(Reducer):
    """
    All reduce reducer with NUQSGD compression and without encoding.
    All gathers norms, normalizing with max norm, all reduces sign array * xi vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(NUQSGDMaxNormReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = NUQSGDMaxNormCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.norm()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            sign_xi_array = compressor.compress(max_norm, flat_grad.buffer)

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
                sign_xi_reduce_op.wait()
                sign_xi_array.true_divide(self.n_workers)
            else:
                sign_xi_array = sign_xi_array

        bits_communicated += self.n_bits(norm) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, sign_xi_array)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDMaxNormBiasedReducer(Reducer):
    """
    All reduce reducer with Biased QSGD compression and without Elias encoding.
    All gathers norms, normalizing with max norm, all reduces floored vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(QSGDMaxNormBiasedReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDMaxNormBiasedCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            l_array_floored = compressor.compress(max_norm, flat_grad.buffer)

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                l_array_floored_op = torch.distributed.all_reduce(tensor=l_array_floored, async_op=True)
                l_array_floored_op.wait()
                l_array_floored.true_divide(self.n_workers)
            else:
                l_array_floored = l_array_floored

        bits_communicated += self.n_bits(norm) + self.n_bits(l_array_floored)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, l_array_floored)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDMaxNormBiasedMemoryReducer(Reducer):
    """
    All reduce reducer with Biased QSGD compression with memory and without Elias encoding.
    All gathers norms, normalizing with max norm, all reduces floored vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(QSGDMaxNormBiasedMemoryReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level
        self._memory = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDMaxNormBiasedCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._memory:
            self._memory = [torch.zeros_like(grad_tensor) for grad_tensor in grad_in]
            self._memory = TensorBuffer(self._memory)
        else:
            flat_grad.buffer[:] += self._memory.buffer

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            l_array_floored = compressor.compress(max_norm, flat_grad.buffer)

        with self._timer("reduce.set_memory", verbosity=2):
            self._memory.buffer[:] = flat_grad.buffer - l_array_floored

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                l_array_floored_op = torch.distributed.all_reduce(tensor=l_array_floored, async_op=True)
                l_array_floored_op.wait()
                l_array_floored.true_divide(self.n_workers)
            else:
                l_array_floored = l_array_floored

        bits_communicated += self.n_bits(norm) + self.n_bits(l_array_floored)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, l_array_floored)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class NUQSGDMaxNormBiasedReducer(Reducer):
    """
    All reduce reducer with Biased NUQSGD compression and without encoding.
    All gathers norms, normalizing with max norm, all reduces sign array * xi vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(NUQSGDMaxNormBiasedReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = NUQSGDMaxNormBiasedCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            l_array_floored = compressor.compress(max_norm, flat_grad.buffer)

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                l_array_floored_op = torch.distributed.all_reduce(tensor=l_array_floored, async_op=True)
                l_array_floored_op.wait()
                l_array_floored.true_divide(self.n_workers)
            else:
                l_array_floored = l_array_floored

        bits_communicated += self.n_bits(norm) + self.n_bits(l_array_floored)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, l_array_floored)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class NUQSGDMaxNormBiasedMemoryReducer(Reducer):
    """
    All reduce reducer with Biased NUQSGD compression with memory and without encoding.
    All gathers norms, normalizing with max norm, all reduces floored vector.
    """

    def __init__(self, device, timer, quantization_level=8):
        super(NUQSGDMaxNormBiasedMemoryReducer, self).__init__(device, timer)
        self._quantization_level = quantization_level
        self._memory = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = NUQSGDMaxNormBiasedCompressor(self._device, self._quantization_level)

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._memory:
            self._memory = [torch.zeros_like(grad_tensor) for grad_tensor in grad_in]
            self._memory = TensorBuffer(self._memory)
        else:
            flat_grad.buffer[:] += self._memory.buffer

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            l_array_floored = compressor.compress(max_norm, flat_grad.buffer)

        with self._timer("reduce.set_memory", verbosity=2):
            self._memory.buffer[:] = flat_grad.buffer - l_array_floored

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                l_array_floored_op = torch.distributed.all_reduce(tensor=l_array_floored, async_op=True)
                l_array_floored_op.wait()
                l_array_floored.true_divide(self.n_workers)
            else:
                l_array_floored = l_array_floored

        bits_communicated += self.n_bits(norm) + self.n_bits(l_array_floored)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, l_array_floored)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class TopKReducer(Reducer):
    """
    TopK reducer with K most important gradient updates layerwise.
    All gathers values and indices of top-K from each worker and updates.
    """

    def __init__(self, device, timer, K=100):
        super(TopKReducer, self).__init__(device, timer)
        self._K = K
        self._memory = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        if not self._memory:
            self._memory = [torch.zeros_like(grad_tensor) for grad_tensor in grad_in]
        else:
            for grad, memory in zip(grad_in, self._memory):
                grad.add_(other=memory, alpha=1)

        with self._timer("reduce.flatpack", verbosity=2):
            flat_grad_size = 0
            tensor_topK_indices = [0]
            for tensor in grad_in:
                top_size = min(tensor.nelement(), self._K)
                flat_grad_size += top_size
                tensor_topK_indices.append(tensor_topK_indices[-1] + top_size)

            flat_grad_start_indices = tensor_topK_indices[:-1]
            flat_grad_end_indices = tensor_topK_indices[1:]
            flat_values = torch.empty(flat_grad_size, device=self._device)
            flat_positions = torch.empty(flat_grad_size, device=self._device, dtype=torch.int)

        with self._timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flat_grad_start_indices, flat_grad_end_indices):
                top_size = min(tensor.nelement(), self._K)
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self._timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(grad_in, self._memory, flat_grad_start_indices, flat_grad_end_indices):
                positions = flat_positions[start:end]
                mem[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_values = [torch.empty_like(flat_values) for _ in range(self.n_workers)]
                values_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_values, tensor=flat_values, async_op=True
                )

                collected_positions = [torch.empty_like(flat_positions) for _ in range(self.n_workers)]
                positions_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_positions,
                    tensor=flat_positions,
                    async_op=True,
                )

                values_gather_op.wait()
                positions_gather_op.wait()
            else:
                collected_values = [flat_values]
                collected_positions = [flat_positions]

        bits_communicated += self.n_bits(flat_values) + self.n_bits(flat_positions)

        with self._timer("reduce.combine", verbosity=2):
            for out, start, end in zip(grad_out, flat_grad_start_indices, flat_grad_end_indices):
                out[:] = 0

                for pos, val in zip(collected_positions, collected_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    out.view(-1)[positions.long()] += values / self.n_workers

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class TopKReducerRatio(Reducer):
    """
    TopK reducer with ratio most important gradient updates layerwise.
    All gathers values and indices of top-K from each worker and updates.
    """

    def __init__(self, device, timer, compression=1 / 100):
        super(TopKReducerRatio, self).__init__(device, timer)
        self._compression = compression
        self._memory = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        if not self._memory:
            self._memory = [torch.zeros_like(grad_tensor) for grad_tensor in grad_in]
        else:
            for grad, memory in zip(grad_in, self._memory):
                grad.add_(other=memory, alpha=1)

        with self._timer("reduce.flatpack", verbosity=2):
            flat_grad_size = 0
            tensor_topK_indices = [0]
            for tensor in grad_in:
                top_size = max(1, int(self._compression * tensor.nelement()))
                flat_grad_size += top_size
                tensor_topK_indices.append(tensor_topK_indices[-1] + top_size)

            flat_grad_start_indices = tensor_topK_indices[:-1]
            flat_grad_end_indices = tensor_topK_indices[1:]
            flat_values = torch.empty(flat_grad_size, device=self._device)
            flat_positions = torch.empty(flat_grad_size, device=self._device, dtype=torch.int)

        with self._timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flat_grad_start_indices, flat_grad_end_indices):
                top_size = max(1, int(self._compression * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self._timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(grad_in, self._memory, flat_grad_start_indices, flat_grad_end_indices):
                positions = flat_positions[start:end]
                mem[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_values = [torch.empty_like(flat_values) for _ in range(self.n_workers)]
                values_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_values, tensor=flat_values, async_op=True
                )

                collected_positions = [torch.empty_like(flat_positions) for _ in range(self.n_workers)]
                positions_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_positions,
                    tensor=flat_positions,
                    async_op=True,
                )

                values_gather_op.wait()
                positions_gather_op.wait()
            else:
                collected_values = [flat_values]
                collected_positions = [flat_positions]

        bits_communicated += self.n_bits(flat_values) + self.n_bits(flat_positions)

        with self._timer("reduce.combine", verbosity=2):
            for out, start, end in zip(grad_out, flat_grad_start_indices, flat_grad_end_indices):
                out[:] = 0

                for pos, val in zip(collected_positions, collected_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    out.view(-1)[positions.long()] += values / self.n_workers

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class GlobalTopKReducer(Reducer):
    """
    TopK reducer with K most important gradient updates global.
    All gathers values and indices of top-K from each worker and updates.
    """

    def __init__(self, device, timer, K=10000):
        super(GlobalTopKReducer, self).__init__(device, timer)
        self._K = K
        self._memory = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._memory:
            self._memory = [torch.zeros_like(grad_tensor) for grad_tensor in grad_in]
            self._memory = TensorBuffer(self._memory)
        else:
            flat_grad.buffer[:] += self._memory.buffer

        top_size = min(flat_grad.buffer.nelement(), self._K)

        with self._timer("reduce.topk", verbosity=2):
            _, positions = torch.topk(flat_grad.buffer.abs(), top_size, sorted=False)
            values = flat_grad.buffer[positions].contiguous()

        with self._timer("reduce.set_memory", verbosity=2):
            self._memory.buffer[:] = flat_grad.buffer
            self._memory.buffer[positions] = 0.0

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_values = [torch.empty_like(values) for _ in range(self.n_workers)]
                values_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_values, tensor=values, async_op=True
                )

                collected_positions = [torch.empty_like(positions) for _ in range(self.n_workers)]
                positions_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_positions, tensor=positions, async_op=True
                )

                values_gather_op.wait()
                positions_gather_op.wait()
            else:
                collected_values = [values]
                collected_positions = [positions]

        bits_communicated += self.n_bits(values) + self.n_bits(positions)

        with self._timer("reduce.combine", verbosity=2):
            for pos, val in zip(collected_positions, collected_values):
                flat_grad.buffer[pos] += val / self.n_workers

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class GlobalTopKReducerRatio(Reducer):
    """
    TopK reducer with ratio most important gradient updates global.
    All gathers values and indices of top-K from each worker and updates.
    """

    def __init__(self, device, timer, compression=1 / 100):
        super(GlobalTopKReducerRatio, self).__init__(device, timer)
        self._compression = compression
        self._memory = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._memory:
            self._memory = [torch.zeros_like(grad_tensor) for grad_tensor in grad_in]
            self._memory = TensorBuffer(self._memory)
        else:
            flat_grad.buffer[:] += self._memory.buffer

        top_size = max(1, int(self._compression * flat_grad.buffer.nelement()))

        with self._timer("reduce.topk", verbosity=2):
            _, positions = torch.topk(flat_grad.buffer.abs(), top_size, sorted=False)
            values = flat_grad.buffer[positions].contiguous()

        with self._timer("reduce.set_memory", verbosity=2):
            self._memory.buffer[:] = flat_grad.buffer
            self._memory.buffer[positions] = 0.0

        with self._timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                collected_values = [torch.empty_like(values) for _ in range(self.n_workers)]
                values_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_values, tensor=values, async_op=True
                )

                collected_positions = [torch.empty_like(positions) for _ in range(self.n_workers)]
                positions_gather_op = torch.distributed.all_gather(
                    tensor_list=collected_positions, tensor=positions, async_op=True
                )

                values_gather_op.wait()
                positions_gather_op.wait()
            else:
                collected_values = [values]
                collected_positions = [positions]

        bits_communicated += self.n_bits(values) + self.n_bits(positions)

        with self._timer("reduce.combine", verbosity=2):
            for pos, val in zip(collected_positions, collected_values):
                flat_grad.buffer[pos] += val / self.n_workers

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDMaxNormTwoScaleReducer(Reducer):
    """
    All reduce reducer with QSGD MaxNorm Two Level compression.
    All gathers norms, normalizing with max norm, find common low resolution mask,
    All reduces two scale sign array * xi vector.
    """

    def __init__(self, device, timer, lower_quantization_level=6, higher_quantization_level=10):
        super(QSGDMaxNormTwoScaleReducer, self).__init__(device, timer)
        self._lower_quantization_level = lower_quantization_level
        self._higher_quantization_level = higher_quantization_level

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDMaxNormTwoScaleCompressor(
            self._device,
            self._lower_quantization_level,
            self._higher_quantization_level,
        )

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            sign_xi_array_lower = compressor.compress_lower(max_norm, flat_grad.buffer)
            sign_xi_array_higher, higher_resolution_mask = compressor.compress_higher(max_norm, flat_grad.buffer)

            if self.n_workers > 1:
                high_mask_op = torch.distributed.all_reduce(
                    tensor=higher_resolution_mask,
                    op=torch.distributed.ReduceOp.PRODUCT,
                    async_op=True,
                )
                high_mask_op.wait()
            else:
                higher_resolution_mask = higher_resolution_mask

            sign_xi_array = (
                higher_resolution_mask * sign_xi_array_higher + (1 - higher_resolution_mask) * sign_xi_array_lower
            )

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
                sign_xi_reduce_op.wait()
                sign_xi_array.true_divide(self.n_workers)
            else:
                sign_xi_array = sign_xi_array

        bits_communicated += self.n_bits(norm) + self.n_bits(higher_resolution_mask) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, sign_xi_array, higher_resolution_mask)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class GlobalRandKMaxNormTwoScaleReducer(Reducer):
    """
    All reduce reducer with QSGD MaxNorm Two Level compression of random K indices.
    All gathers norms, normalizing with max norm, find common low resolution mask,
    All reduces two scale sign array * xi vector.
    """

    def __init__(
        self,
        device,
        timer,
        K=10000,
        lower_quantization_level=6,
        higher_quantization_level=10,
    ):
        super(GlobalRandKMaxNormTwoScaleReducer, self).__init__(device, timer)
        self._lower_quantization_level = lower_quantization_level
        self._higher_quantization_level = higher_quantization_level
        self._K = K
        self._indices_queue = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = GlobalRandKMaxNormTwoScaleCompressor(
            self._device,
            self._lower_quantization_level,
            self._higher_quantization_level,
        )

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._indices_queue:
            self._indices_queue = torch.randperm(len(flat_grad.buffer)).split(self._K)
            self._indices_queue = list(self._indices_queue)

        RandK_indices = self._indices_queue.pop().numpy()
        RandK_flat_grad = flat_grad.buffer[RandK_indices]

        with self._timer("reduce.norm", verbosity=2):
            norm = RandK_flat_grad.abs().max()

            if self.n_workers > 1:
                collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
                norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)

                norms_gather_op.wait()
                max_norm = max(collected_norms)
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            sign_xi_array_lower = compressor.compress_lower(max_norm, RandK_flat_grad)
            sign_xi_array_higher, higher_resolution_mask = compressor.compress_higher(max_norm, RandK_flat_grad)

            if self.n_workers > 1:
                high_mask_op = torch.distributed.all_reduce(
                    tensor=higher_resolution_mask,
                    op=torch.distributed.ReduceOp.PRODUCT,
                    async_op=True,
                )
                high_mask_op.wait()

            else:
                higher_resolution_mask = higher_resolution_mask

            sign_xi_array = (
                higher_resolution_mask * sign_xi_array_higher + (1 - higher_resolution_mask) * sign_xi_array_lower
            )

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
                sign_xi_reduce_op.wait()
                sign_xi_array.true_divide(self.n_workers)
            else:
                sign_xi_array = sign_xi_array

        bits_communicated += self.n_bits(norm) + self.n_bits(higher_resolution_mask) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            RandK_decompressed = compressor.decompress(max_norm, sign_xi_array, higher_resolution_mask)

        with self._timer("reduce.setgrad", verbosity=2):
            flat_grad.buffer[RandK_indices] = RandK_decompressed

            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


class QSGDMaxNormMultiScaleReducer(Reducer):
    """
    All reduce reducer with QSGD MaxNorm Multi Level compression.
    All gathers norms, normalizing with max norm, find common low resolution mask,
    All reduces two scale sign array * xi vector.
    """

    def __init__(self, device, timer, quantization_levels=None):
        super(QSGDMaxNormMultiScaleReducer, self).__init__(device, timer)

        if not quantization_levels:
            quantization_levels = [6, 10]

        quantization_levels.sort()
        self._quantization_levels = quantization_levels

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0
        compressor = QSGDMaxNormMultiScaleCompressor(
            self._device,
            self._quantization_levels,
        )

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.norm", verbosity=2):
            norm = flat_grad.buffer.abs().max()

            if self.n_workers > 1:
                norm_op = torch.distributed.all_reduce(
                    tensor=norm,
                    op=torch.distributed.ReduceOp.MAX,
                    async_op=True,
                )
                norm_op.wait()
                max_norm = norm
            else:
                max_norm = norm

        with self._timer("reduce.compress", verbosity=2):
            resolution_mask = compressor.compress_mask(max_norm, flat_grad.buffer)

            if self.n_workers > 1:
                high_mask_op = torch.distributed.all_reduce(
                    tensor=resolution_mask,
                    op=torch.distributed.ReduceOp.MIN,
                    async_op=True,
                )
                high_mask_op.wait()
            else:
                resolution_mask = resolution_mask

            sign_xi_array = compressor.compress(resolution_mask)

        with self._timer("reduce.reduce.vector", verbosity=2):
            if self.n_workers > 1:
                sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
                sign_xi_reduce_op.wait()
                sign_xi_array.true_divide(self.n_workers)
            else:
                sign_xi_array = sign_xi_array

        bits_communicated += self.n_bits(norm) + self.n_bits(resolution_mask) + self.n_bits(sign_xi_array)

        with self._timer("reduce.decompress", verbosity=2):
            flat_grad.buffer = compressor.decompress(max_norm, sign_xi_array, resolution_mask)

        with self._timer("reduce.setgrad", verbosity=2):
            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()


# class GlobalRandKMaxNormMultiScaleReducer(Reducer):
#     """
#     All reduce reducer with QSGD MaxNorm Multi Level compression of random K indices.
#     All gathers norms, normalizing with max norm, find common low resolution mask,
#     All reduces two scale sign array * xi vector.
#     """
#
#     def __init__(
#         self,
#         device,
#         timer,
#         K=10000,
#         quantization_levels=None,
#     ):
#         super(GlobalRandKMaxNormMultiScaleReducer, self).__init__(device, timer)
#         self._K = K
#
#         if not quantization_levels:
#             quantization_levels = [6, 10]
#
#         quantization_levels.sort()
#         self._quantization_levels = quantization_levels
#
#         self._indices_queue = []
#
#     def reduce(self, grad_in, grad_out):
#         bits_communicated = 0
#         compressor = GlobalRandKMaxNormMultiScaleReducer(
#             self._device,
#             self._quantization_levels,
#         )
#
#         # From here
#         with self._timer("reduce.flat_pack"):
#             flat_grad = TensorBuffer(grad_in)
#
#         if not self._indices_queue:
#             self._indices_queue = torch.randperm(len(flat_grad.buffer)).split(self._K)
#             self._indices_queue = list(self._indices_queue)
#
#         RandK_indices = self._indices_queue.pop().numpy()
#         RandK_flat_grad = flat_grad.buffer[RandK_indices]
#
#         with self._timer("reduce.norm", verbosity=2):
#             norm = RandK_flat_grad.abs().max()
#
#             if self.n_workers > 1:
#                 collected_norms = [torch.empty_like(norm) for _ in range(self.n_workers)]
#                 norms_gather_op = torch.distributed.all_gather(tensor_list=collected_norms, tensor=norm, async_op=True)
#
#                 norms_gather_op.wait()
#                 max_norm = max(collected_norms)
#             else:
#                 max_norm = norm
#
#         with self._timer("reduce.compress", verbosity=2):
#             sign_xi_array_lower = compressor.compress_lower(max_norm, RandK_flat_grad)
#             sign_xi_array_higher, higher_resolution_mask = compressor.compress_higher(max_norm, RandK_flat_grad)
#
#             if self.n_workers > 1:
#                 high_mask_op = torch.distributed.all_reduce(
#                     tensor=higher_resolution_mask,
#                     op=torch.distributed.ReduceOp.PRODUCT,
#                     async_op=True,
#                 )
#                 high_mask_op.wait()
#
#             else:
#                 higher_resolution_mask = higher_resolution_mask
#
#             sign_xi_array = (
#                 higher_resolution_mask * sign_xi_array_higher + (1 - higher_resolution_mask) * sign_xi_array_lower
#             )
#
#         with self._timer("reduce.reduce.vector", verbosity=2):
#             if self.n_workers > 1:
#                 sign_xi_reduce_op = torch.distributed.all_reduce(tensor=sign_xi_array, async_op=True)
#                 sign_xi_reduce_op.wait()
#                 sign_xi_array.true_divide(self.n_workers)
#             else:
#                 sign_xi_array = sign_xi_array
#
#         bits_communicated += self.n_bits(norm) + self.n_bits(higher_resolution_mask) + self.n_bits(sign_xi_array)
#
#         with self._timer("reduce.decompress", verbosity=2):
#             RandK_decompressed = compressor.decompress(max_norm, sign_xi_array, higher_resolution_mask)
#
#         with self._timer("reduce.setgrad", verbosity=2):
#             flat_grad.buffer[RandK_indices] = RandK_decompressed
#
#             for out in grad_out:
#                 out[:] = 0.0
#
#             for grad, out in zip(flat_grad, grad_out):
#                 # TODO Average or Sum
#                 grad = grad.to(self._device)
#                 out.add_(other=grad, alpha=1)
#
#         return bits_communicated
#
#     def n_bits(self, tensor):
#         return 8 * tensor.nelement() * tensor.element_size()



class RankKReducer(Reducer):
    def __init__(self, device, timer, n_power_iterations=0, reuse_query=False, rank=1):
        super().__init__(device, timer)

        assert n_power_iterations == 0
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query

        self._memory = []

    def reduce(self, grad_in, grad_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        if not self._memory:
            self._memory = [torch.zeros_like(grad) for grad in grad_in]
            self._memory = TensorBuffer(self._memory)
        else:
            for grad, mem in zip(grad_in, self._memory):
                grad[:] += mem

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, self._memory)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, self._memory)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        with self._timer("reduce.allocate_memory", verbosity=2):
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                p_total_size += n * rank
                q_total_size += m * rank
            if self.p_memory is None:
                self.p_memory = torch.empty(p_total_size, device=self._device)
                self.q_memory = torch.empty(q_total_size, device=self._device)

            # Find them again and make lists of pointers
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx : p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx : q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

        with self._timer("reduce.prepare.q", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape

                if self.reuse_query and not memory_is_uninitialized:
                    # orthogonalize(q)
                    pass
                else:
                    # Sample a query vector q
                    # self.set_random(q)
                    q.normal_()

        with self._timer("reduce.compute.p", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix, q, out=p)

        with self._timer("reduce.p", verbosity=2):
            if self.n_workers > 1:
                p_memory_reduce_op = torch.distributed.all_reduce(tensor=self.p_memory, async_op=True)
                p_memory_reduce_op.wait()
            else:
                self.p_memory = self.p_memory

            bits_communicated += self.n_bits(self.p_memory)

        # Start communicating rank 1 tensors
        with self._timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])

        with self._timer("reduce.rank1.all_reduce", verbosity=2):
            if self.n_workers > 1:
                tensor_reduce_op = torch.distributed.all_reduce(tensor=rank1_tensor_list.buffer, async_op=True)
                tensor_reduce_op.wait()
            else:
                rank1_tensor_list = rank1_tensor_list

            bits_communicated += self.n_bits(rank1_tensor_list.buffer)

        with self._timer("reduce.normalize.p", verbosity=2):
            for p in ps:
                self.orthogonalize(p)

        with self._timer("reduce.compute.q", verbosity=2):
            for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix.t(), p, out=q)

        with self._timer("reduce.q", verbosity=2):
            if self.n_workers > 1:
                q_memory_reduce_op = torch.distributed.all_reduce(tensor=self.q_memory, async_op=True)
                q_memory_reduce_op.wait()
            else:
                self.q_memory = self.q_memory

            bits_communicated += self.n_bits(self.q_memory)
            self.q_memory.data[:] /= self.n_workers

        with self._timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                torch.matmul(p, q.t(), out=out[:])
                mem[:] = tensor - out

        with self._timer("reduce.rank1.unpack", verbosity=2):
            tensor_reduce_op.wait()
            rank1_tensor_list.buffer /= self.n_workers
            # rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

            # for out in grad_out:
            #     out[:] = 0.0

            for grad, out in zip(rank1_tensor_list, [out for (_, out, _) in rank1_tensors]):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

                # print(out)
                # exit(99)

        return bits_communicated

    # @torch.jit.script
    def orthogonalize(self, matrix, eps=torch.tensor(1e-8)):
        n, m = matrix.shape
        for i in range(m):
            # Normalize the i'th column
            col = matrix[:, i: i + 1]
            col /= torch.sqrt(torch.sum(col ** 2)) + eps
            # Project it on the rest and remove it
            if i + 1 < m:
                rest = matrix[:, i + 1:]
                # rest -= torch.matmul(col.t(), rest) * col
                rest -= torch.sum(col * rest, dim=0) * col

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()
