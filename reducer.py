import torch
import torch.distributed
from compressors import (
    NoneCompressor, QSGDCompressor, QSGDWECCompressor,
    QSGDWECModCompressor, TernGradCompressor
)


class Reducer:
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
        return self.buffer[self._start_idx[index]: self._end_idx[index]].view(self._tensor_shapes[index])

    def __len__(self):
        return self._len_tensors


class NoneReducer(Reducer):
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
                size_gather_op = torch.distributed.all_gather(collected_tensor_sizes, compressed_tensor_size,
                                                              async_op=True)
                size_gather_op.wait()

                max_size = max(collected_tensor_sizes).item()
                padded_compressed_tensors = torch.zeros(max_size, dtype=torch.int64, device=self._device)
                padded_compressed_tensors[:compressed_tensor_size] = compressed_tensor

                collected_tensors = [torch.zeros(max_size, dtype=torch.int64, device=self._device) for _ in
                                     range(self.n_workers)]
                tensor_gather_op = torch.distributed.all_gather(collected_tensors, padded_compressed_tensors,
                                                                async_op=True)
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
    def __init__(self, device, timer):
        super(NoneAllReducer, self).__init__(device, timer)

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.allreduce", verbosity=2):
            if self.n_workers > 1:
                tensor_reduce_op = torch.distributed.all_reduce(flat_grad.buffer, async_op=True)
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
                size_gather_op = torch.distributed.all_gather(collected_tensor_sizes, compressed_tensor_size, async_op=True)
                size_gather_op.wait()

                max_size = max(collected_tensor_sizes).item()
                padded_compressed_tensors = torch.zeros(max_size, dtype=torch.int64, device=self._device)
                padded_compressed_tensors[:compressed_tensor_size] = compressed_tensor

                collected_tensors = [torch.zeros(max_size, dtype=torch.int64, device=self._device) for _ in range(self.n_workers)]
                tensor_gather_op = torch.distributed.all_gather(collected_tensors, padded_compressed_tensors, async_op=True)
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
                norms_gather_op = torch.distributed.all_gather(collected_norms, norm, async_op=True)

                collected_signs = [torch.empty_like(sign_array) for _ in range(self.n_workers)]
                signs_gather_op = torch.distributed.all_gather(collected_signs, sign_array, async_op=True)

                collected_xis = [torch.empty_like(xi_array) for _ in range(self.n_workers)]
                xi_gather_op = torch.distributed.all_gather(collected_xis, xi_array, async_op=True)

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
                norms_gather_op = torch.distributed.all_gather(collected_norms, norm, async_op=True)

                collected_sign_xis = [torch.empty_like(sign_xi_array) for _ in range(self.n_workers)]
                sign_xis_gather_op = torch.distributed.all_gather(collected_sign_xis, sign_xi_array, async_op=True)

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
                scaler_gather_op = torch.distributed.all_gather(collected_scalers, scaler, async_op=True)

                collected_signs = [torch.empty_like(sign_array) for _ in range(self.n_workers)]
                signs_gather_op = torch.distributed.all_gather(collected_signs, sign_array, async_op=True)

                collected_bs = [torch.empty_like(b_array) for _ in range(self.n_workers)]
                b_gather_op = torch.distributed.all_gather(collected_bs, b_array, async_op=True)

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
                scaler_gather_op = torch.distributed.all_gather(collected_scalers, scaler, async_op=True)

                collected_sign_bs = [torch.empty_like(sign_b_array) for _ in range(self.n_workers)]
                sign_bs_gather_op = torch.distributed.all_gather(collected_sign_bs, sign_b_array, async_op=True)

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