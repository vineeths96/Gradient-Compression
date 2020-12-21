import sys
import torch
from torch.utils.cpp_extension import load


bitpacking = load(name="gpu_bitpacking", sources=["gpu_bitpacking.cpp"], verbose=True)

src = torch.randint(low=0, high=220, size=[100000], dtype=torch.int32, device="cuda")
src_copy = src.clone()
print("Source tensor ", src)

packed = gpu_bitpacking.packing(src)
print("Packed tensor ", packed)

dst = gpu_bitpacking.unpacking(packed)
dst = dst[: src.shape[0]]
print("Unpacked tensor", dst)
print((src_copy == dst))

print(
    "Original Num Elements: {:5}, Packed Num Elements: {:5}".format(
        src.nelement(), packed.nelement()
    )
)
print(
    "Original Tensor Size: {:6}, Packed Tenor Size: {:7}".format(
        sys.getsizeof(src.storage()), sys.getsizeof(packed.storage())
    )
)
