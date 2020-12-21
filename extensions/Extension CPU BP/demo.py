import sys
import torch
from torch.utils.cpp_extension import load


bytepacking = load(name="bytepacking", sources=["bytepacking.cpp"], verbose=True)

src = torch.randint(low=-127, high=127, size=[10000], dtype=torch.int32)
src_copy = src.clone()
print("Source tensor ", src)

packed = bytepacking.packing(src)
print("Packed tensor ", packed)

dst = bytepacking.unpacking(packed)
dst = dst[: src.shape[0]]
print("Unpacked tensor", dst)
print((src_copy == dst))

print("Original Num Elements: {:5}, Packed Num Elements: {:5}".format(src.nelement(), packed.nelement()))
print(
    "Original Tensor Size: {:6}, Packed Tenor Size: {:7}".format(
        sys.getsizeof(src.storage()), sys.getsizeof(packed.storage())
    )
)
