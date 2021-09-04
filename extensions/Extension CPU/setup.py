#!/usr/bin/env python

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

"""
Greedily packs varying number of multiple coordinates into condensed tensors.
"""

setup(
    name="bitpacking",
    ext_modules=[CppExtension("bitpacking", ["bitpacking.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
