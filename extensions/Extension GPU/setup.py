#!/usr/bin/env python

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(name='gpu_bitpacking',
      ext_modules=[CppExtension('gpu_bitpacking', ['gpu_bitpacking.cpp'])],
      cmdclass={'build_ext': BuildExtension})
