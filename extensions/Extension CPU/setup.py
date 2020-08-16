#!/usr/bin/env python

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(name='bitpacking',
      ext_modules=[CppExtension('bitpacking', ['bitpacking.cpp'])],
      cmdclass={'build_ext': BuildExtension})
