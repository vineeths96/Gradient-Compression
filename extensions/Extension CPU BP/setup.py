#!/usr/bin/env python

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(
    name="bytepacking",
    ext_modules=[CppExtension("bytepacking", ["bytepacking.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
