from distutils.core import setup, Extension
from setuptools import find_packages
import glob
import os
import numpy as np
import sys

USE_LITTLE_ENDIAN = 1

def collect_cpp(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".cpp"):
                yield os.path.join(root, file)


native_path = os.path.join('bintable', 'native')
print(glob.glob(os.path.join(native_path, "**/*.cpp"), recursive=True))
native = Extension('bintable.native',
                    define_macros = [("SYSTEM_IS_LITTLE_ENDIAN", int(sys.byteorder=='little')),
                                    ("USE_LITTLE_ENDIAN", USE_LITTLE_ENDIAN)],
                    include_dirs = ['pybind11/include', native_path, np.get_include()],
                    sources = list(collect_cpp(native_path)),
                    extra_compile_args=['-std=c++11', '-O3'])

setup (name = 'bintable',
       version = '0.1',
       description = 'This package for binary table serialization',
       packages=find_packages(exclude=["pybind11"]),
       ext_modules = [native])