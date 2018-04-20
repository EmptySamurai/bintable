from distutils.core import setup, Extension
from setuptools import find_packages
import glob
import os
import numpy as np

native_path = os.path.join('bintable', 'native')
native = Extension('bintable.native',
                    include_dirs = ['pybind11/include', native_path, np.get_include()],
                    sources = glob.glob(os.path.join(native_path, "*.cpp")),
                    extra_compile_args=['-std=c++11'])

setup (name = 'bintable',
       version = '0.1',
       description = 'This package for binary table serialization',
       packages=find_packages(exclude=["pybind11"]),
       ext_modules = [native])