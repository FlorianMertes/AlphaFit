from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

ext = Extension(
    "Tail",
    sources=['wrapper.pyx','c/src/Tail.c','extern/src/Faddeeva.c'],
    include_dirs=['.','c/include','extern/include','extern/src',np.get_include()],
    extra_compile_args=["-O3","-fopenmp"],
    extra_link_args=["-fopenmp"]
)

setup(
    ext_modules = cythonize(ext)
)