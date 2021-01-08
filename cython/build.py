from distutils.core import Distribution, Extension
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
import numpy as np

ext = Extension(
    "Tail",
    sources=['wrapper.pyx','c/**'],
    include_dirs=['.','c/**',np.get_include()]
)

d = Distribution()
d.extensions = cythonize(ext)

build_ext(d)