from distutils.core import setup, Extension
from Cython.Build import cythonize


ext = [
    Extension("pyx_files.optimized_functions", sources=["pyx_files/optimized_functions.pyx"])
]

setup(ext_modules=cythonize(ext))