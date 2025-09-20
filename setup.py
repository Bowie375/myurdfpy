import os
import numpy
from setuptools import setup, Extension

ext = Extension(
    "myurdfpy.IK",
    sources=[
        "./src/myurdfpy/core/main.cpp", 
        "./src/myurdfpy/core/ik.cpp", 
        "./src/myurdfpy/core/methods.cpp", 
        "./src/myurdfpy/core/linalg.cpp"
    ],
    include_dirs=[
        numpy.get_include(), 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "myurdfpy", "core")
    ], 
    extra_compile_args=["-std=c++17", "-O3", "-w"],
    language="c++"
)

setup(
    name="IK",
    version="0.1",
    license="MIT",
    ext_modules=[ext]
)
