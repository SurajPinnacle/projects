from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

opencv_include = 'dllenv/include'  # Update this path
opencv_lib = 'dllenv/Lib'          # Update this path

extensions = [
    Extension(
        "mycode",
        sources=["dllopencv.pyx"],
        include_dirs=[numpy.get_include(), opencv_include],
        library_dirs=[opencv_lib],
        libraries=["opencv_core", "opencv_imgcodecs", "opencv_imgproc"],  # Add necessary OpenCV libraries
        language="c++",
    )
]

setup(
    name="mycode",
    ext_modules=cythonize(extensions),
)
