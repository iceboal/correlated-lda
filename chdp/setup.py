from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl

setup(
    name = "chdp",
    include_dirs = [cython_gsl.get_include()],
    ext_modules = cythonize(
        [Extension("chdp",
                   ["chdp.pyx", "evaluate.cpp", "sampler.cpp", "alias.cpp"],
                   libraries=cython_gsl.get_libraries(),
                   library_dirs=[cython_gsl.get_library_dir()],
                   include_dirs=[cython_gsl.get_cython_include_dir(), "/usr/include/gsl"],
                   extra_compile_args=["-fopenmp"],
                   extra_link_args=["-fopenmp"],
                   language="c++"
                   )
        ])
    )
