"""Set up cython accelerator
"""
import platform
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


def _get_open_mp_flag():
    if platform.system() == "Windows":
        print("Windows: Compile with -openmp")
        return "-openmp"
    else:
        print("*nix: Compile with -fopenmp")
        return "-fopenmp"


OPENMP_FLAG = _get_open_mp_flag()

EXT_MODULES = [
    Extension(
        "colocation.genetic_algorithm.core.corr_score.c_score_func",
        ["colocation/genetic_algorithm/core/corr_score/c_score_func.pyx"],
        extra_compile_args=[OPENMP_FLAG],
        extra_link_args=[OPENMP_FLAG],
    )
]

setup(
    name="colocation",
    ext_modules=cythonize(EXT_MODULES),
    version="0.1.0",
    packages=["colocation"],
    entry_points={"console_scripts": ["colocation = colocation.__main__:main"]},
)
