"""Build the FastVideo Green Context extension in place."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

SOURCE_DIR = Path(__file__).resolve().parent

setup(
    name="fastvideo-green-context",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="_greenctx",
            sources=[
                str(SOURCE_DIR / "binding.cpp"),
                str(SOURCE_DIR / "green_context.cpp"),
            ],
            include_dirs=[
                str(SOURCE_DIR),
                "/usr/local/cuda/include",
            ],
            extra_compile_args=["-O3", "-std=c++17"],
            extra_link_args=[
                "/lib/x86_64-linux-gnu/libcuda.so.1",
                "-lc10_cuda",
                "-lc10",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
