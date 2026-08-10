from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="greenctx",

    ext_modules=[
        CppExtension(
            name="greenctx",
            sources=[
                "binding.cpp",
                "green_context.cpp",
            ],

            include_dirs=[
                "/usr/local/cuda/include",
            ],

            extra_compile_args=[
                "-O3",
                "-std=c++17",
            ],

            extra_link_args=[
                "/lib/x86_64-linux-gnu/libcuda.so.1",
                "-lc10_cuda",
                "-lc10",
            ]
        )
    ],

    cmdclass={
        "BuildExtension": BuildExtension
    },
)