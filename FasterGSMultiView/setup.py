# Multi-view version of the Faster-GS rasterizer (Apache-2.0, see ../LICENSE).
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.absolute()
# Gaussians per backward checkpoint = 32 * FGS_BUCKET_PASSES (default 8)
defines = [f"-DFGS_BUCKET_PASSES={os.environ['FGS_BUCKET_PASSES']}"] if "FGS_BUCKET_PASSES" in os.environ else []
setup(
    name="fastergs_mv",
    packages=["fastergs_mv"],
    ext_modules=[
        CUDAExtension(
            name="fastergs_mv._C",
            sources=[str(root / "csrc" / "mv_api.cu")],
            # helper headers (helper_math.h, kernel_utils.cuh, sh_utils.cuh, ...) come unchanged from FasterGSCudaBackend
            include_dirs=[str(root / "csrc"),
                          str(root.parent / "FasterGSCudaBackend" / "FasterGSCudaBackend" / "utils"),
                          str(root.parent / "FasterGSCudaBackend" / "FasterGSCudaBackend" / "rasterization" / "include")],
            extra_compile_args={"cxx": ["-std=c++17", "-O3"] + defines, "nvcc": ["-std=c++17", "-O3", "-use_fast_math"] + defines},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
