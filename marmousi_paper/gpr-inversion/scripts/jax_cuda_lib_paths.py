#!/usr/bin/env python
"""Print CUDA library paths from pip-installed nvidia packages."""

from __future__ import annotations

import os
import site


def main() -> None:
    package_order = [
        "cuda_runtime",
        "cuda_nvcc",
        "cuda_nvrtc",
        "cuda_cupti",
        "cublas",
        "cudnn",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "cusparselt",
        "nccl",
        "nvjitlink",
        "nvshmem",
    ]
    roots: list[str] = []
    site_dirs = site.getsitepackages()
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(user_site)
    for site_dir in site_dirs:
        nvidia_root = os.path.join(site_dir, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for package_name in package_order:
            lib_dir = os.path.join(nvidia_root, package_name, "lib")
            if os.path.isdir(lib_dir):
                roots.append(lib_dir)
        for package_name in sorted(os.listdir(nvidia_root)):
            lib_dir = os.path.join(nvidia_root, package_name, "lib")
            if os.path.isdir(lib_dir) and lib_dir not in roots:
                roots.append(lib_dir)

    deduped: list[str] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    print(":".join(deduped))


if __name__ == "__main__":
    main()
