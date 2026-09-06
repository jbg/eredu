"""Identify installed native tools independently of Cargo package versions."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def output(*command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    # cl prints its version and usage with exit code 2 when invoked without input.
    if result.returncode and not (command == ("cl",) and result.returncode == 2):
        raise RuntimeError(f"{' '.join(command)} failed: {result.stdout}")
    return result.stdout.strip()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def metal_version(banner):
    # Apple's downloaded Metal toolchain is mounted under a random cryptex path
    # on each host. Its location is not part of the compiler's identity; CMake
    # discovers the current executable again when it configures the build.
    return "\n".join(line for line in banner.strip().splitlines()
                     if not line.startswith("InstalledDir:"))


def main():
    native = {
        "system": platform.system(),
        "machine": platform.machine(),
        "image": os.environ.get("ImageVersion", "local"),
        "cmake": output("cmake", "--version"),
        "compiler": output("cl") if os.name == "nt" else output("c++", "--version"),
        "flags": {name: os.environ.get(name, "") for name in
                  ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "CL", "_CL_",
                   "CUDAFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                   "CMAKE_GENERATOR", "CMAKE_TOOLCHAIN_FILE", "SDKROOT",
                   "MACOSX_DEPLOYMENT_TARGET", "IPHONEOS_DEPLOYMENT_TARGET",
                   "TVOS_DEPLOYMENT_TARGET", "XROS_DEPLOYMENT_TARGET",
                   "SAFEMLX_CUDA_ARCHITECTURES")},
    }
    if platform.system() == "Darwin":
        native["xcode"] = output("xcodebuild", "-version")
        for command in (("xcrun", "--toolchain", "Metal", "metal", "--version"),
                        ("xcrun", "metal", "--version")):
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
            if result.returncode == 0:
                native["metal"] = metal_version(result.stdout)
                break
        else:
            raise RuntimeError("Install the Metal toolchain before restoring native builds")
    if os.environ.get("SAFEMLX_CUDA_ARCHITECTURES"):
        native["nvcc"] = output("nvcc", "--version")
        if platform.system() == "Linux":
            # The Linux toolkit installation uses apt, including cuDNN and NCCL.
            # Include installed package versions, not just the nvcc banner.
            packages = output("dpkg-query", "-W", "-f=${Package}=${Version}\n")
            native["cuda_packages"] = sorted(line for line in packages.splitlines()
                if line.startswith(("cuda-", "libcudnn", "cudnn", "libnccl", "libcublas",
                                    "libcufft", "libcurand", "libcusolver", "libcusparse",
                                    "libblas", "liblapack", "libopenblas")))
    native_id = digest(native)
    rust_id = digest(output("rustc", "-vV"))
    root = (Path.cwd() / ".native-build" / native_id).as_posix()
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"native={native_id}\nrust={rust_id}\n")
    with open(os.environ["GITHUB_ENV"], "a") as stream:
        stream.write(f"SAFEMLX_NATIVE_BUILD_ROOT={root}\n")


if __name__ == "__main__":
    main()
