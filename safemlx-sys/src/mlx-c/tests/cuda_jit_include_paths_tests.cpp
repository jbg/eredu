#include "doctest/doctest.h"
#include "mlx/backend/cuda/jit_include_paths.h"

#include <chrono>
#include <fstream>

namespace {
namespace fs = std::filesystem;

struct HeaderTree {
  fs::path root = fs::temp_directory_path() /
      ("eredu-cuda-headers-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));

  ~HeaderTree() {
    std::error_code error;
    fs::remove_all(root, error);
  }

  void add(const fs::path& relative) {
    auto path = root / relative;
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    REQUIRE(file.good());
  }
};
} // namespace

TEST_CASE("CUDA JIT discovers Arch toolkit and split CCCL headers") {
  using namespace mlx::core::cu::detail;
  HeaderTree tree;
  // An existing but incomplete default installation must not hide Arch's CUDA.
  fs::create_directories(tree.root / "usr/local/cuda/include");
  tree.add("opt/cuda/include/cuda_runtime.h");
  tree.add("usr/include/cccl/cuda/std/tuple");
  const auto toolkit = find_cuda_toolkit(
      {tree.root / "usr/local/cuda", tree.root / "opt/cuda"});
  REQUIRE(toolkit == tree.root / "opt/cuda");
  CHECK(find_cccl_include_path(
      tree.root / "bundled/cccl", {}, toolkit / "include",
      {tree.root / "usr/include/cccl", tree.root / "usr/local/include/cccl"}) ==
      tree.root / "usr/include/cccl");

  tree.add("usr/local/cuda/include/cuda_runtime.h");
  CHECK(find_cuda_toolkit(
      {tree.root / "usr/local/cuda", tree.root / "opt/cuda"}) ==
      tree.root / "usr/local/cuda");
}

TEST_CASE("CUDA JIT discovers headers beside the resolved runtime compiler library") {
  using namespace mlx::core::cu::detail;
  HeaderTree tree;
  tree.add("custom-toolkit/lib64/libnvrtc.so");
  tree.add("custom-toolkit/include/cuda_runtime.h");
  CHECK(cuda_include_from_library(tree.root / "custom-toolkit/lib64/libnvrtc.so") ==
      fs::canonical(tree.root / "custom-toolkit/include"));
  tree.add("other-toolkit/targets/x86_64-linux/lib/libnvrtc.so");
  tree.add("other-toolkit/targets/x86_64-linux/include/cuda_runtime.h");
  CHECK(cuda_include_from_library(
      tree.root / "other-toolkit/targets/x86_64-linux/lib/libnvrtc.so") ==
      fs::canonical(tree.root / "other-toolkit/targets/x86_64-linux/include"));
#if !defined(_WIN32)
  fs::create_directories(tree.root / "loader");
  fs::create_symlink(tree.root / "custom-toolkit/lib64/libnvrtc.so",
      tree.root / "loader/libnvrtc.so");
  CHECK(cuda_include_from_library(tree.root / "loader/libnvrtc.so") ==
      fs::canonical(tree.root / "custom-toolkit/include"));
#endif
  tree.add("runtime-only/lib/libnvrtc.so");
  CHECK(cuda_include_from_library(tree.root / "runtime-only/lib/libnvrtc.so").empty());
  CHECK(cuda_include_from_library(tree.root / "missing/libnvrtc.so").empty());
}

TEST_CASE("CUDA JIT prefers packaged CCCL and selected toolkit over system headers") {
  using namespace mlx::core::cu::detail;
  HeaderTree tree;
  const auto bundled = tree.root / "bundle/include/cccl";
  const auto compiled = tree.root / "build/cccl";
  const auto runtime = tree.root / "cuda/include";
  const std::vector<fs::path> system{tree.root / "system/cccl"};
  CHECK(find_cccl_include_path(bundled, compiled, runtime, system).empty());
  tree.add("system/cccl/cuda/std/tuple");
  tree.add("cuda/include/cuda/std/tuple");
  CHECK(find_cccl_include_path(bundled, compiled, runtime, system) == runtime);
  tree.add("cuda/include/cccl/cuda/std/tuple");
  CHECK(find_cccl_include_path(bundled, compiled, runtime, system) == runtime / "cccl");
  tree.add("build/cccl/cuda/std/tuple");
  CHECK(find_cccl_include_path(bundled, compiled, runtime, system) == compiled);
  tree.add("bundle/include/cccl/cuda/std/tuple");
  CHECK(find_cccl_include_path(bundled, compiled, runtime, system) == bundled);
}
