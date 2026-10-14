#include "doctest/doctest.h"
#include "mlx/host_transfer.h"
#include "mlx/mlx.h"

#if defined(MLX_TEST_CUDA)
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#endif

using namespace mlx::core;

TEST_CASE("device storage placement shares CPU views without allocation") {
  auto stream = new_thread_unsafe_stream(Device::cpu);
  auto source = array({1.0f, -2.0f, 3.0f, 4.0f}, Shape{2, 2});
  auto view = transpose(source, stream);
  auto placed = ensure_device_storage(view, stream);
  eval(placed);
  CHECK(placed.data_shared_ptr() == source.data_shared_ptr());
  CHECK(placed.strides() == view.strides());
  CHECK(sum(placed, stream).item<float>() == 6.0f);
}

#if defined(MLX_TEST_CUDA)
TEST_CASE("device storage placement promotes pinned CUDA weights once") {
  auto stream = new_thread_unsafe_stream(Device::gpu);
  auto& device = cu::device(stream.device);
  device.make_current();
  void* host = nullptr;
  REQUIRE(cudaMallocHost(&host, 4096) == cudaSuccess);
  for (int i = 0; i < 1024; ++i) {
    static_cast<float*>(host)[i] = 0.25f * (i % 8);
  }
  auto source = array(
      allocator::Buffer{new cu::CudaBuffer{host, 4096, -1}},
      {2, 512}, float32, [](allocator::Buffer buffer) {
        auto* storage = static_cast<cu::CudaBuffer*>(buffer.ptr());
        cudaFreeHost(storage->data);
        delete storage;
      });
  // Seed the size-only allocator cache with host backing of the same size.
  clear_cache();
  { auto cached_host = zeros({2, 512}, float32, Device::cpu); eval(cached_host); }
  auto placed = ensure_device_storage(transpose(source, stream), stream);
  auto again = ensure_device_storage(placed, stream);
  eval(again);
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, gpu_ptr<void>(placed)) == cudaSuccess);
  if (device.coherent_host_memory()) {
    CHECK(placed.data_shared_ptr() == source.data_shared_ptr());
  } else {
    CHECK(attributes.type == cudaMemoryTypeDevice);
    CHECK(attributes.device == device.cuda_device());
    CHECK(placed.data_shared_ptr() != source.data_shared_ptr());
  }
  CHECK(again.data_shared_ptr() == placed.data_shared_ptr());
  CHECK(sum(placed, stream).item<float>() == 896.0f);
  REQUIRE(cudaPointerGetAttributes(&attributes, host) == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeHost);

  auto strided = slice(source, {0, 0}, {2, 512}, {1, 2}, stream);
  auto compact = ensure_device_storage(strided, stream);
  CHECK(sum(compact, stream).item<float>() == 384.0f);

  if (device.concurrent_managed_access()) {
    // Ordinary array initialization uses managed memory when available.
    auto managed = array({1.0f, 2.0f, 3.0f, 4.0f});
    REQUIRE(cudaPointerGetAttributes(&attributes, gpu_ptr<void>(managed)) == cudaSuccess);
    if (attributes.type == cudaMemoryTypeManaged) {
      auto shared = ensure_device_storage(managed, stream);
      eval(shared);
      CHECK(shared.data_shared_ptr() == managed.data_shared_ptr());
    }
  }
}
#endif
