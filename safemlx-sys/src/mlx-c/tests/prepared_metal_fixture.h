#pragma once
// Isolated native positive cases use the same retained Device/Scheduler/stream
// constructors as production. Ordinary fixture file/compression allocations
// occur before any original Role. Run each selected case in a fresh process.
#ifdef MLX_C_PATCH_TEST_METAL
#include "mlx/c/device_initialization.h"
#include "mlx/c/scheduler_initialization.h"
#include "mlx/c/gpu_stream_registration.h"
#include "mlx/c/stream_copy.h"
#include "mlx/stream.h"
#include <compression.h>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>
namespace prepared_metal_fixture {
struct DeviceSource {
  std::vector<uint8_t> compressed;
  size_t decoded=0;
};
inline mlx_gpu_stream_target source() {
  static const auto target=[] {
    const auto* path=std::getenv("EREDU_NATIVE_METALLIB_PATH");
    REQUIRE(path);
    std::ifstream file(path,std::ios::binary);
    REQUIRE(bool(file));
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),{});
    REQUIRE(!bytes.empty());
    auto owner=std::make_unique<DeviceSource>();
    owner->decoded=bytes.size();
    owner->compressed.resize(bytes.size()+4096);
    std::vector<uint8_t> scratch(compression_encode_scratch_buffer_size(COMPRESSION_LZFSE));
    const auto count=compression_encode_buffer(owner->compressed.data(),owner->compressed.size(),
        bytes.data(),bytes.size(),scratch.empty()?nullptr:scratch.data(),COMPRESSION_LZFSE);
    REQUIRE(count!=0);owner->compressed.resize(count);
    const mlx_device_initialization_source image{owner->compressed.data(),count,owner->decoded};
    mlx_device_initialization_layout device{};
    REQUIRE(mlx_device_initialization_layout_for(&device,image)==0);
    mlx_gpu_stream_target result{};
    REQUIRE(mlx_device_initialize(&result.device_identity,image,device,owner.get(),
        [](void* p){delete static_cast<DeviceSource*>(p);})==0);
    owner.release();
    mlx_scheduler_initialization_layout scheduler{};
    REQUIRE(mlx_scheduler_initialization_layout_for(&scheduler)==0);
    auto scheduler_owner=std::make_unique<int>(0);
    REQUIRE(mlx_scheduler_initialize(&result.scheduler_identity,scheduler,scheduler_owner.get(),
        [](void* p){delete static_cast<int*>(p);})==0);
    scheduler_owner.release();
    return result;
  }();
  REQUIRE(mlx_device_initialized_borrow(target.device_identity)==0);
  REQUIRE(mlx_scheduler_initialized_borrow(target.scheduler_identity)==0);
  return target;
}
class StreamOwner {
  mlx_stream value_{};
public:
  StreamOwner() {
    const auto target=source();
    mlx_gpu_stream_registration_layout layout{};
    REQUIRE(mlx_gpu_stream_registration_layout_for(&layout)==0);
    auto owner=std::make_unique<int>(0);
    REQUIRE(mlx_stream_register_gpu(&value_,layout,target,owner.get(),
        [](void* p){delete static_cast<int*>(p);})==0);
    const auto* retained=owner.release();
    REQUIRE(mlx_stream_gpu_registration_borrow(value_,retained,target)==0);
  }
  StreamOwner(const StreamOwner&)=delete;
  StreamOwner& operator=(const StreamOwner&)=delete;
  ~StreamOwner(){mlx_stream_copy_free(value_);}
  mlx::core::Stream stream() const {return *static_cast<mlx::core::Stream*>(value_.ctx);}
};
} // namespace prepared_metal_fixture
#endif
