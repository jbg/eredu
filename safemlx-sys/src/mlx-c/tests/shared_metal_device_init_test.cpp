// Isolated native mechanism cases. File/compression fixture storage is ordinary;
// genuine source-account admission is exercised by the backend component cases.
#include "mlx/c/device_initialization.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/device_initialization.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef MLX_C_PATCH_TEST_METAL
#include "mlx/backend/metal/device.h"
#include <compression.h>
#endif
using namespace mlx::core;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Owner { std::shared_ptr<std::atomic<unsigned>> retired; };
void retire(void* ptr) { std::unique_ptr<Owner> owner(static_cast<Owner*>(ptr)); ++*owner->retired; }
constexpr uint8_t invalid_data[]{0, 0, 0, 0, 0, 0, 0, 0};
mlx_device_initialization_source invalid_source() { return {invalid_data, sizeof(invalid_data), 1024}; }
#ifdef MLX_C_PATCH_TEST_METAL
struct FixtureSource {
  std::vector<uint8_t> compressed;
  size_t decoded_size;
  mlx_device_initialization_source source() const { return {compressed.data(), compressed.size(), decoded_size}; }
};
FixtureSource fixture_source(const char* path) {
  std::ifstream stream(path, std::ios::binary);
  check(bool(stream), "fixture metallib open");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)), {});
  check(!bytes.empty(), "fixture metallib bytes");
  FixtureSource result{std::vector<uint8_t>(bytes.size() + 4096), bytes.size()};
  std::vector<uint8_t> scratch(compression_encode_scratch_buffer_size(COMPRESSION_LZFSE));
  const auto count = compression_encode_buffer(result.compressed.data(), result.compressed.size(),
      bytes.data(), bytes.size(), scratch.empty() ? nullptr : scratch.data(), COMPRESSION_LZFSE);
  check(count != 0, "fixture compression"); result.compressed.resize(count); return result;
}
void success(const char* path, bool long_override) {
  // A real selected file becomes immutable fixture source before the operation.
  auto fixture = fixture_source(path);
  std::string override;
  if (long_override) { override.assign(8192, 'x'); override += "14g"; check(setenv("MLX_METAL_GPU_ARCH", override.c_str(), 1) == 0, "fixture environment"); }
  mlx_device_initialization_layout layout{};
  check(mlx_device_initialization_layout_for(&layout, fixture.source()) == 0, "selected layout");
  check(layout.decoded_bytes == fixture.decoded_size && layout.dispatch_copy_bytes == fixture.decoded_size,
      "actual doubled decoded/copy overlap");
  if (long_override) check(layout.override_bytes == override.size() + 1, "actual override bytes");
  auto retired = std::make_shared<std::atomic<unsigned>>(0);
  auto owner = std::make_unique<Owner>(Owner{retired});
  uint64_t identity = 0;
  check(mlx_device_initialize(&identity, fixture.source(), layout, owner.get(), retire) == 0, "actual embedded Device");
  owner.release();
  auto* actual = metal::prepared_device(Device::gpu);
  check(actual && identity && mlx_device_initialized_borrow(identity) == 0, "same completed Device");
  check(mlx_device_initialized_borrow(identity + 1) == 8, "wrong Device identity");
  const auto name = actual->get_architecture(); // explicitly ordinary report storage
  check(!name.empty() && actual->architecture_suffix() == name.back(), "retained full-name/suffix parity");
  if (long_override) check(name == override && actual->get_architecture_gen() == 14, "long actual override parity");
  uint64_t second = 0;
  auto duplicate = std::make_unique<Owner>(Owner{retired});
  check(mlx_device_initialize(&second, fixture.source(), layout, duplicate.get(), retire) == 7, "no second constructor");
  check(!second && *retired == 0, "refusal and permanent custody");
  retire(duplicate.release()); check(*retired == 1, "refused owner retires once");
  check(&metal::device(Device::gpu) == actual, "ordinary borrow is the same actual Device");
}
#endif
}
int main(int argc, char** argv) {
  try {
    check(argc >= 2, "explicit isolated case required");
    const char* mode = argv[1];
    mlx_device_initialization_static_layout fixed{};
    mlx_device_initialization_static_layout_for(&fixed);
    if (std::getenv("EREDU_REQUIRE_METAL_DEVICE_INITIALIZATION_QUALIFICATION"))
      check(fixed.qualified && fixed.required, "positive selected Metal qualification required");
    mlx_device_initialization_layout layout{11, 13, 17, 19, 23, 29};
    if (!fixed.required || !fixed.qualified) {
      check(mlx_device_initialization_layout_for(&layout, invalid_source()) == 1, "unsupported fixed status");
      check(layout.object_bytes == 11 && layout.decoded_bytes == 13 && layout.scratch_bytes == 17 &&
          layout.dispatch_copy_bytes == 19 && layout.override_bytes == 23 && layout.controls == 29, "unsupported output unchanged");
      std::cout << "unqualified fixed refusal passed\n"; return 0;
    }
#ifdef MLX_C_PATCH_TEST_METAL
    check(!metal::prepared_device(Device::gpu), "fresh Device process required");
    if (std::strcmp(mode, "source-refusal") == 0) {
      check(mlx_device_initialization_layout_for(&layout, {}) == 10, "missing source refusal");
      check(layout.object_bytes == 11 && layout.decoded_bytes == 13 && layout.scratch_bytes == 17 &&
          layout.dispatch_copy_bytes == 19 && layout.override_bytes == 23 && layout.controls == 29, "missing output unchanged");
      mlx_device_initialization_layout prior{};
      check(mlx_device_initialization_layout_for(&prior, invalid_source()) == 0, "retained original layout");
      const char* current = std::getenv("MLX_METAL_GPU_ARCH");
      std::string changed = current ? current : ""; changed += "x";
      check(setenv("MLX_METAL_GPU_ARCH", changed.c_str(), 1) == 0, "changed-length fixture");
      auto retired = std::make_shared<std::atomic<unsigned>>(0);
      auto owner = std::make_unique<Owner>(Owner{retired}); uint64_t identity = 0;
      check(mlx_device_initialize(&identity, invalid_source(), prior, owner.get(), retire) == 11,
          "changed reached extent cannot replace the admitted layout");
      check(!identity && !metal::prepared_device(Device::gpu) && *retired == 0, "changed extent before construction");
      retire(owner.release()); check(*retired == 1, "changed extent preserves owner");
      metal::set_metallib_path(std::string(512, 'x'));
      check(mlx_device_initialization_layout_for(&layout, invalid_source()) == 10, "ordinary path precedence");
      metal::set_metallib_path("");
      check(mlx_device_initialization_layout_for(&layout, invalid_source()) == 10, "cleared ordinary backing cannot promote");
      check(!metal::prepared_device(Device::gpu), "source refusal initialized Device");
    } else if (std::strcmp(mode, "failed-prefix") == 0) {
      check(mlx_device_initialization_layout_for(&layout, invalid_source()) == 0, "real malformed source layout");
      auto retired = std::make_shared<std::atomic<unsigned>>(0);
      auto owner = std::make_unique<Owner>(Owner{retired});
      uint64_t identity = 0;
      check(mlx_device_initialize(&identity, invalid_source(), layout, owner.get(), retire) == 13,
          "actual decode failure after Device/residency/decoded/scratch creation");
      check(!identity && !metal::prepared_device(Device::gpu) && *retired == 0, "failed prefix not published/consumed");
      retire(owner.release()); check(*retired == 1, "failed owner exact retirement");
      check(argc == 3, "actual metallib path for later independent attempt");
      success(argv[2], false); // previous partial Device does not poison the slot
    } else if (std::strcmp(mode, "long-override") == 0) {
      check(argc == 3, "actual metallib path"); success(argv[2], true);
    } else { throw std::runtime_error("unknown isolated case"); }
#endif
    std::cout << "shared Device case passed: " << mode << '\n'; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
