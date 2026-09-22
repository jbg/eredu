// Copyright © 2026 Eredu contributors.
#include "mlx/c/ordinary_control.h"
#include "mlx/ordinary_control.h"
#include "mlx/array.h"
namespace {
using namespace mlx::core::submission;
bool publish(mlx_ordinary_control_population* out, const OrdinaryControlPopulation& source, bool valid) {
  if (!out || !valid) return false;
  *out={source.observed_host_controls,source.control_allocations,source.platform_events};
  return true;
}
}
extern "C" bool mlx_ordinary_frontend_control_layout(mlx_ordinary_control_population* out,
    size_t entries,size_t seeds,size_t rank,size_t operands) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_frontend_control_layout(entries,seeds,rank,operands,value);
  return publish(out,value,valid);
}
extern "C" bool mlx_ordinary_cpu_eval_control_layout(mlx_ordinary_control_population* out,
    size_t roots,size_t arrays,size_t tape,size_t edges,size_t outputs,size_t streams,size_t captures) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_cpu_eval_control_layout({roots,arrays,tape,edges,outputs,streams,captures},value);
  return publish(out,value,valid);
}
extern "C" bool mlx_ordinary_metal_eval_control_layout(mlx_ordinary_control_population* out,
    size_t roots,size_t arrays,size_t tape,size_t edges,size_t outputs,size_t streams,size_t captures) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_metal_eval_control_layout({roots,arrays,tape,edges,outputs,streams,captures},value);
  return publish(out,value,valid);
}
extern "C" bool mlx_ordinary_metal_router_eval_control_layout(mlx_ordinary_control_population* out,
    size_t roots,size_t arrays,size_t tape,size_t edges,size_t outputs,size_t streams,size_t captures) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_metal_router_eval_control_layout({roots,arrays,tape,edges,outputs,streams,captures},value);
  return publish(out,value,valid);
}
extern "C" bool mlx_ordinary_cpu_dispatch_envelope(mlx_ordinary_control_population* out,size_t extents) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_cpu_dispatch_envelope(extents,value);
  return publish(out,value,valid);
}
extern "C" bool mlx_ordinary_array_vector_control_layout(mlx_ordinary_control_population* out,size_t count) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_array_vector_control_layout(count,value);
  return publish(out,value,valid);
}

extern "C" bool mlx_ordinary_cpu_wait_control_layout(mlx_ordinary_control_population* out) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_cpu_wait_control_layout(value);
  return publish(out,value,valid);
}

extern "C" bool mlx_ordinary_metal_wait_record_control_layout(mlx_ordinary_control_population* out) {
  OrdinaryControlPopulation value;
  const auto valid=ordinary_metal_wait_record_control_layout(value);
  return publish(out,value,valid);
}

extern "C" bool mlx_ordinary_reserved_array_vector_control_layout(
    mlx_ordinary_control_population* out, size_t elements) {
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106 && __cplusplus == 202002L
  using mlx::core::array;
  OrdinaryControlPopulation value;
  const bool valid = elements <= size_t(PTRDIFF_MAX) && elements <= SIZE_MAX / sizeof(array) &&
      (!elements || append_ordinary_control(value, elements * sizeof(array), alignof(array)));
  return publish(out,value,valid);
#else
  return false;
#endif
}
