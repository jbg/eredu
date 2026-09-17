/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/fast.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/fast.h"
#include "mlx/utils.h"
#include "mlx/submission.h"
#include <array>
#include <cstring>

extern "C" int mlx_fast_sdpa_blocks_override(int* res) {
  try {
    *res = mlx::core::env::sdpa_blocks_override();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_cuda_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new_() {
  return mlx_fast_cuda_kernel_config({new mlx_fast_cuda_kernel_config_cpp_()});
}

inline mlx_fast_cuda_kernel_config_cpp_& mlx_fast_cuda_kernel_config_get_(
    mlx_fast_cuda_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_cuda_kernel_config");
  }
  return *static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_cuda_kernel_config_free_(mlx_fast_cuda_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void) {
  try {
    return mlx_fast_cuda_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_cuda_kernel_config_free(
    mlx_fast_cuda_kernel_config cls) {
  mlx_fast_cuda_kernel_config_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_cuda_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_cuda_kernel_cpp_ {
  mlx::core::fast::BorrowedCustomKernelFunction mkf;
  mlx_fast_cuda_kernel_cpp_(mlx::core::fast::BorrowedCustomKernelFunction mkf)
      : mkf(std::move(mkf)) {};
};

inline mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    int shared_memory) {
  return mlx_fast_cuda_kernel({new mlx_fast_cuda_kernel_cpp_(
      mlx::core::fast::cuda_kernel_borrowed(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          shared_memory))});
}

extern "C" mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) {
  try {
    return mlx_fast_cuda_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        shared_memory);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::BorrowedCustomKernelFunction& mlx_fast_cuda_kernel_get_(
    mlx_fast_cuda_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_cuda_kernel");
  }
  return static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_cuda_kernel_free_(mlx_fast_cuda_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls) {
  mlx_fast_cuda_kernel_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream) {
  try {
    const auto& config_ctx = mlx_fast_cuda_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_cuda_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::layer_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            (bias.ctx ? std::make_optional(mlx_array_get_(bias))
                      : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

extern "C" size_t mlx_fast_metal_single_output_control_bytes(
    size_t templates, size_t rank, size_t maximum_name) {
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106 && __cplusplus == 202002L
  // Actual fixed bridge: one output, up to four SSO template names. The
  // borrowed invocation retains none of these configuration references.
  if (templates > 4 || maximum_name > 22 || rank > mlx::core::Shape::inline_capacity)
    return 0;
  using Template = std::pair<std::string, mlx::core::fast::TemplateArg>;
  // libc++ fresh vector growth requests 1, 2, 4 elements. Count every issued
  // destination, including the failed-prefix/old-new growth overlap.
  const size_t template_slots = templates == 0 ? 0 : templates == 1 ? 1 : templates == 2 ? 3 : 7;
  return sizeof(mlx_fast_metal_kernel_config_cpp_) + sizeof(mlx::core::Shape) +
      sizeof(mlx::core::Dtype) + template_slots * sizeof(Template) +
      // Input and output heap vector headers, new() temporary and returned
      // output header. Their element buffers are owned by the existing Graph.
      4 * sizeof(mlx::core::ArrayVector) + sizeof(Template) + sizeof(std::string) +
      sizeof(mlx::core::Shape) + sizeof(mlx::core::StreamOrDevice) +
      3 * sizeof(mlx_fast_metal_kernel_config) + 3 * sizeof(mlx_vector_array) +
      sizeof(mlx_fast_metal_kernel) + sizeof(mlx_array) + sizeof(mlx_stream) +
      sizeof(const char*) + sizeof(const int*) + sizeof(mlx_dtype) +
      6 * sizeof(int) + 4 * sizeof(size_t) + sizeof(bool);
#else
  return 0;
#endif
}

inline mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new_() {
  return mlx_fast_metal_kernel_config(
      {new mlx_fast_metal_kernel_config_cpp_()});
}

inline mlx_fast_metal_kernel_config_cpp_& mlx_fast_metal_kernel_config_get_(
    mlx_fast_metal_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_metal_kernel_config");
  }
  return *static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_metal_kernel_config_free_(mlx_fast_metal_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void) {
  try {
    return mlx_fast_metal_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_metal_kernel_config_free(
    mlx_fast_metal_kernel_config cls) {
  mlx_fast_metal_kernel_config_free_(cls);
}

extern "C" int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_metal_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_cpp_ {
  mlx::core::fast::ViewCustomKernelFunction mkf;
  mlx_fast_metal_kernel_cpp_(mlx::core::fast::ViewCustomKernelFunction mkf)
      : mkf(std::move(mkf)) {};
};

inline mlx_fast_metal_kernel mlx_fast_metal_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  return mlx_fast_metal_kernel({new mlx_fast_metal_kernel_cpp_(
      mlx::core::fast::metal_kernel_view(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          atomic_outputs))});
}

extern "C" mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  try {
    return mlx_fast_metal_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::ViewCustomKernelFunction& mlx_fast_metal_kernel_get_(
    mlx_fast_metal_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_metal_kernel");
  }
  return static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_metal_kernel_free_(mlx_fast_metal_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls) {
  mlx_fast_metal_kernel_free_(cls);
}

namespace {
constexpr size_t fixed_inputs = 8, fixed_outputs = 4, fixed_templates = 4;
struct FixedMetalInvocation {
  std::array<mlx::core::Shape, fixed_outputs> shapes;
  std::array<mlx::core::Dtype, fixed_outputs> dtypes{
      mlx::core::float32, mlx::core::float32, mlx::core::float32, mlx::core::float32};
  std::array<mlx::core::fast::NamedTemplateView, fixed_templates> templates;
};
}
namespace {
struct NativeDefinitionInput {
  std::array<std::string_view, fixed_inputs> inputs;
  std::array<std::string_view, fixed_outputs> outputs;
  mlx::core::fast::MetalKernelDefinitionPlan plan;
  bool borrow(const mlx_fast_definition_view* view) noexcept {
    if (!view || view->input_count > fixed_inputs || view->output_count == 0 ||
        view->output_count > fixed_outputs || (view->input_count && !view->inputs) ||
        !view->outputs) return false;
    auto text = [](mlx_fast_text_view value, std::string_view& target) {
      if (value.size && !value.data) return false;
      target = {value.data ? value.data : "", value.size}; return true;
    };
    if (!text(view->name, plan.name) || !text(view->source, plan.source) ||
        !text(view->header, plan.header)) return false;
    for (size_t i = 0; i < view->input_count; ++i)
      if (!text(view->inputs[i], inputs[i])) return false;
    for (size_t i = 0; i < view->output_count; ++i)
      if (!text(view->outputs[i], outputs[i])) return false;
    plan.inputs = {inputs.data(), view->input_count};
    plan.outputs = {outputs.data(), view->output_count};
    plan.ensure_row_contiguous = view->ensure_row_contiguous;
    plan.atomic_outputs = view->atomic_outputs;
    plan.compile_options = mlx::core::CompileOptions{}.serialize();
    return true;
  }
};
}
extern "C" int mlx_fast_metal_definition_layout(
    mlx_fast_definition_layout* result, const mlx_fast_definition_view* view) {
  if (!result) return 2;
  *result = {};
  NativeDefinitionInput input;
  if (!input.borrow(view)) return 2;
  mlx::core::fast::MetalKernelDefinitionLayout layout;
  const auto status = mlx::core::fast::metal_kernel_definition_layout(input.plan, layout);
  if (status != mlx::core::fast::MetalKernelDefinitionStatus::success)
    return static_cast<int>(status);
  const size_t controls = layout.construction_controls + sizeof(NativeDefinitionInput) +
      sizeof(mlx_fast_definition_view) + sizeof(mlx_fast_definition_layout) +
      sizeof(mlx_fast_prepared_definition) + sizeof(decltype(layout)) +
      sizeof(mlx_fast_text_view) + sizeof(void*) * 3 + sizeof(size_t) * 2 + sizeof(int);
  if (controls < layout.construction_controls) return 3;
  *result = {layout.requested_bytes, layout.alignment, controls};
  return 0;
}
extern "C" size_t mlx_fast_metal_definition_static_bytes(void) {
  return mlx::core::fast::metal_kernel_definition_static_bytes();
}
extern "C" int mlx_fast_metal_definition_new(
    mlx_fast_prepared_definition* result, const mlx_fast_definition_view* view) {
  if (!result) return 2;
  result->ctx = nullptr;
  NativeDefinitionInput input;
  if (!input.borrow(view)) return 2;
  mlx::core::fast::PreparedMetalKernelDefinition* definition = nullptr;
  const auto status = mlx::core::fast::prepare_metal_kernel_definition(input.plan, definition);
  result->ctx = definition;
  return static_cast<int>(status);
}
extern "C" void mlx_fast_metal_definition_free(mlx_fast_prepared_definition definition) {
  mlx::core::fast::free_metal_kernel_definition(
      static_cast<mlx::core::fast::PreparedMetalKernelDefinition*>(definition.ctx));
}

namespace {
constexpr size_t maximum_kernel_specializations = 64;
struct NativeKernelFamilyInput {
  NativeDefinitionInput definition;
  std::array<mlx::core::fast::KernelSpecialization, maximum_kernel_specializations> signatures;
  bool borrow(const mlx_fast_definition_view* plan, const mlx_fast_specialization* values, size_t count) {
    if (!definition.borrow(plan) || !values || !count || count > maximum_kernel_specializations) return false;
    for (size_t i = 0; i < count; ++i) {
      auto& target = signatures[i];
      const auto& source = values[i];
      if (source.template_count > 4) return false;
      target.template_count = source.template_count;
      for (size_t j = 0; j < plan->input_count; ++j)
        target.inputs[j] = {mlx_dtype_to_cpp(source.inputs[j].dtype), source.inputs[j].scalar, source.inputs[j].constant};
      for (size_t j = 0; j < plan->output_count; ++j) target.outputs[j] = mlx_dtype_to_cpp(source.outputs[j]);
      for (size_t j = 0; j < source.template_count; ++j) {
        const auto& value = source.templates[j];
        if (!value.name || strnlen(value.name, 23) > 22) return false;
        target.templates[j].first = value.name;
        switch (value.kind) {
          case 0: target.templates[j].second = value.value; break;
          case 1:
            if (value.value != 0 && value.value != 1) return false;
            target.templates[j].second = value.value != 0; break;
          case 2: target.templates[j].second = mlx_dtype_to_cpp(static_cast<mlx_dtype>(value.value)); break;
          default: return false;
        }
      }
    }
    return true;
  }
};
}
extern "C" int mlx_fast_kernel_family_layout_for(mlx_fast_kernel_family_layout* out,
    const mlx_fast_definition_view* plan, const mlx_fast_specialization* values, size_t count) {
  if (!out) return 2;
  *out = {};
  try {
    NativeKernelFamilyInput input;
    if (!input.borrow(plan, values, count)) return 2;
    mlx::core::fast::KernelFamilyLayout layout;
    auto status = mlx::core::fast::kernel_family_layout(input.definition.plan,
        std::span<const mlx::core::fast::KernelSpecialization>(input.signatures.data(), count), layout);
    if (status != mlx::core::fast::MetalKernelDefinitionStatus::success) return static_cast<int>(status);
    const size_t controls = sizeof(NativeKernelFamilyInput) + sizeof(layout) + sizeof(*out) +
        sizeof(mlx_fast_specialization) + sizeof(void*) * 6 + sizeof(size_t) * 4 + sizeof(int);
    if (controls > SIZE_MAX - layout.control_bytes) return 3;
    *out = {layout.definition_bytes, layout.family_bytes, layout.alignment, layout.control_bytes + controls};
    return 0;
  } catch (...) { return 2; }
}
extern "C" size_t mlx_fast_kernel_family_static_bytes(void) {
  return mlx::core::fast::kernel_family_static_bytes();
}
extern "C" size_t mlx_fast_kernel_family_control_bytes(void) {
  return mlx::core::fast::kernel_family_invocation_control_bytes();
}
extern "C" int mlx_fast_kernel_family_new(mlx_fast_kernel_family* out,
    const mlx_fast_definition_view* plan, const mlx_fast_specialization* values, size_t count,
    void* owner, void (*retire)(void*)) {
  if (!out) return 2;
  out->ctx = nullptr;
  try {
    NativeKernelFamilyInput input;
    if (!input.borrow(plan, values, count) || !owner || !retire) return 2;
    mlx::core::fast::PreparedKernelFamily* family = nullptr;
    auto status = mlx::core::fast::prepare_kernel_family(input.definition.plan,
        std::span<const mlx::core::fast::KernelSpecialization>(input.signatures.data(), count), owner, retire, family);
    out->ctx = family;
    return static_cast<int>(status);
  } catch (...) { return 2; }
}
extern "C" void mlx_fast_kernel_family_free(mlx_fast_kernel_family value) {
  mlx::core::fast::release_kernel_family(static_cast<mlx::core::fast::PreparedKernelFamily*>(value.ctx));
}

extern "C" size_t mlx_fast_metal_fixed_control_bytes(
    size_t inputs, size_t outputs, size_t templates, size_t maximum_rank) {
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106 && __cplusplus == 202002L
  if (inputs > fixed_inputs || outputs == 0 || outputs > fixed_outputs ||
      templates > fixed_templates || maximum_rank > mlx::core::Shape::inline_capacity)
    return 0;
  return sizeof(FixedMetalInvocation) + sizeof(mlx::core::ArrayVector) * 3 +
      sizeof(mlx::core::fast::TemplateArgsView) +
      sizeof(std::span<const mlx::core::Shape>) + sizeof(std::span<const mlx::core::Dtype>) +
      sizeof(std::span<const mlx::core::fast::NamedTemplateView>) +
      sizeof(mlx::core::StreamOrDevice) + sizeof(std::tuple<int, int, int>) * 2 +
      sizeof(mlx::core::fast::TemplateArg) + sizeof(std::string_view) +
      sizeof(mlx_fast_metal_kernel) + sizeof(mlx_stream) + sizeof(mlx_array) +
      sizeof(void*) * 5 + sizeof(size_t) * 8;
#else
  return 0;
#endif
}
template <typename Apply>
static int apply_fixed_impl(
    mlx_array* outputs, size_t output_count,
    Apply&& apply, const mlx_array* inputs, size_t input_count,
    const mlx_fast_output_view* output_shapes,
    const mlx_fast_template_view* templates, size_t template_count,
    const int* grid, const int* thread_group, mlx_stream stream) {
  try {
    using namespace mlx::core;
    auto invalid = [] { throw submission::NativeControlError(
        submission::NativeControlFailure::unsupported_backend); };
    if (input_count > fixed_inputs || output_count == 0 || output_count > fixed_outputs ||
        template_count > fixed_templates || !outputs || !output_shapes || !grid ||
        !thread_group || (input_count && !inputs) || (template_count && !templates)) invalid();
    FixedMetalInvocation config;
    for (size_t i = 0; i < output_count; ++i) {
      const auto& output = output_shapes[i];
      if (output.ndim > Shape::inline_capacity || (output.ndim && !output.shape)) invalid();
      if (output.ndim) config.shapes[i] = Shape(output.shape, output.shape + output.ndim);
      config.dtypes[i] = mlx_dtype_to_cpp(output.dtype);
    }
    for (size_t i = 0; i < template_count; ++i) {
      const auto& value = templates[i];
      if (!value.name || strnlen(value.name, 23) > 22) invalid();
      fast::TemplateArg argument;
      switch (value.kind) {
        case 0: argument = value.value; break;
        case 1:
          if (value.value != 0 && value.value != 1) invalid();
          argument = value.value != 0; break;
        case 2: argument = mlx_dtype_to_cpp(static_cast<mlx_dtype>(value.value)); break;
        default: invalid();
      }
      config.templates[i] = {value.name, argument};
    }
    ArrayVector borrowed_inputs;
    borrowed_inputs.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) borrowed_inputs.push_back(mlx_array_get_(inputs[i]));
    auto values = apply(borrowed_inputs,
        std::span<const Shape>(config.shapes.data(), output_count),
        std::span<const Dtype>(config.dtypes.data(), output_count),
        std::tuple<int, int, int>{grid[0], grid[1], grid[2]},
        std::tuple<int, int, int>{thread_group[0], thread_group[1], thread_group[2]},
        fast::TemplateArgsView(std::span<const fast::NamedTemplateView>(
            config.templates.data(), template_count)), std::nullopt, false,
        mlx_stream_get_(stream));
    if (values.size() != output_count) invalid();
    for (size_t i = 0; i < output_count; ++i) mlx_array_set_(outputs[i], std::move(values[i]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_apply_fixed(
    mlx_array* outputs, size_t output_count,
    mlx_fast_metal_kernel kernel, const mlx_array* inputs, size_t input_count,
    const mlx_fast_output_view* output_shapes,
    const mlx_fast_template_view* templates, size_t template_count,
    const int* grid, const int* thread_group, mlx_stream stream) {
  return apply_fixed_impl(outputs, output_count,
      [kernel](auto&&... args) { return mlx_fast_metal_kernel_get_(kernel)(
          std::forward<decltype(args)>(args)...); },
      inputs, input_count, output_shapes, templates, template_count, grid, thread_group, stream);
}
extern "C" int mlx_fast_metal_definition_apply_fixed(
    mlx_array* outputs, size_t output_count,
    mlx_fast_prepared_definition kernel, const mlx_array* inputs, size_t input_count,
    const mlx_fast_output_view* output_shapes,
    const mlx_fast_template_view* templates, size_t template_count,
    const int* grid, const int* thread_group, mlx_stream stream) {
  return apply_fixed_impl(outputs, output_count,
      [kernel](const mlx::core::ArrayVector& values,
          std::span<const mlx::core::Shape> shapes, std::span<const mlx::core::Dtype> dtypes,
          std::tuple<int, int, int> grid, std::tuple<int, int, int> group,
          mlx::core::fast::TemplateArgsView templates, std::optional<float>, bool,
          mlx::core::StreamOrDevice stream) {
        if (!kernel.ctx) throw mlx::core::submission::NativeControlError(
            mlx::core::submission::NativeControlFailure::unsupported_backend);
        return mlx::core::fast::apply_metal_kernel_definition(
            *static_cast<const mlx::core::fast::PreparedMetalKernelDefinition*>(kernel.ctx),
            values, shapes, dtypes, grid, group, templates, stream);
      }, inputs, input_count, output_shapes, templates, template_count, grid, thread_group, stream);
}

extern "C" int mlx_fast_kernel_family_apply_fixed(
    mlx_array* outputs, size_t output_count,
    mlx_fast_kernel_family kernel, const mlx_array* inputs, size_t input_count,
    const mlx_fast_output_view* output_shapes,
    const mlx_fast_template_view* templates, size_t template_count,
    const int* grid, const int* thread_group, mlx_stream stream) {
  return apply_fixed_impl(outputs, output_count,
      [kernel](const mlx::core::ArrayVector& values,
          std::span<const mlx::core::Shape> shapes, std::span<const mlx::core::Dtype> dtypes,
          std::tuple<int, int, int> grid, std::tuple<int, int, int> group,
          mlx::core::fast::TemplateArgsView templates, std::optional<float>, bool,
          mlx::core::StreamOrDevice stream) {
        if (!kernel.ctx) throw mlx::core::submission::NativeControlError(
            mlx::core::submission::NativeControlFailure::unsupported_backend);
        return mlx::core::fast::apply_kernel_family(
            *static_cast<const mlx::core::fast::PreparedKernelFamily*>(kernel.ctx),
            values, shapes, dtypes, grid, group, templates, stream);
      }, inputs, input_count, output_shapes, templates, template_count, grid, thread_group, stream);
}

extern "C" int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream) {
  try {
    const auto& config_ctx = mlx_fast_metal_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_metal_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rms_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            offset,
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            mlx_array_get_(offset),
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::scaled_dot_product_attention(
            mlx_array_get_(queries),
            mlx_array_get_(keys),
            mlx_array_get_(values),
            scale,
            std::string(mask_mode),
            (mask_arr.ctx ? std::make_optional(mlx_array_get_(mask_arr))
                          : std::nullopt),
            (sinks.ctx ? std::make_optional(mlx_array_get_(sinks))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
