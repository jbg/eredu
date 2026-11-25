#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {
using namespace mlx::core;

// Bound recording work and retained allocations independently of input size.
constexpr size_t max_operations = 256;
constexpr size_t max_slots = 512;
constexpr size_t max_constant_bytes = 4096;

struct TensorSignature {
  Shape shape;
  Dtype dtype;
  size_t source_slot;
};

struct Operation {
  std::shared_ptr<Primitive> primitive;
  std::vector<size_t> inputs;
  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  std::vector<size_t> outputs;
};

struct Tape {
  explicit Tape(Stream stream) : stream(stream) {}
  Stream stream;
  std::vector<TensorSignature> inputs;
  std::vector<std::pair<size_t, array>> constants;
  std::vector<Operation> operations;
  std::vector<size_t> outputs;
  size_t slots = 0;
};

bool supported(
    const Primitive& primitive,
    const char* const* kernel_names,
    size_t num_kernel_names) {
  const auto name = std::string_view(primitive.name());
  if (name == "CustomKernel") {
    auto* kernel = dynamic_cast<const fast::CustomKernel*>(&primitive);
    if (!kernel) {
      return false;
    }
    const auto state = kernel->state();
    auto kernel_name = std::string_view(std::get<0>(state));
    constexpr std::string_view prefix = "custom_kernel_";
    if (!kernel_name.starts_with(prefix)) {
      return false;
    }
    kernel_name.remove_prefix(prefix.size());
    kernel_name = kernel_name.substr(0, kernel_name.find("__"));
    for (size_t i = 0; i < num_kernel_names; ++i) {
      if (kernel_name == kernel_names[i]) {
        return true;
      }
    }
    return false;
  }
  return name == "Add" || name == "AsType" || name == "Broadcast" ||
      name == "Concatenate" || name == "Contiguous" || name == "Copy" ||
      name == "Divide" || name == "Exp" || name == "Full" ||
      name == "Gather" || name == "GatherAxis" || name == "Multiply" ||
      name == "Negative" || name == "Reshape" || name == "Sigmoid" ||
      name == "Slice" || name == "Split" || name == "Subtract" ||
      name == "Transpose";
}

struct Recorder {
  Tape& tape;
  const char* const* kernel_names;
  size_t num_kernel_names;
  std::unordered_map<uintptr_t, size_t> known;
  size_t constant_bytes = 0;

  std::optional<size_t> record(const array& value, size_t depth = 0) {
    if (auto found = known.find(value.id()); found != known.end()) {
      return found->second;
    }
    if (depth >= max_operations || tape.slots >= max_slots || value.is_tracer()) {
      return std::nullopt;
    }
    if (!value.has_primitive()) {
      // In particular, a tiny view into a large backing buffer is ineligible.
      if (!value.is_available() || !value.data_shared_ptr()) {
        return std::nullopt;
      }
      const auto bytes = value.buffer().ptr() ? value.buffer_size() : 0;
      if (bytes > max_constant_bytes - constant_bytes) {
        return std::nullopt;
      }
      const auto slot = tape.slots++;
      constant_bytes += bytes;
      tape.constants.emplace_back(slot, value);
      known.emplace(value.id(), slot);
      return slot;
    }
    if (value.primitive().stream() != tape.stream ||
        !supported(value.primitive(), kernel_names, num_kernel_names)) {
      return std::nullopt;
    }
    Operation op;
    op.primitive = value.primitive_ptr();
    op.inputs.reserve(value.inputs().size());
    for (const auto& input : value.inputs()) {
      auto slot = record(input, depth + 1);
      if (!slot) {
        return std::nullopt;
      }
      op.inputs.push_back(*slot);
    }
    const auto outputs = value.outputs();
    if (tape.operations.size() >= max_operations ||
        outputs.size() > max_slots - tape.slots) {
      return std::nullopt;
    }
    for (const auto& output : outputs) {
      const auto slot = tape.slots++;
      known.emplace(output.id(), slot);
      op.shapes.push_back(output.shape());
      op.dtypes.push_back(output.dtype());
      op.outputs.push_back(slot);
    }
    tape.operations.push_back(std::move(op));
    return known.at(value.id());
  }
};
} // namespace

extern "C" int mlx_record_fixed_shape_graph(
    void** result,
    mlx_vector_array inputs,
    mlx_vector_array outputs,
    mlx_stream stream,
    const char* const* kernel_names,
    size_t num_kernel_names) {
  try {
    *result = nullptr;
    const auto& bindings = mlx_vector_array_get_(inputs);
    const auto& results = mlx_vector_array_get_(outputs);
    if (bindings.size() > max_slots || results.empty() ||
        results.size() > max_slots) {
      return 0;
    }

    auto tape = std::make_unique<Tape>(mlx_stream_get_(stream));
    Recorder recorder{*tape, kernel_names, num_kernel_names, {}};
    for (const auto& value : bindings) {
      if (value.is_tracer()) {
        return 0;
      }
      auto binding = recorder.known.try_emplace(value.id(), tape->slots).first;
      ++tape->slots;
      tape->inputs.push_back({value.shape(), value.dtype(), binding->second});
    }
    for (const auto& value : results) {
      auto slot = recorder.record(value);
      if (!slot) {
        return 0;
      }
      tape->outputs.push_back(*slot);
    }
    if (!tape->operations.empty()) {
      *result = tape.release();
    }
    return 0;
  } catch (const std::exception& error) {
    mlx_error(error.what());
    return 1;
  }
}

extern "C" int mlx_apply_fixed_shape_graph(
    mlx_vector_array* output,
    bool* matched,
    const void* tape_ptr,
    mlx_vector_array inputs,
    mlx_stream stream) {
  try {
    *matched = false;
    const auto& tape = *static_cast<const Tape*>(tape_ptr);
    const auto& bindings = mlx_vector_array_get_(inputs);
    if (bindings.size() != tape.inputs.size() ||
        mlx_stream_get_(stream) != tape.stream) {
      return 0;
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto& input = tape.inputs[i];
      if (bindings[i].is_tracer() || bindings[i].shape() != input.shape ||
          bindings[i].dtype() != input.dtype ||
          (input.source_slot != i &&
           bindings[i].id() != bindings[input.source_slot].id())) {
        return 0;
      }
    }

    std::vector<std::optional<array>> values(tape.slots);
    for (size_t i = 0; i < bindings.size(); ++i) {
      values[i] = bindings[i];
    }
    for (const auto& [slot, value] : tape.constants) {
      values[slot] = value;
    }
    std::vector<array> input_values;
    for (const auto& op : tape.operations) {
      input_values.clear();
      for (auto slot : op.inputs) {
        input_values.push_back(*values[slot]);
      }
      if (op.outputs.size() == 1) {
        values[op.outputs.front()].emplace(
            op.shapes.front(), op.dtypes.front(), op.primitive, input_values);
      } else {
        auto outputs = array::make_arrays(
            op.shapes, op.dtypes, op.primitive, input_values);
        for (size_t j = 0; j < op.outputs.size(); ++j) {
          values[op.outputs[j]] = std::move(outputs[j]);
        }
      }
    }

    std::vector<array> results;
    results.reserve(tape.outputs.size());
    for (auto slot : tape.outputs) {
      results.push_back(*values[slot]);
    }
    mlx_vector_array_set_(*output, std::move(results));
    *matched = true;
    return 0;
  } catch (const std::exception& error) {
    mlx_error(error.what());
    return 1;
  }
}

extern "C" void mlx_free_fixed_shape_graph(void* tape) {
  delete static_cast<Tape*>(tape);
}
