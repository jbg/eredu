// Reuse the actual original Graph/Record fixture and native allocation probe.
namespace constant_pad_tests {
using namespace pointwise_graph_tests;
struct ResultArray {
  mlx_array value{nullptr};
  ~ResultArray() { mlx_array_free(value); }
};
struct Call {
  ResultArray* output;
  array* input;
  array* fill;
  Stream* stream;
  int status{-1};
};
void construct(void* raw) {
  auto& c = *static_cast<Call*>(raw);
  const int axes[]{0, 1, 2}, lows[]{1, 2, 0}, highs[]{0, 1, 2};
  c.status = mlx_pad(&c.output->value, mlx_array{c.input}, axes, 3,
      lows, 3, highs, 3, mlx_array{c.fill}, "constant", mlx_stream{c.stream});
}
void numerical(Device device) {
  auto stream = new_stream(device);
  prepare(stream, stream);
  for (bool empty : {false, true}) {
    std::vector<float> values(24);
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(i) - 9.f;
    array source = empty ? array(values.begin(), {2, 0, 3}, float32)
        : transpose(array(values.begin(), {2, 3, 4}, float32), {0, 2, 1}, stream);
    eval(source);
    array fill(-5, int32);
    auto expected = pad(source, std::vector<std::pair<int, int>>{{1, 0}, {2, 1}, {0, 2}},
        fill, "constant", stream);
    eval(expected);
    Role role;
    Observer observer;
    Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 3, 1, 3) == 0);
    const auto reserved = role.graph->occupied_bytes();
    ResultArray result;
    Call call{&result, &source, &fill, &stream};
    CHECK(native_recovery_without_allocations(construct, &call) == 0);
    REQUIRE(call.status == 0);
    auto& value = mlx_array_get_(result.value);
    CHECK(value.shape() == expected.shape());
    CHECK(role.graph->occupied_bytes() <= reserved);
    bank.reset();
    Operation operation;
    operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1, 7, 4, 6, 4, 1, 8};
    REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
    eval_traversal_tests::complete(role, operation, value);
    REQUIRE(value.size() == expected.size());
    for (size_t i = 0; i < value.size(); ++i) {
      CHECK(value.data<float>()[i] == expected.data<float>()[i]);
    }
    // The padding is nonzero even for the empty-input branch.
    CHECK(value.data<float>()[0] == -5.f);
    if (!empty) CHECK(value.data<float>()[35 + 10] == -9.f);
  }
}
}
TEST_CASE("constant pad uses prepared controls and borrowed CPU copy geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  constant_pad_tests::numerical(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("constant pad uses prepared controls and borrowed Metal copy geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  constant_pad_tests::numerical(Device::gpu);
}
#endif
