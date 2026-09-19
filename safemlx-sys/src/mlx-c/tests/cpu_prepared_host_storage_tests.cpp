// Included with the original-operation fixtures.
TEST_CASE("CPU host backing refuses GPU submission before stream preparation") {
  using namespace eval_record_facts;
  const auto cpu = new_stream(Device::cpu);
  prepare(cpu, cpu);
  HostTransferBuffer source({3}, float32, HostTransferPolicy::transfer);
  if (source.storage_kind() != HostTransferStorageKind::cpu) return;
  const float values[]{2.f, -3.f, 7.f};
  std::memcpy(source.data(), values, sizeof(values));
  array input(values, Shape{3}, float32);
  // A descriptor only: no GPU runtime, stream registration or encoder exists.
  const Stream gpu{0, Device::gpu};
  Role role;
  const auto graph = role.graph->occupied_bytes();
  for (bool store : {false, true}) {
    bool refused = false;
    try {
      if (store) copy_to_host_into(input, source, gpu);
      else copy_from_host(source, gpu);
    } catch (const ScopedEvaluationError& error) {
      CHECK(error.outcome() == ScopedEvaluation::unobservable);
      refused = true;
    }
    CHECK(refused);
    CHECK(role.graph->occupied_bytes() == graph);
    CHECK_FALSE(role.scope->query().failed);
  }
  CHECK(std::memcmp(source.data(), values, sizeof(values)) == 0);
}
