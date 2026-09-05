#[test]
fn every_layered_family_exposes_authoritative_geometry_and_parameter_binding() {
    fn assert_bindable<A: eredu_runtime::ArchitectureParameters<MlxNeuralBackend>>() {}

    assert_bindable::<eredu_architectures::llama::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::muse_glimmer::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>>(
    );
    assert_bindable::<eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::deepseek::v3::Model<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::deepseek::v4::Model<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::moshi::LayeredModel<MlxNeuralBackend>>();
}
#[test]
fn native_mlx_execution_is_available() {
    let _ = mlx_execution();
}
