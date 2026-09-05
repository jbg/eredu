#[test]
fn mlx_declares_every_supported_architecture_operator_set() {
    let declared = <MlxNeuralBackend as NeuralBackend>::OPERATOR_CAPABILITIES;
    for required in [
        operator_requirements::KIMI_LINEAR,
        operator_requirements::QWEN_HYBRID,
        operator_requirements::NEMOTRON_H,
        operator_requirements::QWEN_VISION,
        operator_requirements::QWEN_VL,
        operator_requirements::DEEPSEEK_V3,
        operator_requirements::DEEPSEEK_V4,
        operator_requirements::INKLING,
        operator_requirements::GEMMA4,
        operator_requirements::MUSE_GLIMMER,
        eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS,
        eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL,
    ] {
        assert!(declared.contains(required));
    }
}
