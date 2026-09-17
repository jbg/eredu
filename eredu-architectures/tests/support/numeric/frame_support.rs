// Facts describe the actual resident and lease-backed scalar fixture policies.
struct Support;
impl RealtimeMechanismSupport for Support {
    fn facts(&self) -> RealtimeMechanismFacts {
        RealtimeMechanismFacts::new(
            NumericBackend::OPERATOR_CAPABILITIES,
            [
                RealtimeMechanism::TensorOperations,
                RealtimeMechanism::NeuralOperations,
                RealtimeMechanism::ParameterMaterialization,
                RealtimeMechanism::ParameterStorage,
                RealtimeMechanism::StateStorage,
                RealtimeMechanism::CoordinateStorage,
                RealtimeMechanism::Sampling,
                RealtimeMechanism::Randomness,
                RealtimeMechanism::HostConversion,
                RealtimeMechanism::ExactCompletion,
                RealtimeMechanism::ResourceRetention,
                RealtimeMechanism::Transfer,
                RealtimeMechanism::Observation,
            ],
            [
                ExecutionResidency::FullyResident,
                ExecutionResidency::LayerwiseHost,
                ExecutionResidency::DenseDiskStream,
            ],
            NonZeroUsize::new(1).unwrap(),
            CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .unwrap(),
            SessionCapabilities::new(true, true, true),
        )
        .with_state_lifecycle(
            eredu_runtime::StateLifecycleCapabilities::new()
                .with_transactions(true, true)
                .with_reset(true)
                .with_observation_retention(true),
        )
    }
    fn supports_lowering(
        &self,
        d: &eredu_runtime::WeightLoweringDescriptor,
        k: WeightLoweringKind,
    ) -> bool {
        matches!(
            d.source(),
            eredu_checkpoint::SourceTensorEncoding::Safetensors(_)
        ) && matches!(k, WeightLoweringKind::Direct | WeightLoweringKind::Derived)
    }
    fn state_component_placements(
        &self,
        _: &eredu_core::cache::StateComponentPolicy,
    ) -> (
        Option<StateComponentPlacement>,
        Option<StateComponentPlacement>,
    ) {
        (Some(StateComponentPlacement::Device), None)
    }
}
