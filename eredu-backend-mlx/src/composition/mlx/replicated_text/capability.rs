use super::*;

/// Reports the exact MLX mechanisms applicable to one neutral requirement set.
///
/// The report is derived only from source encodings, executable formats, and
/// implemented backend facilities. It does not receive architecture identity.
pub(crate) const GROUPED_OPERATION_CAPABILITIES: [GroupedOperationRequirement; 4] = [
    GroupedOperationRequirement::GatedProduct,
    GroupedOperationRequirement::GatedProductTensorParallelPartial,
    GroupedOperationRequirement::Relu2,
    GroupedOperationRequirement::Relu2TensorParallelPartial,
];

pub(crate) fn capabilities(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
) -> BackendMechanismCapabilities {
    let mut weight_lowerings = Vec::new();
    for parameter in requirements
        .parameters()
        .iter()
        .chain(requirements.auxiliary_parameters())
    {
        if !parameter.has_lowering_source() {
            continue;
        }
        let requested = request
            .quantization()
            .and_then(|requested| parameter.transform_target(requested).ok().flatten())
            .map(|target| target.executable());
        for executable in std::iter::once(parameter.native_executable()).chain(requested) {
            let descriptor = parameter
                .lowering_descriptor(executable)
                .expect("validated replicated parameter forms a lowering query");
            let direct_is_semantically_valid = supports_direct(&descriptor)
                && !(parameter.role() == eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                    && matches!(
                        descriptor.source(),
                        SourceTensorEncoding::Safetensors(StoredDtype::U8)
                    )
                    && executable == LinearFormat::Dense);
            let kind =
                if executable == parameter.native_executable() && direct_is_semantically_valid {
                    Some(WeightLoweringKind::Direct)
                } else if supports_transform(&descriptor) {
                    Some(WeightLoweringKind::Transform)
                } else {
                    None
                };
            if let Some(kind) = kind {
                let capability = WeightLoweringCapability::new(descriptor, kind);
                if !weight_lowerings.contains(&capability) {
                    weight_lowerings.push(capability);
                }
            }
        }
    }
    let state =
        StateMechanismCapabilities::new((0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .expect("validated state layout exposes every layer")
                .iter()
                .filter_map(move |component| {
                    let paged = match component.residency() {
                        StateResidencyClass::SealablePaged => StateComponentPlacement::Paged,
                        StateResidencyClass::AlwaysDeviceMutable
                        | StateResidencyClass::LayerScopedOffloadable => {
                            StateComponentPlacement::Device
                        }
                    };
                    mlx_supports_state_component(component).then(|| {
                        StateComponentMechanism::new(
                            layer,
                            component.clone(),
                            Some(StateComponentPlacement::Device),
                            Some(paged),
                        )
                    })
                })
        }))
        .with_transactions(true, true)
        .with_reset(true)
        .with_prompt_cache(matches!(request.state(), CacheResidencyPolicy::Paged(_)))
        .with_observation_retention(true);
    BackendMechanismCapabilities::new(
        MlxNeuralBackend::OPERATOR_CAPABILITIES,
        weight_lowerings,
        vec![
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
            WeightResidencyMechanism::DiskStreamed,
        ],
        state,
    )
    .with_session(eredu_core::SessionCapabilities::new(true, true, true))
    .with_grouped_operations(GROUPED_OPERATION_CAPABILITIES)
    .with_indexed_movement(true)
    .with_addressable_storage(
        eredu_runtime::AddressableStorageCapabilities::new(true, true, true, u64::MAX).with_tiers(
            eredu_runtime::AddressableStorageTiers::new(true, true, true),
        ),
    )
    .with_prompt_cache(true)
    .with_exact_completion(true)
}

pub(super) fn mlx_supports_state_component(
    component: &eredu_core::cache::StateComponentPolicy,
) -> bool {
    use eredu_core::cache::{StateTensorDimension, StateTensorDtype};

    !component.shape().is_empty()
        && component.shape().iter().all(|dimension| match dimension {
            StateTensorDimension::Fixed(value)
            | StateTensorDimension::PrefixTokensDiv(value)
            | StateTensorDimension::PrefixTokensRem(value) => value.get() > 0,
            StateTensorDimension::Batch | StateTensorDimension::PrefixTokens => true,
            StateTensorDimension::Scalar => component.shape().len() == 1,
        })
        && matches!(
            component.dtype(),
            StateTensorDtype::Floating
                | StateTensorDtype::Float32
                | StateTensorDtype::Int32
                | StateTensorDtype::Uint32
        )
}
