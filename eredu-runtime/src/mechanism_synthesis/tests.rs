use std::cell::RefCell;
use std::collections::BTreeMap;

use super::*;
use crate::{
    select_replicated_text_realization, ArchitectureGroupKind, ArchitectureGroupPlacement,
    ArchitectureGroupTransport, ArchitectureMergeDestination, ExecutionGraph, ExecutionUnitLayout,
    LayerWeightResidency, ParameterTransformConstraint, ReplicatedTextParameterOwner,
    ReplicatedTextParameterPresence, ReplicatedTextParameterRequirement,
    ReplicatedTextPhysicalSource, ReplicatedTextStateAccess, StateLayout,
};
use eredu_checkpoint::AffineQuantization;
use eredu_core::{
    cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    },
    AttentionPolicy, LayerSchedule, QuantizationRequest,
};

#[derive(Default)]
struct Support {
    direct: bool,
    transform: bool,
    paged: bool,
    rejected_dtype: Option<StateTensorDtype>,
    maximum_element_bytes: Option<u8>,
    storage_queries: RefCell<Vec<StateStorageDtype>>,
    queries: RefCell<Vec<(WeightLoweringKind, WeightLoweringDescriptor)>>,
    state_queries: RefCell<Vec<(StateComponentPolicy, StateComponentPlacement)>>,
}

impl ReplicatedTextMechanismSupport for Support {
    fn facts(&self, _: &CacheResidencyPolicy) -> BackendMechanismFacts {
        BackendMechanismFacts::new(
            NeuralOperatorCapabilities::NONE,
            [WeightResidencyMechanism::Resident],
            StateLifecycleCapabilities::new()
                .with_transactions(true, true)
                .with_reset(true),
        )
    }

    fn supports_direct(&self, descriptor: &WeightLoweringDescriptor) -> bool {
        self.queries
            .borrow_mut()
            .push((WeightLoweringKind::Direct, descriptor.clone()));
        self.direct
    }

    fn supports_transform(&self, descriptor: &WeightLoweringDescriptor) -> bool {
        self.queries
            .borrow_mut()
            .push((WeightLoweringKind::Transform, descriptor.clone()));
        self.transform
            && matches!(
                descriptor.executable(),
                LinearFormat::Affine(_) | LinearFormat::MxFp4
            )
    }

    fn floating_state_dtype(&self, source: &TensorDtype) -> Option<StateStorageDtype> {
        match source {
            TensorDtype::F16 => Some(StateStorageDtype::F16),
            TensorDtype::Bf16 => Some(StateStorageDtype::Bf16),
            TensorDtype::F32 => Some(StateStorageDtype::F32),
            _ => None,
        }
    }

    fn supports_state_component(
        &self,
        component: &StateComponentPolicy,
        storage_dtype: StateStorageDtype,
        placement: StateComponentPlacement,
    ) -> bool {
        self.storage_queries.borrow_mut().push(storage_dtype);
        self.state_queries
            .borrow_mut()
            .push((component.clone(), placement));
        self.maximum_element_bytes
            .is_none_or(|maximum| storage_dtype.bytes().get() <= maximum)
            && self.rejected_dtype != Some(component.dtype())
            && (placement == StateComponentPlacement::Device || self.paged)
    }
}

fn parameter(
    name: &str,
    shape: Vec<usize>,
    dtype: StoredDtype,
) -> ReplicatedTextParameterRequirement {
    let source = SourceTensorEncoding::Safetensors(dtype);
    ReplicatedTextParameterRequirement::new(
        name,
        vec![name.into()],
        vec![ReplicatedTextPhysicalSource::new(
            name,
            name,
            "/checkpoint/weights.safetensors",
            name,
            source.clone(),
            1,
        )
        .unwrap()],
        Vec::new(),
        Some(source),
        Some(shape.clone()),
        shape,
        LinearFormat::Dense,
        ReplicatedTextParameterRole::LinearWeight,
        ReplicatedTextParameterOwner::ExecutionUnit {
            group: "units".into(),
            unit: 0,
        },
        ReplicatedTextParameterPresence::Required,
        ParameterTransformConstraint::Linear { packed_axis: 1 },
    )
    .unwrap()
    .with_transform_companions(format!("{name}.scale"), format!("{name}.offset"))
    .unwrap()
}

fn requirements(
    parameters: Vec<ReplicatedTextParameterRequirement>,
    state: LayerCachePolicy,
    access: ReplicatedTextStateAccess,
) -> ReplicatedTextRequirements {
    let graph = ExecutionGraph::chain(["units"]).unwrap();
    let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
    ReplicatedTextRequirements::new(
        "test.mechanism-synthesis",
        NeuralOperatorCapabilities::NONE,
        graph,
        units,
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Decoder,
            first_owner_static_roles: Vec::new(),
            last_owner_static_roles: Vec::new(),
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        StateLayout::new(LayerSchedule::new(1, vec![state]).unwrap()).unwrap(),
        access,
        parameters,
    )
    .unwrap()
    .with_floating_state_source(TensorDtype::F32)
}

fn dense_requirements(
    parameters: Vec<ReplicatedTextParameterRequirement>,
) -> ReplicatedTextRequirements {
    requirements(
        parameters,
        LayerCachePolicy::NoState,
        ReplicatedTextStateAccess::Stateless,
    )
}

fn request() -> ReplicatedTextSelectionRequest {
    ReplicatedTextSelectionRequest::new(
        LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    )
}

#[test]
fn malformed_catalog_ranks_and_zero_rank_packing_fail_without_native_queries() {
    let make = |logical: Vec<usize>, physical: Vec<usize>, executable, role| {
        let source = SourceTensorEncoding::Safetensors(StoredDtype::F32);
        ReplicatedTextParameterRequirement::new(
            "weight",
            vec!["weight".into()],
            vec![ReplicatedTextPhysicalSource::new(
                "weight",
                "weight",
                "/checkpoint/weights.safetensors",
                "weight",
                source.clone(),
                4,
            )
            .unwrap()],
            Vec::new(),
            Some(source),
            Some(physical),
            logical,
            executable,
            role,
            ReplicatedTextParameterOwner::ExecutionUnit {
                group: "units".into(),
                unit: 0,
            },
            ReplicatedTextParameterPresence::Required,
            ParameterTransformConstraint::None,
        )
    };
    let rank_mismatch = make(
        vec![2, 3, 4],
        vec![2, 3],
        LinearFormat::Dense,
        ReplicatedTextParameterRole::LinearWeight,
    )
    .unwrap();
    let malformed = make(
        Vec::new(),
        Vec::new(),
        LinearFormat::Affine(AffineQuantization::new(16, 4).unwrap()),
        ReplicatedTextParameterRole::Embedding,
    )
    .unwrap();
    for parameter in [rank_mismatch, malformed] {
        let requirements = dense_requirements(vec![parameter]);
        let support = Support {
            direct: true,
            transform: true,
            ..Support::default()
        };
        for request in [
            request(),
            request().with_quantization(QuantizationRequest::MxFp4),
        ] {
            let capabilities =
                synthesize_replicated_text_capabilities(&requirements, &request, &support);
            assert!(capabilities.weight_lowerings().is_empty());
            assert!(support.queries.borrow().is_empty());
            assert!(
                select_replicated_text_realization(&requirements, &request, &capabilities).is_err()
            );
        }
    }
}

#[test]
fn gguf_source_padding_does_not_admit_an_invalid_executable_format() {
    let descriptor = WeightLoweringDescriptor::new(
        SourceTensorEncoding::Gguf {
            ggml_type: eredu_gguf::GgmlType::Q4_0,
            endian: eredu_gguf::Endian::Little,
        },
        LinearFormat::Affine(AffineQuantization {
            group_size: 0,
            ..AffineQuantization::default()
        }),
        vec![64, 32],
        vec![64, 32],
        Some(1),
    )
    .unwrap();
    assert!(!descriptor.has_valid_direct_geometry());
    assert!(!descriptor.has_valid_transform_geometry());
}

#[test]
fn packed_scalars_cannot_bypass_axis_validation_in_direct_source_branches() {
    for source in [
        SourceTensorEncoding::Safetensors(StoredDtype::U8),
        SourceTensorEncoding::RecipeOutput(StoredDtype::U8),
        SourceTensorEncoding::Gguf {
            ggml_type: eredu_gguf::GgmlType::Q4_0,
            endian: eredu_gguf::Endian::Little,
        },
    ] {
        let descriptor = WeightLoweringDescriptor::new(
            source,
            LinearFormat::MxFp4,
            Vec::new(),
            Vec::new(),
            None,
        )
        .unwrap();
        assert!(!descriptor.has_valid_direct_geometry());
        assert!(!descriptor.has_valid_transform_geometry());
    }
    let scalar = WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(StoredDtype::F32),
        LinearFormat::Dense,
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();
    assert!(scalar.has_valid_direct_geometry());
}

#[test]
fn synthesis_preserves_candidate_order_and_deduplicates_primary_and_auxiliary_descriptors() {
    let primary = parameter("primary", vec![64, 64], StoredDtype::F16);
    let auxiliary = parameter("auxiliary", vec![64, 64], StoredDtype::F16);
    let requirements = dense_requirements(vec![primary.clone()])
        .with_auxiliary_parameters(vec![auxiliary], BTreeMap::new(), BTreeMap::new())
        .unwrap();
    let request = request().with_quantization(QuantizationRequest::Affine {
        group_size: 32,
        bits: 4,
    });
    let support = Support {
        direct: true,
        transform: true,
        ..Support::default()
    };
    let report = synthesize_replicated_text_capabilities(&requirements, &request, &support);
    let affine = LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap());
    assert_eq!(
        report.weight_lowerings(),
        [
            WeightLoweringCapability::new(
                primary.lowering_descriptor(LinearFormat::Dense).unwrap(),
                WeightLoweringKind::Direct
            ),
            WeightLoweringCapability::new(
                primary.lowering_descriptor(affine).unwrap(),
                WeightLoweringKind::Transform
            ),
        ]
    );
    assert_eq!(
        report,
        synthesize_replicated_text_capabilities(&requirements, &request, &support)
    );
    assert!(support.state_queries.borrow().is_empty());
    assert!(!report.exact_completion());
    assert!(!report.prompt_cache());
    assert_eq!(report.session(), SessionCapabilities::default());
    assert_eq!(report.addressable_storage(), None);
}

#[test]
fn direct_only_and_transform_only_support_remain_distinct_selection_facts() {
    let requirements =
        dense_requirements(vec![parameter("weight", vec![64, 64], StoredDtype::F16)]);
    let direct = Support {
        direct: true,
        ..Support::default()
    };
    let transform = Support {
        transform: true,
        ..Support::default()
    };
    let plain = request();
    let quantized = request().with_quantization(QuantizationRequest::MxFp4);
    let direct_report = synthesize_replicated_text_capabilities(&requirements, &quantized, &direct);
    let transform_report =
        synthesize_replicated_text_capabilities(&requirements, &quantized, &transform);
    assert!(select_replicated_text_realization(&requirements, &plain, &direct_report).is_ok());
    assert!(select_replicated_text_realization(&requirements, &quantized, &direct_report).is_err());
    assert!(select_replicated_text_realization(&requirements, &plain, &transform_report).is_err());
    let selected =
        select_replicated_text_realization(&requirements, &quantized, &transform_report).unwrap();
    assert_eq!(
        selected.parameters()[0].lowering(),
        WeightLoweringKind::Transform
    );
    assert_eq!(selected.parameters()[0].executable(), LinearFormat::MxFp4);
}

#[test]
fn invalid_transform_preserves_architecture_diagnostic_without_querying_that_target() {
    let requirements =
        dense_requirements(vec![parameter("weight", vec![64, 48], StoredDtype::F16)]);
    let request = request().with_quantization(QuantizationRequest::MxFp4);
    let support = Support {
        direct: true,
        transform: true,
        ..Support::default()
    };
    let report = synthesize_replicated_text_capabilities(&requirements, &request, &support);
    assert!(support
        .queries
        .borrow()
        .iter()
        .all(|(_, descriptor)| descriptor.executable() == LinearFormat::Dense));
    let error = select_replicated_text_realization(&requirements, &request, &report).unwrap_err();
    assert!(error.to_string().contains("not divisible by block size"));
}

#[test]
fn raw_byte_linear_is_rejected_before_native_lowering_queries() {
    let requirements = dense_requirements(vec![parameter("weight", vec![64, 64], StoredDtype::U8)]);
    let support = Support {
        direct: true,
        transform: true,
        ..Support::default()
    };
    let report = synthesize_replicated_text_capabilities(&requirements, &request(), &support);
    assert!(report.weight_lowerings().is_empty());
    assert!(support.queries.borrow().is_empty());
    assert!(select_replicated_text_realization(&requirements, &request(), &report).is_err());
}

#[test]
fn exact_state_components_retain_dtype_shape_and_semantic_placement() {
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(8).unwrap(),
        ],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    let requirements = requirements(
        Vec::new(),
        LayerCachePolicy::key_value_with_fixed_state(AttentionPolicy::Full, 2, 8, vec![fixed])
            .unwrap(),
        ReplicatedTextStateAccess::AttentionWithFixed,
    );
    let paged = CacheResidencyPolicy::Paged(
        crate::PagedCacheOptions::new(4, 1024, 1024, 1)
            .unwrap()
            .with_full_attention(true),
    );
    let request =
        ReplicatedTextSelectionRequest::new(LayerWeightResidency::FullyResident, paged.clone());
    let support = Support {
        paged: true,
        ..Support::default()
    };
    let report = synthesize_replicated_text_capabilities(&requirements, &request, &support);
    let declared = requirements.state_layout().components(0).unwrap();
    assert_eq!(
        report
            .state()
            .components()
            .iter()
            .map(|component| component.component())
            .collect::<Vec<_>>(),
        declared.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        report
            .state()
            .components()
            .iter()
            .map(|component| component.placement(&paged))
            .collect::<Vec<_>>(),
        [
            Some(StateComponentPlacement::Paged),
            Some(StateComponentPlacement::Paged),
            Some(StateComponentPlacement::Device)
        ]
    );
    assert!(support
        .state_queries
        .borrow()
        .iter()
        .all(
            |(component, placement)| component.residency() == StateResidencyClass::SealablePaged
                || *placement == StateComponentPlacement::Device
        ));
    assert!(select_replicated_text_realization(&requirements, &request, &report).is_ok());

    let unsupported = Support {
        paged: true,
        rejected_dtype: Some(StateTensorDtype::Float32),
        ..Support::default()
    };
    let report = synthesize_replicated_text_capabilities(&requirements, &request, &unsupported);
    let error = select_replicated_text_realization(&requirements, &request, &report).unwrap_err();
    assert!(error.to_string().contains("Float32"));
    let device_only =
        synthesize_replicated_text_capabilities(&requirements, &request, &Support::default());
    assert!(select_replicated_text_realization(&requirements, &request, &device_only).is_err());
}

#[test]
fn malformed_state_geometry_is_rejected_before_support_can_receive_a_component() {
    for shape in [
        Vec::new(),
        vec![StateTensorDimension::Scalar, StateTensorDimension::Batch],
    ] {
        assert!(StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            shape,
            StateTensorDtype::Float32,
            MutableStateResidency::LayerScopedOffloadable,
        )
        .is_err());
    }
}

#[test]
fn portable_lowering_geometry_preserves_padding_packing_and_overflow_rejections() {
    let affine = LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap());
    let gguf = |shape| {
        WeightLoweringDescriptor::new(
            SourceTensorEncoding::Gguf {
                ggml_type: eredu_gguf::GgmlType::Q4_0,
                endian: eredu_gguf::Endian::Little,
            },
            affine,
            shape,
            vec![64, 8],
            Some(1),
        )
        .unwrap()
    };
    assert!(gguf(vec![64, 32]).has_valid_direct_geometry());
    for shape in [vec![63, 32], vec![64, 31], vec![64, 64]] {
        assert!(!gguf(shape).has_valid_direct_geometry());
    }
    for source in [
        SourceTensorEncoding::Safetensors(StoredDtype::U32),
        SourceTensorEncoding::RecipeOutput(StoredDtype::U32),
    ] {
        let descriptor = |physical, logical| {
            WeightLoweringDescriptor::new(
                source.clone(),
                affine,
                vec![64, physical],
                vec![64, logical],
                Some(1),
            )
            .unwrap()
        };
        assert!(descriptor(8, 64).has_valid_direct_geometry());
        assert!(!descriptor(7, 64).has_valid_direct_geometry());
        let overflowing = usize::MAX - 31;
        assert!(!descriptor(overflowing, overflowing).has_valid_direct_geometry());
    }
    let nonfinal = WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(StoredDtype::F16),
        affine,
        vec![64, 64],
        vec![64, 64],
        Some(0),
    )
    .unwrap();
    assert!(!nonfinal.has_valid_direct_geometry());
    assert!(!nonfinal.has_valid_transform_geometry());
    assert!(WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(StoredDtype::F16),
        affine,
        vec![64, 0],
        vec![64, 64],
        Some(1)
    )
    .is_err());
    assert!(WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(StoredDtype::U8),
        LinearFormat::MxFp4,
        vec![64, 2, 16],
        vec![64, 64],
        Some(1)
    )
    .is_err());
}

#[test]
fn floating_storage_width_changes_admission_and_cannot_reuse_another_sources_report() {
    let requirements = requirements(
        Vec::new(),
        LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
        ReplicatedTextStateAccess::KeyValue,
    );
    let half = requirements
        .clone()
        .with_floating_state_source(TensorDtype::F16);
    let support = Support {
        maximum_element_bytes: Some(2),
        ..Support::default()
    };
    let half_report = synthesize_replicated_text_capabilities(&half, &request(), &support);
    let selected = select_replicated_text_realization(&half, &request(), &half_report).unwrap();
    assert!(selected
        .state()
        .components()
        .iter()
        .all(|component| component.storage_dtype() == StateStorageDtype::F16));
    assert!(support
        .storage_queries
        .borrow()
        .iter()
        .all(|dtype| *dtype == StateStorageDtype::F16));
    let full_report = synthesize_replicated_text_capabilities(&requirements, &request(), &support);
    assert!(select_replicated_text_realization(&requirements, &request(), &full_report).is_err());
    let stale =
        select_replicated_text_realization(&requirements, &request(), &half_report).unwrap_err();
    assert!(stale
        .to_string()
        .contains("differs from the selected source"));
    let unknown = requirements.with_floating_state_source(TensorDtype::Encoded("unknown".into()));
    support.storage_queries.borrow_mut().clear();
    let report = synthesize_replicated_text_capabilities(&unknown, &request(), &support);
    assert!(support.storage_queries.borrow().is_empty());
    assert!(select_replicated_text_realization(&unknown, &request(), &report).is_err());
}

#[test]
fn half_precision_activations_preserve_fixed_float_and_integer_state_dtypes() {
    let fixed = [
        (StateTensorRole::Recurrent, StateTensorDtype::Float32),
        (
            StateTensorRole::Convolution { slot: 0 },
            StateTensorDtype::Int32,
        ),
    ]
    .into_iter()
    .map(|(role, dtype)| {
        StateTensorPolicy::new(
            role,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(8).unwrap(),
            ],
            dtype,
            match role {
                StateTensorRole::Convolution { .. } => MutableStateResidency::AlwaysDeviceMutable,
                _ => MutableStateResidency::LayerScopedOffloadable,
            },
        )
        .unwrap()
    })
    .collect();
    let requirements = requirements(
        Vec::new(),
        LayerCachePolicy::key_value_with_fixed_state(AttentionPolicy::Full, 2, 8, fixed).unwrap(),
        ReplicatedTextStateAccess::AttentionWithFixed,
    )
    .with_floating_state_source(TensorDtype::F16);
    let report =
        synthesize_replicated_text_capabilities(&requirements, &request(), &Support::default());
    let selected = select_replicated_text_realization(&requirements, &request(), &report).unwrap();
    assert_eq!(
        selected
            .state()
            .components()
            .iter()
            .map(|c| c.storage_dtype())
            .collect::<Vec<_>>(),
        [
            StateStorageDtype::F16,
            StateStorageDtype::F16,
            StateStorageDtype::F32,
            StateStorageDtype::I32
        ]
    );
}
