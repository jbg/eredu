//! Multi-family structural and portable-lifecycle tests using a shape-only backend.

use std::{cell::RefCell, collections::BTreeMap};

use eredu_architectures::{
    configuration,
    decoder::{self, AttentionProjectionLayout, GatedProjectionLayout, TransformerBlock},
    deepseek, gemma4, gpt_oss, inkling, kimi_linear, lfm2,
    llama::{self, LayeredInput, ModelArgs},
    moshi, muse_glimmer, qwen,
    replicated_text::{
        replicated_text_requirements, visit_replicated_text_architecture,
        PreparedReplicatedTextArchitecture, ReplicatedTextArchitectureVisitor,
    },
};
use eredu_core::{AttentionPolicy, Completion, LayerSchedule, TokenFilter};
use eredu_nn::{
    validate_parameter_topology, AttentionCache, AttentionMask, AttentionRequest,
    DistributedNeuralBackend, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error,
    GroupSelection, GroupSelectionOperator, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, GroupedRelu2Operator, GroupedRelu2Spec, Index, LinearOperator,
    LinearSpec, NeuralBackend, NormalizationConstructionSpec, NormalizationOperator,
    NormalizationScale, PadMode, ParameterMetadata, ParameterVisitor, ParameterVisitorMut,
    Parameterized, RotaryOperator, RotaryPosition, RotarySpec, Tensor,
    TensorParallelGroupedGatedProductOperator, TensorParallelGroupedOutput,
    TensorParallelGroupedRelu2Operator, TopKGroupSelectorSpec, VocabularyParallelRange,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_bindings, ArchitectureParameterDescription,
    ArchitectureParameters, ArchitecturePartition, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, DeviceState, ExpertPass, LayerRuntimeState,
    LayeredArchitecture, LayerwiseRuntime, LocalModelLayout, LocalTensorLayout, MemberSharding,
    NoAuxiliaryBoundarySchema, ParameterBackend, ParameterGroupOwner, ParameterGroupSpec,
    PartitionOwnership, PenaltyConfig, PredictionDirective, ReplicatedTextParameterOwner,
    ReplicatedTextParameterPresence, ReplicatedTextParameterRole, ResettableRuntimeLayerState,
    ResidentRuntime, ResidentUnitWindow, RoutedExpertProvider, RoutedExpertRequest,
    RuntimeLayerState, RuntimeState, RuntimeStateComponents, Sampler, SamplingBackend,
    SequentialDecisionDriver, SequentialDecisionMode, SequentialDecisionPlan,
    SequentialDecisionSource, SequentialDecisionTraversal, StateError, StaticParameterVisitor,
    SubmissionBackend, TensorPlacement, TokenDomain, WeightBinding,
};

struct ReferenceReplicatedVisitor {
    construction_started: bool,
}

impl
    ReplicatedTextArchitectureVisitor<
        ReferenceBackend,
        DeviceState<ReferenceBackend, ReferenceCache>,
    > for ReferenceReplicatedVisitor
{
    type Output = ReferenceTensor;
    type Error = Error;

    fn construction_started(&mut self) {
        self.construction_started = true;
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        _: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                Error = Error,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        assert!(self.construction_started);
        let layout = prepared.requirements().state_layout().clone();
        let unit_count = prepared.requirements().execution_units().len();
        let architecture = prepared.into_modules().take_architecture();
        let units = (0..unit_count)
            .map(|index| A::build_unit(&architecture, 0, index, &()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
        let mut state = DeviceState::create(layout, |_, policy| {
            Ok::<_, std::convert::Infallible>(ReferenceCache {
                offset: 0,
                window: policy
                    .attention()
                    .and_then(|attention| attention.window())
                    .map(|window| window.get() as i32),
                resets: 0,
                fixed: None,
            })
        })
        .expect("reference state layout is valid");
        let tokens = ReferenceTensor(vec![1, 3]);
        runtime
            .forward(A::text_input(&tokens, None), &mut state, &())
            .map_err(|error| Error::backend(error.to_string()))
    }
}

include!("support/reference_backend.rs");
fn tiny_args() -> ModelArgs {
    ModelArgs {
        model_type: "llama".into(),
        hidden_size: 8,
        num_hidden_layers: 2,
        intermediate_size: 16,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 32,
        num_key_value_heads: 1,
        max_position_embeddings: 128,
        rope_theta: 10_000.0,
        rope_traditional: false,
        head_dim: 4,
        tie_word_embeddings: true,
        attention_bias: false,
        mlp_bias: false,
        rope_scaling: None,
        attention_schedule: LayerSchedule::new(2, vec![AttentionPolicy::Full; 2]).unwrap(),
        quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    }
}

fn llama_parallel_layout(
    args: &ModelArgs,
    vocabulary: std::ops::Range<usize>,
    local_query_heads: i32,
    local_key_value_heads: i32,
) -> LocalModelLayout {
    let architecture = llama::LayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
    let static_modules = architecture.static_modules();
    let mut groups = llama::static_parallel_parameter_groups::<ReferenceBackend>(
        &static_modules.embeddings,
        &static_modules.norm,
        static_modules.lm_head.as_ref(),
        "model",
    )
    .unwrap();
    for layer in 0..args.num_hidden_layers as usize {
        let block = TransformerBlock::<ReferenceBackend>::new(args, layer, &()).unwrap();
        groups.extend(
            llama::layer_parallel_parameter_groups::<ReferenceBackend>(&block, args, layer)
                .unwrap(),
        );
    }

    let query_width = usize::try_from(local_query_heads * args.head_dim).unwrap();
    let key_value_width = usize::try_from(local_key_value_heads * args.head_dim).unwrap();
    let mut layout = LocalModelLayout::default();
    for group in groups {
        for member in group.members() {
            let target = member.target();
            let mut local_shape = member.global_shape().to_vec();
            let (placement, logical_range) = if group.role()
                == eredu_runtime::ParameterRole::Vocabulary
            {
                local_shape[0] = vocabulary.len();
                (
                    TensorPlacement::Range {
                        axis: 0,
                        start: vocabulary.start,
                        end: vocabulary.end,
                    },
                    Some(vocabulary.clone()),
                )
            } else if target.contains(".self_attn.q_proj") {
                local_shape[0] = query_width;
                (
                    TensorPlacement::Range {
                        axis: 0,
                        start: 0,
                        end: query_width,
                    },
                    Some(0..usize::try_from(local_query_heads).unwrap()),
                )
            } else if target.contains(".self_attn.k_proj") || target.contains(".self_attn.v_proj") {
                local_shape[0] = key_value_width;
                (
                    TensorPlacement::Range {
                        axis: 0,
                        start: 0,
                        end: key_value_width,
                    },
                    Some(0..usize::try_from(local_key_value_heads).unwrap()),
                )
            } else {
                (
                    TensorPlacement::Replicated,
                    group.partition_units().map(|units| 0..units),
                )
            };
            layout.insert(
                target.to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    local_shape,
                    placement,
                    group.partition_units(),
                    logical_range,
                    false,
                ),
            );
        }
    }
    layout
}

fn replicated_parallel_layout(groups: &[ParameterGroupSpec]) -> LocalModelLayout {
    let mut layout = LocalModelLayout::default();
    for group in groups {
        for member in group.members() {
            layout.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    member.global_shape().to_vec(),
                    TensorPlacement::Replicated,
                    group.partition_units(),
                    group.partition_units().map(|units| 0..units),
                    false,
                ),
            );
        }
    }
    layout
}

fn assert_geometry_identity_error<T>(result: Result<T, Error>, family: &str) {
    match result {
        Err(error) => assert!(
            error
                .to_string()
                .contains("different normalized configuration"),
            "unexpected {family} geometry error: {error}"
        ),
        Ok(_) => panic!("{family} accepted geometry derived from a different configuration"),
    }
}

#[test]
fn shared_decoder_lifecycle_validates_before_construction() {
    let mut args = tiny_args();
    args.num_hidden_layers = -1;
    let error = llama::LayeredModel::<ReferenceBackend>::new(args, &())
        .err()
        .expect("negative decoder layer count must be rejected");
    assert!(
        error
            .to_string()
            .contains("num_hidden_layers must be positive, got -1"),
        "unexpected validation error: {error}"
    );
}

#[test]
fn shared_decoder_parallel_geometry_rejects_cross_config_reuse() {
    let llama_args = tiny_args();
    let llama_layout = llama_parallel_layout(&llama_args, 0..32, 2, 1);
    let llama_geometry = llama::local_geometry(&llama_args, &llama_layout).unwrap();
    let mut changed_llama = llama_args;
    changed_llama.rope_theta = 20_000.0;
    assert_geometry_identity_error(
        llama::LayeredModel::<ReferenceBackend>::new_parallel(changed_llama, llama_geometry, &()),
        "Llama",
    );

    let qwen_args = qwen::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let qwen_model = qwen::LayeredModel::<ReferenceBackend>::new(qwen_args.clone(), &()).unwrap();
    let mut qwen_groups = qwen::static_parallel_parameter_groups::<ReferenceBackend>(
        &qwen_model.static_modules().embeddings,
        &qwen_model.static_modules().norm,
        qwen_model.static_modules().lm_head.as_ref(),
        &qwen_args.parameter_root,
    )
    .unwrap();
    let qwen_unit = qwen_model.construct_unit(0, &()).unwrap();
    qwen_groups.extend(qwen::layer_parallel_parameter_groups(&qwen_unit, &qwen_args, 0).unwrap());
    let qwen_geometry =
        qwen::local_geometry(&qwen_args, &replicated_parallel_layout(&qwen_groups)).unwrap();
    let mut changed_qwen = qwen_args;
    changed_qwen.rms_norm_eps = 0.00002;
    assert_geometry_identity_error(
        qwen::LayeredModel::<ReferenceBackend>::new_parallel(changed_qwen, qwen_geometry, &()),
        "Qwen",
    );

    let gpt_args = gpt_oss::model_args_from_config_value(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 32,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001,
        "sliding_window": 4,
        "max_position_embeddings": 64,
        "rope_theta": 150000.0,
        "layer_types": ["full_attention"],
        "quantization_config": {"quant_method": "mxfp4"},
        "swiglu_limit": 7.0
    }))
    .unwrap();
    let gpt_model = gpt_oss::LayeredModel::<ReferenceBackend>::new(gpt_args.clone(), &()).unwrap();
    let mut gpt_groups =
        gpt_oss::static_parameter_groups(gpt_model.static_modules(), &gpt_args).unwrap();
    let gpt_unit = gpt_model.construct_unit(0, &()).unwrap();
    gpt_groups.extend(gpt_oss::layer_parallel_parameter_groups(&gpt_unit, &gpt_args, 0).unwrap());
    let gpt_geometry =
        gpt_oss::local_geometry(&gpt_args, &replicated_parallel_layout(&gpt_groups)).unwrap();
    let mut changed_gpt = gpt_args;
    changed_gpt.rms_norm_eps = 0.00002;
    assert_geometry_identity_error(
        gpt_oss::LayeredModel::<ReferenceBackend>::new_parallel(changed_gpt, gpt_geometry, &()),
        "GPT-OSS",
    );
}

struct StaticTopologyCollector(BTreeMap<String, Vec<String>>);

impl StaticParameterVisitor<ReferenceBackend> for StaticTopologyCollector {
    type Error = Error;

    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<ReferenceTensor>,
    {
        let parameters = validate_parameter_topology(module)
            .map_err(Error::backend)?
            .into_iter()
            .map(|metadata| metadata.id.as_str().to_owned())
            .collect();
        assert!(self.0.insert(role.to_owned(), parameters).is_none());
        Ok(())
    }
}

#[test]
fn shared_decoder_exposes_architecture_owned_static_parameter_bindings() {
    let args = qwen::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let model = qwen::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let mut visitor = StaticTopologyCollector(BTreeMap::new());
    model.visit_static_parameters(&mut visitor).unwrap();

    assert_eq!(
        visitor.0,
        BTreeMap::from([
            ("embedding".into(), vec!["model.embed_tokens.weight".into()]),
            ("norm".into(), vec!["model.norm.weight".into()]),
            ("output".into(), vec!["lm_head.weight".into()]),
        ])
    );
}

#[test]
fn replicated_requirement_catalog_matches_authoritative_architecture_parameters() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    });
    let args = qwen::model_args_from_config_value(&config).unwrap();
    let model = qwen::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let description = model.parameter_description(&()).unwrap();
    let mut authoritative = BTreeMap::new();
    for owned in description.groups() {
        for member in owned.group().members() {
            assert!(authoritative
                .insert(
                    member.target().to_owned(),
                    (member.global_shape().to_vec(), owned.owner().clone()),
                )
                .is_none());
        }
    }

    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tensors = authoritative
        .iter()
        .map(|(name, (shape, _))| {
            let elements = shape.iter().product::<usize>();
            (name.clone(), shape.clone(), vec![0_u8; elements * 4])
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &artifact.path().join("model.safetensors")).unwrap();
    let inspection = configuration::inspect_artifact(artifact.path()).unwrap();
    let requirements = replicated_text_requirements(&inspection).unwrap();
    let physical_requirements = requirements
        .parameters()
        .iter()
        .filter(|parameter| parameter.presence().has_physical_source())
        .map(|parameter| parameter.name().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        physical_requirements,
        authoritative.keys().cloned().collect()
    );

    for (target, (shape, owner)) in &authoritative {
        let requirement = requirements
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == target)
            .unwrap_or_else(|| panic!("requirement catalog omitted {target}"));
        assert_eq!(requirement.logical_shape(), shape);
        assert_eq!(requirement.physical_shape(), Some(shape.as_slice()));
        assert_eq!(requirement.sources(), std::slice::from_ref(target));
        assert_eq!(
            requirement.presence(),
            &ReplicatedTextParameterPresence::Required
        );
        match (owner, requirement.owner()) {
            (
                ParameterGroupOwner::StaticRole(expected),
                ReplicatedTextParameterOwner::StaticRole(actual),
            ) => assert_eq!(actual, expected),
            (
                ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                },
                ReplicatedTextParameterOwner::ExecutionUnit {
                    group: actual_group,
                    unit,
                },
            ) => {
                assert_eq!(actual_group, group.as_str());
                assert_eq!(unit, global_unit);
            }
            (expected, actual) => panic!("owner mismatch for {target}: {expected:?} != {actual:?}"),
        }
    }
    assert!(requirements.parameters().iter().any(|parameter| {
        parameter.role() == ReplicatedTextParameterRole::Normalization
            && parameter.logical_shape().len() == 1
    }));

    let weight_lowerings = requirements
        .parameters()
        .iter()
        .filter(|parameter| parameter.has_lowering_source())
        .map(|parameter| {
            eredu_runtime::WeightLoweringCapability::new(
                parameter
                    .lowering_descriptor(parameter.native_executable())
                    .unwrap(),
                eredu_runtime::WeightLoweringKind::Direct,
            )
        })
        .collect();
    let state = eredu_runtime::StateMechanismCapabilities::new(
        (0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .unwrap()
                .iter()
                .cloned()
                .map(move |component| {
                    eredu_runtime::StateComponentMechanism::new(
                        layer,
                        component,
                        Some(eredu_runtime::StateComponentPlacement::Device),
                        None,
                    )
                })
        }),
    )
    .with_transactions(true, true)
    .with_reset(true);
    let capabilities = eredu_runtime::BackendMechanismCapabilities::new(
        eredu_nn::NeuralOperatorCapabilities::ALL,
        weight_lowerings,
        vec![eredu_runtime::WeightResidencyMechanism::Resident],
        state,
    );
    let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        eredu_runtime::CacheResidencyPolicy::Device,
    );
    let selected =
        eredu_runtime::select_replicated_text_realization(&requirements, &request, &capabilities)
            .unwrap();
    let store: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
    );
    let logits = visit_replicated_text_architecture::<
        ReferenceBackend,
        DeviceState<ReferenceBackend, ReferenceCache>,
        _,
    >(
        inspection.architecture_plan(),
        selected,
        store,
        &(),
        ReferenceReplicatedVisitor {
            construction_started: false,
        },
    )
    .unwrap();
    assert_eq!(logits, ReferenceTensor(vec![1, 3, 32]));
}

#[test]
fn neutral_llama_parallel_geometry_owns_uneven_tied_and_untied_vocabularies() {
    let mut tied_args = tiny_args();
    tied_args.vocab_size = 7;
    let tied_layout = llama_parallel_layout(&tied_args, 4..7, 2, 1);
    let tied_geometry = llama::local_geometry(&tied_args, &tied_layout).unwrap();
    assert_eq!(tied_geometry.embedding_range().global_vocabulary, 7);
    assert_eq!(tied_geometry.embedding_range().local, 4..7);
    assert_eq!(tied_geometry.output_range(), None);

    let replicated: llama::LayeredModel<ReferenceBackend> =
        llama::LayeredModel::new(tied_args.clone(), &()).unwrap();
    let parallel: llama::LayeredModel<ReferenceBackend> =
        llama::LayeredModel::new_parallel(tied_args.clone(), tied_geometry, &()).unwrap();
    assert!(replicated.parallel_geometry().is_none());
    assert!(parallel.parallel_geometry().is_some());
    assert_eq!(parallel.static_modules().embeddings.weight.0, [3, 8]);
    assert!(parallel.static_modules().lm_head.is_none());

    let mut untied_args = tied_args;
    untied_args.tie_word_embeddings = false;
    let untied_layout = llama_parallel_layout(&untied_args, 4..7, 2, 1);
    let untied_geometry = llama::local_geometry(&untied_args, &untied_layout).unwrap();
    assert_eq!(
        untied_geometry
            .output_range()
            .map(|range| range.local.clone()),
        Some(4..7)
    );
    let untied: llama::LayeredModel<ReferenceBackend> =
        llama::LayeredModel::new_parallel(untied_args, untied_geometry, &()).unwrap();
    assert_eq!(untied.static_modules().embeddings.weight.0, [3, 8]);
    assert_eq!(
        untied.static_modules().lm_head.as_ref().unwrap().weight.0,
        [3, 8]
    );
}

#[test]
fn llama_static_parameter_topology_matches_constructed_dense_model() {
    let args = tiny_args();
    let static_description = llama::dense_parameter_description(&args).unwrap();
    let constructed = llama::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let constructed_description = constructed.parameter_description(&()).unwrap();
    assert_eq!(static_description, constructed_description);
}

fn dense_qwen_partition_args(model_type: &str) -> qwen::ModelArgs {
    qwen::model_args_from_config_value(&serde_json::json!({
        "model_type": model_type,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rms_norm_eps": 0.00001,
        "vocab_size": 7,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    }))
    .unwrap()
}

fn local_qwen_partition(
    args: &qwen::ModelArgs,
    tensor_parts: usize,
    tensor_rank: usize,
) -> (
    LocalModelLayout,
    ArchitecturePartition<qwen::PartitionLocalGeometry<qwen::ModelArgs>, NoAuxiliaryBoundarySchema>,
) {
    let description = qwen::dense_parameter_description(args).unwrap();
    let topology = eredu_core::ParallelTopology::new(tensor_parts, 1, 1, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, tensor_rank).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &description,
        rank,
    )
    .unwrap();
    let units = 0..usize::try_from(args.num_hidden_layers).unwrap();
    let geometry = qwen::partition_local_geometry(args, &layout, units.clone()).unwrap();
    let state = geometry.complete_state_layout().clone();
    let state_plan =
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..state.len(),
        )]);
    let partition = ArchitecturePartition::from_description(
        &description,
        [(decoder::TEXT_DECODER_EXECUTION_GROUP, units)],
        PartitionOwnership::new(true, true, ["embedding", "norm", "output"]).unwrap(),
        &state,
        &state_plan,
        geometry,
        NoAuxiliaryBoundarySchema::new(args.hidden_size),
    )
    .unwrap();
    (layout, partition)
}

#[test]
fn dense_qwen_partition_topology_preserves_qwen2_biases_and_qwen3_qk_norms() {
    for model_type in ["qwen2", "qwen3"] {
        let args = dense_qwen_partition_args(model_type);
        let description = qwen::dense_parameter_description(&args).unwrap();
        let constructed = qwen::LayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
        if model_type == "qwen3" {
            assert_eq!(description, constructed.parameter_description(&()).unwrap());
        }

        let (layout, partition) = local_qwen_partition(&args, 2, 1);
        let model = qwen::PartitionedLayeredModel::<ReferenceBackend>::from_partition(
            args.clone(),
            &description,
            &partition,
            &(),
        )
        .unwrap();
        let unit = model.construct_unit(0, &()).unwrap();
        assert_eq!(unit.self_attention.query_heads, 2);
        assert_eq!(unit.self_attention.key_value_heads, 1);
        assert_eq!(model.local_geometry().local_unit_count(), 2);

        let prefix = "model.layers.0.self_attn";
        if model_type == "qwen2" {
            assert_eq!(
                layout
                    .tensor(&format!("{prefix}.q_proj.bias"))
                    .unwrap()
                    .local_shape(),
                [4]
            );
            for projection in ["k_proj", "v_proj"] {
                assert_eq!(
                    layout
                        .tensor(&format!("{prefix}.{projection}.bias"))
                        .unwrap()
                        .local_shape(),
                    [2]
                );
            }
            assert!(layout.tensor(&format!("{prefix}.q_norm.weight")).is_none());
        } else {
            assert!(layout.tensor(&format!("{prefix}.q_proj.bias")).is_none());
            for norm in ["q_norm", "k_norm"] {
                let tensor = layout.tensor(&format!("{prefix}.{norm}.weight")).unwrap();
                assert_eq!(tensor.global_shape(), [2]);
                assert_eq!(tensor.local_shape(), [2]);
                assert_eq!(tensor.placement(), &TensorPlacement::Replicated);
            }
        }
    }
}

#[test]
fn dense_qwen_partition_geometry_rejects_routed_moe_before_construction() {
    let value = serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 0,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rms_norm_eps": 0.00001,
        "vocab_size": 7,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true
    });
    let args = qwen::model_args_from_config_value(&value).unwrap();
    let description = qwen::dense_parameter_description(&args).unwrap();
    let topology = eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, 0).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &description,
        rank,
    )
    .unwrap();
    let error = qwen::partition_local_geometry(&args, &layout, 0..2).unwrap_err();
    assert!(
        error.to_string().contains("does not accept routed MoE"),
        "unexpected dense Qwen MoE error: {error}"
    );
}

fn routed_qwen_partition_args() -> qwen::ModelArgs {
    qwen::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "intermediate_size": 0,
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rms_norm_eps": 0.00001,
        "vocab_size": 8,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    }))
    .unwrap()
}

fn gpt_oss_partition_args() -> gpt_oss::ModelArgs {
    gpt_oss::model_args_from_config_value(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 0.00001,
        "sliding_window": 128,
        "max_position_embeddings": 4096,
        "rope_theta": 150000.0,
        "quantization_config": { "quant_method": "mxfp4" },
        "swiglu_limit": 7.0
    }))
    .unwrap()
}

fn gpt_oss_parameter_description(args: &gpt_oss::ModelArgs) -> ArchitectureParameterDescription {
    gpt_oss::LayeredModel::<ReferenceBackend>::new(args.clone(), &())
        .unwrap()
        .parameter_description(&())
        .unwrap()
}
fn qwen_realization(
    args: &qwen::ModelArgs,
    layout: &LocalModelLayout,
    rank: eredu_core::ParallelRankTopology,
) -> eredu_architectures::ExpertRealizationPlan<GroupedGatedProductSpec> {
    let local_experts = eredu_core::balanced_contiguous_range(
        args.num_experts as usize,
        rank.expert_parallel_size(),
        rank.expert_parallel_rank(),
        false,
    )
    .unwrap()
    .len() as i32;
    let group =
        eredu_runtime::ExecutionGroupId::new(decoder::TEXT_DECODER_EXECUTION_GROUP).unwrap();
    let specs = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            let local = qwen::local_block_args(args, layer, layout).unwrap();
            let spec = qwen::expert_bank_spec(args, layer)
                .unwrap()
                .with_group_geometry(local_experts, local.moe_intermediate_size)
                .unwrap();
            ((group.clone(), layer), spec)
        })
        .collect::<BTreeMap<_, _>>();
    eredu_architectures::ExpertRealizationPlan::balanced(args.num_experts as usize, rank, specs)
        .unwrap()
}

fn gpt_oss_realization(
    args: &gpt_oss::ModelArgs,
    layout: &LocalModelLayout,
    rank: eredu_core::ParallelRankTopology,
) -> eredu_architectures::ExpertRealizationPlan<GroupedGatedProductSpec> {
    let local_experts = eredu_core::balanced_contiguous_range(
        args.num_local_experts as usize,
        rank.expert_parallel_size(),
        rank.expert_parallel_rank(),
        false,
    )
    .unwrap()
    .len() as i32;
    let group =
        eredu_runtime::ExecutionGroupId::new(decoder::TEXT_DECODER_EXECUTION_GROUP).unwrap();
    let specs = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            let local = gpt_oss::local_block_args(args, layer, layout).unwrap();
            let spec = gpt_oss::expert_bank_spec(args, layer)
                .unwrap()
                .with_group_geometry(local_experts, local.intermediate_size)
                .unwrap();
            ((group.clone(), layer), spec)
        })
        .collect::<BTreeMap<_, _>>();
    eredu_architectures::ExpertRealizationPlan::balanced(
        args.num_local_experts as usize,
        rank,
        specs,
    )
    .unwrap()
}

#[test]
fn routed_qwen_and_gpt_oss_cartesian_partitions_own_exact_units_state_and_banks() {
    let topologies = [(2, 1, 1), (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2)];
    for (tensor, pipeline, expert) in topologies {
        let topology = eredu_core::ParallelTopology::new(tensor, pipeline, expert, 1).unwrap();
        for global_rank in 0..topology.world_size() {
            let rank = eredu_core::ParallelRankTopology::new(topology, global_rank).unwrap();
            let owned = eredu_core::balanced_contiguous_range(
                4,
                rank.pipeline_parallel_size(),
                rank.pipeline_parallel_rank(),
                false,
            )
            .unwrap();
            let mut static_roles = Vec::new();
            if rank.pipeline_parallel_rank() == 0 {
                static_roles.push("embedding");
            }
            if rank.pipeline_parallel_rank() + 1 == rank.pipeline_parallel_size() {
                static_roles.extend(["norm", "output"]);
            }
            let ownership = PartitionOwnership::new(
                rank.pipeline_parallel_rank() == 0,
                rank.pipeline_parallel_rank() + 1 == rank.pipeline_parallel_size(),
                static_roles,
            )
            .unwrap();

            let qwen_args = routed_qwen_partition_args();
            let qwen_global =
                qwen::RoutedLayeredModel::<ReferenceBackend>::new(qwen_args.clone(), &()).unwrap();
            let qwen_description = qwen_global.parameter_description(&()).unwrap();
            let qwen_layout =
                eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    &qwen_description,
                    rank,
                )
                .unwrap();
            let qwen_plan = qwen_realization(&qwen_args, &qwen_layout, rank);
            let qwen_geometry = qwen::partition_local_routed_geometry(
                &qwen_args,
                &qwen_layout,
                owned.clone(),
                rank,
                &qwen_plan,
            )
            .unwrap();
            let qwen_state = qwen_geometry.complete_state_layout().clone();
            let qwen_partition = ArchitecturePartition::from_description(
                &qwen_description,
                [(decoder::TEXT_DECODER_EXECUTION_GROUP, owned.clone())],
                ownership.clone(),
                &qwen_state,
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(0, 0..qwen_state.len()),
                ]),
                qwen_geometry,
                NoAuxiliaryBoundarySchema::new(qwen_args.hidden_size),
            )
            .unwrap();
            let qwen_model =
                qwen::PartitionedRoutedLayeredModel::<ReferenceBackend>::from_partition(
                    qwen_args.clone(),
                    &qwen_description,
                    &qwen_partition,
                    &(),
                )
                .unwrap();
            assert_eq!(qwen_model.local_geometry().owned_units(), owned);
            assert_eq!(
                qwen_model.static_modules().embeddings.is_some(),
                rank.pipeline_parallel_rank() == 0
            );
            assert_eq!(
                qwen_model.static_modules().norm.is_some(),
                rank.pipeline_parallel_rank() + 1 == rank.pipeline_parallel_size()
            );
            assert_eq!(
                qwen_partition.state().unwrap().global_layers(),
                qwen_model.local_geometry().owned_units()
            );
            assert_eq!(
                qwen_plan.local_global_group_indices(),
                eredu_core::balanced_contiguous_range(
                    4,
                    rank.expert_parallel_size(),
                    rank.expert_parallel_rank(),
                    false,
                )
                .unwrap()
                .collect::<Vec<_>>()
            );
            let qwen_unit = qwen_model
                .construct_unit(qwen_model.local_geometry().owned_units().start, &())
                .unwrap();
            if owned.start > 0 {
                assert!(qwen_model.construct_unit(owned.start - 1, &()).is_err());
            }
            if owned.end < 4 {
                assert!(qwen_model.construct_unit(owned.end, &()).is_err());
            }
            assert_eq!(qwen_unit.self_attention.query_heads, 4 / tensor as i32);
            assert!(qwen_unit.self_attention.query_norm.is_some());
            assert!(qwen_unit.self_attention.key_norm.is_some());
            let qwen::FeedForward::Routed(qwen_moe) = qwen_unit.mlp else {
                panic!("partitioned Qwen unit lost routed policy");
            };
            assert_eq!(qwen_moe.router.weight.0, [4, 8]);
            assert_eq!(qwen_moe.experts.weight.0[0], 4 / expert as i32);
            assert_eq!(qwen_moe.experts.weight.0[1], 16 / tensor as i32);

            let gpt_args = gpt_oss_partition_args();
            let gpt_description = gpt_oss_parameter_description(&gpt_args);
            let gpt_layout =
                eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    &gpt_description,
                    rank,
                )
                .unwrap();
            let gpt_plan = gpt_oss_realization(&gpt_args, &gpt_layout, rank);
            let gpt_geometry = gpt_oss::partition_local_routed_geometry(
                &gpt_args,
                &gpt_layout,
                owned.clone(),
                rank,
                &gpt_plan,
            )
            .unwrap();
            let gpt_state = gpt_geometry.complete_state_layout().clone();
            let gpt_partition = ArchitecturePartition::from_description(
                &gpt_description,
                [(decoder::TEXT_DECODER_EXECUTION_GROUP, owned.clone())],
                ownership,
                &gpt_state,
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(0, 0..gpt_state.len()),
                ]),
                gpt_geometry,
                NoAuxiliaryBoundarySchema::new(gpt_args.hidden_size),
            )
            .unwrap();
            let gpt_model = gpt_oss::PartitionedLayeredModel::<ReferenceBackend>::from_partition(
                gpt_args,
                &gpt_description,
                &gpt_partition,
                &(),
            )
            .unwrap();
            assert_eq!(gpt_model.local_geometry().owned_units(), owned);
            assert_eq!(
                gpt_model.static_modules().embeddings.is_some(),
                rank.pipeline_parallel_rank() == 0
            );
            assert_eq!(
                gpt_model.static_modules().norm.is_some(),
                rank.pipeline_parallel_rank() + 1 == rank.pipeline_parallel_size()
            );
            assert_eq!(gpt_partition.state().unwrap().global_layers(), owned);
            let gpt_unit = gpt_model.construct_unit(owned.start, &()).unwrap();
            if owned.start > 0 {
                assert!(gpt_model.construct_unit(owned.start - 1, &()).is_err());
            }
            if owned.end < 4 {
                assert!(gpt_model.construct_unit(owned.end, &()).is_err());
            }
            assert!(gpt_unit.self_attention.sinks.is_some());
            assert_eq!(gpt_unit.mlp.router.weight.0, [4, 64]);
            assert_eq!(gpt_unit.mlp.experts.weight.0[0], 4 / expert as i32);
            assert_eq!(gpt_unit.mlp.experts.weight.0[1], 128 / tensor as i32);
            let spec = gpt_unit.mlp.experts.expert_spec.unwrap();
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
                panic!("GPT-OSS partition lost packed expert layout");
            };
            assert!(gate_up.bias().is_some());
            assert!(down.bias().is_some());
            assert_eq!(
                gate_up.format().encoding(),
                eredu_checkpoint::LinearFormat::MxFp4
            );
            assert_eq!(
                down.format().encoding(),
                eredu_checkpoint::LinearFormat::MxFp4
            );
        }
    }
}

#[test]
fn routed_partition_geometry_rejects_wrong_expert_owner_and_state() {
    let args = routed_qwen_partition_args();
    let global = qwen::RoutedLayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
    let description = global.parameter_description(&()).unwrap();
    let topology = eredu_core::ParallelTopology::new(1, 2, 2, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, 3).unwrap();
    let wrong_rank = eredu_core::ParallelRankTopology::new(topology, 2).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &description,
        rank,
    )
    .unwrap();
    let wrong_plan = qwen_realization(&args, &layout, wrong_rank);
    assert!(
        qwen::partition_local_routed_geometry(&args, &layout, 2..4, rank, &wrong_plan)
            .unwrap_err()
            .to_string()
            .contains("ownership")
    );

    let plan = qwen_realization(&args, &layout, rank);
    let geometry =
        qwen::partition_local_routed_geometry(&args, &layout, 2..4, rank, &plan).unwrap();
    let complete = geometry.complete_state_layout().clone();
    let valid_partition = ArchitecturePartition::from_description(
        &description,
        [(decoder::TEXT_DECODER_EXECUTION_GROUP, 2..4)],
        PartitionOwnership::new(false, true, ["norm", "output"]).unwrap(),
        &complete,
        &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..complete.len(),
        )]),
        geometry.clone(),
        NoAuxiliaryBoundarySchema::new(args.hidden_size),
    )
    .unwrap();
    let dense_parameters = qwen::dense_parameter_description(&args).unwrap();
    let task_error = qwen::PartitionedRoutedLayeredModel::<ReferenceBackend>::from_partition(
        args.clone(),
        &dense_parameters,
        &valid_partition,
        &(),
    );
    assert!(matches!(task_error, Err(error) if error.to_string().contains("omits")));

    let state_error = ArchitecturePartition::from_description(
        &description,
        [(decoder::TEXT_DECODER_EXECUTION_GROUP, 2..4)],
        PartitionOwnership::new(false, true, ["norm", "output"]).unwrap(),
        &complete,
        &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..1,
        )]),
        geometry,
        NoAuxiliaryBoundarySchema::new(args.hidden_size),
    );
    assert!(matches!(state_error, Err(error) if error.to_string().contains("state")));
}

fn local_llama_partition(
    args: &ModelArgs,
    tensor_parts: usize,
    tensor_rank: usize,
    units: std::ops::Range<usize>,
    ownership: PartitionOwnership,
) -> (
    eredu_runtime::LocalModelLayout,
    ArchitecturePartition<llama::PartitionLocalGeometry<ModelArgs>, NoAuxiliaryBoundarySchema>,
) {
    let description = llama::dense_parameter_description(args).unwrap();
    let topology = eredu_core::ParallelTopology::new(tensor_parts, 1, 1, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, tensor_rank).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &description,
        rank,
    )
    .unwrap();
    let geometry = llama::partition_local_geometry(args, &layout, units.clone()).unwrap();
    let state = geometry.complete_state_layout().clone();
    let state_plan =
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..state.len(),
        )]);
    let partition = ArchitecturePartition::from_description(
        &description,
        [(decoder::TEXT_DECODER_EXECUTION_GROUP, units)],
        ownership,
        &state,
        &state_plan,
        geometry,
        NoAuxiliaryBoundarySchema::new(args.hidden_size),
    )
    .unwrap();
    (layout, partition)
}

#[test]
fn llama_pipeline_local_model_allocates_only_owned_units_and_static_roles() {
    let mut args = tiny_args();
    args.num_hidden_layers = 4;
    args.attention_schedule = LayerSchedule::new(4, vec![AttentionPolicy::Full; 4]).unwrap();
    let input_ownership = PartitionOwnership::new(true, false, ["embedding"]).unwrap();
    let (layout, input_partition) = local_llama_partition(&args, 1, 0, 0..2, input_ownership);
    let description = llama::dense_parameter_description(&args).unwrap();
    let input = llama::PartitionedLayeredModel::<ReferenceBackend>::from_partition(
        args.clone(),
        &description,
        &input_partition,
        &(),
    )
    .unwrap();
    assert_eq!(input.local_geometry().owned_units(), 0..2);
    assert_eq!(input.local_geometry().local_unit_count(), 2);
    assert!(input.static_modules().embeddings.is_some());
    assert!(input.static_modules().norm.is_none());
    assert!(input.static_modules().lm_head.is_none());
    assert!(input.construct_unit(0, &()).is_ok());
    assert!(input.construct_unit(2, &()).is_err());

    let output_ownership = PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
    let (_, output_partition) = local_llama_partition(&args, 1, 0, 2..4, output_ownership);
    let mut output = llama::PartitionedLayeredModel::<ReferenceBackend>::from_partition(
        args.clone(),
        &description,
        &output_partition,
        &(),
    )
    .unwrap();
    assert_eq!(output.local_geometry().owned_units(), 2..4);
    assert!(output.static_modules().embeddings.is_some());
    assert!(output.static_modules().norm.is_some());
    assert!(output.construct_unit(1, &()).is_err());
    assert!(output.construct_unit(3, &()).is_ok());
    let local_state = output_partition.state().unwrap().layout().clone();
    let mut state = DeviceState::create(local_state.clone(), |_, policy| {
        Ok::<_, std::convert::Infallible>(ReferenceCache {
            offset: 0,
            window: policy
                .attention()
                .and_then(|attention| attention.window())
                .map(|window| window.get() as i32),
            resets: 0,
            fixed: None,
        })
    })
    .unwrap();
    let hidden = ReferenceTensor(vec![1, 3, 8]);
    <llama::PartitionedLayeredModel<ReferenceBackend> as eredu_runtime::PartitionedLayeredArchitecture<
        ReferenceBackend,
        DeviceState<ReferenceBackend, ReferenceCache>,
    >>::begin_partition(
        &mut output,
        eredu_runtime::LayeredPartitionInput::Hidden {
            hidden,
            auxiliary: eredu_runtime::NoAuxiliaryBoundary,
        },
        None,
        &mut state,
        &local_state,
        2,
        &(),
    )
    .unwrap();
    assert_eq!(
        layout
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .local_shape(),
        [8, 8]
    );
}

#[test]
fn llama_tp_pp_local_model_uses_rank_local_shapes_for_owned_stage_only() {
    let mut args = tiny_args();
    args.num_hidden_layers = 4;
    args.num_attention_heads = 4;
    args.num_key_value_heads = 2;
    args.head_dim = 2;
    args.attention_schedule = LayerSchedule::new(4, vec![AttentionPolicy::Full; 4]).unwrap();
    let ownership = PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
    let (layout, partition) = local_llama_partition(&args, 2, 1, 2..4, ownership);
    let description = llama::dense_parameter_description(&args).unwrap();
    let model = llama::PartitionedLayeredModel::<ReferenceBackend>::from_partition(
        args,
        &description,
        &partition,
        &(),
    )
    .unwrap();
    assert_eq!(model.local_geometry().local_unit_count(), 2);
    assert_eq!(
        layout
            .tensor("model.layers.2.self_attn.q_proj.weight")
            .unwrap()
            .local_shape(),
        [4, 8]
    );
    assert!(model.construct_unit(0, &()).is_err());
    let unit = model.construct_unit(2, &()).unwrap();
    assert_eq!(unit.self_attention.query_heads, 2);
    assert_eq!(unit.self_attention.key_value_heads, 1);
}

fn dense_lfm2_partition_args() -> lfm2::ModelArgs {
    lfm2::model_args_from_config_value(&serde_json::json!({
        "model_type":"lfm2", "vocab_size":8, "hidden_size":8,
        "intermediate_size":12, "num_hidden_layers":4,
        "num_attention_heads":4, "num_key_value_heads":2,
        "max_position_embeddings":64,
        "layer_types":["conv","full_attention","conv","full_attention"],
        "conv_L_cache":3, "block_auto_adjust_ff_dim":false,
        "tie_word_embeddings":false
    }))
    .unwrap()
}

fn local_lfm2_partition(
    args: &lfm2::ModelArgs,
    tensor_parts: usize,
    tensor_rank: usize,
    units: std::ops::Range<usize>,
    ownership: PartitionOwnership,
) -> (
    ArchitectureParameterDescription,
    LocalModelLayout,
    ArchitecturePartition<lfm2::PartitionLocalGeometry, NoAuxiliaryBoundarySchema>,
) {
    let complete = lfm2::LayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
    let description = complete.parameter_description(&()).unwrap();
    assert_eq!(
        lfm2::dense_parameter_description(args).unwrap(),
        description
    );
    let topology = eredu_core::ParallelTopology::new(tensor_parts, 1, 1, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, tensor_rank).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &description,
        rank,
    )
    .unwrap();
    let geometry = lfm2::partition_local_geometry(args, &layout, units.clone()).unwrap();
    let state = geometry.complete_state_layout().clone();
    let state_plan =
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..state.len(),
        )]);
    let partition = ArchitecturePartition::from_description(
        &description,
        [(decoder::TARGET_EXECUTION_GROUP, units)],
        ownership,
        &state,
        &state_plan,
        geometry,
        NoAuxiliaryBoundarySchema::new(args.hidden_size),
    )
    .unwrap();
    (description, layout, partition)
}

#[test]
fn dense_lfm2_tp_pp_model_owns_exact_units_static_roles_and_state_offset() {
    let args = dense_lfm2_partition_args();
    let input_ownership = PartitionOwnership::new(true, false, ["embedding"]).unwrap();
    let (input_description, _, input_partition) =
        local_lfm2_partition(&args, 2, 1, 0..2, input_ownership);
    let input = lfm2::PartitionedLayeredModel::<ReferenceBackend>::from_partition(
        args.clone(),
        &input_description,
        &input_partition,
        &(),
    )
    .unwrap();
    assert!(input.static_modules().embeddings.is_some());
    assert!(input.static_modules().norm.is_none());
    assert!(input.static_modules().lm_head.is_none());

    let ownership = PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
    let (description, layout, partition) = local_lfm2_partition(&args, 2, 1, 2..4, ownership);
    let model = lfm2::PartitionedLayeredModel::<ReferenceBackend>::from_partition(
        args,
        &description,
        &partition,
        &(),
    )
    .unwrap();

    assert_eq!(model.local_geometry().owned_units(), 2..4);
    assert_eq!(model.local_geometry().local_unit_count(), 2);
    assert_eq!(partition.state().unwrap().global_layer_offset(), 2);
    assert_eq!(partition.state().unwrap().layout().len(), 2);
    assert!(model.static_modules().embeddings.is_none());
    assert!(model.static_modules().norm.is_some());
    assert!(model.static_modules().lm_head.is_some());
    assert!(model.construct_unit(1, &()).is_err());
    assert!(model.construct_unit(2, &()).is_ok());
    assert!(model.construct_unit(3, &()).is_ok());
    assert_eq!(
        layout
            .tensor("model.layers.2.conv.conv.weight")
            .unwrap()
            .local_shape()[0],
        4
    );
    assert_eq!(
        layout
            .tensor("model.layers.3.self_attn.k_proj.weight")
            .unwrap()
            .local_shape()[0],
        2
    );
}

#[test]
fn dense_lfm2_partition_rejects_routed_and_invalid_ranges_before_model_construction() {
    let args = dense_lfm2_partition_args();
    let ownership = PartitionOwnership::new(true, false, ["embedding"]).unwrap();
    let (_, layout, _) = local_lfm2_partition(&args, 2, 0, 0..2, ownership);
    for range in [0..0, 3..5] {
        assert!(lfm2::partition_local_geometry(&args, &layout, range).is_err());
    }
    let mut malformed = layout.clone();
    malformed.insert(
        "model.layers.1.self_attn.q_proj.weight".into(),
        LocalTensorLayout::new(
            "model.layers.1.self_attn.heads",
            eredu_runtime::ParameterRole::AttentionHeads,
            vec![8, 8],
            vec![3, 8],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 3,
            },
            None,
            None,
            false,
        ),
    );
    let error = lfm2::partition_local_geometry(&args, &malformed, 0..2).unwrap_err();
    assert!(error.to_string().contains("split head dimension"));

    let routed = lfm2::model_args_from_config_value(&serde_json::json!({
        "model_type":"lfm2_moe", "vocab_size":8, "hidden_size":8,
        "intermediate_size":12, "num_hidden_layers":4,
        "num_attention_heads":4, "num_key_value_heads":2,
        "max_position_embeddings":64,
        "layer_types":["conv","full_attention","conv","full_attention"],
        "conv_L_cache":3, "block_auto_adjust_ff_dim":false,
        "num_dense_layers":1, "moe_intermediate_size":8,
        "num_experts":2, "num_experts_per_tok":1,
        "tie_word_embeddings":false
    }))
    .unwrap();
    let error = lfm2::partition_local_geometry(&routed, &layout, 0..2).unwrap_err();
    assert!(error.to_string().contains("does not accept routed"));
}

#[derive(Debug)]
struct DirectLlamaAdmission;

impl eredu_architectures::partitioned_execution::PartitionedAdmissionDispatcher
    for DirectLlamaAdmission
{
    type Output = eredu_architectures::partitioned_execution::DirectPartitionedAdmission;
    type Error = String;

    fn direct(
        self,
        requirements: eredu_architectures::partitioned_execution::DirectPartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        Ok(requirements)
    }

    fn routed(
        self,
        _requirements: eredu_architectures::partitioned_execution::RoutedPartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        Err("expected direct Llama admission".into())
    }

    fn composite(
        self,
        _requirements: eredu_architectures::partitioned_execution::CompositePartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        Err("expected direct Llama admission".into())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct LlamaPartitionSummary {
    units: std::ops::Range<usize>,
    global_state_offset: usize,
    owns_input: bool,
    owns_output: bool,
    embedding: bool,
    norm: bool,
    source_architecture: bool,
    task_names: Vec<String>,
    state_policies: Vec<eredu_core::cache::LayerCachePolicy>,
}

struct InspectLlamaPartition;

impl
    eredu_architectures::partitioned_execution::PartitionedArchitectureVisitor<
        ReferenceBackend,
        DeviceState<ReferenceBackend, ReferenceCache>,
    > for InspectLlamaPartition
{
    type Output = LlamaPartitionSummary;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
            ReferenceBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
            >>::Boundary,
        >,
        _store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
            > + eredu_runtime::ReplicatedTextArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                Error = eredu_nn::Error,
            > + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        let summary = std::rc::Rc::new(RefCell::new(None));
        let captured = std::rc::Rc::clone(&summary);
        let _binding = prepared
            .prepare_session_runtime(
                eredu_core::cache::PromptCacheTopology::default(),
                &(),
                move |input, source, _layout, _selected, _context| {
                    let (_architecture, partition, _communication, tasks) = input.into_parts();
                    let task_names = tasks
                        .iter()
                        .map(|task| task.name().to_owned())
                        .collect::<Vec<_>>();
                    let state = partition.state();
                    *captured.borrow_mut() = Some(LlamaPartitionSummary {
                        units: partition.groups()[0].global_units(),
                        global_state_offset: state
                            .map(eredu_runtime::PartitionState::global_layer_offset)
                            .unwrap_or(0),
                        owns_input: partition.ownership().owns_input(),
                        owns_output: partition.ownership().owns_output(),
                        embedding: task_names
                            .iter()
                            .any(|name| name.contains("embed_tokens.weight")),
                        norm: task_names.iter().any(|name| {
                            name == "model.norm.weight" || name == "model.embedding_norm.weight"
                        }),
                        source_architecture: source.is_some(),
                        task_names,
                        state_policies: state
                            .into_iter()
                            .flat_map(|state| state.layout().layers().iter())
                            .cloned()
                            .collect(),
                    });
                    let state = match partition.state() {
                        Some(state) => DeviceState::create(state.layout().clone(), |_, policy| {
                            Ok::<_, std::convert::Infallible>(ReferenceCache {
                                offset: 0,
                                window: policy
                                    .attention()
                                    .and_then(|attention| attention.window())
                                    .map(|window| window.get() as i32),
                                resets: 0,
                                fixed: None,
                            })
                        })
                        .unwrap(),
                        None => DeviceState::stateless(),
                    };
                    Ok::<_, Error>(((), state))
                },
            )
            .map_err(|error| Error::backend(error.to_string()))?;
        let summary = summary
            .borrow_mut()
            .take()
            .ok_or_else(|| Error::backend("partitioned session factory did not run"));
        summary
    }
}

fn resident_selection_for(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
) -> eredu_runtime::SelectedReplicatedTextRealization {
    selection_for(
        inspection,
        eredu_runtime::LayerWeightResidency::FullyResident,
        None,
    )
}

fn selection_for(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    residency: eredu_runtime::LayerWeightResidency,
    quantization: Option<eredu_core::QuantizationRequest>,
) -> eredu_runtime::SelectedReplicatedTextRealization {
    let requirements = replicated_text_requirements(inspection).unwrap();
    let affine = eredu_checkpoint::LinearFormat::Affine(
        eredu_checkpoint::AffineQuantization::new(16, 4).unwrap(),
    );
    let weight_lowerings = requirements
        .parameters()
        .iter()
        .flat_map(|parameter| {
            let mut capabilities = Vec::new();
            if parameter.has_lowering_source() {
                capabilities.push(eredu_runtime::WeightLoweringCapability::new(
                    parameter
                        .lowering_descriptor(parameter.native_executable())
                        .unwrap(),
                    eredu_runtime::WeightLoweringKind::Direct,
                ));
                if let Ok(descriptor) = parameter.lowering_descriptor(affine) {
                    capabilities.push(eredu_runtime::WeightLoweringCapability::new(
                        descriptor.clone(),
                        eredu_runtime::WeightLoweringKind::Transform,
                    ));
                    capabilities.push(eredu_runtime::WeightLoweringCapability::new(
                        descriptor,
                        eredu_runtime::WeightLoweringKind::DerivedTransform,
                    ));
                }
            }
            capabilities
        })
        .collect();
    let state = eredu_runtime::StateMechanismCapabilities::new(
        (0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .unwrap()
                .iter()
                .cloned()
                .map(move |component| {
                    eredu_runtime::StateComponentMechanism::new(
                        layer,
                        component,
                        Some(eredu_runtime::StateComponentPlacement::Device),
                        None,
                    )
                })
        }),
    )
    .with_transactions(true, true)
    .with_reset(true);
    let backend = eredu_runtime::BackendMechanismCapabilities::new(
        eredu_nn::NeuralOperatorCapabilities::ALL,
        weight_lowerings,
        vec![
            eredu_runtime::WeightResidencyMechanism::Resident,
            eredu_runtime::WeightResidencyMechanism::Windowed,
            eredu_runtime::WeightResidencyMechanism::DiskStreamed,
        ],
        state,
    );
    let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        residency,
        eredu_runtime::CacheResidencyPolicy::Device,
    );
    let request = match quantization {
        Some(quantization) => request.with_quantization(quantization),
        None => request,
    };
    eredu_runtime::select_replicated_text_realization(&requirements, &request, &backend).unwrap()
}

fn resident_partition_communication() -> eredu_runtime::CommunicationCapabilities {
    let limits = eredu_runtime::CommunicationTensorLimits::new(8, 8, 1 << 20, None).unwrap();
    eredu_runtime::CommunicationCapabilities::new([
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllReduceSum,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllGatherUneven,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::SendReceive,
            [
                eredu_core::checkpoint::TensorDtype::F32,
                eredu_core::checkpoint::TensorDtype::I32,
            ],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::Broadcast,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::failure_agreement(true),
    ])
    .unwrap()
    .with_boundary_framing([eredu_runtime::BoundaryFramingProtocol::RoleExactV1])
    .unwrap()
    .with_completion_capabilities(
        eredu_runtime::CommunicationCompletionCapabilities::new([
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap(),
    )
}

#[test]
fn authoritative_lfm2_visitor_selects_indexed_tp_pp_partition_and_mixed_state() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let config = serde_json::json!({
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "layer_types": ["conv", "full_attention", "conv", "full_attention"],
        "conv_L_cache": 3,
        "block_auto_adjust_ff_dim": false,
        "tie_word_embeddings": false
    });
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let description = lfm2::dense_parameter_description(&args).unwrap();
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tensors = description
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .map(|member| {
            let bytes = vec![0_u8; member.global_shape().iter().product::<usize>() * 4];
            (
                member.target().to_owned(),
                member.global_shape().to_vec(),
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        views,
        None,
        &artifact.path().join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        artifact.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    let inspection = configuration::inspect_artifact(artifact.path()).unwrap();
    let selected_base = resident_selection_for(&inspection);
    let communication = resident_partition_communication();
    let topology = eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap();
    let store: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
    );

    for rank in 0..topology.world_size() {
        let admission = eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                topology,
                rank,
                1,
                8,
                eredu_runtime::PipelineActivationDtype::Float32,
            )
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            ),
            DirectLlamaAdmission,
        )
        .unwrap();
        let selected =
            eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
                admission,
                selected_base.clone(),
                &communication,
            )
            .unwrap();
        assert_eq!(
            eredu_architectures::partitioned_execution::dense_decoder_partitioned_production_route(
                &inspection,
                &selected,
            ),
            eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
        );
        let coordinates = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
        let summary =
            eredu_architectures::partitioned_execution::visit_resident_partitioned_architecture::<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                _,
            >(
                &inspection,
                selected,
                std::sync::Arc::clone(&store),
                &(),
                InspectLlamaPartition,
            )
            .unwrap();
        let expected = if coordinates.pipeline_parallel_rank() == 0 {
            0..2
        } else {
            2..4
        };
        assert_eq!(summary.units, expected.clone());
        assert_eq!(summary.global_state_offset, expected.start);
        assert_eq!(summary.owns_input, expected.start == 0);
        assert_eq!(summary.owns_output, expected.end == 4);
        assert_eq!(summary.embedding, expected.start == 0);
        assert_eq!(summary.norm, expected.end == 4);
        assert_eq!(summary.state_policies.len(), 2);
        assert!(matches!(
            summary.state_policies[0],
            eredu_core::cache::LayerCachePolicy::FixedState { .. }
        ));
        assert!(matches!(
            summary.state_policies[1],
            eredu_core::cache::LayerCachePolicy::KeyValue { .. }
        ));
        assert!(summary.task_names.iter().all(|name| {
            expected
                .clone()
                .any(|layer| name.contains(&format!("model.layers.{layer}.")))
                || !name.contains("model.layers.")
        }));
    }
}

#[test]
fn authoritative_kimi_visitor_selects_indexed_tp_pp_partition_and_exact_mixed_state() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let config = serde_json::json!({
        "model_type":"kimi_linear", "vocab_size":16, "hidden_size":8,
        "num_hidden_layers":4, "num_attention_heads":4, "num_key_value_heads":2,
        "intermediate_size":12, "head_dim":2, "model_max_length":64,
        "linear_attn_config":{
            "kda_layers":[1,3], "full_attn_layers":[2,4], "num_heads":4,
            "head_dim":2, "short_conv_kernel_size":3
        },
        "num_experts":2, "moe_intermediate_size":6, "kv_lora_rank":4,
        "qk_nope_head_dim":2, "qk_rope_head_dim":2, "v_head_dim":2,
        "mla_use_nope":true, "num_experts_per_token":1, "num_shared_experts":1,
        "routed_scaling_factor":1.0, "first_k_dense_replace":4,
        "num_expert_group":1, "topk_group":1, "tie_word_embeddings":false,
        "num_nextn_predict_layers":0
    });
    let args = kimi_linear::model_args_from_config_value(&config).unwrap();
    let architecture = kimi_linear::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let description = architecture.parameter_description(&()).unwrap();
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tensors = description
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .map(|member| {
            let bytes = vec![0_u8; member.global_shape().iter().product::<usize>() * 4];
            (
                member.target().to_owned(),
                member.global_shape().to_vec(),
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        views,
        None,
        &artifact.path().join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        artifact.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    let inspection = configuration::inspect_artifact(artifact.path()).unwrap();
    let selected_base = resident_selection_for(&inspection);
    let communication = resident_partition_communication();
    let topology = eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap();
    let store: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
    );

    for rank in 0..topology.world_size() {
        let admission = eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                topology,
                rank,
                1,
                8,
                eredu_runtime::PipelineActivationDtype::Float32,
            )
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            ),
            DirectLlamaAdmission,
        )
        .unwrap();
        let selected =
            eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
                admission,
                selected_base.clone(),
                &communication,
            )
            .unwrap();
        assert_eq!(
            eredu_architectures::partitioned_execution::dense_decoder_partitioned_production_route(
                &inspection,
                &selected,
            ),
            eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
        );
        let coordinates = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
        let summary =
            eredu_architectures::partitioned_execution::visit_resident_partitioned_architecture::<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                _,
            >(
                &inspection,
                selected,
                std::sync::Arc::clone(&store),
                &(),
                InspectLlamaPartition,
            )
            .unwrap();
        let expected = if coordinates.pipeline_parallel_rank() == 0 {
            0..2
        } else {
            2..4
        };
        assert_eq!(summary.units, expected.clone());
        assert_eq!(summary.global_state_offset, expected.start);
        assert_eq!(summary.owns_input, expected.start == 0);
        assert_eq!(summary.owns_output, expected.end == 4);
        assert_eq!(summary.embedding, expected.start == 0);
        assert_eq!(summary.norm, expected.end == 4);
        assert_eq!(summary.state_policies.len(), 2);
        let eredu_core::cache::LayerCachePolicy::FixedState { tensors } =
            &summary.state_policies[0]
        else {
            panic!("KDA state must retain its fixed recurrent and convolution tensors")
        };
        assert_eq!(tensors.len(), 4);
        assert!(tensors[..3]
            .iter()
            .all(|tensor| { tensor.dtype == eredu_core::cache::StateTensorDtype::Floating }));
        assert_eq!(
            tensors[3].dtype,
            eredu_core::cache::StateTensorDtype::Float32
        );
        assert!(matches!(
            summary.state_policies[1],
            eredu_core::cache::LayerCachePolicy::CompressedLatentRotary { .. }
        ));
        assert!(summary.task_names.iter().all(|name| {
            expected
                .clone()
                .any(|layer| name.contains(&format!("model.layers.{layer}.")))
                || !name.contains("model.layers.")
        }));
    }

    std::fs::copy(
        artifact.path().join("model-00001-of-00001.safetensors"),
        artifact.path().join("model.safetensors"),
    )
    .unwrap();
    std::fs::remove_file(artifact.path().join("model.safetensors.index.json")).unwrap();
    let unindexed = configuration::inspect_artifact(artifact.path()).unwrap();
    let admission = eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
        &unindexed,
        eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
            topology,
            0,
            1,
            8,
            eredu_runtime::PipelineActivationDtype::Float32,
        )
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        ),
        DirectLlamaAdmission,
    )
    .unwrap();
    let selected = eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
        admission,
        resident_selection_for(&unindexed),
        &communication,
    )
    .unwrap();
    assert_eq!(
        eredu_architectures::partitioned_execution::dense_decoder_partitioned_production_route(
            &unindexed,
            &selected,
        ),
        eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
    );
}

#[test]
fn authoritative_qwen_visitor_selects_tp_pp_partition_and_exact_mixed_state() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let config = serde_json::json!({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "use_sliding_window": true,
        "sliding_window": 32,
        "max_window_layers": 2
    });
    let args = eredu_architectures::qwen::model_args_from_config_value(&config).unwrap();
    let description = eredu_architectures::qwen::dense_parameter_description(&args).unwrap();
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tensors = description
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .map(|member| {
            let bytes = vec![0_u8; member.global_shape().iter().product::<usize>() * 4];
            (
                member.target().to_owned(),
                member.global_shape().to_vec(),
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        views,
        None,
        &artifact.path().join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        artifact.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    let inspection = configuration::inspect_artifact(artifact.path()).unwrap();
    let requirements = replicated_text_requirements(&inspection).unwrap();
    let weight_lowerings = requirements
        .parameters()
        .iter()
        .filter(|parameter| parameter.has_lowering_source())
        .map(|parameter| {
            eredu_runtime::WeightLoweringCapability::new(
                parameter
                    .lowering_descriptor(parameter.native_executable())
                    .unwrap(),
                eredu_runtime::WeightLoweringKind::Direct,
            )
        })
        .collect();
    let state = eredu_runtime::StateMechanismCapabilities::new(
        (0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .unwrap()
                .iter()
                .cloned()
                .map(move |component| {
                    eredu_runtime::StateComponentMechanism::new(
                        layer,
                        component,
                        Some(eredu_runtime::StateComponentPlacement::Device),
                        None,
                    )
                })
        }),
    )
    .with_transactions(true, true)
    .with_reset(true);
    let backend = eredu_runtime::BackendMechanismCapabilities::new(
        eredu_nn::NeuralOperatorCapabilities::ALL,
        weight_lowerings,
        vec![eredu_runtime::WeightResidencyMechanism::Resident],
        state,
    );
    let selected = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        ),
        &backend,
    )
    .unwrap();
    let limits = eredu_runtime::CommunicationTensorLimits::new(8, 8, 1 << 20, None).unwrap();
    let communication = eredu_runtime::CommunicationCapabilities::new([
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllReduceSum,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllGatherUneven,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::SendReceive,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::Broadcast,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::failure_agreement(true),
    ])
    .unwrap()
    .with_boundary_framing([eredu_runtime::BoundaryFramingProtocol::RoleExactV1])
    .unwrap()
    .with_completion_capabilities(
        eredu_runtime::CommunicationCompletionCapabilities::new([
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap(),
    );
    let topology = eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap();
    let missing_failure_agreement = eredu_runtime::CommunicationCapabilities::new([
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllReduceSum,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllGatherUneven,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::SendReceive,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
        eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::Broadcast,
            [eredu_core::checkpoint::TensorDtype::F32],
            limits,
            true,
        )
        .unwrap(),
    ])
    .unwrap()
    .with_boundary_framing([eredu_runtime::BoundaryFramingProtocol::RoleExactV1])
    .unwrap()
    .with_completion_capabilities(
        eredu_runtime::CommunicationCompletionCapabilities::new([
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap(),
    );
    let missing_admission =
        eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                topology,
                0,
                1,
                8,
                eredu_runtime::PipelineActivationDtype::Float32,
            )
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            ),
            DirectLlamaAdmission,
        )
        .unwrap();
    let error = eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
        missing_admission,
        selected.clone(),
        &missing_failure_agreement,
    )
    .expect_err("PP admission must require explicit failure agreement capability");
    assert!(error.to_string().contains("FailureAgreement"));
    let store: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
    );
    let residencies = [
        eredu_runtime::LayerWeightResidency::FullyResident,
        eredu_runtime::LayerWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::LayerWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1).unwrap(),
        ),
    ];
    for admitted_topology in [
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap(),
        topology,
    ] {
        for residency in residencies {
            for quantization in [
                None,
                Some(eredu_core::QuantizationRequest::Affine {
                    group_size: 16,
                    bits: 4,
                }),
            ] {
                let selected_base = selection_for(&inspection, residency, quantization);
                let admission = eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
                    &inspection,
                    eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                        admitted_topology,
                        0,
                        1,
                        8,
                        eredu_runtime::PipelineActivationDtype::Float32,
                    )
                    .unwrap()
                    .with_completion_policy(
                        eredu_runtime::CommunicationCompletionPolicy::new(
                            std::time::Duration::from_secs(1),
                            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                        )
                        .unwrap(),
                    ),
                    DirectLlamaAdmission,
                )
                .unwrap();
                let selected = eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
                    admission,
                    selected_base,
                    &communication,
                )
                .unwrap();
                assert_eq!(
                    eredu_architectures::partitioned_execution::dense_decoder_partitioned_production_route(
                        &inspection,
                        &selected,
                    ),
                    eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
                    "{admitted_topology:?} {residency:?} {quantization:?}",
                );
            }
        }
    }

    let transformed = selection_for(
        &inspection,
        eredu_runtime::LayerWeightResidency::FullyResident,
        Some(eredu_core::QuantizationRequest::Affine {
            group_size: 16,
            bits: 4,
        }),
    );
    let transformed_admission =
        eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                topology,
                2,
                1,
                8,
                eredu_runtime::PipelineActivationDtype::Float32,
            )
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            ),
            DirectLlamaAdmission,
        )
        .unwrap();
    let transformed =
        eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
            transformed_admission,
            transformed,
            &communication,
        )
        .unwrap();
    let transformed_summary =
        eredu_architectures::partitioned_execution::visit_resident_partitioned_architecture::<
            ReferenceBackend,
            DeviceState<ReferenceBackend, ReferenceCache>,
            _,
        >(
            &inspection,
            transformed,
            std::sync::Arc::clone(&store),
            &(),
            InspectLlamaPartition,
        )
        .unwrap();
    assert!(transformed_summary.source_architecture);
    assert_eq!(transformed_summary.units, 2..4);
    assert!(transformed_summary
        .task_names
        .iter()
        .any(|name| name.contains("layers.2") || name.contains("layers.3")));
    for rank in 0..topology.world_size() {
        let admission = eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                topology,
                rank,
                1,
                8,
                eredu_runtime::PipelineActivationDtype::Float32,
            )
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            ),
            DirectLlamaAdmission,
        )
        .unwrap();
        let selected =
            eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
                admission,
                selected.clone(),
                &communication,
            )
            .unwrap();
        assert_eq!(
            eredu_architectures::partitioned_execution::dense_decoder_partitioned_production_route(
                &inspection,
                &selected,
            ),
            eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
        );
        let coordinates = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
        let summary =
            eredu_architectures::partitioned_execution::visit_resident_partitioned_architecture::<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                _,
            >(
                &inspection,
                selected,
                std::sync::Arc::clone(&store),
                &(),
                InspectLlamaPartition,
            )
            .unwrap();
        let expected_units = if coordinates.pipeline_parallel_rank() == 0 {
            0..2
        } else {
            2..4
        };
        assert_eq!(summary.units, expected_units);
        assert_eq!(
            summary.owns_input,
            coordinates.pipeline_parallel_rank() == 0
        );
        assert_eq!(
            summary.owns_output,
            coordinates.pipeline_parallel_rank() == 1
        );
        assert!(summary.embedding);
        assert_eq!(summary.norm, summary.owns_output);
        let expected_window = (coordinates.pipeline_parallel_rank() == 1).then_some(32);
        assert!(summary.state_policies.iter().all(|policy| {
            matches!(
                policy,
                eredu_core::cache::LayerCachePolicy::KeyValue {
                    attention,
                    num_key_value_heads,
                    ..
                } if attention.window().map(|window| window.get()) == expected_window
                    && num_key_value_heads.get() == 1
            )
        }));
        assert!(summary.task_names.iter().all(|name| {
            !name.contains("layers.0") && !name.contains("layers.1")
                || coordinates.pipeline_parallel_rank() == 0
        }));
        assert!(summary.task_names.iter().all(|name| {
            !name.contains("layers.2") && !name.contains("layers.3")
                || coordinates.pipeline_parallel_rank() == 1
        }));
    }

    std::fs::copy(
        artifact.path().join("model-00001-of-00001.safetensors"),
        artifact.path().join("model.safetensors"),
    )
    .unwrap();
    std::fs::remove_file(artifact.path().join("model.safetensors.index.json")).unwrap();
    let unindexed = configuration::inspect_artifact(artifact.path()).unwrap();
    let admission = eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
        &unindexed,
        eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            0,
            1,
            8,
            eredu_runtime::PipelineActivationDtype::Float32,
        )
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        ),
        DirectLlamaAdmission,
    )
    .unwrap();
    let selected = eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
        admission,
        resident_selection_for(&unindexed),
        &communication,
    )
    .unwrap();
    assert_eq!(
        eredu_architectures::partitioned_execution::dense_decoder_partitioned_production_route(
            &unindexed,
            &selected,
        ),
        eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
    );
}

#[test]
fn neutral_llama_parallel_geometry_drives_local_gqa_and_cache_shape() {
    let mut args = tiny_args();
    args.num_attention_heads = 4;
    args.num_key_value_heads = 2;
    args.head_dim = 2;
    let layout = llama_parallel_layout(&args, 0..16, 2, 1);
    let geometry = llama::local_geometry(&args, &layout).unwrap();

    assert!(geometry
        .blocks()
        .iter()
        .all(|block| { block.num_attention_heads == 2 && block.num_key_value_heads == 1 }));
    for layer in 0..args.num_hidden_layers as usize {
        match geometry.state_layout().layer(layer).unwrap() {
            eredu_core::cache::LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            } => {
                assert_eq!(num_key_value_heads.get(), 1);
                assert_eq!(head_dim.get(), 2);
            }
            policy => panic!("unexpected local Llama cache policy {policy:?}"),
        }
    }

    let architecture =
        llama::LayeredModel::<ReferenceBackend>::new_parallel(args, geometry, &()).unwrap();
    let block = <llama::LayeredModel<ReferenceBackend> as LayeredArchitecture<
        ReferenceBackend,
        DeviceState<ReferenceBackend, ReferenceCache>,
    >>::build_unit(&architecture, 0, 0, &())
    .unwrap();
    assert_eq!(block.self_attention.query_heads, 2);
    assert_eq!(block.self_attention.key_value_heads, 1);
}

#[test]
fn neutral_llama_tensor_parallel_size_one_matches_replicated_lifecycle() {
    let args = tiny_args();
    let state = || {
        DeviceState::<ReferenceBackend, ReferenceCache>::create(
            llama::state_layout(&args).unwrap(),
            |_, policy| {
                Ok::<_, std::convert::Infallible>(ReferenceCache {
                    offset: 0,
                    window: policy
                        .attention()
                        .and_then(|attention| attention.window())
                        .map(|window| window.get() as i32),
                    resets: 0,
                    fixed: None,
                })
            },
        )
        .unwrap()
    };

    let replicated = llama::LayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
    let units = (0..args.num_hidden_layers as usize)
        .map(|index| {
            <llama::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
            >>::build_unit(&replicated, 0, index, &())
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut replicated = LayerwiseRuntime::new(replicated, ResidentUnitWindow::new(units));
    let mut replicated_state = state();

    let layout = llama_parallel_layout(
        &args,
        0..args.vocab_size as usize,
        args.num_attention_heads,
        args.num_key_value_heads,
    );
    let geometry = llama::local_geometry(&args, &layout).unwrap();
    let parallel =
        llama::LayeredModel::<ReferenceBackend>::new_parallel(args.clone(), geometry, &()).unwrap();
    let units = (0..parallel.args().num_hidden_layers as usize)
        .map(|index| {
            <llama::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
            >>::build_unit(&parallel, 0, index, &())
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut parallel = LayerwiseRuntime::new(parallel, ResidentUnitWindow::new(units));
    let mut parallel_state = state();
    let tokens = ReferenceTensor(vec![1, 3]);

    let replicated_logits = replicated
        .forward(
            LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut replicated_state,
            &(),
        )
        .unwrap();
    let parallel_logits = parallel
        .forward_parallel(
            LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut parallel_state,
            &(),
            &(),
        )
        .unwrap();

    assert_eq!(parallel_logits, replicated_logits);
    assert_eq!(parallel_state.layer(0).unwrap().offset(), 3);
    assert_eq!(parallel_state.layer(1).unwrap().offset(), 3);
}

#[test]
fn neutral_llama_local_geometry_rejects_bad_vocabulary_companion_selection() {
    let mut args = tiny_args();
    args.vocab_size = 7;
    let mut layout = llama_parallel_layout(&args, 4..7, 2, 1);
    layout.insert(
        "model.embed_tokens.scales".into(),
        LocalTensorLayout::new(
            "model.embed_tokens",
            eredu_runtime::ParameterRole::Vocabulary,
            vec![7, 1],
            vec![3, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 3,
            },
            None,
            Some(0..3),
            false,
        ),
    );
    let error = llama::local_geometry(&args, &layout).unwrap_err();
    assert!(error
        .to_string()
        .contains("inconsistent companion selections"));

    let mut malformed = llama_parallel_layout(&args, 4..7, 2, 1);
    malformed.insert(
        "model.embed_tokens.weight".into(),
        LocalTensorLayout::new(
            "model.embed_tokens",
            eredu_runtime::ParameterRole::Vocabulary,
            vec![7, 8],
            vec![7, 3],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: 3,
            },
            None,
            Some(0..3),
            false,
        ),
    );
    let error = llama::local_geometry(&args, &malformed).unwrap_err();
    assert!(error.to_string().contains("non-row placement"));
}

struct ProjectionLayoutConfig {
    args: ModelArgs,
    fused: bool,
    empty_field: bool,
    alternate_fields: bool,
}

impl decoder::Config for ProjectionLayoutConfig {
    fn model_family(&self) -> &'static str {
        "projection_layout_fixture"
    }

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn architecture_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "reference_projection_layout_decoder",
            [
                (
                    "base",
                    decoder::Config::architecture_fingerprint(&self.args),
                ),
                ("fused", self.fused.to_string()),
                ("empty_field", self.empty_field.to_string()),
                ("alternate_fields", self.alternate_fields.to_string()),
                ("attention_bias", "false".into()),
                ("mlp_bias", "false".into()),
                ("weight_quantization", "dense".into()),
            ],
        )
    }

    fn validate_config(&self) -> Result<(), Error> {
        Ok(())
    }

    fn block_parameter_fields(&self) -> decoder::BlockParameterFields<'_> {
        if self.alternate_fields {
            decoder::BlockParameterFields {
                attention_output: "out_proj",
                feed_forward: "gating",
                feed_forward_output: "linear_out",
                input_norm: "norm1",
                post_attention_norm: "norm2",
                ..decoder::BlockParameterFields::default()
            }
        } else {
            decoder::BlockParameterFields::default()
        }
    }

    fn hidden_size(&self) -> i32 {
        self.args.hidden_size
    }

    fn num_hidden_layers(&self) -> i32 {
        self.args.num_hidden_layers
    }

    fn intermediate_size(&self) -> i32 {
        self.args.intermediate_size
    }

    fn num_attention_heads(&self) -> i32 {
        self.args.num_attention_heads
    }

    fn num_key_value_heads(&self) -> i32 {
        self.args.num_key_value_heads
    }

    fn head_dim(&self) -> i32 {
        self.args.head_dim
    }

    fn rms_norm_epsilon(&self) -> f32 {
        self.args.rms_norm_eps
    }

    fn vocabulary_size(&self) -> i32 {
        self.args.vocab_size
    }

    fn attention_bias(&self, _: decoder::AttentionProjection) -> bool {
        false
    }

    fn attention_projection_layout(&self) -> AttentionProjectionLayout<'_> {
        if self.fused {
            AttentionProjectionLayout::Fused {
                field: if self.empty_field { "" } else { "in_proj" },
            }
        } else {
            AttentionProjectionLayout::Split
        }
    }

    fn mlp_bias(&self) -> bool {
        false
    }

    fn gated_projection_layout(&self) -> GatedProjectionLayout<'_> {
        if self.fused {
            GatedProjectionLayout::Fused {
                field: if self.empty_field {
                    ""
                } else if self.alternate_fields {
                    "linear_in"
                } else {
                    "gate_up_proj"
                },
            }
        } else {
            GatedProjectionLayout::Split
        }
    }

    fn tie_word_embeddings(&self) -> bool {
        self.args.tie_word_embeddings
    }

    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.args.attention_schedule
    }

    fn weight_quantization(&self, _: &str) -> Option<eredu_checkpoint::WeightQuantization> {
        None
    }

    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        RotarySpec {
            dimensions,
            base: self.args.rope_theta,
            traditional: self.args.rope_traditional,
            algorithm: eredu_nn::RotaryAlgorithm::Default,
        }
    }
}

#[test]
fn tied_vocabulary_parallel_projection_preserves_embedding_storage_and_global_logits() {
    let spec = EmbeddingSpec {
        vocabulary: 7,
        dimensions: 4,
        weight: eredu_nn::ParameterSpec::trainable("model.embed_tokens.weight").unwrap(),
        format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)
            .unwrap(),
    };
    let mut embedding = ReferenceBackend::vocabulary_parallel_embedding(
        spec,
        VocabularyParallelRange {
            global_vocabulary: 7,
            local: 0..4,
        },
        &(),
    )
    .unwrap();
    assert_eq!(embedding.weight.0, [4, 4]);

    let hidden = ReferenceTensor(vec![1, 2, 4]);
    let logits =
        ReferenceBackend::vocabulary_parallel_embedding_project(&mut embedding, &hidden, &(), &())
            .unwrap();

    assert_eq!(embedding.weight.0, [4, 4]);
    assert_eq!(logits.0, [1, 2, 7]);
}

#[test]
fn fused_and_split_decoder_blocks_publish_equivalent_projection_geometry() {
    let split = TransformerBlock::<ReferenceBackend>::new(
        &ProjectionLayoutConfig {
            args: tiny_args(),
            fused: false,
            empty_field: false,
            alternate_fields: false,
        },
        0,
        &(),
    )
    .unwrap();
    let fused = TransformerBlock::<ReferenceBackend>::new(
        &ProjectionLayoutConfig {
            args: tiny_args(),
            fused: true,
            empty_field: false,
            alternate_fields: false,
        },
        0,
        &(),
    )
    .unwrap();
    let split = topology(&split);
    let fused = topology(&fused);
    let elements = |topology: &[(String, Vec<usize>)]| {
        topology
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>()
    };
    assert_eq!(elements(&split), elements(&fused));
    assert!(split
        .iter()
        .any(|(name, shape)| name.ends_with("self_attn.q_proj.weight") && shape == &[8, 8]));
    assert!(split
        .iter()
        .any(|(name, shape)| name.ends_with("self_attn.k_proj.weight") && shape == &[4, 8]));
    assert!(split
        .iter()
        .any(|(name, shape)| name.ends_with("mlp.gate_proj.weight") && shape == &[16, 8]));
    assert!(fused
        .iter()
        .any(|(name, shape)| name.ends_with("self_attn.in_proj.weight") && shape == &[16, 8]));
    assert!(fused
        .iter()
        .any(|(name, shape)| name.ends_with("mlp.gate_up_proj.weight") && shape == &[32, 8]));
    assert!(!fused.iter().any(|(name, _)| {
        name.ends_with("q_proj.weight")
            || name.ends_with("k_proj.weight")
            || name.ends_with("gate_proj.weight")
    }));
}

#[test]
fn alternate_block_fields_drive_parameter_topology_and_parallel_groups() {
    let config = ProjectionLayoutConfig {
        args: tiny_args(),
        fused: true,
        empty_field: false,
        alternate_fields: true,
    };
    let block = TransformerBlock::<ReferenceBackend>::new(&config, 0, &()).unwrap();
    let expected = std::collections::BTreeSet::from([
        "model.layers.0.gating.linear_in.weight".to_string(),
        "model.layers.0.gating.linear_out.weight".to_string(),
        "model.layers.0.norm1.weight".to_string(),
        "model.layers.0.norm2.weight".to_string(),
        "model.layers.0.self_attn.in_proj.weight".to_string(),
        "model.layers.0.self_attn.out_proj.weight".to_string(),
    ]);
    let topology_names = topology(&block)
        .into_iter()
        .map(|(name, _)| name)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(topology_names, expected);

    let groups = decoder::layer_parallel_parameter_groups(&block, &config, 0).unwrap();
    let group_names = groups
        .iter()
        .map(|group| group.logical_name())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        group_names,
        std::collections::BTreeSet::from([
            "model.layers.0.gating.projections",
            "model.layers.0.norm1",
            "model.layers.0.norm2",
            "model.layers.0.self_attn.projections",
        ])
    );
    let member_names = groups
        .iter()
        .flat_map(|group| group.members())
        .map(|member| member.target().to_string())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(member_names, expected);
}

fn tiny_moshi_config() -> moshi::MoshiConfig {
    moshi::MoshiConfig::from_json(
        r#"{
            "model_type": "moshi",
            "dim": 32,
            "text_card": 101,
            "n_q": 4,
            "dep_q": 3,
            "generated_audio_codebooks": 2,
            "card": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 48,
            "causal": true,
            "context": 7,
            "max_period": 10000.0,
            "positional_embedding": "rope",
            "depformer_dim": 24,
            "depformer_dim_feedforward": 36,
            "depformer_num_heads": 4,
            "depformer_num_layers": 2,
            "depformer_context": 3,
            "depformer_max_period": 10000.0,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 2, 1]
        }"#,
    )
    .unwrap()
}

fn minimal_moshi_config() -> moshi::MoshiConfig {
    moshi::MoshiConfig::from_json(
        r#"{
            "model_type": "moshi",
            "dim": 32,
            "text_card": 101,
            "n_q": 2,
            "dep_q": 1,
            "generated_audio_codebooks": 1,
            "card": 64,
            "num_heads": 4,
            "num_layers": 1,
            "dim_feedforward": 48,
            "causal": true,
            "context": 7,
            "max_period": 10000.0,
            "positional_embedding": "rope",
            "depformer_dim": 32,
            "depformer_dim_feedforward": 48,
            "depformer_num_heads": 4,
            "depformer_num_layers": 1,
            "depformer_context": 3,
            "depformer_max_period": 10000.0,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1]
        }"#,
    )
    .unwrap()
}

type ReferenceMoshiState = DeviceState<ReferenceBackend, ReferenceCache>;

fn reference_moshi_state(config: &moshi::MoshiConfig) -> ReferenceMoshiState {
    let layout = moshi::state_layout(config).unwrap();
    DeviceState::create(layout, |_, policy| {
        Ok::<_, std::convert::Infallible>(ReferenceCache {
            offset: 0,
            window: policy
                .attention()
                .and_then(|attention| attention.window())
                .map(|window| window.get() as i32),
            resets: 0,
            fixed: None,
        })
    })
    .unwrap()
}

fn reference_decision_driver(
    retain_diagnostics: bool,
) -> SequentialDecisionDriver<ReferenceBackend, ReferenceSampler> {
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Sample,
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        retain_diagnostics,
        true,
    )
    .unwrap();
    SequentialDecisionDriver::new(plan, vec![ReferenceSampler; 4], vec![0.0; 4], Some(0)).unwrap()
}

fn execute_reference_moshi_frame(
    config: &moshi::MoshiConfig,
    directives: Vec<PredictionDirective<ReferenceTensor>>,
    retain_diagnostics: bool,
    allow_tail_skip: bool,
    random: Option<i32>,
) -> (
    ReferenceTensor,
    moshi::ForwardContext<ReferenceTensor>,
    SequentialDecisionDriver<ReferenceBackend, ReferenceSampler>,
    ReferenceMoshiState,
    ReferenceTrace,
) {
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let decision_count = architecture.decision_count();
    assert_eq!(directives.len(), decision_count);
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let mut state = reference_moshi_state(config);
    let plan =
        SequentialDecisionPlan::new(directives, retain_diagnostics, allow_tail_skip).unwrap();
    let mut driver = SequentialDecisionDriver::new(
        plan,
        vec![ReferenceSampler; decision_count],
        vec![0.0; decision_count],
        random,
    )
    .unwrap();
    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = (0..config.frame_schedule().total_audio_codebooks())
        .map(|_| ReferenceTensor(vec![1, 1]))
        .collect::<Vec<_>>();
    let audio = audio_values.iter().collect::<Vec<_>>();
    clear_reference_trace();
    let mut boundary = moshi::DecisionBoundary::new(config).unwrap();
    let (logits, forward) = {
        let mut hook = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap()
    };
    driver.finish().unwrap();
    (logits, forward, driver, state, reference_trace())
}

fn replicated_moshi_parallel_layout(config: &moshi::MoshiConfig) -> LocalModelLayout {
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut groups = architecture.static_parameter_groups().unwrap();
    for group in 0..2 {
        let count = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
            ReferenceBackend,
            ReferenceMoshiState,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            let unit = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                ReferenceBackend,
                ReferenceMoshiState,
            >>::build_unit(&architecture, group, index, &())
            .unwrap();
            groups.extend(moshi::unit_parameter_groups(&unit, config, group, index).unwrap());
        }
    }
    let mut layout = LocalModelLayout::default();
    for group in groups {
        let logical_range = group.partition_units().map(|units| 0..units);
        for member in group.members() {
            layout.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    member.global_shape().to_vec(),
                    TensorPlacement::Replicated,
                    group.partition_units(),
                    logical_range.clone(),
                    false,
                ),
            );
        }
    }
    layout
}

fn constructed_moshi_parameter_groups(config: &moshi::MoshiConfig) -> Vec<ParameterGroupSpec> {
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut groups = architecture.static_parameter_groups().unwrap();
    for group in 0..2 {
        let count = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
            ReferenceBackend,
            ReferenceMoshiState,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            let unit = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                ReferenceBackend,
                ReferenceMoshiState,
            >>::build_unit(&architecture, group, index, &())
            .unwrap();
            groups.extend(moshi::unit_parameter_groups(&unit, config, group, index).unwrap());
        }
    }
    groups
}

#[test]
fn moshi_symbolic_parameter_topology_matches_constructed_dense_modules_and_affine_companions() {
    let dense = minimal_moshi_config();
    let symbolic = moshi::parameter_description(&dense)
        .unwrap()
        .groups()
        .iter()
        .map(|group| group.group().clone())
        .collect::<Vec<_>>();
    assert_eq!(symbolic, constructed_moshi_parameter_groups(&dense));

    let affine = dense
        .with_native_quantization(Some(eredu_checkpoint::WeightQuantization::Affine(
            eredu_checkpoint::AffineQuantization::new(16, 4).unwrap(),
        )))
        .unwrap();
    let affine = moshi::parameter_description(&affine).unwrap();
    for primary in [
        "text_emb.weight",
        "transformer.layers.0.self_attn.in_proj.weight",
        "transformer.layers.0.gating.linear_in.weight",
        "depformer.slices.0.linear_in.weight",
        "depformer.slices.0.transformer.layers.0.self_attn.out_proj.weight",
        "depformer.slices.0.linear_out.weight",
    ] {
        let group = affine
            .groups()
            .iter()
            .find(|group| {
                group
                    .members()
                    .iter()
                    .any(|member| member.target() == primary)
            })
            .unwrap_or_else(|| panic!("missing affine Moshi primary {primary}"));
        let prefix = primary.strip_suffix(".weight").unwrap();
        for (suffix, role) in [
            ("scales", eredu_nn::LinearCompanionRole::Scale),
            ("biases", eredu_nn::LinearCompanionRole::AffineBias),
        ] {
            let companion = group
                .members()
                .iter()
                .find(|member| member.target() == format!("{prefix}.{suffix}"))
                .unwrap_or_else(|| panic!("missing affine Moshi {suffix} for {primary}"));
            assert_eq!(companion.linear_companion(), Some(role));
            assert_eq!(companion.linear_companion_of(), Some(primary));
        }
    }
}

#[test]
fn one_portable_moshi_model_runs_replicated_and_parallel_lifecycles() {
    let config = minimal_moshi_config();
    let replicated = execute_reference_moshi_frame(
        &config,
        vec![
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
        None,
    );

    let layout = replicated_moshi_parallel_layout(&config);
    let geometry = moshi::local_geometry(&config, &layout, std::iter::empty()).unwrap();
    let architecture =
        moshi::LayeredModel::<ReferenceBackend>::new_parallel(config.clone(), geometry, &())
            .unwrap();
    let mut units = Vec::new();
    for group in 0..2 {
        let count = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
            ReferenceBackend,
            ReferenceMoshiState,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            units.push(
                <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                    ReferenceBackend,
                    ReferenceMoshiState,
                >>::build_unit(&architecture, group, index, &())
                .unwrap(),
            );
        }
    }
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut state = reference_moshi_state(&config);
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
    )
    .unwrap();
    let mut driver =
        SequentialDecisionDriver::new(plan, vec![ReferenceSampler; 2], vec![0.0; 2], None).unwrap();
    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = (0..config.frame_schedule().total_audio_codebooks())
        .map(|_| ReferenceTensor(vec![1, 1]))
        .collect::<Vec<_>>();
    let audio = audio_values.iter().collect::<Vec<_>>();
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    let (parallel_logits, _) = {
        let mut traversal = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_parallel_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &(),
                &mut traversal,
            )
            .unwrap()
    };
    driver.finish().unwrap();

    assert_eq!(parallel_logits, replicated.0);
    assert_eq!(driver.decisions(), replicated.2.decisions());
    assert_eq!(
        state.layout(),
        &runtime.architecture().state_layout().unwrap()
    );
}

#[test]
fn moshi_parallel_geometry_rejects_policy_compatible_cross_config_reuse() {
    let config = minimal_moshi_config();
    let layout = replicated_moshi_parallel_layout(&config);
    let geometry = moshi::local_geometry(&config, &layout, std::iter::empty()).unwrap();
    let changed = moshi::MoshiConfig::from_json(
        r#"{
            "model_type": "moshi",
            "dim": 32,
            "text_card": 101,
            "n_q": 2,
            "dep_q": 1,
            "generated_audio_codebooks": 1,
            "card": 64,
            "num_heads": 4,
            "num_layers": 1,
            "dim_feedforward": 48,
            "causal": true,
            "context": 7,
            "max_period": 20000.0,
            "positional_embedding": "rope",
            "depformer_dim": 24,
            "depformer_dim_feedforward": 36,
            "depformer_num_heads": 4,
            "depformer_num_layers": 1,
            "depformer_context": 3,
            "depformer_max_period": 10000.0,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1]
        }"#,
    )
    .unwrap();
    match moshi::LayeredModel::<ReferenceBackend>::new_parallel(changed, geometry, &()) {
        Err(error) => assert!(
            error
                .to_string()
                .contains("different normalized configuration"),
            "unexpected Moshi geometry error: {error}"
        ),
        Ok(_) => panic!("Moshi accepted geometry derived from a different configuration"),
    }
}

#[test]
fn tiny_moshi_executes_one_temporal_block_and_one_depth_slice_with_exact_logit_geometry() {
    let config = minimal_moshi_config();
    let forced_text = ReferenceTensor(vec![1, 1]);
    let forced_audio = ReferenceTensor(vec![1, 1]);
    let (text_logits, forward, driver, state, trace) = execute_reference_moshi_frame(
        &config,
        vec![
            PredictionDirective::Force(forced_text),
            PredictionDirective::Force(forced_audio.clone()),
        ],
        true,
        true,
        None,
    );

    assert_eq!(driver.plan().mode(), SequentialDecisionMode::TeacherForced);
    assert_eq!(text_logits, ReferenceTensor(vec![1, 1, 101]));
    assert_eq!(
        driver
            .diagnostics()
            .iter()
            .map(|diagnostic| (diagnostic.prediction(), diagnostic.logits().clone()))
            .collect::<Vec<_>>(),
        [
            (0, ReferenceTensor(vec![1, 1, 101])),
            (1, ReferenceTensor(vec![1, 1, 64])),
        ]
    );
    assert_eq!(forward.previous_depth_token(), Some(&forced_audio));
    assert_eq!(state.as_ref()[0].offset, 1);
    assert_eq!(state.as_ref()[1].offset, 1);
    assert_eq!(state.as_ref()[0].resets, 0);
    assert_eq!(state.as_ref()[1].resets, 1);
    assert!(trace
        .linear_outputs
        .contains(&("text_linear.weight".into(), vec![1, 1, 101])));
    assert!(trace.linear_outputs.contains(&(
        "depformer.slices.0.linear_out.weight".into(),
        vec![1, 1, 64]
    )));
}

#[test]
fn moshi_parallel_selection_owns_local_geometry_and_opaque_group_projection() {
    let config = tiny_moshi_config();
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let parameters = architecture.parameter_description(&()).unwrap();
    let topology = eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, 1).unwrap();
    let completion = eredu_runtime::CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let selected = moshi::select_parallel_execution(
        &config,
        &parameters,
        rank,
        2,
        7,
        eredu_runtime::PipelineActivationDtype::Float32,
        completion,
        std::iter::empty(),
    )
    .unwrap();

    assert_eq!(selected.geometry().temporal()[0].attention_heads(), 2);
    assert_eq!(selected.communication().groups().len(), 1);
    assert_eq!(
        selected.communication().groups()[0].id(),
        selected.tensor_group()
    );
    assert_eq!(selected.communication().groups()[0].members(), [0, 1]);
    assert!(selected
        .layout()
        .tensor("transformer.layers.0.self_attn.in_proj.weight")
        .is_some());
    assert_eq!(selected.execution_plan().drivers().len(), 2);
    assert!(selected
        .execution_plan()
        .drivers()
        .iter()
        .all(Option::is_some));
    assert!(selected.execution_plan().routes().is_empty());
    assert!(selected.execution_plan().publication().is_none());

    let pipeline = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap(),
        0,
    )
    .unwrap();
    assert!(moshi::select_parallel_execution(
        &config,
        &parameters,
        pipeline,
        2,
        7,
        eredu_runtime::PipelineActivationDtype::Float32,
        eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
        std::iter::empty(),
    )
    .is_err());
}

#[test]
fn moshi_frame_decisions_cover_greedy_seeded_partial_forced_and_tail_skip() {
    let config = tiny_moshi_config();
    let count = config.frame_schedule().depth_audio_codebooks() + 1;
    let sampled = vec![PredictionDirective::Sample; count];

    let (_, _, greedy, _, _) =
        execute_reference_moshi_frame(&config, sampled.clone(), true, true, None);
    assert_eq!(greedy.plan().mode(), SequentialDecisionMode::Autoregressive);
    assert!(greedy.random_state().is_none());
    assert!(greedy
        .decisions()
        .iter()
        .all(|decision| decision.source() == SequentialDecisionSource::Sampled));

    let (_, _, seeded, _, _) =
        execute_reference_moshi_frame(&config, sampled, true, true, Some(17));
    assert_eq!(seeded.random_state(), Some(&21));
    assert_eq!(seeded.diagnostics().len(), count);

    let partial = vec![
        PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        PredictionDirective::Sample,
        PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        PredictionDirective::Sample,
    ];
    let (_, _, partial, _, _) =
        execute_reference_moshi_frame(&config, partial, true, true, Some(4));
    assert_eq!(
        partial.plan().mode(),
        SequentialDecisionMode::PartiallyForced
    );
    assert_eq!(partial.random_state(), Some(&6));
    assert_eq!(
        partial
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Sampled,
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Sampled,
        ]
    );

    let forced = (0..count)
        .map(|_| PredictionDirective::Force(ReferenceTensor(vec![1, 1])))
        .collect::<Vec<_>>();
    let (_, _, forced, state, _) =
        execute_reference_moshi_frame(&config, forced, false, true, None);
    assert_eq!(forced.plan().mode(), SequentialDecisionMode::TeacherForced);
    assert_eq!(
        forced
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
        ]
    );
    assert_eq!(state.as_ref()[2].offset, 0);
    assert_eq!(state.as_ref()[3].offset, 0);
    assert_eq!(state.as_ref()[2].resets, 1);
    assert_eq!(state.as_ref()[3].resets, 1);
}

#[test]
fn moshi_depth_codebooks_receive_the_immediately_preceding_decision() {
    let config = tiny_moshi_config();
    let symbolic_tokens = [
        ReferenceTensor(vec![1, 2]),
        ReferenceTensor(vec![1, 3]),
        ReferenceTensor(vec![1, 4]),
        ReferenceTensor(vec![1, 5]),
    ];
    let directives = symbolic_tokens
        .iter()
        .cloned()
        .map(PredictionDirective::Force)
        .collect::<Vec<_>>();
    let (_, forward, _, _, trace) =
        execute_reference_moshi_frame(&config, directives, true, false, None);
    let depth_lookups = trace
        .embedding_lookups
        .iter()
        .filter(|(name, _)| name.starts_with("depformer.slices."))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        depth_lookups,
        [
            ("depformer.slices.0.emb.weight".into(), vec![1, 2]),
            ("depformer.slices.1.emb.weight".into(), vec![1, 3]),
            ("depformer.slices.2.emb.weight".into(), vec![1, 4]),
        ]
    );
    assert_eq!(forward.previous_depth_token(), Some(&symbolic_tokens[3]));
}

#[test]
fn moshi_depth_resets_once_per_frame_and_temporal_rope_uses_absolute_offset() {
    let config = minimal_moshi_config();
    assert_eq!(config.temporal().context(), 7);
    assert_eq!(config.temporal().attention_window(), 8);
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let mut state = reference_moshi_state(&config);
    assert_eq!(state.as_ref()[0].window, Some(8));
    assert_eq!(state.as_ref()[1].window, Some(3));

    let prefill_text = ReferenceTensor(vec![1, 3]);
    let prefill_audio_values = [ReferenceTensor(vec![1, 3]), ReferenceTensor(vec![1, 3])];
    let prefill_audio = prefill_audio_values.iter().collect::<Vec<_>>();
    let prefill_plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
    )
    .unwrap();
    let mut prefill_driver =
        SequentialDecisionDriver::new(prefill_plan, vec![ReferenceSampler; 2], vec![0.0; 2], None)
            .unwrap();
    clear_reference_trace();
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    {
        let mut hook = SequentialDecisionTraversal::new(&mut prefill_driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &prefill_text,
                    audio: &prefill_audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap();
    }
    prefill_driver.finish().unwrap();
    let prefill_trace = reference_trace();
    assert_eq!(prefill_trace.rotary_offsets, [0, 0]);
    assert_eq!(prefill_trace.sliding_attention, [(8, 0), (3, 0)]);
    assert_eq!(prefill_trace.causal_masks, [(3, 0, None), (3, 0, None)]);
    assert_eq!(state.as_ref()[0].offset, 3);
    assert_eq!(state.as_ref()[1].offset, 3);
    assert_eq!(state.as_ref()[0].resets, 0);
    assert_eq!(state.as_ref()[1].resets, 1);

    let decode_text = ReferenceTensor(vec![1, 1]);
    let decode_audio_values = [ReferenceTensor(vec![1, 1]), ReferenceTensor(vec![1, 1])];
    let decode_audio = decode_audio_values.iter().collect::<Vec<_>>();
    let decode_plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
    )
    .unwrap();
    let mut decode_driver =
        SequentialDecisionDriver::new(decode_plan, vec![ReferenceSampler; 2], vec![0.0; 2], None)
            .unwrap();
    clear_reference_trace();
    {
        let mut hook = SequentialDecisionTraversal::new(&mut decode_driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &decode_text,
                    audio: &decode_audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap();
    }
    decode_driver.finish().unwrap();
    let decode_trace = reference_trace();
    assert_eq!(decode_trace.rotary_offsets, [3, 3]);
    assert!(decode_trace.sliding_attention.is_empty());
    assert!(decode_trace.causal_masks.is_empty());
    assert_eq!(state.as_ref()[0].offset, 4);
    assert_eq!(state.as_ref()[1].offset, 1);
    assert_eq!(state.as_ref()[0].resets, 0);
    assert_eq!(state.as_ref()[1].resets, 2);
}

#[test]
fn moshi_rejects_mismatched_cache_layout_and_depth_block_drift() {
    let config = minimal_moshi_config();
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = [ReferenceTensor(vec![1, 1]), ReferenceTensor(vec![1, 1])];
    let audio = audio_values.iter().collect::<Vec<_>>();
    let malformed_audio_values = [ReferenceTensor(vec![1, 1]), ReferenceTensor(vec![1, 2])];
    let malformed_audio = malformed_audio_values.iter().collect::<Vec<_>>();
    let mut valid_state = reference_moshi_state(&config);
    let error = runtime
        .forward(
            moshi::Input {
                text: &text,
                audio: &malformed_audio,
                mask: None,
            },
            &mut valid_state,
            &(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("token shape"));
    assert!(valid_state.as_ref().iter().all(|cache| cache.resets == 0));

    let mut wrong_state = reference_moshi_state(&tiny_moshi_config());
    let error = runtime
        .forward(
            moshi::Input {
                text: &text,
                audio: &audio,
                mask: None,
            },
            &mut wrong_state,
            &(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("state layout mismatch"));
    assert!(wrong_state.as_ref().iter().all(|cache| cache.resets == 0));

    let moshi::Unit::Depth(depth) = &mut runtime.units_mut()[1][0] else {
        panic!("depth group owns depth units");
    };
    depth.blocks.pop();
    let mut state = reference_moshi_state(&config);
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        false,
    )
    .unwrap();
    let mut driver =
        SequentialDecisionDriver::new(plan, vec![ReferenceSampler; 2], vec![0.0; 2], None).unwrap();
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    let error = {
        let mut hook = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .err()
            .expect("drifted depth block count must fail")
    };
    assert!(error.to_string().contains("depth block count drifted"));
}

#[test]
fn portable_moshi_topology_state_and_decision_order_are_backend_independent() {
    let config = tiny_moshi_config();
    let layout = moshi::state_layout(&config).unwrap();
    assert_eq!(layout.layers().len(), 4);
    assert_eq!(layout.segments().len(), 2);
    assert_eq!(layout.segments()[0].id().as_str(), "temporal");
    assert_eq!(layout.segments()[0].layers(), 0..2);
    assert_eq!(layout.segments()[1].id().as_str(), "depth");
    assert_eq!(layout.segments()[1].layers(), 2..4);

    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    assert_eq!(runtime.units().len(), 2);
    assert_eq!(runtime.units()[0].len(), 2);
    assert_eq!(runtime.units()[1].len(), 3);
    assert!(runtime.units()[0].iter().all(|unit| match unit {
        moshi::Unit::Temporal(block) => block.self_attention.rotary.is_some(),
        moshi::Unit::Depth(_) => false,
    }));
    assert!(runtime.units()[1].iter().all(|unit| {
        match unit {
            moshi::Unit::Depth(slice) => slice
                .blocks
                .iter()
                .all(|block| block.self_attention.rotary.is_none()),
            moshi::Unit::Temporal(_) => false,
        }
    }));
    let graph = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
        ReferenceBackend,
        ReferenceMoshiState,
    >>::execution_graph(runtime.architecture())
    .unwrap();
    assert_eq!(graph.execution_order(), &[0, 1]);
    assert_eq!(
        graph.groups()[1].dependencies(),
        &["temporal_transformer".to_string()]
    );
    let retained = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
        ReferenceBackend,
        ReferenceMoshiState,
    >>::retained_state_ordinals(runtime.architecture(), 1, 2, 4);
    assert_eq!(retained, 2..4);

    let mut names = topology(runtime.architecture().static_modules());
    for group in runtime.units() {
        for unit in group {
            names.extend(topology(unit));
        }
    }
    let names = names
        .into_iter()
        .map(|(name, _)| name)
        .collect::<std::collections::BTreeSet<_>>();
    for expected in [
        "text_emb.weight",
        "audio_embs.3.weight",
        "out_norm.weight",
        "text_linear.weight",
        "transformer.layers.0.norm1.weight",
        "transformer.layers.0.self_attn.in_proj.weight",
        "transformer.layers.0.self_attn.out_proj.weight",
        "transformer.layers.0.gating.linear_in.weight",
        "transformer.layers.0.gating.linear_out.weight",
        "depformer.slices.2.emb.weight",
        "depformer.slices.2.linear_in.weight",
        "depformer.slices.2.linear_out.weight",
        "depformer.slices.2.transformer.layers.1.norm2.weight",
    ] {
        assert!(names.contains(expected), "missing {expected}");
    }
    assert!(!names.iter().any(|name| {
        name.contains("input_layernorm")
            || name.contains("post_attention_layernorm")
            || name.contains(".mlp.")
            || name.contains(".o_proj.")
    }));

    let static_groups = runtime.architecture().static_parameter_groups().unwrap();
    assert_eq!(static_groups.len(), 7);
    let depth_groups = moshi::unit_parameter_groups(&runtime.units()[1][0], &config, 1, 0).unwrap();
    assert!(depth_groups
        .iter()
        .any(|group| group.logical_name() == "depformer.slices.0.linear_in"));
    assert!(depth_groups
        .iter()
        .any(|group| group.logical_name() == "depformer.slices.0.transformer.layers.1.norm2"));
    for name in ["depformer.slices.0.emb", "depformer.slices.0.linear_out"] {
        let group = depth_groups
            .iter()
            .find(|group| group.logical_name() == name)
            .unwrap();
        assert_eq!(group.role(), eredu_runtime::ParameterRole::Vocabulary);
        assert!(group.members().iter().all(|member| matches!(
            member.sharding(),
            eredu_runtime::MemberSharding::Balanced { axis: 0 }
        )));
    }
    let linear_in = depth_groups
        .iter()
        .find(|group| group.logical_name() == "depformer.slices.0.linear_in")
        .unwrap();
    assert_eq!(linear_in.role(), eredu_runtime::ParameterRole::Replicated);
    assert!(linear_in
        .members()
        .iter()
        .all(|member| member.sharding() == &eredu_runtime::MemberSharding::Replicated));

    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = [
        ReferenceTensor(vec![1, 1]),
        ReferenceTensor(vec![1, 1]),
        ReferenceTensor(vec![1, 1]),
        ReferenceTensor(vec![1, 1]),
    ];
    let audio = audio_values.iter().collect::<Vec<_>>();
    let mut state = reference_moshi_state(&config);
    state.as_mut()[2].offset = 9;
    state.as_mut()[3].offset = 9;
    let error = runtime
        .forward(
            moshi::Input {
                text: &text,
                audio: &audio[..3],
                mask: None,
            },
            &mut state,
            &(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("3 audio codebooks, expected 4"));
    assert_eq!(state.as_ref()[2].offset, 9);
    assert_eq!(state.as_ref()[3].offset, 9);

    let mut driver = reference_decision_driver(true);
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    let (_, forward) = {
        let mut hook = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap()
    };
    driver.finish().unwrap();
    assert_eq!(driver.diagnostics().len(), 4);
    assert_eq!(driver.random_state(), Some(&1));
    assert_eq!(
        driver
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Sampled,
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Forced,
        ]
    );
    assert_eq!(
        forward.previous_depth_token(),
        Some(&ReferenceTensor(vec![1, 1]))
    );
    assert_eq!(state.as_ref()[2].offset, 3);
    assert_eq!(state.as_ref()[3].offset, 3);
    assert_eq!(state.as_ref()[2].resets, 1);
    assert_eq!(state.as_ref()[3].resets, 1);

    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let mut tail_driver = reference_decision_driver(false);
    let (_, forward) = {
        let mut hook = SequentialDecisionTraversal::new(&mut tail_driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap()
    };
    tail_driver.finish().unwrap();
    assert_eq!(
        tail_driver
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Sampled,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
        ]
    );
    assert_eq!(
        forward.previous_depth_token(),
        Some(&ReferenceTensor(vec![1, 1]))
    );
    assert_eq!(state.as_ref()[2].offset, 0);
    assert_eq!(state.as_ref()[3].offset, 0);
    assert_eq!(state.as_ref()[2].resets, 2);
    assert_eq!(state.as_ref()[3].resets, 2);
    assert_eq!(state.as_ref()[0].offset, 2);
    assert_eq!(state.as_ref()[1].offset, 2);
}

#[test]
fn fused_decoder_projection_fields_must_be_named() {
    let error = match TransformerBlock::<ReferenceBackend>::new(
        &ProjectionLayoutConfig {
            args: tiny_args(),
            fused: true,
            empty_field: true,
            alternate_fields: false,
        },
        0,
        &(),
    ) {
        Ok(_) => panic!("empty fused projection fields must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("field must not be empty"));
}

fn topology<M: Parameterized<ReferenceTensor>>(module: &M) -> Vec<(String, Vec<usize>)> {
    struct Collector(Vec<(String, Vec<usize>)>);
    impl<'a> ParameterVisitor<'a, ReferenceTensor> for Collector {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a ReferenceTensor) {
            self.0.push((
                metadata.id.to_string(),
                value
                    .shape()
                    .iter()
                    .map(|dimension| *dimension as usize)
                    .collect(),
            ));
        }
    }
    let mut collector = Collector(Vec::new());
    module.visit_parameters(&mut collector);
    collector.0
}

fn load_module<M: Parameterized<ReferenceTensor>>(
    module: &mut M,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) {
    let bindings = topology(module)
        .into_iter()
        .map(|(name, shape)| {
            let bytes = shape.iter().product::<usize>() as u64 * 4;
            WeightBinding::new(
                name.clone(),
                name,
                eredu_checkpoint::store::TensorSelection::Full,
                bytes,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let unit = materialize_bindings::<ReferenceBackend>(source, &bindings, &()).unwrap();
    bind_materialized_unit::<ReferenceBackend, _>(module, unit).unwrap();
}

#[test]
fn shared_llama_runs_prefill_and_decode_without_mlx() {
    let args = tiny_args();
    let layout = llama::state_layout(&args).unwrap();
    let mut state = DeviceState::<ReferenceBackend, ReferenceCache>::create(layout, |_, policy| {
        Ok::<_, std::convert::Infallible>(ReferenceCache {
            offset: 0,
            window: policy
                .attention()
                .and_then(|attention| attention.window())
                .map(|window| window.get() as i32),
            resets: 0,
            fixed: None,
        })
    })
    .unwrap();
    let architecture = llama::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let mut runtime = ResidentRuntime::new(architecture, &()).unwrap();

    let mut catalog = topology(runtime.architecture().static_modules());
    for unit in runtime.units() {
        catalog.extend(topology(unit));
    }
    let storage = catalog
        .into_iter()
        .map(|(name, shape)| {
            let bytes = vec![0; shape.iter().product::<usize>() * 4];
            (name, shape, bytes)
        })
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = directory.path().join("model.safetensors");
    safetensors::tensor::serialize_to_file(
        storage.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        }),
        None,
        &checkpoint,
    )
    .unwrap();
    let source = eredu_checkpoint::store::SafetensorsWeightStore::open(&checkpoint).unwrap();
    load_module(runtime.architecture_mut().static_modules_mut(), &source);
    for unit in runtime.units_mut() {
        load_module(unit, &source);
    }

    let prefill = ReferenceTensor(vec![1, 3]);
    let logits = runtime
        .forward(
            LayeredInput {
                tokens: &prefill,
                mask: None,
            },
            &mut state,
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 3, 32]);
    assert_eq!(state.layer(0).unwrap().offset(), 3);
    assert_eq!(state.layer(1).unwrap().offset(), 3);

    let decode = ReferenceTensor(vec![1, 1]);
    let logits = runtime
        .forward(
            LayeredInput {
                tokens: &decode,
                mask: None,
            },
            &mut state,
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 1, 32]);
    assert_eq!(state.layer(0).unwrap().offset(), 4);
    assert_eq!(state.layer(1).unwrap().offset(), 4);

    let (architecture, units) = runtime.into_parts();
    let mut layerwise = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let logits = layerwise
        .forward(
            LayeredInput {
                tokens: &decode,
                mask: None,
            },
            &mut state,
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 1, 32]);
    assert_eq!(state.layer(0).unwrap().offset(), 5);
    assert_eq!(state.layer(1).unwrap().offset(), 5);
}

#[test]
fn shared_decoder_runs_qwen2_and_qwen3_without_mlx() {
    for model_type in ["qwen2", "qwen3", "qwen3_moe"] {
        let mut config = serde_json::json!({
            "model_type": model_type,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false
        });
        if model_type == "qwen3_moe" {
            config["intermediate_size"] = 0.into();
            config["moe_intermediate_size"] = 8.into();
            config["num_experts"] = 4.into();
            config["num_experts_per_tok"] = 2.into();
            config["norm_topk_prob"] = true.into();
        } else if model_type == "qwen2" {
            config["use_sliding_window"] = true.into();
            config["sliding_window"] = 4.into();
            config["max_window_layers"] = 1.into();
        }
        let args = qwen::model_args_from_config_value(&config).unwrap();
        let layout = qwen::state_layout(&args).unwrap();
        let mut state =
            DeviceState::<ReferenceBackend, ReferenceCache>::create(layout, |_, policy| {
                Ok::<_, std::convert::Infallible>(ReferenceCache {
                    offset: 0,
                    window: policy
                        .attention()
                        .and_then(|attention| attention.window())
                        .map(|window| window.get() as i32),
                    resets: 0,
                    fixed: None,
                })
            })
            .unwrap();
        let architecture = qwen::RoutedLayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
        let mut runtime = ResidentRuntime::new(architecture, &()).unwrap();
        let parameters = runtime
            .units()
            .iter()
            .flat_map(topology)
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        if model_type == "qwen2" {
            assert!(!parameters
                .iter()
                .any(|name| name.ends_with("q_norm.weight")));
        } else {
            assert!(parameters
                .iter()
                .any(|name| name.ends_with("q_norm.weight")));
        }
        if model_type == "qwen3_moe" {
            assert!(parameters
                .iter()
                .any(|name| name.ends_with("mlp.gate.weight")));
            assert!(parameters
                .iter()
                .any(|name| name.ends_with("mlp.experts.gate_up_proj")));
        }
        let logits = runtime
            .forward(
                LayeredInput {
                    tokens: &ReferenceTensor(vec![1, 3]),
                    mask: None,
                },
                &mut state,
                &(),
            )
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);
        assert_eq!(state.layer(0).unwrap().offset(), 3);
        assert_eq!(state.layer(1).unwrap().offset(), 3);
        if model_type == "qwen2" {
            assert_eq!(state.layer(0).unwrap().max_size(), None);
            assert_eq!(state.layer(1).unwrap().max_size(), Some(4));
        }
        let logits = runtime
            .forward(
                LayeredInput {
                    tokens: &ReferenceTensor(vec![1, 1]),
                    mask: None,
                },
                &mut state,
                &(),
            )
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(state.layer(0).unwrap().offset(), 4);
        assert_eq!(state.layer(1).unwrap().offset(), 4);
    }
}

#[derive(Default)]
struct ProbeExpertProvider {
    calls: Vec<(usize, ExpertPass, Vec<i32>, Vec<i32>)>,
}

impl RoutedExpertProvider<ReferenceBackend> for ProbeExpertProvider {
    type Error = std::convert::Infallible;

    fn forward_grouped(
        &mut self,
        _resident_bank: &mut ReferenceLinear,
        request: RoutedExpertRequest<'_, ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Self::Error> {
        self.calls.push((
            request.layer,
            request.pass,
            request.input.shape().to_vec(),
            request.routes.group_indices().shape().to_vec(),
        ));
        Ok(request.input.clone())
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut ReferenceLinear,
        request: RoutedExpertRequest<'_, ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Self::Error> {
        self.calls.push((
            request.layer,
            request.pass,
            request.input.shape().to_vec(),
            request.routes.group_indices().shape().to_vec(),
        ));
        Ok(request.input.clone())
    }
}

#[derive(Default)]
struct ProbeObserver {
    route_shapes: Vec<(Vec<i32>, Vec<i32>, i32)>,
}

impl eredu_runtime::ActivationObserver<ReferenceTensor, Error> for ProbeObserver {
    fn observe(&mut self, _: &str, _: &ReferenceTensor) -> Result<(), Error> {
        Ok(())
    }

    fn observe_routing(
        &mut self,
        selection: eredu_runtime::RoutingObservation<'_, ReferenceTensor>,
    ) -> Result<(), Error> {
        self.route_shapes.push((
            selection.selected_experts.shape().to_vec(),
            selection.coefficients.shape().to_vec(),
            selection.expert_count,
        ));
        Ok(())
    }
}

#[test]
fn qwen_routed_execution_uses_the_runtime_provider_and_observer_contract() {
    let config = serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 0,
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "tie_word_embeddings": false
    });
    let args = qwen::model_args_from_config_value(&config).unwrap();
    let mut policy = qwen::FeedForward::<ReferenceBackend>::new(&args, 0, &()).unwrap();
    let input = ReferenceTensor(vec![3, 8]);
    let mut provider = ProbeExpertProvider::default();
    let output = policy
        .forward_with_provider(0, ExpertPass::Prefill, &input, &(), &mut provider)
        .unwrap();
    assert_eq!(output.shape(), input.shape());
    assert_eq!(
        provider.calls,
        vec![(0, ExpertPass::Prefill, vec![3, 8], vec![3, 2])]
    );

    let mut observer = ProbeObserver::default();
    let point = args.routed_observation_point("model.layers.0", 0).unwrap();
    let mut observed_provider =
        eredu_runtime::ObservedExpertProvider::new(&mut provider, &mut observer, point);
    let output = policy
        .forward_with_provider(0, ExpertPass::Decode, &input, &(), &mut observed_provider)
        .unwrap();
    drop(observed_provider);
    assert_eq!(output.shape(), input.shape());
    assert_eq!(observer.route_shapes, vec![(vec![3, 2], vec![3, 2], 4)]);
    assert_eq!(provider.calls[1].1, ExpertPass::Decode);
}

#[test]
fn released_moshi_profiles_share_one_portable_model_contract() {
    let native = moshi::MoshiConfig::native_v0_1().unwrap();
    let persona =
        moshi::MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
    for config in [&native, &persona] {
        assert_eq!(config.family(), "moshi");
        assert_eq!(config.temporal().parameter_root(), "transformer");
        assert_eq!(
            config.temporal().attention_window(),
            config.temporal().context() + 1
        );
        assert_eq!(
            config.depth_template().attention_window(),
            config.depth_template().context()
        );
        assert_eq!(
            config.depth_transformer(0).unwrap().parameter_root(),
            "depformer.slices.0.transformer"
        );
        let layout = moshi::state_layout(config).unwrap();
        assert_eq!(layout.segments()[0].id().as_str(), "temporal");
        assert_eq!(layout.segments()[1].id().as_str(), "depth");
    }
    assert_ne!(
        native.architecture_fingerprint(),
        persona.architecture_fingerprint()
    );
}

#[test]
fn moshi_decision_domains_include_exact_released_padding_rows() {
    for config in [
        moshi::MoshiConfig::native_v0_1().unwrap(),
        moshi::MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap(),
    ] {
        let boundary = moshi::DecisionBoundary::new(&config).unwrap();
        assert_eq!(
            boundary.text_token_domain().cardinality(),
            config.text_vocabulary_size() as usize + 1
        );
        assert_eq!(
            boundary.audio_token_domain().cardinality(),
            config.audio_vocabulary_size() as usize + 1
        );
    }
}

#[test]
fn external_assistant_safetensors_schemas_equal_neutral_parameter_topology() {
    let gemma_config = gemma4::AssistantConfig::from_json(
        br#"{
          "model_type":"gemma4_assistant","backbone_hidden_size":32,
          "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
          "text_config":{"model_type":"gemma4_text","hidden_size":32,
            "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
            "vocab_size":32,"max_position_embeddings":128,
            "tie_word_embeddings":false,"attention_k_eq_v":false,
            "layer_types":["full_attention"]}
        }"#,
    )
    .unwrap();
    let gemma = gemma4::Assistant::<ReferenceBackend>::new(gemma_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &gemma,
        gemma4::assistant_safetensors_plan(&gemma_config).unwrap(),
    );
    let mut tied_value = serde_json::from_slice::<serde_json::Value>(
        br#"{
          "model_type":"gemma4_assistant","backbone_hidden_size":32,
          "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
          "text_config":{"model_type":"gemma4_text","hidden_size":32,
            "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
            "vocab_size":32,"max_position_embeddings":128,
            "tie_word_embeddings":false,"attention_k_eq_v":false,
            "layer_types":["full_attention"]}
        }"#,
    )
    .unwrap();
    tied_value["tie_word_embeddings"] = true.into();
    let tied_config =
        gemma4::AssistantConfig::from_json(&serde_json::to_vec(&tied_value).unwrap()).unwrap();
    let tied = gemma4::Assistant::<ReferenceBackend>::new(tied_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &tied,
        gemma4::assistant_safetensors_plan(&tied_config).unwrap(),
    );
    tied_value["tie_word_embeddings"] = false.into();
    tied_value["use_ordered_embeddings"] = true.into();
    tied_value["num_centroids"] = 4.into();
    tied_value["centroid_intermediate_top_k"] = 2.into();
    let ordered_config =
        gemma4::AssistantConfig::from_json(&serde_json::to_vec(&tied_value).unwrap()).unwrap();
    let ordered = gemma4::Assistant::<ReferenceBackend>::new(ordered_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &ordered,
        gemma4::assistant_safetensors_plan(&ordered_config).unwrap(),
    );
    let dflash_config = muse_glimmer::DFlashConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
          "model_type":"muse_glimmer_assistant","hidden_size":6656,
          "intermediate_size":19968,"num_hidden_layers":5,"num_attention_heads":32,
          "num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.000001,
          "max_position_embeddings":131072,"sliding_window":2048,"block_size":16,
          "mask_token_id":201818,"target_layer_ids":[1,13,25,37,49],
          "layer_types":["sliding_attention","sliding_attention","sliding_attention",
            "sliding_attention","sliding_attention"],
          "hidden_act":"silu","attention_dropout":0.0,
          "rope_parameters":{"rope_theta":500000.0}
        }))
        .unwrap(),
    )
    .unwrap();
    let dflash = muse_glimmer::DFlash::<ReferenceBackend>::new(dflash_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &dflash,
        muse_glimmer::dflash_safetensors_plan(&dflash_config).unwrap(),
    );
}

fn assert_assistant_plan_matches_topology<M: Parameterized<ReferenceTensor>>(
    module: &M,
    plan: eredu_checkpoint::schema::SafetensorsCheckpointPlan,
) {
    assert!(plan.layout_groups.is_empty());
    let declared = plan
        .common_tensors
        .into_iter()
        .map(|tensor| (tensor.key, tensor.shape))
        .collect::<BTreeMap<_, _>>();
    let actual = topology(module).into_iter().collect::<BTreeMap<_, _>>();
    assert_eq!(declared, actual);
}

fn partition_test_range(units: usize, size: usize, rank: usize) -> std::ops::Range<usize> {
    let base = units / size;
    let remainder = units % size;
    let start = rank * base + rank.min(remainder);
    start..start + base + usize::from(rank < remainder)
}

fn partition_test_layout(
    groups: &[ParameterGroupSpec],
    size: usize,
    rank: usize,
) -> Result<LocalModelLayout, Error> {
    let mut layout = LocalModelLayout::default();
    for group in groups {
        let logical_range = group
            .partition_units()
            .map(|units| partition_test_range(units, size, rank));
        for member in group.members() {
            let mut local_shape = member.global_shape().to_vec();
            let (placement, member_logical_range) = match member.sharding() {
                MemberSharding::Replicated => (TensorPlacement::Replicated, None),
                MemberSharding::Equal { axis } => {
                    let width = member.global_shape()[*axis];
                    if !width.is_multiple_of(size) {
                        return Err(Error::backend("test equal shard does not divide"));
                    }
                    local_shape[*axis] = width / size;
                    (
                        TensorPlacement::Shard {
                            axis: *axis,
                            index: rank,
                            parts: size,
                        },
                        Some(rank * (width / size)..(rank + 1) * (width / size)),
                    )
                }
                MemberSharding::Balanced { axis } => {
                    let range = partition_test_range(member.global_shape()[*axis], size, rank);
                    local_shape[*axis] = range.len();
                    (
                        TensorPlacement::Range {
                            axis: *axis,
                            start: range.start,
                            end: range.end,
                        },
                        Some(range),
                    )
                }
                MemberSharding::Partitioned { axis } => {
                    let units = group
                        .partition_units()
                        .ok_or_else(|| Error::backend("test partitioned member has no units"))?;
                    let range = logical_range.as_ref().unwrap();
                    let width = member.global_shape()[*axis];
                    if !width.is_multiple_of(units) {
                        return Err(Error::backend("test partitioned shard does not divide"));
                    }
                    let per_unit = width / units;
                    let start = range.start * per_unit;
                    let end = range.end * per_unit;
                    local_shape[*axis] = end - start;
                    (
                        TensorPlacement::Range {
                            axis: *axis,
                            start,
                            end,
                        },
                        Some(range.clone()),
                    )
                }
                MemberSharding::PartitionedSegments { axis, segments } => {
                    let units = group
                        .partition_units()
                        .ok_or_else(|| Error::backend("test segmented member has no units"))?;
                    let range = logical_range.as_ref().unwrap();
                    let mut indices = Vec::new();
                    for segment in segments {
                        if !segment.len().is_multiple_of(units) {
                            return Err(Error::backend("test segment does not divide"));
                        }
                        let width = segment.len() / units;
                        indices.extend(
                            segment.start + range.start * width..segment.start + range.end * width,
                        );
                    }
                    local_shape[*axis] = indices.len();
                    (
                        TensorPlacement::Indices {
                            axis: *axis,
                            indices,
                        },
                        Some(range.clone()),
                    )
                }
                MemberSharding::Segmented { axis, segments } => {
                    let mut indices = Vec::new();
                    for segment in segments {
                        let range = partition_test_range(segment.len(), size, rank);
                        indices.extend(segment.start + range.start..segment.start + range.end);
                    }
                    local_shape[*axis] = indices.len();
                    (
                        TensorPlacement::Indices {
                            axis: *axis,
                            indices,
                        },
                        None,
                    )
                }
            };
            layout.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    local_shape,
                    placement,
                    group.partition_units(),
                    member_logical_range,
                    false,
                ),
            );
        }
    }
    Ok(layout)
}

fn inkling_partition_args() -> inkling::ModelArgs {
    inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "model_type":"inkling_mm_model", "image_token_id":5,
            "text_config":{
                "hidden_size":16,"num_hidden_layers":2,"vocab_size":19,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":4,"local_layer_ids":[1],
                "mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,
                "d_rel":2,"rel_extent":8,"intermediate_size":12,
                "dense_intermediate_size":12,"n_routed_experts":3,
                "num_experts_per_tok":2,"n_shared_experts":1,
                "unpadded_vocab_size":19
            },
            "vision_config":{"text_hidden_size":16,"patch_size":40,
                "temporal_patch_size":2,"num_channels":3,"num_hidden_layers":4}
        }))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn inkling_dense_partition_foundation_owns_optional_vision_text_and_state_exactly() {
    let args = inkling_partition_args();
    let architecture = inkling::LayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
    let parameters = architecture.parameter_description(&()).unwrap();
    let groups = parameters
        .groups()
        .iter()
        .map(|owned| owned.group().clone())
        .collect::<Vec<_>>();
    let layout = partition_test_layout(&groups, 2, 0).unwrap();
    let target_state = inkling::local_geometry(&args, &layout)
        .unwrap()
        .state_layout()
        .clone();
    let complete_state = inkling::composite_state_layout(&target_state, None).unwrap();
    let state_plan =
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(2, 0..2)]);

    let first_ownership = PartitionOwnership::new(
        true,
        false,
        ["vision", "audio", "embedding", "embedding_norm"],
    )
    .unwrap();
    let first_geometry = inkling::partition_local_geometry(
        &args,
        &layout,
        [
            (inkling::VISION_EXECUTION_GROUP, 0..2),
            (inkling::TEXT_EXECUTION_GROUP, 0..1),
        ],
        &first_ownership,
    )
    .unwrap();
    let first_partition = ArchitecturePartition::from_description(
        &parameters,
        [
            (inkling::VISION_EXECUTION_GROUP, 0..2),
            (inkling::TEXT_EXECUTION_GROUP, 0..1),
        ],
        first_ownership,
        &complete_state,
        &state_plan,
        first_geometry,
        NoAuxiliaryBoundarySchema::new(args.text_config.hidden_size),
    )
    .unwrap();
    let first = inkling::PartitionLocalFoundation::from_partition(&args, &first_partition).unwrap();
    assert_eq!(first.geometry().vision_units(), Some(0..2));
    assert_eq!(first.geometry().text_units(), 0..1);
    assert_eq!(
        first.geometry().text_layer(0).unwrap().num_attention_heads,
        2
    );
    assert_eq!(first.geometry().local_state_layout().unwrap().len(), 1);
    assert!(first
        .parameter_targets()
        .iter()
        .any(|target| target == "visual.final_norm.weight"));
    assert!(first
        .parameter_targets()
        .iter()
        .any(|target| target.starts_with("visual.layers.0.")));
    assert!(first
        .parameter_targets()
        .iter()
        .any(|target| target.starts_with("model.layers.0.")));
    assert!(!first
        .parameter_targets()
        .iter()
        .any(|target| target.starts_with("audio.")));
    assert!(!first
        .parameter_targets()
        .iter()
        .any(|target| target.starts_with("model.layers.1.")));

    let last_ownership =
        PartitionOwnership::new(false, true, ["norm", "output", inkling::MTP_STATIC_ROLE]).unwrap();
    let last_geometry = inkling::partition_local_geometry(
        &args,
        &layout,
        [
            (inkling::VISION_EXECUTION_GROUP, 2..4),
            (inkling::TEXT_EXECUTION_GROUP, 1..2),
        ],
        &last_ownership,
    )
    .unwrap();
    let last_partition = ArchitecturePartition::from_description(
        &parameters,
        [
            (inkling::VISION_EXECUTION_GROUP, 2..4),
            (inkling::TEXT_EXECUTION_GROUP, 1..2),
        ],
        last_ownership,
        &complete_state,
        &state_plan,
        last_geometry,
        NoAuxiliaryBoundarySchema::new(args.text_config.hidden_size),
    )
    .unwrap();
    let last = inkling::PartitionLocalFoundation::from_partition(&args, &last_partition).unwrap();
    assert_eq!(last_partition.state().unwrap().global_layer_offset(), 1);
    assert_eq!(
        last.geometry().static_roles(),
        ["norm", "output", inkling::MTP_STATIC_ROLE]
    );
    assert!(last
        .parameter_targets()
        .iter()
        .any(|target| target.starts_with("visual.layers.3.")));
    assert!(last
        .parameter_targets()
        .iter()
        .any(|target| target.starts_with("model.layers.1.")));
    assert!(last
        .parameter_targets()
        .iter()
        .any(|target| target == "lm_head.weight"));
    assert!(!last
        .parameter_targets()
        .iter()
        .any(|target| target.starts_with("model.mtp.")));

    let mut prediction = args.clone();
    prediction.mtp_config = Some(inkling::MtpConfig {
        num_nextn_predict_layers: 1,
        ..Default::default()
    });
    assert!(inkling::partition_local_geometry(
        &prediction,
        &layout,
        [(inkling::TEXT_EXECUTION_GROUP, 0..1)],
        &PartitionOwnership::new(
            true,
            false,
            ["vision", "audio", "embedding", "embedding_norm"],
        )
        .unwrap(),
    )
    .is_err());

    let mut routed = args.clone();
    let mut routed_schedule = routed
        .text_config
        .layer_schedule
        .iter()
        .copied()
        .collect::<Vec<_>>();
    routed_schedule[1].feed_forward = inkling::FeedForwardPolicy::SparseMoe;
    routed.text_config.layer_schedule = LayerSchedule::new(2, routed_schedule).unwrap();
    assert!(inkling::partition_local_geometry(
        &routed,
        &layout,
        [(inkling::TEXT_EXECUTION_GROUP, 0..1)],
        &PartitionOwnership::new(
            true,
            false,
            ["vision", "audio", "embedding", "embedding_norm"],
        )
        .unwrap(),
    )
    .is_err());
}

fn deepseek_v3_partition_args() -> deepseek::V3Args {
    deepseek::parse_v3_config(&serde_json::json!({
        "model_type": "deepseek_v3", "hidden_size": 8,
        "intermediate_size": 16, "moe_intermediate_size": 8,
        "num_hidden_layers": 4, "num_attention_heads": 4,
        "vocab_size": 16, "max_position_embeddings": 64,
        "q_lora_rank": 4, "kv_lora_rank": 4,
        "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 2,
        "first_k_dense_replace": 1, "n_routed_experts": 4,
        "n_shared_experts": 1, "num_experts_per_tok": 2,
        "n_group": 2, "topk_group": 1, "num_nextn_predict_layers": 0,
        "tie_word_embeddings": false
    }))
    .unwrap()
}

fn deepseek_v4_partition_args() -> deepseek::V4Args {
    deepseek::parse_v4_config(&serde_json::json!({
        "model_type": "deepseek_v4", "hidden_size": 8,
        "moe_intermediate_size": 8, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 4,
        "qk_rope_head_dim": 2, "q_lora_rank": 4,
        "o_lora_rank": 2, "o_groups": 4, "vocab_size": 16,
        "max_position_embeddings": 64, "sliding_window": 8,
        "compress_ratios": [0, 4, 128, 0], "index_n_heads": 4,
        "index_head_dim": 4, "index_topk": 1, "hc_mult": 2,
        "hc_sinkhorn_iters": 2, "n_routed_experts": 4,
        "n_shared_experts": 1, "num_experts_per_tok": 2,
        "num_hash_layers": 1, "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc", "norm_topk_prob": true,
        "routed_scaling_factor": 1.0, "swiglu_limit": 4.0,
        "num_nextn_predict_layers": 0
    }))
    .unwrap()
}

fn deepseek_partition_ownership(
    rank: eredu_core::ParallelRankTopology,
    v4: bool,
) -> PartitionOwnership {
    let first = rank.pipeline_parallel_rank() == 0;
    let last = rank.pipeline_parallel_rank() + 1 == rank.pipeline_parallel_size();
    let mut roles = Vec::new();
    if first {
        roles.push("embedding");
    }
    if last {
        roles.extend(["norm", "output"]);
        if v4 {
            roles.push("hyper_head");
        }
    }
    PartitionOwnership::new(first, last, roles).unwrap()
}

#[test]
fn deepseek_v3_v4_partition_foundations_cover_cartesian_ownership_and_compact_banks() {
    let matrices = [
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ];
    for (tensor, pipeline, expert) in matrices {
        let topology = eredu_core::ParallelTopology::new(tensor, pipeline, expert, 1).unwrap();
        for global_rank in 0..topology.world_size() {
            let rank = eredu_core::ParallelRankTopology::new(topology, global_rank).unwrap();
            let owned = eredu_core::balanced_contiguous_range(
                4,
                rank.pipeline_parallel_size(),
                rank.pipeline_parallel_rank(),
                false,
            )
            .unwrap();

            let v3_args = deepseek_v3_partition_args();
            let v3_parameters = deepseek::parallel::v3_parameter_description(&v3_args).unwrap();
            let v3_layout =
                eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    &v3_parameters,
                    rank,
                )
                .unwrap();
            let v3_local = deepseek::parallel::v3_local_geometry(&v3_args, &v3_layout).unwrap();
            let v3_realization =
                deepseek::v3_partition_expert_realization_plan(&v3_args, &v3_local, rank).unwrap();
            let v3_ownership = deepseek_partition_ownership(rank, false);
            let v3_geometry = deepseek::v3_partition_local_geometry(
                &v3_args,
                &v3_layout,
                [(decoder::TARGET_EXECUTION_GROUP, owned.clone())],
                &v3_ownership,
                rank,
                &v3_realization,
            )
            .unwrap();
            let v3_complete_state = v3_local.state_layout().clone();
            let v3_partition = ArchitecturePartition::from_description(
                &v3_parameters,
                [(decoder::TARGET_EXECUTION_GROUP, owned.clone())],
                v3_ownership,
                &v3_complete_state,
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(0, 0..4),
                ]),
                v3_geometry,
                deepseek::v3::TargetBoundarySchema::from_args(&v3_args),
            )
            .unwrap();
            let v3 = deepseek::V3PartitionLocalFoundation::from_partition(&v3_args, &v3_partition)
                .unwrap();
            assert_eq!(v3.geometry().target_units(), owned);
            assert_eq!(
                v3_partition.state().unwrap().global_layer_offset(),
                owned.start
            );
            assert_eq!(
                v3.geometry().local_state_layout().unwrap().len(),
                owned.len()
            );
            assert_eq!(
                v3.geometry().local_state_layout().unwrap(),
                v3_complete_state.slice(owned.clone()).unwrap()
            );
            assert_eq!(
                v3.geometry().static_roles(),
                v3_partition.ownership().static_roles()
            );
            let expected_experts = eredu_core::balanced_contiguous_range(
                4,
                rank.expert_parallel_size(),
                rank.expert_parallel_rank(),
                false,
            )
            .unwrap()
            .collect::<Vec<_>>();
            let sparse_owned = owned.clone().filter(|unit| *unit != 0).count();
            assert_eq!(
                v3.geometry().expert_banks().len(),
                sparse_owned * expected_experts.len()
            );
            for bank in v3.geometry().expert_banks() {
                assert!(owned.contains(&bank.global_unit()));
                assert_eq!(
                    expected_experts[bank.owner_local_expert()],
                    bank.global_expert()
                );
                assert_eq!(bank.bank_key().unit(), bank.global_unit());
                assert_eq!(bank.bank_key().member(), bank.global_expert());
                assert_eq!(bank.intermediate_range().len(), 8 / tensor);
            }
            assert!(v3
                .resident_parameter_targets()
                .iter()
                .any(|target| target.contains("shared_experts")));
            if sparse_owned > 0 {
                assert!(!v3.routed_parameter_targets().is_empty());
            }

            let v4_args = deepseek_v4_partition_args();
            let v4_parameters = deepseek::parallel::v4_parameter_description(&v4_args).unwrap();
            let v4_layout =
                eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    &v4_parameters,
                    rank,
                )
                .unwrap();
            let v4_local = deepseek::parallel::v4_local_geometry(&v4_args, &v4_layout).unwrap();
            let v4_realization =
                deepseek::v4_partition_expert_realization_plan(&v4_args, &v4_local, rank).unwrap();
            let v4_ownership = deepseek_partition_ownership(rank, true);
            let v4_geometry = deepseek::v4_partition_local_geometry(
                &v4_args,
                &v4_layout,
                [(decoder::TARGET_EXECUTION_GROUP, owned.clone())],
                &v4_ownership,
                rank,
                &v4_realization,
            )
            .unwrap();
            let v4_complete_state = v4_local.state_layout().clone();
            let v4_partition = ArchitecturePartition::from_description(
                &v4_parameters,
                [(decoder::TARGET_EXECUTION_GROUP, owned.clone())],
                v4_ownership,
                &v4_complete_state,
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(0, 0..4),
                ]),
                v4_geometry,
                deepseek::v4::TargetBoundarySchema::from_args(&v4_args).unwrap(),
            )
            .unwrap();
            let v4 = deepseek::V4PartitionLocalFoundation::from_partition(&v4_args, &v4_partition)
                .unwrap();
            assert_eq!(v4.geometry().target_units(), owned);
            assert_eq!(
                v4_partition.state().unwrap().global_layer_offset(),
                owned.start
            );
            assert_eq!(
                v4.geometry().local_state_layout().unwrap().len(),
                owned.len()
            );
            assert_eq!(
                v4.geometry().local_state_layout().unwrap(),
                v4_complete_state.slice(owned.clone()).unwrap()
            );
            assert_eq!(
                v4.geometry().static_roles(),
                v4_partition.ownership().static_roles()
            );
            assert_eq!(
                v4.geometry().expert_banks().len(),
                owned.len() * expected_experts.len()
            );
            assert!(v4
                .resident_parameter_targets()
                .iter()
                .any(|target| target.contains("shared_experts")));
            assert!(!v4.routed_parameter_targets().is_empty());
        }
    }
}

#[test]
fn deepseek_partition_foundations_reject_prediction_and_wrong_expert_owner_before_layout() {
    let args = deepseek_v3_partition_args();
    let topology = eredu_core::ParallelTopology::new(1, 2, 2, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, 3).unwrap();
    let wrong_rank = eredu_core::ParallelRankTopology::new(topology, 2).unwrap();
    let parameters = deepseek::parallel::v3_parameter_description(&args).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &parameters,
        rank,
    )
    .unwrap();
    let local = deepseek::parallel::v3_local_geometry(&args, &layout).unwrap();
    let realization = deepseek::v3_partition_expert_realization_plan(&args, &local, rank).unwrap();
    let ownership = deepseek_partition_ownership(rank, false);
    assert!(deepseek::v3_partition_local_geometry(
        &args,
        &layout,
        [(decoder::TARGET_EXECUTION_GROUP, 2..4)],
        &ownership,
        wrong_rank,
        &realization,
    )
    .is_err());

    let mut prediction = args;
    prediction.num_nextn_predict_layers = 1;
    let error = deepseek::v3_partition_local_geometry(
        &prediction,
        &LocalModelLayout::default(),
        [(decoder::TARGET_EXECUTION_GROUP, 0..2)],
        &PartitionOwnership::new(true, false, ["embedding"]).unwrap(),
        rank,
        &realization,
    )
    .unwrap_err();
    assert!(error.to_string().contains("embedded prediction"));
}

#[derive(Debug)]
struct CompositeOnlyAdmission;

impl eredu_architectures::partitioned_execution::PartitionedAdmissionDispatcher
    for CompositeOnlyAdmission
{
    type Output = eredu_architectures::partitioned_execution::CompositePartitionedAdmission;
    type Error = Error;

    fn direct(
        self,
        _: eredu_architectures::partitioned_execution::DirectPartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        Err(Error::backend("expected composite partition admission"))
    }

    fn routed(
        self,
        _: eredu_architectures::partitioned_execution::RoutedPartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        Err(Error::backend("expected composite partition admission"))
    }

    fn composite(
        self,
        admission: eredu_architectures::partitioned_execution::CompositePartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        Ok(admission)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CompositePartitionSummary {
    groups: Vec<(String, std::ops::Range<usize>)>,
    owns_input: bool,
    owns_output: bool,
    state_offset: usize,
    tasks: Vec<String>,
    layout_len: usize,
    publication_width: i32,
}

struct InspectCompositePartition;

impl
    eredu_architectures::composite_partitioned::AuthoritativeCompositePartitionVisitor<
        ReferenceBackend,
        DeviceState<ReferenceBackend, ReferenceCache>,
    > for InspectCompositePartition
{
    type Output = CompositePartitionSummary;
    type Error = Error;

    fn visit<A, G, W>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::composite_execution::CompositeArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
            > + eredu_runtime::PartitionedLayeredArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                Boundary = W,
            >,
        A::Error: std::fmt::Display,
    {
        let layout_len = prepared.layout().len();
        let tasks = prepared
            .materialization_tasks()
            .iter()
            .map(|task| task.name().to_owned())
            .collect();
        let partition = prepared.prepared().selected().partition();
        Ok(CompositePartitionSummary {
            groups: partition
                .groups()
                .iter()
                .map(|group| (group.group().as_str().to_owned(), group.global_units()))
                .collect(),
            owns_input: partition.ownership().owns_input(),
            owns_output: partition.ownership().owns_output(),
            state_offset: partition
                .state()
                .map(eredu_runtime::PartitionState::global_layer_offset)
                .unwrap_or(0),
            tasks,
            layout_len,
            publication_width: prepared.publication().output_width(),
        })
    }
}

fn composite_reference_selection(
    requirements: &eredu_architectures::replicated_text::CompositeTextRequirements,
) -> eredu_architectures::replicated_text::SelectedCompositeTextRealization {
    let execution = requirements.execution();
    let lowerings = execution
        .parameters()
        .iter()
        .filter(|parameter| parameter.has_lowering_source())
        .map(|parameter| {
            eredu_runtime::WeightLoweringCapability::new(
                parameter
                    .lowering_descriptor(parameter.native_executable())
                    .unwrap(),
                eredu_runtime::WeightLoweringKind::Direct,
            )
        })
        .collect();
    let state = eredu_runtime::StateMechanismCapabilities::new(
        (0..execution.state_layout().len()).flat_map(|layer| {
            execution
                .state_layout()
                .components(layer)
                .unwrap()
                .iter()
                .cloned()
                .map(move |component| {
                    eredu_runtime::StateComponentMechanism::new(
                        layer,
                        component,
                        Some(eredu_runtime::StateComponentPlacement::Device),
                        None,
                    )
                })
        }),
    )
    .with_transactions(true, true)
    .with_reset(true);
    let mechanisms = eredu_runtime::BackendMechanismCapabilities::new(
        eredu_nn::NeuralOperatorCapabilities::ALL,
        lowerings,
        vec![eredu_runtime::WeightResidencyMechanism::Resident],
        state,
    );
    let processor = eredu_runtime::select_processor_execution(
        requirements.processor_execution(),
        &eredu_runtime::ProcessorSelectionRequest::new([eredu_core::InputModality::Text]),
        &eredu_runtime::MediaPrimitiveCapabilities::new(
            [],
            [eredu_core::InputModality::Text],
            [],
            [],
            u64::MAX,
        ),
    )
    .unwrap();
    eredu_architectures::replicated_text::select_composite_text_realization_with_processor(
        requirements,
        &eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        ),
        eredu_runtime::WeightResidency::fully_resident(),
        &mechanisms,
        processor,
    )
    .unwrap()
}

fn indexed_composite_artifact(
    config: &serde_json::Value,
    description: &ArchitectureParameterDescription,
) -> tempfile::TempDir {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    let mut tensors = description
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .map(|member| {
            (
                member.target().to_owned(),
                member.global_shape().to_vec(),
                vec![0_u8; member.global_shape().iter().product::<usize>() * 4],
            )
        })
        .collect::<Vec<_>>();
    if matches!(
        config.get("model_type").and_then(serde_json::Value::as_str),
        Some("qwen3_vl" | "qwen3_5")
    ) {
        let vision = &config["vision_config"];
        let depth = vision["depth"].as_u64().unwrap() as usize;
        let hidden = vision["hidden_size"].as_u64().unwrap() as usize;
        let intermediate = vision["intermediate_size"].as_u64().unwrap() as usize;
        let merge = vision["spatial_merge_size"].as_u64().unwrap() as usize;
        let output = vision["out_hidden_size"].as_u64().unwrap() as usize;
        let mut required = Vec::new();
        for layer in 0..depth {
            let root = format!("model.visual.blocks.{layer}");
            required.extend([
                (format!("{root}.attn.qkv.bias"), vec![3 * hidden]),
                (format!("{root}.attn.proj.bias"), vec![hidden]),
                (format!("{root}.mlp.linear_fc1.bias"), vec![intermediate]),
                (format!("{root}.mlp.linear_fc2.bias"), vec![hidden]),
            ]);
        }
        let merged = hidden * merge * merge;
        required.extend([
            (
                "model.visual.merger.linear_fc1.bias".to_owned(),
                vec![merged],
            ),
            (
                "model.visual.merger.linear_fc2.bias".to_owned(),
                vec![output],
            ),
        ]);
        let deepstack = vision
            .get("deepstack_visual_indexes")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        for index in 0..deepstack {
            required.extend([
                (
                    format!("model.visual.deepstack_merger_list.{index}.linear_fc1.bias"),
                    vec![merged],
                ),
                (
                    format!("model.visual.deepstack_merger_list.{index}.linear_fc2.bias"),
                    vec![output],
                ),
            ]);
        }
        for (name, shape) in required {
            if !tensors.iter().any(|(target, _, _)| target == &name) {
                let bytes = vec![0_u8; shape.iter().product::<usize>() * 4];
                tensors.push((name, shape, bytes));
            }
        }
    }
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        views,
        None,
        &artifact.path().join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        artifact.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    artifact
}

fn authoritative_composite_pp_summaries(
    artifact: &std::path::Path,
    label: &str,
) -> Vec<CompositePartitionSummary> {
    let inspection = configuration::inspect_artifact(artifact).unwrap();
    let requirements =
        eredu_architectures::replicated_text::composite_text_requirements(&inspection)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
    let base = composite_reference_selection(&requirements);
    let topology = eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap();
    let communication = resident_partition_communication();
    [0, topology.world_size() - 1]
        .into_iter()
        .map(|rank| {
            let admission =
                eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
                    &inspection,
                    eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                        topology,
                        rank,
                        1,
                        8,
                        eredu_runtime::PipelineActivationDtype::Float32,
                    )
                    .unwrap()
                    .with_completion_policy(
                        eredu_runtime::CommunicationCompletionPolicy::new(
                            std::time::Duration::from_secs(1),
                            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                        )
                        .unwrap(),
                    ),
                    CompositeOnlyAdmission,
                )
                .unwrap_or_else(|error| panic!("{label} partition admission: {error}"));
            let selected =
                eredu_architectures::partitioned_execution::select_composite_partitioned_admission(
                    admission,
                    base.clone(),
                    &communication,
                )
                .unwrap();
            eredu_architectures::composite_partitioned::visit_authoritative_composite_partition::<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                _,
            >(selected, &(), InspectCompositePartition)
            .unwrap()
        })
        .collect()
}

#[test]
fn authoritative_composite_dispatch_preserves_selected_gemma_partition_and_local_payload() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let config = serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4_unified", "tie_word_embeddings": false,
        "image_token_id": 60, "audio_token_id": 61,
        "text_config": {
            "model_type": "gemma4_text", "hidden_size": 16,
            "num_hidden_layers": 4, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 8, "rms_norm_eps": 0.000001, "vocab_size": 64,
            "max_position_embeddings": 128,
            "hidden_size_per_layer_input": 4, "vocab_size_per_layer_input": 64,
            "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"]
        },
        "vision_config": {
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "patch_size": 4, "pooling_kernel_size": 2, "position_embedding_size": 16,
            "rms_norm_eps": 0.000001
        },
        "audio_config": {
            "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
            "output_proj_dims": 8, "conv_kernel_size": 3, "attention_chunk_size": 4,
            "attention_context_left": 5, "attention_context_right": 0,
            "attention_invalid_logits_value": -1000000000.0, "attention_logit_cap": 50.0,
            "residual_weight": 0.5, "rms_norm_eps": 0.000001,
            "subsampling_conv_channels": [4, 8]
        }
    });
    let args = gemma4::FamilyConfig::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
    let model = gemma4::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let description = ArchitectureParameters::parameter_description(&model, &()).unwrap();
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tensors = description
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .map(|member| {
            (
                member.target().to_owned(),
                member.global_shape().to_vec(),
                vec![0_u8; member.global_shape().iter().product::<usize>() * 4],
            )
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        views,
        None,
        &artifact.path().join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        artifact.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    let inspection = configuration::inspect_artifact(artifact.path()).unwrap();
    let requirements =
        eredu_architectures::replicated_text::composite_text_requirements(&inspection).unwrap();
    let base = composite_reference_selection(&requirements);
    let topology = eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap();
    let communication = resident_partition_communication();

    for rank in [0, topology.world_size() - 1] {
        let admission = eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                topology,
                rank,
                1,
                8,
                eredu_runtime::PipelineActivationDtype::Float32,
            )
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            ),
            CompositeOnlyAdmission,
        )
        .unwrap();
        let selected =
            eredu_architectures::partitioned_execution::select_composite_partitioned_admission(
                admission,
                base.clone(),
                &communication,
            )
            .unwrap();
        let summary =
            eredu_architectures::composite_partitioned::visit_authoritative_composite_partition::<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
                _,
            >(selected, &(), InspectCompositePartition)
            .unwrap();
        assert!(summary.layout_len > 0);
        assert!(summary
            .groups
            .iter()
            .any(|(group, _)| group == gemma4::TEXT_EXECUTION_GROUP));
        assert!(!summary.tasks.is_empty());
        if rank == 0 {
            assert_eq!(summary.groups.len(), 3);
            assert!(summary.owns_input);
            assert!(!summary.owns_output);
            assert_eq!(summary.state_offset, 0);
            assert!(summary
                .tasks
                .iter()
                .any(|name| name.contains("embed_tokens.weight")));
        } else {
            assert_eq!(summary.groups.len(), 1);
            assert!(!summary.owns_input);
            assert!(summary.owns_output);
            assert_eq!(summary.state_offset, 2);
            assert!(summary
                .tasks
                .iter()
                .any(|name| name.ends_with("norm.weight")));
        }
    }
}

fn assert_two_stage_composite_dispatch(
    label: &str,
    config: serde_json::Value,
    description: ArchitectureParameterDescription,
    expected_publication_width: i32,
) {
    let artifact = indexed_composite_artifact(&config, &description);
    let summaries = authoritative_composite_pp_summaries(artifact.path(), label);
    assert_eq!(summaries.len(), 2);
    assert!(summaries[0].owns_input);
    assert!(!summaries[0].owns_output);
    assert_eq!(summaries[0].state_offset, 0);
    assert!(!summaries[1].owns_input);
    assert!(summaries[1].owns_output);
    assert!(summaries[1].state_offset > 0);
    assert!(summaries.iter().all(|summary| {
        summary.layout_len > 0 && !summary.groups.is_empty() && !summary.tasks.is_empty()
    }));
    assert!(summaries
        .iter()
        .all(|summary| summary.publication_width == expected_publication_width));
}

#[test]
fn authoritative_composite_dispatch_reaches_muse_qwen_vl_conditional_and_inkling() {
    let muse_config = serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "image_token_id": 22, "video_token_id": 23,
        "out_hidden_size": 32, "projector_hidden_size": 16,
        "text_config": {
            "model_type": "muse_glimmer_text", "hidden_size": 16,
            "num_hidden_layers": 2, "intermediate_size": 24,
            "moe_intermediate_size": 0, "num_experts": 0, "num_experts_per_tok": 0,
            "norm_topk_prob": false, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 4, "rms_norm_eps": 0.00001, "post_norm_eps": 0.00001,
            "vocab_size": 24, "max_position_embeddings": 64, "rope_theta": 10000.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [10000.0, 0.0], "sliding_window": 8,
            "tie_word_embeddings": false, "hidden_act": "silu", "attention_dropout": 0.0,
            "qk_scale_factor": 1.0, "output_multiplier": 1.0,
            "final_logit_softcapping": 30.0
        },
        "vision_config": {
            "model_type": "muse_glimmer_vision", "hidden_size": 8,
            "intermediate_size": 12, "num_attention_heads": 2, "num_hidden_layers": 1,
            "patch_size": 2, "patch_temporal": 1, "merge_size": 2,
            "pos_emb_height": 2, "pos_emb_width": 2, "max_position_embeddings": 4,
            "layer_norm_eps": 0.00001, "hidden_act": "gelu",
            "layer_types": ["full_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        }
    });
    let muse_args = muse_glimmer::DecoderConfig::from_hf_value(&muse_config).unwrap();
    let muse_model = muse_glimmer::LayeredModel::<ReferenceBackend>::new(muse_args, &()).unwrap();
    assert_two_stage_composite_dispatch(
        "muse",
        muse_config,
        ArchitectureParameters::parameter_description(&muse_model, &()).unwrap(),
        24,
    );

    let qwen_vl_config = serde_json::json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl", "image_token_id": 61, "video_token_id": 62,
        "text_config": {
            "model_type": "qwen3_vl_text", "hidden_size": 32,
            "num_hidden_layers": 2, "intermediate_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "rms_norm_eps": 0.000001, "vocab_size": 64,
            "max_position_embeddings": 128, "tie_word_embeddings": false,
            "rope_scaling": {"mrope_section": [2, 1, 1]}
        },
        "vision_config": {
            "depth": 1, "hidden_size": 16, "intermediate_size": 24, "num_heads": 4,
            "num_position_embeddings": 16, "in_channels": 3, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 32,
            "deepstack_visual_indexes": [0]
        }
    });
    let qwen_vl_args = qwen::vl::model_args_from_config_value(&qwen_vl_config).unwrap();
    let qwen_vl_model = qwen::vl::LayeredModel::<ReferenceBackend>::new(qwen_vl_args, &()).unwrap();
    assert_two_stage_composite_dispatch(
        "qwen-vl",
        qwen_vl_config,
        ArchitectureParameters::parameter_description(&qwen_vl_model, &()).unwrap(),
        64,
    );

    let conditional_config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5", "image_token_id": 60, "video_token_id": 61,
        "text_config": {
            "model_type": "qwen3_5_text", "vocab_size": 64, "hidden_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "max_position_embeddings": 128, "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "intermediate_size": 64, "layer_types": ["linear_attention", "full_attention"],
            "mtp_num_hidden_layers": 0, "tie_word_embeddings": false
        },
        "vision_config": {
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "num_position_embeddings": 16, "in_channels": 3, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 32
        }
    });
    let conditional_args = qwen::hybrid::model_args_from_config_value(&conditional_config).unwrap();
    let conditional_model =
        qwen::hybrid::ConditionalLayeredModel::<ReferenceBackend>::new(conditional_args, &())
            .unwrap();
    assert_two_stage_composite_dispatch(
        "conditional-qwen",
        conditional_config,
        ArchitectureParameters::parameter_description(&conditional_model, &()).unwrap(),
        64,
    );

    let inkling_config = serde_json::json!({
        "architectures": ["InklingForConditionalGeneration"],
        "model_type": "inkling_mm_model", "image_token_id": 60, "audio_token_id": 61,
        "text_config": {
            "hidden_size": 16, "num_hidden_layers": 2, "vocab_size": 32,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "sliding_window_size": 8, "layer_types": ["sliding_attention", "full_attention"],
            "mlp_layer_types": ["dense", "dense"], "sconv_kernel_size": 4,
            "d_rel": 2, "rel_extent": 16, "intermediate_size": 32,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "unpadded_vocab_size": 30, "tie_word_embeddings": true
        },
        "audio_config": {"text_hidden_size": 16, "num_codebooks": 4, "codebook_size": 8},
        "vision_config": {"text_hidden_size": 16, "patch_size": 40,
            "temporal_patch_size": 2, "num_channels": 3, "num_hidden_layers": 4}
    });
    let inkling_args =
        inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&inkling_config).unwrap()).unwrap();
    assert!(!inkling_args.text_config.has_sparse_moe_layers());
    let inkling_model = inkling::LayeredModel::<ReferenceBackend>::new(inkling_args, &()).unwrap();
    assert_two_stage_composite_dispatch(
        "inkling",
        inkling_config,
        ArchitectureParameters::parameter_description(&inkling_model, &()).unwrap(),
        30,
    );
}
