//! Published header catalogs, with only the tiny literal control buffers readable.
use super::*;
use eredu_checkpoint::validation::{validate_safetensors_plan, CheckpointValidation};
use serde::Deserialize;

#[derive(Deserialize)]
struct Record {
    name: String,
    axes: Vec<Vec<usize>>,
    dtype: String,
    shape: Vec<usize>,
}
#[derive(Deserialize)]
struct Constant {
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: String,
}
#[derive(Deserialize)]
struct Fixture {
    tensor_count: usize,
    tensors: Vec<Record>,
    constants: Vec<Constant>,
}
struct PublishedSource {
    metadata: BTreeMap<String, TensorMetadata>,
    constants: eredu_checkpoint::store::SafetensorsWeightStore,
    _directory: tempfile::TempDir,
    reads: Mutex<Vec<String>>,
}
impl CheckpointSource for PublishedSource {
    fn source_keys(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.metadata
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.reads.lock().unwrap().push(request.key.clone());
        // No model/table weight bytes exist in this fixture. Such a read fails closed.
        self.constants.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.constants.source_diagnostics()
    }
}
fn dtype(name: &str) -> (safetensors::Dtype, StoredDtype, usize) {
    match name {
        "BF16" => (safetensors::Dtype::BF16, StoredDtype::BF16, 2),
        "F16" => (safetensors::Dtype::F16, StoredDtype::F16, 2),
        "F32" => (safetensors::Dtype::F32, StoredDtype::F32, 4),
        "I64" => (safetensors::Dtype::I64, StoredDtype::I64, 8),
        "F8_E4M3" => (safetensors::Dtype::F8_E4M3, StoredDtype::F8E4M3, 1),
        _ => panic!("unexpected published dtype {name}"),
    }
}
fn published(fp8: bool) -> PublishedSource {
    let fixture: Fixture = serde_json::from_str(if fp8 {
        include_str!("../fixtures/fp8.json")
    } else {
        include_str!("../fixtures/bf16.json")
    })
    .unwrap();
    let mut metadata = BTreeMap::new();
    for record in fixture.tensors {
        let mut names = vec![record.name];
        for (axis, values) in record.axes.iter().enumerate() {
            names = names
                .into_iter()
                .flat_map(|name| {
                    values
                        .iter()
                        .map(move |value| name.replace(&format!("{{{axis}}}"), &value.to_string()))
                })
                .collect();
        }
        let (_, stored_dtype, bytes) = dtype(&record.dtype);
        for name in names {
            let meta = TensorMetadata {
                name: name.clone(),
                logical_shape: record.shape.clone(),
                physical_shape: record.shape.clone(),
                stored_dtype: stored_dtype.clone(),
                encoded_byte_len: record.shape.iter().product::<usize>() as u64 * bytes as u64,
                backing_shard: None,
            };
            assert!(metadata.insert(name, meta).is_none());
        }
    }
    assert_eq!(metadata.len(), fixture.tensor_count);
    let constants: Vec<_> = fixture
        .constants
        .into_iter()
        .map(|c| {
            let bytes: Vec<u8> = (0..c.bytes.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&c.bytes[i..i + 2], 16).unwrap())
                .collect();
            (c.tensor, dtype(&c.dtype).0, c.shape, bytes)
        })
        .collect();
    // Only the tiny literal controls exist physically. Keep their metadata and
    // leases honest while declaring the remaining model/table headers without
    // allocating or acquiring those payloads.
    let directory = tempfile::tempdir().unwrap();
    let values = constants.iter().map(|(name, dtype, shape, bytes)| {
        (
            name.as_str(),
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
        )
    });
    safetensors::tensor::serialize_to_file(
        values,
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let constants =
        eredu_checkpoint::store::SafetensorsWeightStore::open(directory.path()).unwrap();
    for key in constants.source_keys() {
        metadata.insert(key.clone(), constants.source_metadata(&key).unwrap());
    }
    PublishedSource {
        metadata,
        constants,
        _directory: directory,
        reads: Mutex::new(vec![]),
    }
}
fn policy(fp8: bool) -> schema::SafetensorsEncoding {
    let mut root: serde_json::Value =
        serde_json::from_str(include_str!("../../config/released.json")).unwrap();
    if fp8 {
        root["quantization_config"] =
            serde_json::from_str(include_str!("../fixtures/fp8-policy.json")).unwrap();
    }
    schema::SafetensorsEncoding::from_json(&root).unwrap()
}

#[test]
fn ordinary_load_defaults_preserve_explicit_policy_and_follow_state_pages() {
    use crate::qwen4_exp::prepared::normalize_load_request;
    use eredu_runtime::*;
    let config = Config::from_json(
        &serde_json::from_str(include_str!("../../config/released.json")).unwrap(),
    )
    .unwrap();
    let paged = NormalizedLoadRequest::default().with_state_residency(CacheResidencyPolicy::Paged(
        PagedCacheOptions::new(32, 1 << 24, 1 << 24, 1).unwrap(),
    ));
    let normalized = normalize_load_request(&paged, &config).unwrap();
    assert_eq!(normalized.state_residency(), paged.state_residency());
    assert_eq!(
        normalized
            .bounded_execution()
            .unwrap()
            .append()
            .limits()
            .page_entries,
        32
    );
    assert_eq!(
        normalized
            .bounded_execution()
            .unwrap()
            .invocation()
            .history_tokens(),
        config.max_positions,
    );

    let policy = normalized.bounded_execution().unwrap();
    let custom = BoundedExecutionPolicy::new(
        InvocationLimits::new(2, 16, 1024).unwrap(),
        policy.selection(),
        policy.rows(),
        policy.append(),
    )
    .unwrap();
    let explicit = normalized
        .with_bounded_execution(custom)
        .with_weight_residency(WeightResidency::with_layers(
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
            ),
        ))
        .with_drafting(DraftingLoadRequest::Disabled);
    assert_eq!(
        normalize_load_request(&explicit, &config).unwrap(),
        explicit
    );
}

#[test]
fn pinned_bf16_and_fp8_catalogs_match_complete_family_contract_without_weight_reads() {
    let config = Config::from_json(
        &serde_json::from_str(include_str!("../../config/released.json")).unwrap(),
    )
    .unwrap();
    for fp8 in [false, true] {
        let source = Arc::new(published(fp8));
        let header = SafetensorsTableSourcePlan::prepare(
            source.as_ref() as &dyn CheckpointSource,
            &config,
            1,
        )
        .unwrap();
        let tables = BTreeMap::from([(1, header.clone())]);
        let plan = schema::safetensors_plan(&config, &policy(fp8), &tables).unwrap();
        let validation = validate_safetensors_plan(source.as_ref() as &dyn CheckpointSource, &plan);
        assert_eq!(validation, CheckpointValidation::Exact, "fp8={fp8}");
        assert!(source.reads.lock().unwrap().is_empty());
        let table = header
            .bind(source.clone(), 2, 1, TensorElementType::Bf16)
            .unwrap();
        assert_eq!(table.lookup.rows, 320001536);
        assert_eq!(table.lookup.dimensions, 160);
        assert_eq!(table.rows.source_keys().len(), 128);
        let NGramControls::Safetensors(controls) = &table.controls else {
            panic!("expected retained SafeTensors controls");
        };
        assert_eq!(controls.source_keys().len(), 3);
        let lexical = crate::qwen4_exp::ple::LexicalInjectionSpec::from_header(
            &config,
            1,
            "model.layers.1",
            &table.lookup,
            64,
            |_| eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense),
        )
        .unwrap();
        let recipes = recipes::parameter_recipes(
            source.source_keys(),
            &config,
            recipes::ParameterScope::Lexical(1),
        )
        .unwrap();
        for linear in [&lexical.key, &lexical.value] {
            let shape = recipes[linear.weight.id.as_str()]
                .infer(source.as_ref() as &dyn CheckpointSource)
                .unwrap()
                .shape;
            assert_eq!(shape, [linear.output as usize, linear.input as usize]);
        }
        let shape = recipes[lexical.convolution.weight.id.as_str()]
            .infer(source.as_ref() as &dyn CheckpointSource)
            .unwrap()
            .shape;
        assert_eq!(
            shape,
            [
                lexical.geometry.flattened_width() as usize,
                1,
                config.ngram.kernel as usize
            ]
        );
        for norm in [
            &lexical.key_norm,
            &lexical.query_norm,
            &lexical.convolution_norm,
        ] {
            let eredu_nn::NormalizationScale::LearnedOffset { weight, offset } = &norm.scale else {
                panic!("released norm uses offsets")
            };
            assert_eq!(*offset, 1.);
            assert_eq!(
                recipes[weight.id.as_str()]
                    .infer(source.as_ref() as &dyn CheckpointSource)
                    .unwrap()
                    .shape,
                [norm.dimensions as usize]
            );
        }
        assert_eq!(lexical.convolution.dilation, 3);
        assert_eq!(lexical.embedding.output_width(), 2560);
        assert_eq!(lexical.state_policies().unwrap().len(), 2);
        let plan = table
            .rows
            .plan(
                &[2500011, 2500012, 2500011],
                eredu_checkpoint::rows::RowReadLimits {
                    requests: 3,
                    rows_per_read: 2,
                },
            )
            .unwrap();
        assert_eq!(plan.reads().len(), 1);
        assert_eq!(source.reads.lock().unwrap().len(), if fp8 { 4 } else { 3 });
        for scope in [
            recipes::ParameterScope::Target(0),
            recipes::ParameterScope::Target(1),
            recipes::ParameterScope::Target(3),
            recipes::ParameterScope::Lexical(1),
            recipes::ParameterScope::Prediction(0),
            recipes::ParameterScope::Vision(0),
            recipes::ParameterScope::Static,
        ] {
            let ordinary =
                recipes::parameter_recipes(source.source_keys(), &config, scope).unwrap();
            assert!(!ordinary.is_empty());
            for recipe in ordinary.values() {
                assert!(recipe
                    .infer(source.as_ref() as &dyn CheckpointSource)
                    .is_ok());
            }
            assert!(ordinary.keys().all(
                |name| !name.contains(".mlp.experts.") && !name.contains(".ple.ple_embedding.")
            ));
            if matches!(
                scope,
                recipes::ParameterScope::Target(_) | recipes::ParameterScope::Prediction(_)
            ) {
                for expert in [0, 511] {
                    let bank = recipes::expert_recipes(
                        source.as_ref() as &dyn CheckpointSource,
                        &config,
                        scope,
                        expert,
                    )
                    .unwrap();
                    assert_eq!(bank.len(), if fp8 { 4 } else { 2 });
                    for (name, recipe) in bank {
                        let meta = recipe
                            .infer(source.as_ref() as &dyn CheckpointSource)
                            .unwrap();
                        assert_eq!(meta.shape[0], 1, "one expert: {name}");
                        let is_gate = name.starts_with("gate_up");
                        assert_eq!(
                            &meta.shape[1..],
                            if name.ends_with("_scales") {
                                if is_gate {
                                    &[10, 20]
                                } else {
                                    &[20, 5]
                                }
                            } else if is_gate {
                                &[1280, 2560]
                            } else {
                                &[2560, 640]
                            }
                        );
                    }
                }
            }
        }
        assert_eq!(source.reads.lock().unwrap().len(), if fp8 { 4 } else { 3 });
        if fp8 {
            let mut source = Arc::try_unwrap(source).ok();
            // Prepared tables retain provenance and prevent premature source destruction.
            assert!(source.take().is_none());
        }
    }
}
#[test]
fn fp8_policy_preserves_exclusions_and_rejects_other_scale_contracts() {
    use eredu_checkpoint::LinearFormat;
    let policy = policy(true);
    for name in [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "mtp.fc_hidden.weight",
        "lm_head.weight",
    ] {
        assert_eq!(policy.linear_format(name), LinearFormat::Dense);
    }
    assert!(matches!(
        policy.linear_format("model.layers.0.mlp.experts.0.gate_proj.weight"),
        LinearFormat::E4M3BlockFp8(_)
    ));
    let mut root = serde_json::json!({"quantization_config":serde_json::from_str::<serde_json::Value>(include_str!("../fixtures/fp8-policy.json")).unwrap()});
    root["quantization_config"]["weight_block_size"] = serde_json::json!([1, 128]);
    assert!(schema::SafetensorsEncoding::from_json(&root).is_err());
}

#[test]
fn pinned_target_cold_requirements_cover_sources_banks_and_state_without_payload_reads() {
    use crate::qwen4_exp::{
        prepared::PreparedTarget,
        qsa::QsaExecutionLimits,
        target::{MixerSpec, TargetLimits, UnitSpec},
    };
    use eredu_runtime::*;
    struct Support;
    impl RowLookupMechanismSupport for Support {
        fn storage(&self) -> Option<AddressableStorageCapabilities> {
            Some(AddressableStorageCapabilities::new(
                true,
                true,
                true,
                64 << 20,
            ))
        }
        fn workspace(
            &self,
            _: &eredu_runtime::RowLookupDescriptor,
        ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
            Ok(Some(RowLookupWorkspace {
                decode_bytes: 4096,
                scalar_bytes: 16,
            }))
        }
    }
    let config = Config::from_json(
        &serde_json::from_str(include_str!("../../config/released.json")).unwrap(),
    )
    .unwrap();
    let limits = TargetLimits {
        history_tokens: 128,
        qsa: QsaExecutionLimits {
            batch: 1,
            tokens: 4,
            workspace_bytes: 1 << 24,
        },
        tile_blocks: 16,
        selection_workspace: 1 << 20,
        invocation_tokens: 4,
        lookup_rows: 64,
        element: TensorElementType::Bf16,
    };
    let row_limits = RowLookupLimits {
        requests: 64,
        rows_per_acquisition: 4,
        acquisition_bytes: 4096,
        host_bytes: 8192,
        output_bytes: 61440,
    };
    let options = ParameterBankLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(0), 1).unwrap(),
        1 << 20,
        1 << 20,
    )
    .unwrap();
    for fp8 in [false, true] {
        let mut source = published(fp8);
        let controls = source.constants.source_keys();
        for (name, metadata) in &mut source.metadata {
            if !controls.contains(name) {
                // Header-only model/table fixture provenance. Tiny controls
                // retain the actual temporary SafeTensors shard declared above.
                metadata.backing_shard = Some("header-fixture.safetensors".into());
            }
        }
        let source = Arc::new(source);
        let header = crate::qwen4_exp::prepared::SafetensorsTargetPlan::prepare(
            source.as_ref() as &dyn CheckpointSource,
            config.clone(),
            policy(fp8),
        )
        .unwrap();
        let cold_spec = header.target_spec(limits).unwrap();
        let streams: Vec<_> = cold_spec
            .units
            .iter()
            .enumerate()
            .flat_map(|(layer, unit)| {
                let UnitSpec::Decoder {
                    mixer: MixerSpec::Indexed(a),
                    ..
                } = unit
                else {
                    return vec![];
                };
                a.state
                    .streams()
                    .into_iter()
                    .map(|spec| AppendStreamBinding {
                        layer,
                        lanes: 1,
                        spec,
                        limits: AppendStreamLimits {
                            entries: 1024,
                            page_entries: 16,
                            read_entries: 16,
                        },
                        payload_bytes: 1 << 24,
                        scratch_bytes: 1 << 20,
                        catalog_bytes: 1 << 20,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let physical: BTreeMap<_, _> = header
            .resolution()
            .source_keys()
            .iter()
            .map(|key| {
                (
                    key.clone(),
                    crate::replicated_text::exact_physical_source(
                        source.as_ref() as &dyn CheckpointSource,
                        key,
                    )
                    .unwrap(),
                )
            })
            .collect();
        // The published artifacts contain MTP weights, but ordinary loading
        // prepares the target without requiring callers to disable drafting or
        // author internal workspace budgets. Both encodings stay header-only.
        assert!(config.prediction.is_some());
        let default = crate::qwen4_exp::prepared::normalize_load_request(
            &NormalizedLoadRequest::default(),
            &config,
        )
        .unwrap();
        assert_eq!(default.drafting(), DraftingLoadRequest::Disabled);
        let ordinary = header
            .execution_plan_for_load(
                &default,
                TensorElementType::Bf16,
                &Support,
                physical.clone(),
            )
            .unwrap();
        assert!(ordinary
            .requirements()
            .text()
            .auxiliary_parameters()
            .is_empty());
        assert!(ordinary
            .requirements()
            .text()
            .parameters()
            .iter()
            .all(|parameter| !parameter.name().starts_with("mtp.")));
        assert!(ordinary
            .capability_estimate()
            .speculative_draft_source()
            .is_none());
        assert!(ordinary
            .requirements()
            .text()
            .append_streams()
            .iter()
            .all(|stream| stream.limits.entries as u64
                >= config.max_positions as u64 / config.attention.ratio as u64));
        let embedded = default.with_drafting(DraftingLoadRequest::embedded(1).unwrap());
        assert!(matches!(
            header.execution_plan_for_load(
                &embedded,
                TensorElementType::Bf16,
                &Support,
                physical.clone(),
            ),
            Err(crate::qwen4_exp::prepared::TargetLoadError::PredictionPreparationRequired)
        ));
        assert!(source.reads.lock().unwrap().is_empty());
        let rows = eredu_runtime::SelectedRowLookupPlans::select(
            header
                .row_descriptors(
                    limits,
                    row_limits,
                    eredu_core::residency::ResidencyPolicy::Cacheable,
                )
                .unwrap(),
            options,
            16,
            &Support,
        )
        .unwrap();
        let cold_execution = header
            .execution_plan(limits, streams.clone(), rows, physical.clone())
            .unwrap();
        assert!(
            source.reads.lock().unwrap().is_empty(),
            "complete released BF16/FP8 requirements must not acquire controls or weights"
        );
        let prepared =
            PreparedTarget::safetensors(source.clone(), config.clone(), policy(fp8), limits)
                .unwrap();
        let reads_before_prediction = source.reads.lock().unwrap().len();
        let prediction = prepared
            .prediction(crate::qwen4_exp::mtp::PredictionLimits {
                qsa: limits.qsa,
                tile_blocks: limits.tile_blocks,
                selection_workspace: limits.selection_workspace,
                element: limits.element,
            })
            .unwrap();
        assert_eq!(source.reads.lock().unwrap().len(), reads_before_prediction);
        assert_eq!(prediction.spec().units.len(), 1);
        assert_eq!(prediction.spec().fusion.hidden_norm.dimensions, 10240);
        assert_eq!(prediction.spec().fusion.hidden_norm.groups, None);
        assert_eq!(prediction.spec().state_layout().unwrap().len(), 1);
        assert_eq!(
            prediction
                .spec()
                .state_layout()
                .unwrap()
                .layers()
                .get(0)
                .unwrap()
                .append_streams()
                .len(),
            2
        );
        for (name, recipe) in prediction.vocabulary().recipes() {
            assert_eq!(recipe, &prepared.static_parameters().recipes()[name]);
        }
        assert!(prediction
            .shared()
            .recipes()
            .keys()
            .all(|name| name.starts_with("mtp.") && !name.starts_with("mtp.layers.")));
        assert!(prediction
            .unit(0)
            .unwrap()
            .recipes()
            .keys()
            .all(|name| name.starts_with("mtp.layers.0.") && !name.contains(".mlp.experts.")));
        for expert in [0, 511] {
            for recipe in prediction.expert(0, expert).unwrap().recipes().values() {
                assert_eq!(
                    recipe.infer(prepared.artifact().as_ref()).unwrap().shape[0],
                    1
                );
            }
        }
        let select_rows = |scalar_limit| {
            eredu_runtime::SelectedRowLookupPlans::select(
                prepared
                    .row_lookups(
                        row_limits,
                        eredu_core::residency::ResidencyPolicy::Cacheable,
                    )
                    .unwrap()
                    .descriptors()
                    .clone(),
                options,
                scalar_limit,
                &Support,
            )
        };
        let execution = prepared
            .execution_plan(streams.clone(), select_rows(16).unwrap())
            .unwrap();
        assert_eq!(cold_execution.requirements(), execution.requirements());
        assert_eq!(
            cold_execution.capability_estimate(),
            execution.capability_estimate()
        );
        let before_joint = source.reads.lock().unwrap().len();
        let vision = prepared.vision().unwrap();
        let vision_layers = vision.config().layer_count();
        let vision_parameters = vision.static_parameters().recipes().len()
            + (0..vision_layers)
                .map(|i| vision.block(i).unwrap().recipes().len())
                .sum::<usize>();
        let joint = execution.clone().with_vision(vision).unwrap();
        assert_eq!(
            source.reads.lock().unwrap().len(),
            before_joint,
            "joint requirements read no additional payload"
        );
        assert_eq!(
            joint
                .requirements()
                .text()
                .execution_graph()
                .execution_order(),
            [1, 0]
        );
        assert_eq!(
            joint.requirements().text().execution_units().len(),
            prepared.spec().units.len() + vision_layers
        );
        assert_eq!(
            joint
                .requirements()
                .text()
                .parameters()
                .iter()
                .filter(|p| p.name().starts_with("model.visual."))
                .count(),
            vision_parameters
        );
        assert_eq!(
            joint.requirements().row_lookups(),
            execution.requirements().row_lookups(),
            "vision retains exact row ownership and sources"
        );
        let capability = execution.capability_estimate();
        assert_eq!(
            capability.capabilities().effective_model_type,
            "qwen4_exp_text"
        );
        assert_eq!(
            capability.capabilities().modalities,
            eredu_core::InputModalities::TEXT
        );
        assert_eq!(
            capability.speculative_draft_source(),
            None,
            "target role does not construct prediction weights"
        );
        assert_eq!(
            capability.capabilities().state_strategy,
            eredu_core::CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 12,
                sliding_attention: vec![],
                recurrent_layers: 36,
            }
        );
        assert_eq!(
            capability.state_layout().hidden_size,
            2560,
            "media input width precedes residual expansion"
        );
        // Independent released-geometry arithmetic, including the injection unit.
        let recurrent_fixed = 36 * (3 * (2 * 16 * 128 + 48 * 128) * 2 + 48 * 128 * 128 * 4);
        let qsa_partial = 12 * (6 * 4 + 3 * 128 * 2 + 2 * 64 * 4);
        let lexical_fixed = 2 * 4 + 9 * 4 * 2560 * 2;
        let position_delta = 4; // One exact I32 media rotary offset per target state.
        for positions in [0, 3, 4, 2047, 2048, 2049, 8195] {
            let state = eredu_core::estimate_runtime_state(
                capability.state_layout(),
                eredu_core::InputTokenCount::text(positions),
                0,
                1,
                std::num::NonZeroU8::new(2).unwrap(),
            )
            .unwrap();
            assert_eq!(
                state.fixed_state_bytes,
                recurrent_fixed + qsa_partial + lexical_fixed + position_delta
            );
            let kv = 12 * positions * 2 * 2 * 256 * 2;
            let summaries = 12 * (positions / 4) * (128 * 2 + 4 * 4);
            assert_eq!(
                state.context_state_bytes,
                kv + summaries,
                "full K/V grows past the sparse selection budget"
            );
            assert_eq!(
                state.bytes_per_position_per_batch,
                12 * (2 * 2 * 256 * 2 + (128 * 2 + 4 * 4) / 4)
            );
        }
        let requirements = execution.requirements();
        let text = requirements.text();
        let quantization = eredu_core::QuantizationRequest::Affine {
            group_size: 64,
            bits: 4,
        };
        let mut eligible = 0;
        for parameter in text.parameters() {
            let transform = parameter.transform_target(quantization).unwrap();
            if parameter.native_executable() != eredu_checkpoint::LinearFormat::Dense {
                assert!(
                    transform.is_none(),
                    "native encoded companions must remain atomic"
                );
            }
            if transform.is_some() {
                eligible += 1;
                assert!(parameter.transform_companions().is_some());
            }
        }
        assert!(eligible > 10, "released dense projections remain eligible");
        for bank in requirements.banks().values() {
            for unit in bank.catalog().units() {
                for parameter in unit.parameters() {
                    if matches!(parameter.binding_name(), "gate_up_proj" | "down_proj") {
                        assert_eq!(
                            matches!(
                                parameter.role(),
                                crate::ExpertParameterRole::QuantizableProjection { .. }
                            ),
                            !fp8
                        );
                    }
                }
            }
        }
        assert_eq!(
            text.architecture_identity(),
            prepared.spec().geometry_fingerprint()
        );
        assert_eq!(
            text.state_layout(),
            &prepared.spec().state_layout().unwrap()
        );
        assert_eq!(text.append_streams(), streams);
        assert_eq!(
            text.floating_state_source(),
            Some(&eredu_core::checkpoint::TensorDtype::Bf16)
        );
        let rows = requirements.row_lookups().unwrap();
        assert_eq!(rows.descriptors().entries().len(), 1);
        assert_eq!(
            rows.requirements().rows.source_bytes,
            320001536 * 160 * if fp8 { 1 } else { 2 }
        );
        let bank = requirements.bank(RoutedBankId::new(0)).unwrap();
        assert_eq!(
            bank.catalog().units().len(),
            config.layers.len() * config.experts.count as usize
        );
        let first = &bank.catalog().units()[0];
        assert_eq!(first.parameters().len(), if fp8 { 4 } else { 2 });
        let parameters: BTreeMap<_, _> = text.parameters().iter().map(|p| (p.name(), p)).collect();
        let gate = parameters["model.layers.0.mlp.experts.gate_up_proj"];
        assert_eq!(gate.logical_shape(), [512, 1280, 2560]);
        assert!(gate
            .physical_sources()
            .iter()
            .all(|s| s.shard() == std::path::Path::new("header-fixture.safetensors")));
        assert_eq!(text.derived_recipes().contains_key(gate.name()), fp8);
        if fp8 {
            let scales = parameters["model.layers.0.mlp.experts.gate_up_proj_scales"];
            assert_eq!(scales.logical_shape(), [512, 10, 20]);
            assert_eq!(
                scales.linear_companion(),
                Some((eredu_nn::LinearCompanionRole::Scale, gate.name()))
            );
            assert!(matches!(
                gate.native_executable(),
                eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
            ));
        }
        assert!(parameters.keys().all(|name| !name.starts_with("mtp.")
            && !name.starts_with("model.visual.")
            && !name.contains("ple_embedding")));
        let prediction_streams: Vec<_> = streams[..2]
            .iter()
            .cloned()
            .map(|mut s| {
                s.layer = 0;
                s
            })
            .collect();
        let reads = source.reads.lock().unwrap().len();
        let joint = execution
            .clone()
            .with_prediction(prediction.spec().limits, prediction_streams.clone())
            .unwrap();
        assert_eq!(source.reads.lock().unwrap().len(), reads);
        let auxiliary = joint.requirements().text().auxiliary_parameters();
        assert!(auxiliary.iter().all(|p| p.name().starts_with("mtp.")));
        assert!(auxiliary.iter().all(|p| p.auxiliary_residency().is_some()));
        let prediction_gate = auxiliary
            .iter()
            .find(|p| p.name() == "mtp.layers.0.mlp.experts.gate_up_proj")
            .unwrap();
        assert_eq!(prediction_gate.logical_shape(), [512, 1280, 2560]);
        assert_eq!(
            prediction_gate
                .transform_target(quantization)
                .unwrap()
                .is_some(),
            !fp8
        );
        let prediction_bank = joint.requirements().bank(RoutedBankId::new(2)).unwrap();
        assert_eq!(prediction_bank.catalog().units().len(), 512);
        assert_eq!(
            prediction_bank.catalog().units()[0].parameters().len(),
            if fp8 { 4 } else { 2 }
        );
        assert_eq!(joint.requirements().text().parameters(), text.parameters());
        let cold_conditional = cold_execution
            .with_prediction(prediction.spec().limits, prediction_streams)
            .unwrap()
            .with_vision(header.vision_plan(physical).unwrap())
            .unwrap();
        assert_eq!(
            cold_conditional.prediction_state_requirements(),
            joint.prediction_state_requirements(),
        );
        assert_eq!(
            cold_conditional.capability_estimate().state_layout(),
            joint.capability_estimate().state_layout(),
            "adding media preserves independent target and prediction state",
        );
        assert_eq!(
            cold_conditional
                .capability_estimate()
                .speculative_draft_source(),
            Some(eredu_core::SpeculativeDraftSource::Embedded),
        );
        assert!(
            cold_conditional
                .capability_estimate()
                .capabilities()
                .modalities
                .image
        );
        assert!(
            cold_conditional
                .capability_estimate()
                .capabilities()
                .modalities
                .video
        );
        assert_eq!(
            source.reads.lock().unwrap().len(),
            reads,
            "joint media and prediction preparation does not read weights"
        );
        let mut inadequate = streams.clone();
        inadequate[0].limits.read_entries = 15;
        assert!(prepared
            .execution_plan(inadequate, select_rows(16).unwrap())
            .is_err());
        assert!(prepared
            .execution_plan(vec![], select_rows(16).unwrap())
            .is_err());
        if fp8 {
            assert!(matches!(
                select_rows(0),
                Err(RowLookupSelectionError::Lookup(RowLookupError::Budget {
                    resource: "retained row scalars",
                    ..
                }))
            ));
        }
        assert_eq!(source.reads.lock().unwrap().len(), if fp8 { 4 } else { 3 });
    }
}

#[test]
fn pinned_prediction_role_admits_bf16_fp8_without_target_or_table_reads() {
    use crate::qwen4_exp::prepared::PreparedPredictionSource;
    let config = Config::from_json(
        &serde_json::from_str(include_str!("../../config/released.json")).unwrap(),
    )
    .unwrap();
    for fp8 in [false, true] {
        let source = Arc::new(published(fp8));
        let prepared =
            PreparedPredictionSource::safetensors(source.clone(), config.clone(), policy(fp8))
                .unwrap();
        assert!(!prepared.artifact().source_keys().is_empty());
        assert!(prepared
            .artifact()
            .source_keys()
            .iter()
            .all(|n| n.starts_with("mtp.")));
        assert!(source.reads.lock().unwrap().is_empty());
        for key in prepared.artifact().source_keys() {
            assert_eq!(
                prepared.artifact().source_provenance(&key).unwrap(),
                source.source_provenance(&key).unwrap()
            );
        }
        let mut missing = published(fp8);
        let key = missing
            .metadata
            .keys()
            .find(|n| {
                n.starts_with("mtp.")
                    && if fp8 {
                        n.ends_with("_scale_inv")
                    } else {
                        n.ends_with("fc_hidden.weight")
                    }
            })
            .unwrap()
            .clone();
        missing.metadata.remove(&key);
        let missing = Arc::new(missing);
        assert!(PreparedPredictionSource::safetensors(
            missing.clone(),
            config.clone(),
            policy(fp8)
        )
        .is_err());
        assert!(missing.reads.lock().unwrap().is_empty());
    }
}
