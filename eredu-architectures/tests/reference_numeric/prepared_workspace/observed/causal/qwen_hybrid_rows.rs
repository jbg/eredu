use super::*;
use eredu_architectures::qwen::hybrid;
use eredu_runtime::layered::PrefillObservationDeclaration;

type HybridState = DeviceState<NumericBackend, NumericHybridLayerState>;
type HybridModel = hybrid::LayeredModel<NumericBackend>;

// Every declared value, including effective companions, must be emitted by the
// real component/traversal observer. This collector never manufactures aliases.
struct DeclaredRows<'a> {
    declarations: &'a [PrefillObservationDeclaration],
    values: BTreeMap<String, NumericTensor>,
}
impl<'a> DeclaredRows<'a> {
    fn new(declarations: &'a [PrefillObservationDeclaration]) -> Self {
        Self {
            declarations,
            values: BTreeMap::new(),
        }
    }
    fn record(&mut self, path: &str, value: &NumericTensor) {
        if self.declarations.iter().any(|d| d.path() == path) {
            assert!(
                self.values.insert(path.into(), value.clone()).is_none(),
                "duplicate {path}"
            );
        }
    }
    fn assert_complete(&self, positions: i32) {
        assert_eq!(self.values.len(), self.declarations.len());
        for declaration in self.declarations {
            let path = declaration.path();
            let value = self
                .values
                .get(path)
                .unwrap_or_else(|| panic!("missing {path}"));
            assert_eq!(declaration.sequence_axis(), 1);
            assert_eq!(value.shape.len(), 3, "{path}");
            assert_eq!(value.shape[..2], [1, positions], "{path}");
            assert!(value.data.iter().all(|v| v.is_finite()), "finite {path}");
            assert!(value.data.iter().any(|v| v.abs() > 1e-6), "nonzero {path}");
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for DeclaredRows<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.record(path, value);
        Ok(())
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &NumericTensor,
        _: &GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<NumericTensor, Error>,
    ) -> Result<(), Error> {
        assert!(self.declarations.iter().all(|d| d.path() != path));
        Ok(())
    }
}

fn assert_hybrid_state(actual: &HybridState, expected: &HybridState) {
    assert_eq!(actual.layout(), expected.layout());
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (layer, (a, b)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        for (role, a) in &a.fixed {
            let b = &b.fixed[role];
            assert_eq!(a.is_some(), b.is_some(), "layer {layer} {role:?}");
            if let (Some(a), Some(b)) = (a, b) {
                assert_tensor_close(a, b, &format!("layer {layer} {role:?}"));
            }
        }
        // These ordinary configurations have only the declared recurrent fixed
        // slots or contiguous full KV storage; absence is checked explicitly.
        assert!(a.compressed.is_none() && b.compressed.is_none());
        assert!(a.pooling.is_none() && b.pooling.is_none());
        assert_eq!(a.attention.is_some(), b.attention.is_some());
        if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.window, b.window);
            assert!(a.attention_history.is_none() && b.attention_history.is_none());
            for (name, a, b) in [("keys", &a.keys, &b.keys), ("values", &a.values, &b.values)] {
                assert_eq!(a.is_some(), b.is_some());
                if let (Some(a), Some(b)) = (a, b) {
                    assert_tensor_close(a, b, &format!("layer {layer} {name}"));
                }
            }
        }
    }
}

fn assert_populated(state: &HybridState, config: &hybrid::HybridConfig, positions: i32) {
    assert_eq!(state.as_ref().len(), 2);
    let fixed = &state.as_ref()[0];
    assert!(fixed.attention.is_none());
    assert_eq!(fixed.fixed_offset, positions);
    let history = config.linear_conv_kernel_dim - 1;
    assert_eq!(fixed.fixed.len(), 1 + usize::from(history > 0));
    if history == 0 {
        assert!(!fixed
            .fixed
            .contains_key(&StateTensorRole::Convolution { slot: 0 }));
    } else {
        let convolution = fixed.fixed[&StateTensorRole::Convolution { slot: 0 }]
            .as_ref()
            .unwrap();
        let channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        assert_eq!(convolution.shape, [1, history, channels]);
        assert!(convolution.data.iter().all(|v| v.is_finite()));
        assert!(convolution.data.iter().any(|v| *v != 0.0));
    }
    let recurrent = fixed.fixed[&StateTensorRole::Recurrent].as_ref().unwrap();
    assert_eq!(
        recurrent.shape,
        [
            1,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim
        ]
    );
    assert!(recurrent.data.iter().all(|v| v.is_finite()));
    assert!(recurrent.data.iter().any(|v| *v != 0.0));
    let attention = &state.as_ref()[1];
    assert!(attention.fixed.is_empty());
    assert_eq!(attention.fixed_offset, 0);
    let attention = attention.attention.as_ref().unwrap();
    assert_eq!(attention.offset, positions);
    assert_eq!(attention.window, None);
    for tensor in [&attention.keys, &attention.values] {
        let tensor = tensor.as_ref().unwrap();
        assert_eq!(
            tensor.shape,
            [1, config.num_key_value_heads, positions, config.head_dim]
        );
        assert!(tensor.data.iter().all(|v| v.is_finite()));
        assert!(tensor.data.iter().any(|v| *v != 0.0));
    }
}

// Shared raw checkpoint values for direct and selected edge-case fixtures.
pub(super) fn recurrent_parameter_pattern(name: &str) -> Option<(f32, f32)> {
    let (_, field) = name.rsplit_once(".linear_attn.")?;
    match field {
        "conv1d.weight" => Some((0.15, 0.01)),
        "dt_bias" => Some((-1.0, 0.01)),
        "A_log" => Some((-0.6, 0.01)),
        "norm.weight" => Some((1.0, 0.001)),
        _ => None,
    }
}

// Factory-created linear/embedding weights are already deterministic. Raw
// Parameter::unloaded slots use NumericTensor::unloaded_f32 (all zeros), so
// load the actual convolution/scan controls before any state or path token.
// Mutable state itself is never seeded or replaced by this fixture.
fn load_recurrent_parameters<A>(
    model: &mut ResidentRuntime<A, NumericBackend, HybridState>,
    expected_parameters: usize,
) where
    A: LayeredArchitecture<NumericBackend, HybridState, Error = Error>,
    A::Unit: Parameterized<NumericTensor>,
{
    struct Load {
        visited: BTreeMap<String, ()>,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let Some((base, step)) = recurrent_parameter_pattern(metadata.id().as_str()) else {
                return;
            };
            assert!(self
                .visited
                .insert(metadata.id().as_str().into(), ())
                .is_none());
            assert!(
                value.data.iter().all(|value| *value == 0.0),
                "unloaded {}",
                metadata.id().as_str()
            );
            for (index, value) in value.data.iter_mut().enumerate() {
                *value = base + step * (index % 7) as f32;
            }
        }
    }
    let mut loader = Load {
        visited: BTreeMap::new(),
    };
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut loader);
    }
    assert_eq!(loader.visited.len(), expected_parameters);
}

fn compare_equations(config: serde_json::Value) {
    let args = hybrid::model_args_from_config_value(&config).unwrap().text;
    let layout = hybrid::state_layout(&args).unwrap();
    let vocabulary = args.vocab_size;
    let context = NumericContext::default();
    let architecture = HybridModel::new(args.clone(), &context).unwrap();
    let declarations = tensor_row_declarations(<HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
        prefill_observation_declarations(&architecture, None).unwrap());
    assert!(declarations.len() >= 18);
    let mut model = ResidentRuntime::new(architecture, &context).unwrap();
    // The fixture has one gated-delta unit and one attention unit.
    load_recurrent_parameters(&mut model, 4);
    let paths = model.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        assert_eq!(
            paths.source().prefill_observation(declaration.path()),
            Some(declaration)
        );
    }
    let mut full_state = HybridState::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    model
        .forward(
            hybrid::EmbeddedInput::target(&NumericTensor::token_ids(&[4, 2]), None),
            &mut full_state,
            &context,
        )
        .unwrap();
    let prefix_state = full_state.clone();
    assert_populated(&prefix_state, &args, 2);
    let mut chunk_state = full_state.clone();
    let mut full = DeclaredRows::new(&declarations);
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            hybrid::EmbeddedInput::target(&NumericTensor::token_ids(&[1, 3, 5, 2, 6]), None),
            &mut full_state,
            &context,
            &mut full,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
    let scores = eredu_runtime::observe_model_logits(&mut full, &scores.unwrap()).unwrap();
    assert_eq!(scores.shape, [1, 5, vocabulary]);
    full.assert_complete(5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = DeclaredRows::new(&declarations);
        let (scores, _) = model
            .forward_with_prepared_observer_and_context_with_readout(
                hybrid::EmbeddedInput::target(&NumericTensor::token_ids(ids), None),
                &mut chunk_state,
                &context,
                &mut rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
        let scores = eredu_runtime::observe_model_logits(&mut rows, &scores.unwrap()).unwrap();
        assert_eq!(scores.shape, [1, ids.len() as i32, vocabulary]);
        rows.assert_complete(ids.len() as i32);
        consumed.extend_from_slice(ids);
        let mut direct_prefix = prefix_state.clone();
        model
            .forward(
                hybrid::EmbeddedInput::target(&NumericTensor::token_ids(&consumed), None),
                &mut direct_prefix,
                &context,
            )
            .unwrap();
        assert_hybrid_state(&chunk_state, &direct_prefix);
        assert_populated(&chunk_state, &args, 2 + consumed.len() as i32);
        for declaration in &declarations {
            assembled
                .entry(declaration.path().into())
                .or_default()
                .extend_from_slice(&rows.values[declaration.path()].data);
        }
    }
    assert_eq!(assembled.len(), declarations.len());
    for declaration in &declarations {
        let path = declaration.path();
        let expected = &full.values[path];
        assert_tensor_close(
            &NumericTensor::new(expected.shape.clone(), assembled.remove(path).unwrap()),
            expected,
            path,
        );
    }
    assert_hybrid_state(&chunk_state, &full_state);
    for id in [7, 8, 9] {
        let tokens = NumericTensor::token_ids(&[id]);
        let mut expected = DeclaredRows::new(&declarations);
        let mut actual = DeclaredRows::new(&declarations);
        for (state, rows) in [
            (&mut full_state, &mut expected),
            (&mut chunk_state, &mut actual),
        ] {
            let (scores, _) = model
                .forward_with_prepared_observer_and_context_with_readout(
                    hybrid::EmbeddedInput::target(&tokens, None),
                    state,
                    &context,
                    rows,
                    &paths,
                    OutputDemand::Sequence,
                )
                .unwrap();
            let scores = eredu_runtime::observe_model_logits(rows, &scores.unwrap()).unwrap();
            assert_eq!(scores.shape, [1, 1, vocabulary]);
            rows.assert_complete(1);
        }
        for declaration in &declarations {
            let path = declaration.path();
            assert_tensor_close(&actual.values[path], &expected.values[path], path);
        }
        assert_hybrid_state(&chunk_state, &full_state);
    }
    assert!(full_state
        .as_ref()
        .iter()
        .all(|state| state.position() == 10));
}

pub(super) fn configuration(kind: &str, routed: bool, tied: bool) -> serde_json::Value {
    let mut config = heterogeneous_replicated_configs()
        .into_iter()
        .find(|config| config["model_type"] == kind)
        .unwrap();
    config["tie_word_embeddings"] = tied.into();
    if routed {
        if kind == "qwen3_5_text" {
            config["model_type"] = "qwen3_5_moe_text".into();
        }
        config["num_experts"] = 4.into();
        config["num_experts_per_tok"] = 2.into();
        config["moe_intermediate_size"] = 6.into();
        config["shared_expert_intermediate_size"] = 8.into();
        config["norm_topk_prob"] = true.into();
    }
    config
}

#[test]
fn qwen_hybrid_dense_rows_preserve_complete_recurrence_and_kv_across_chunks() {
    for kind in ["qwen3_next", "qwen3_5_text"] {
        compare_equations(configuration(kind, false, false));
        let mut expanded = configuration(kind, false, true);
        expanded["linear_conv_kernel_dim"] = 4.into();
        expanded["linear_num_value_heads"] = 4.into();
        expanded["linear_value_head_dim"] = 3.into();
        compare_equations(expanded);
    }
}

#[test]
fn qwen_hybrid_routed_rows_preserve_shared_and_sparse_outputs_across_chunks() {
    for kind in ["qwen3_next", "qwen3_5_text"] {
        for normalized in [false, true] {
            let mut config = configuration(kind, true, normalized);
            config["norm_topk_prob"] = normalized.into();
            compare_equations(config);
        }
    }
}

#[test]
fn qwen_hybrid_rows_preserve_fixed_rotary_algorithms_after_a_cached_prefix() {
    for rope in [
        serde_json::json!({"rope_type":"linear", "factor":2.0}),
        serde_json::json!({"rope_type":"proportional", "factor":2.0,
            "partial_rotary_factor":0.5}),
        serde_json::json!({"rope_type":"yarn", "factor":2.0,
            "original_max_position_embeddings":16, "beta_fast":8.0, "beta_slow":1.0,
            "mscale":1.0, "mscale_all_dim":0.0}),
        serde_json::json!({"rope_type":"llama3", "factor":2.0,
            "low_freq_factor":1.0, "high_freq_factor":4.0,
            "original_max_position_embeddings":16}),
    ] {
        let mut config = configuration("qwen3_5_text", false, false);
        config["rope_parameters"] = rope;
        compare_equations(config);
    }
}

#[test]
fn qwen_hybrid_actual_sources_bind_all_target_rows_and_original_physical_readout() {
    for (kind, routed) in [
        ("qwen3_next", false),
        ("qwen3_5_text", false),
        ("qwen3_5_text", true),
    ] {
        let config = configuration(kind, routed, false);
        let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(None),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let discovery = sources.prepare_discovery(
            ObservationMechanisms {
                activation_tensors: true,
                floating_to_f32: true,
                ..Default::default()
            },
            CaptureCapabilities {
                transformations: vec![
                    CaptureTransformKind::FullTensor,
                    CaptureTransformKind::Preview,
                ],
                ..Default::default()
            },
        );
        let args = hybrid::model_args_from_config_value(&config).unwrap().text;
        let context = NumericContext::default();
        let architecture = HybridModel::new(args, &context).unwrap();
        let declarations = tensor_row_declarations(<HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
            prefill_observation_declarations(&architecture, None).unwrap());
        assert!(declarations.len() >= 18);
        let runtime =
            ResidentRuntime::<_, NumericBackend, HybridState>::new(architecture, &context).unwrap();
        let paths = runtime.prepare_observation_paths().unwrap();
        let before = sources.target().source_diagnostics().unwrap();
        for declaration in &declarations {
            for preview in [false, true] {
                let source = admission(&discovery, &[declaration.path()], preview, true);
                let selected = discovery
                    .prepare_capture_selection(&source, paths.source())
                    .unwrap();
                assert!(selected.source().same_storage(&source));
                assert!(selected.paths().same_storage(paths.source()));
                assert_eq!(selected.declaration(0).unwrap(), Some(declaration));
                let mut geometry = request(4);
                geometry.prefill_chunk_positions = 2;
                geometry.output = selected.physical_output(OutputDemand::LastPosition);
                let expected = match declaration.readout_stage() {
                    Stage::BeforeReadout => OutputDemand::LastPosition,
                    Stage::ReadoutInput | Stage::VocabularyScores => OutputDemand::Sequence,
                };
                assert_eq!(geometry.output, expected);
                let bound = selected.bind_geometry(geometry).unwrap();
                assert_eq!(bound.geometry(), geometry);
                assert!(std::ptr::eq(bound.selection(), &selected));
                if expected == OutputDemand::Sequence {
                    geometry.output = OutputDemand::LastPosition;
                    assert!(matches!(
                        selected.bind_geometry(geometry),
                        Err(SelectionError::Readout)
                    ));
                }
                let independent = admission(&discovery, &[declaration.path()], preview, true);
                assert!(matches!(
                    selected.validate_sources(&independent, paths.source()),
                    Err(SelectionError::Identity)
                ));
            }
        }
        for path in [
            "model.layers.0.mixer.channels",
            "model.layers.1.attention.channels",
        ] {
            assert!(paths.source().prefill_observation(path).is_none());
            let source = admission(&discovery, &[path], false, true);
            assert!(matches!(
                discovery.prepare_capture_selection(&source, paths.source()),
                Err(SelectionError::Undeclared { index: 0 })
            ));
        }
        let after = sources.target().source_diagnostics().unwrap();
        assert_eq!(before.physical_reads, after.physical_reads);
    }
}

#[test]
fn qwen_hybrid_target_evidence_preserves_existing_mtp_availability_gate() {
    use eredu_runtime::inspection::ObservationHookSite;
    let mut config = configuration("qwen3_5_text", false, false);
    config["mtp_num_hidden_layers"] = 1.into();
    let args = hybrid::model_args_from_config_value(&config).unwrap().text;
    let context = NumericContext::default();
    let architecture = HybridModel::new(args, &context).unwrap();
    let declarations = tensor_row_declarations(<HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
        prefill_observation_declarations(&architecture, None).unwrap());
    assert!(declarations.len() >= 18);
    assert!(declarations
        .iter()
        .all(|point| !point.path().starts_with("mtp.")));
    let hooks =
        <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::observation_hooks(
            &architecture,
        );
    for site in [
        ObservationHookSite::Input,
        ObservationHookSite::Unit,
        ObservationHookSite::Readout,
    ] {
        assert!(!hooks.supports(site));
    }
}

#[path = "qwen_hybrid_rows/width_one.rs"]
mod width_one;
