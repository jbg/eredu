use super::*;
use eredu_architectures::deepseek;
use eredu_runtime::layered::{PrefillObservationDeclaration, PreparedLayeredObservationPaths};

type V3State = DeviceState<NumericBackend, NumericHybridLayerState>;
type V3Architecture = deepseek::v3::Model<NumericBackend>;
type Model = ResidentRuntime<V3Architecture, NumericBackend, V3State>;

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
    fn requires_sequence_readout(&self) -> bool {
        self.declarations
            .iter()
            .any(|d| d.readout_stage() != Stage::BeforeReadout)
    }
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

fn assert_state(actual: &V3State, expected: &V3State) {
    assert_eq!(actual.layout(), expected.layout());
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (layer, (a, b)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        // The actual V3 layout has compressed state only. Do not omit another
        // mutable representation from equality merely because it is unexpected.
        assert!(a.fixed.is_empty() && b.fixed.is_empty());
        assert!(a.attention.is_none() && b.attention.is_none());
        assert!(a.pooling.is_none() && b.pooling.is_none());
        let a = a.compressed.as_ref().unwrap();
        let b = b.compressed.as_ref().unwrap();
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.block_size, b.block_size);
        assert_eq!(a.state.is_some(), b.state.is_some());
        if let (Some(a), Some(b)) = (&a.state, &b.state) {
            assert_tensor_close(&a.latent, &b.latent, &format!("layer {layer} latent"));
            assert_tensor_close(&a.rotary, &b.rotary, &format!("layer {layer} rotary"));
        }
    }
}

fn assert_populated(state: &V3State, config: &deepseek::V3Args, positions: i32, paged: bool) {
    assert_eq!(state.as_ref().len(), config.num_hidden_layers as usize);
    for layer in state.as_ref() {
        assert!(layer.fixed.is_empty());
        assert_eq!(layer.fixed_offset, 0);
        assert_eq!(layer.resets, 0);
        assert!(layer.attention.is_none() && layer.pooling.is_none());
        let cache = layer.compressed.as_ref().unwrap();
        assert_eq!(cache.offset, positions);
        assert_eq!(cache.block_size, paged.then_some(2));
        let retained = cache.state.as_ref().unwrap();
        for (tensor, width) in [
            (&retained.latent, config.kv_lora_rank),
            (&retained.rotary, config.qk_rope_head_dim),
        ] {
            assert_eq!(tensor.shape, [1, positions, width]);
            assert!(tensor.data.iter().all(|v| v.is_finite()));
            assert!(tensor.data.iter().any(|v| v.abs() > 1e-9));
        }
    }
}

fn make_state(config: &deepseek::V3Args, paged: bool) -> V3State {
    V3State::create(deepseek::v3::state_layout(config).unwrap(), |_, policy| {
        let mut layer = NumericHybridLayerState::new(policy);
        if paged {
            *layer.compressed.as_mut().unwrap() = NumericCompressedCache::paged(2);
        }
        Ok::<_, Error>(layer)
    })
    .unwrap()
}

// V3's projections, norms and correction bias are built by NumericBackend
// factories, which initialize actual tensors. There are no raw conv/scan controls.
// Verify the full declared parameter set and every actual value before creating
// a path token or state; a future zero/unvisited raw parameter must fail here.
fn assert_loaded_parameters(model: &Model, config: &deepseek::V3Args) {
    #[derive(Default)]
    struct Loaded(BTreeMap<String, Vec<usize>>);
    impl<'a> ParameterVisitor<'a, NumericTensor> for Loaded {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a NumericTensor) {
            let name = metadata.id().as_str();
            assert!(
                value.data.iter().all(|v| v.is_finite()),
                "finite parameter {name}"
            );
            assert!(
                value.data.iter().any(|v| v.abs() > 1e-9),
                "nonzero parameter {name}"
            );
            assert!(
                self.0
                    .insert(
                        name.into(),
                        value
                            .shape
                            .iter()
                            .map(|dimension| usize::try_from(*dimension).unwrap())
                            .collect()
                    )
                    .is_none(),
                "duplicate physical parameter {name}"
            );
        }
    }
    let mut actual = Loaded::default();
    model
        .architecture()
        .static_modules()
        .visit_parameters(&mut actual);
    for unit in model.units().iter().flatten() {
        unit.visit_parameters(&mut actual);
    }
    let description = deepseek::parallel::v3_parameter_description(config).unwrap();
    let expected = description
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .map(|member| (member.target().to_owned(), member.global_shape().to_vec()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(actual.0, expected);
}

fn make_model(
    config: &deepseek::V3Args,
    context: &NumericContext,
) -> (
    Model,
    Vec<PrefillObservationDeclaration>,
    PreparedLayeredObservationPaths,
) {
    let architecture = V3Architecture::new(config.clone(), context).unwrap();
    let declarations = <V3Architecture as LayeredArchitecture<NumericBackend, V3State>>::
        prefill_observation_declarations(&architecture, None).unwrap();
    // Sparse carrier hooks have their own typed collector. This fixture
    // compares every actual dense tensor declaration, including the new
    // local units and complete/additive writes, with whole-request execution.
    let declarations: Vec<_> = declarations.into_iter()
        .filter(|declaration| !declaration.flattens_batch_tokens()).collect();
    assert!(declarations.len() >= 10 + 8 * config.num_hidden_layers as usize);
    for index in 0..config.num_hidden_layers as usize {
        let path = <V3Architecture as LayeredArchitecture<NumericBackend, V3State>>::unit_path(
            &architecture,
            0,
            index, None)
        .unwrap();
        let component = match config.layer_schedule.get(index).expect("validated target layer policy") {
            deepseek::LayerPolicy::DenseMlp => format!("{path}.feed_forward"),
            deepseek::LayerPolicy::SparseMoe => format!("{path}.feed_forward.shared"),
        };
        for suffix in ["units", "units.effective", "write", "write.effective"] {
            assert!(declarations.iter().any(|d| d.path() == format!("{component}.{suffix}")));
        }
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert!(declarations
                .iter()
                .any(|d| d.path() == format!("{path}.{suffix}")));
        }
    }
    let runtime = ResidentRuntime::new(architecture, context).unwrap();
    assert_loaded_parameters(&runtime, config);
    let paths = runtime.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        assert_eq!(
            paths.source().prefill_observation(declaration.path()),
            Some(declaration)
        );
    }
    (runtime, declarations, paths)
}

fn run(
    model: &mut Model,
    ids: &[usize],
    state: &mut V3State,
    context: &NumericContext,
    rows: &mut DeclaredRows<'_>,
    paths: &PreparedLayeredObservationPaths,
    demand: OutputDemand,
) -> Option<NumericTensor> {
    let tokens = NumericTensor::token_ids(ids);
    let input = deepseek::mtp::EmbeddedInput::target(&tokens, None);
    assert_eq!(
        <V3Architecture as LayeredArchitecture<NumericBackend, V3State>>::inference_input_shape(
            &input
        )
        .unwrap(),
        Some([1, ids.len() as u64])
    );
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            input, state, context, rows, paths, demand,
        )
        .unwrap();
    // The enclosing runtime publishes the final vocabulary scores. All other
    // original/effective observations come directly from actual model traversal.
    scores.map(|scores| eredu_runtime::observe_model_logits(rows, &scores).unwrap())
}

fn compare_equations(value: serde_json::Value, paged: bool) {
    let config = deepseek::parse_v3_config(&value).unwrap();
    let context = NumericContext::default();
    let (mut model, declarations, paths) = make_model(&config, &context);
    let mut full_state = make_state(&config, paged);
    run(
        &mut model,
        &[4, 2],
        &mut full_state,
        &context,
        &mut DeclaredRows::new(&[]),
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    let prefix = full_state.clone();
    assert_populated(&prefix, &config, 2, paged);
    let mut split_state = prefix.clone();
    let mut full = DeclaredRows::new(&declarations);
    let scores = run(
        &mut model,
        &[1, 3, 5, 2, 6],
        &mut full_state,
        &context,
        &mut full,
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    assert_eq!(scores.shape, [1, 5, config.vocab_size]);
    full.assert_complete(5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = DeclaredRows::new(&declarations);
        let scores = run(
            &mut model,
            ids,
            &mut split_state,
            &context,
            &mut rows,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        assert_eq!(scores.shape, [1, ids.len() as i32, config.vocab_size]);
        rows.assert_complete(ids.len() as i32);
        consumed.extend_from_slice(ids);
        let mut direct = prefix.clone();
        run(
            &mut model,
            &consumed,
            &mut direct,
            &context,
            &mut DeclaredRows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        assert_state(&split_state, &direct);
        assert_populated(&split_state, &config, 2 + consumed.len() as i32, paged);
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
    assert_state(&split_state, &full_state);
    for id in [7, 8, 9] {
        let mut expected = DeclaredRows::new(&declarations);
        let mut actual = DeclaredRows::new(&declarations);
        for (state, rows) in [
            (&mut full_state, &mut expected),
            (&mut split_state, &mut actual),
        ] {
            let scores = run(
                &mut model,
                &[id],
                state,
                &context,
                rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
            assert_eq!(scores.shape, [1, 1, config.vocab_size]);
            rows.assert_complete(1);
        }
        for declaration in &declarations {
            let path = declaration.path();
            assert_tensor_close(&actual.values[path], &expected.values[path], path);
        }
        assert_state(&split_state, &full_state);
    }
    assert_populated(&full_state, &config, 10, paged);
}

// Same nonzero MLA geometry as components/deepseek_v3.rs, with two target
// layers so the routed variant exercises a dense prefix and grouped experts.
fn configuration(routed: bool, query_rank: Option<i32>, yarn: bool) -> serde_json::Value {
    let mut value = serde_json::json!({
        "model_type":"deepseek_v3", "hidden_size":8, "intermediate_size":10,
        "moe_intermediate_size":4, "num_hidden_layers":2, "num_attention_heads":2,
        "vocab_size":16, "max_position_embeddings":64, "q_lora_rank":query_rank,
        "kv_lora_rank":3, "qk_nope_head_dim":2, "qk_rope_head_dim":2, "v_head_dim":3,
        "first_k_dense_replace":if routed {1} else {2}, "n_routed_experts":4,
        "n_shared_experts":1, "num_experts_per_tok":2, "n_group":2, "topk_group":1,
        "routed_scaling_factor":1.3, "norm_topk_prob":true, "tie_word_embeddings":false
    });
    if yarn {
        value["rope_scaling"] = serde_json::json!({
            "type":"yarn", "factor":2.0, "original_max_position_embeddings":4,
            "beta_fast":32.0, "beta_slow":1.0, "mscale":1.0, "mscale_all_dim":0.5
        });
    }
    value
}

#[test]
fn deepseek_v3_dense_rows_preserve_all_real_hooks_and_complete_compressed_state() {
    for query_rank in [None, Some(3)] {
        for yarn in [false, true] {
            for paged in [false, true] {
                compare_equations(configuration(false, query_rank, yarn), paged);
            }
        }
    }
}

#[test]
fn deepseek_v3_routed_rows_preserve_token_local_experts_and_blockwise_attention() {
    for query_rank in [None, Some(3)] {
        for yarn in [false, true] {
            for paged in [false, true] {
                compare_equations(configuration(true, query_rank, yarn), paged);
            }
        }
    }
}

#[test]
fn deepseek_v3_prepared_paths_bind_actual_sources_and_physical_readout() {
    for routed in [false, true] {
        let value = configuration(routed, Some(3), false);
        let (artifact, _) = prepared_adapter::payload_fixture_config(&value, 1.0);
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
        let config = deepseek::parse_v3_config(&value).unwrap();
        let context = NumericContext::default();
        let (_, declarations, paths) = make_model(&config, &context);
        let (_, _, independent_paths) = make_model(&config, &context);
        let before = sources.target().source_diagnostics().unwrap();
        assert_eq!(declarations.iter().filter(|d| !d.flattens_batch_tokens()).count(), 26);
        for declaration in &declarations {
            for preview in [false, true] {
                let source = admission(&discovery, &[declaration.path()], preview, true);
                let selected = discovery
                    .prepare_capture_selection(&source, paths.source())
                    .unwrap();
                assert!(selected.source().same_storage(&source));
                assert!(selected.paths().same_storage(paths.source()));
                assert_eq!(selected.declaration(0).unwrap(), Some(declaration));
                selected
                    .validate_sources(&source.clone(), &paths.source().clone())
                    .unwrap();
                assert!(matches!(
                    selected.validate_sources(&source, independent_paths.source()),
                    Err(SelectionError::Identity)
                ));
                assert_eq!(
                    selected.physical_output(OutputDemand::StateOnly),
                    if declaration.readout_stage() == Stage::BeforeReadout {
                        OutputDemand::StateOnly
                    } else {
                        OutputDemand::Sequence
                    }
                );
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
        for path in ["model.layers.0.attention.input"] {
            assert!(paths.source().prefill_observation(path).is_none());
            let source = admission(&discovery, &[path], false, true);
            assert!(matches!(
                discovery.prepare_capture_selection(&source, paths.source()),
                Err(SelectionError::Undeclared { index: 0 })
            ));
        }
        assert_eq!(
            before.physical_reads,
            sources
                .target()
                .source_diagnostics()
                .unwrap()
                .physical_reads
        );
    }
}

#[test]
fn deepseek_v3_body_rows_keep_positions_before_state_only_and_last_position_readout() {
    for routed in [false, true] {
        let config = deepseek::parse_v3_config(&configuration(routed, Some(3), true)).unwrap();
        let context = NumericContext::default();
        let (mut model, declarations, paths) = make_model(&config, &context);
        let body = declarations
            .into_iter()
            .filter(|d| d.readout_stage() == Stage::BeforeReadout)
            .collect::<Vec<_>>();
        assert!(body.len() >= 10);
        let mut prefix = make_state(&config, true);
        run(
            &mut model,
            &[4, 2],
            &mut prefix,
            &context,
            &mut DeclaredRows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        let mut expected_state = prefix.clone();
        let mut expected = DeclaredRows::new(&body);
        let full = run(
            &mut model,
            &[1, 3, 5, 2, 6],
            &mut expected_state,
            &context,
            &mut expected,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        expected.assert_complete(5);
        for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
            let mut state = prefix.clone();
            let mut start = 0;
            for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
                let end = start + ids.len() as i32;
                let mut actual = DeclaredRows::new(&body);
                assert!(!actual.requires_sequence_readout());
                let output = run(
                    &mut model,
                    ids,
                    &mut state,
                    &context,
                    &mut actual,
                    &paths,
                    demand,
                );
                actual.assert_complete(ids.len() as i32);
                for declaration in &body {
                    let path = declaration.path();
                    let expected = expected.values[path]
                        .index(
                            &[Index::Full, Index::Range(start, end), Index::Full],
                            &context,
                        )
                        .unwrap();
                    assert_tensor_close(&actual.values[path], &expected, path);
                }
                if demand == OutputDemand::StateOnly {
                    assert!(output.is_none());
                } else {
                    let last = full
                        .index(
                            &[Index::Full, Index::Range(end - 1, end), Index::Full],
                            &context,
                        )
                        .unwrap();
                    assert_tensor_close(&output.unwrap(), &last, "last prediction");
                }
                assert_populated(&state, &config, 2 + end, true);
                start = end;
            }
            assert_state(&state, &expected_state);
        }
    }
}

#[test]
fn deepseek_v3_target_declarations_preserve_existing_prediction_availability() {
    use eredu_runtime::inspection::ObservationHookSite;
    let context = NumericContext::default();
    for prediction_layers in [0, 1] {
        let mut value = configuration(true, Some(3), false);
        value["num_nextn_predict_layers"] = prediction_layers.into();
        let args = deepseek::parse_v3_config(&value).unwrap();
        let architecture = V3Architecture::new(args.clone(), &context).unwrap();
        let hooks =
            <V3Architecture as LayeredArchitecture<NumericBackend, V3State>>::observation_hooks(
                &architecture,
            );
        for site in [
            ObservationHookSite::Input,
            ObservationHookSite::Unit,
            ObservationHookSite::Readout,
        ] {
            assert_eq!(hooks.supports(site), prediction_layers == 0);
        }
        let declarations = <V3Architecture as LayeredArchitecture<NumericBackend, V3State>>::
            prefill_observation_declarations(&architecture, None).unwrap();
        assert_eq!(declarations.iter().filter(|d| !d.flattens_batch_tokens()).count(), 26);
        assert!(declarations
            .iter()
            .all(|d| !d.path().starts_with("model.layers.2.") && !d.path().starts_with("mtp.")));
        let layout = deepseek::v3::state_layout(&args).unwrap();
        assert_eq!(layout.len(), 2 + prediction_layers as usize);
        // Existing availability is authoritative: row declarations do not turn
        // an embedded prediction bank into an observable ordinary target run.
        if prediction_layers > 0 {
            let prediction_path = <V3Architecture as LayeredArchitecture<
                NumericBackend,
                V3State,
            >>::unit_path(&architecture, 1, 0, None)
            .unwrap();
            assert!(declarations
                .iter()
                .all(|d| !d.path().starts_with(&format!("{prediction_path}."))));
        }
    }
}


#[test]
fn dense_and_shared_component_rows_match_full_cached_request_before_partition_delivery() {
    for routed in [false, true] {
        compare_equations(configuration(routed, Some(3), false), false);
    }
}
