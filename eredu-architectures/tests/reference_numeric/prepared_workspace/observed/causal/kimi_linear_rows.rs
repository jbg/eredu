use super::*;
use eredu_architectures::kimi_linear;
use eredu_runtime::layered::{PrefillObservationDeclaration, PreparedLayeredObservationPaths};
use eredu_runtime::ArchitectureParameters;

type KimiState = DeviceState<NumericBackend, NumericHybridLayerState>;
type KimiArchitecture = kimi_linear::LayeredModel<NumericBackend>;
type Model = ResidentRuntime<KimiArchitecture, NumericBackend, KimiState>;

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

fn assert_state_tensor(actual: &NumericTensor, expected: &NumericTensor, label: &str) {
    assert_eq!(actual.dtype, expected.dtype, "{label} dtype");
    assert_eq!(actual.data.len(), expected.data.len(), "{label} storage");
    assert_tensor_close(actual, expected, label);
}

fn assert_state(actual: &KimiState, expected: &KimiState) {
    assert_eq!(actual.layout(), expected.layout());
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (layer, (a, b)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert!(a.attention.is_none() && b.attention.is_none());
        assert!(a.pooling.is_none() && b.pooling.is_none());
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        for (role, value) in &a.fixed {
            let expected = &b.fixed[role];
            assert_eq!(
                value.is_some(),
                expected.is_some(),
                "layer {layer} {role:?}"
            );
            if let (Some(value), Some(expected)) = (value, expected) {
                assert_state_tensor(value, expected, &format!("layer {layer} {role:?}"));
            }
        }
        assert_eq!(a.compressed.is_some(), b.compressed.is_some());
        if let (Some(a), Some(b)) = (&a.compressed, &b.compressed) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.block_size, b.block_size);
            assert_eq!(a.state.is_some(), b.state.is_some());
            if let (Some(a), Some(b)) = (&a.state, &b.state) {
                assert_state_tensor(&a.latent, &b.latent, &format!("layer {layer} latent"));
                assert_state_tensor(
                    &a.rotary,
                    &b.rotary,
                    &format!("layer {layer} positional channels"),
                );
            }
        }
    }
}

fn assert_nonzero(value: &NumericTensor) {
    assert!(value.data.iter().all(|v| v.is_finite()));
    assert!(value.data.iter().any(|v| v.abs() > 1e-9));
}

fn assert_populated(
    state: &KimiState,
    config: &kimi_linear::ModelArgs,
    positions: i32,
    paged: bool,
) {
    assert_eq!(state.as_ref().len(), config.layer_schedule.len());
    for (layer, policy) in state.as_ref().iter().zip(config.layer_schedule.iter()) {
        assert_eq!(layer.position(), positions);
        assert_eq!(layer.resets, 0);
        assert!(layer.attention.is_none() && layer.pooling.is_none());
        match policy.attention {
            kimi_linear::AttentionKind::Kda => {
                assert!(layer.compressed.is_none());
                assert_eq!(layer.fixed_offset, positions);
                let history = config.kda_config.short_conv_kernel_size - 1;
                assert_eq!(layer.fixed.len(), if history > 0 { 4 } else { 1 });
                for slot in 0..3 {
                    let role = StateTensorRole::Convolution { slot };
                    if history == 0 {
                        assert!(!layer.fixed.contains_key(&role));
                    } else {
                        let value = layer.fixed[&role].as_ref().unwrap();
                        assert_eq!(
                            value.shape,
                            [
                                1,
                                history,
                                config.kda_config.num_heads * config.kda_config.head_dim
                            ]
                        );
                        assert_nonzero(value);
                    }
                }
                let value = layer.fixed[&StateTensorRole::Recurrent].as_ref().unwrap();
                assert_eq!(
                    value.shape,
                    [
                        1,
                        config.kda_config.num_heads,
                        config.kda_config.head_dim,
                        config.kda_config.head_dim
                    ]
                );
                assert_nonzero(value);
            }
            kimi_linear::AttentionKind::Mla => {
                assert!(layer.fixed.is_empty());
                assert_eq!(layer.fixed_offset, 0);
                let cache = layer.compressed.as_ref().unwrap();
                assert_eq!(cache.offset, positions);
                assert_eq!(cache.block_size, paged.then_some(2));
                let retained = cache.state.as_ref().unwrap();
                for (tensor, width) in [
                    (&retained.latent, config.kv_lora_rank),
                    (&retained.rotary, config.qk_rope_head_dim),
                ] {
                    assert_eq!(tensor.shape, [1, positions, width]);
                    assert_nonzero(tensor);
                }
            }
        }
    }
}

fn make_state(config: &kimi_linear::ModelArgs, paged: bool) -> KimiState {
    KimiState::create(kimi_linear::state_layout(config).unwrap(), |_, policy| {
        let mut layer = NumericHybridLayerState::new(policy);
        if paged {
            if let Some(cache) = &mut layer.compressed {
                *cache = NumericCompressedCache::paged(2);
            }
        }
        Ok::<_, Error>(layer)
    })
    .unwrap()
}

// Shared with the selected-owner checkpoint fixture. Only immutable parameter
// values are supplied here; no mutable cache or recurrence state is seeded.
pub(super) fn recurrent_parameter_pattern(name: &str) -> Option<(f32, f32)> {
    let (_, field) = name.rsplit_once(".self_attn.")?;
    match field {
        "q_conv1d.weight" | "k_conv1d.weight" | "v_conv1d.weight" => Some((0.15, 0.01)),
        "A_log" => Some((-0.6, 0.01)),
        "dt_bias" => Some((-1.0, 0.01)),
        _ => None,
    }
}

// Load real raw parameter slots, never mutable state. Factory-created projection,
// router and normalization parameters already have deterministic nonzero values.
fn load_recurrent_parameters(model: &mut Model, config: &kimi_linear::ModelArgs) {
    #[derive(Default)]
    struct Load(BTreeMap<String, ()>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let Some((base, step)) = recurrent_parameter_pattern(metadata.id().as_str()) else {
                return;
            };
            assert!(self.0.insert(metadata.id().as_str().into(), ()).is_none());
            assert!(
                value.data.iter().all(|v| *v == 0.0),
                "unloaded {}",
                metadata.id().as_str()
            );
            for (index, value) in value.data.iter_mut().enumerate() {
                *value = base + step * (index % 7) as f32;
            }
        }
    }
    let mut loader = Load::default();
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut loader);
    }
    assert_eq!(
        loader.0.len(),
        5 * config
            .layer_schedule
            .iter()
            .filter(|p| p.attention == kimi_linear::AttentionKind::Kda)
            .count()
    );
}

// Verify every actual parameter against architecture-owned physical groups after
// loading, before any path token or state is created.
fn assert_loaded_parameters(model: &Model, context: &NumericContext) {
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
    let description = model.architecture().parameter_description(context).unwrap().into_owned();
    let expected = description
        .groups()
        .iter()
        .flat_map(|group| group.group().members())
        .map(|member| (member.target().to_owned(), member.global_shape().to_vec()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(actual.0, expected);
}

fn make_model(
    config: &kimi_linear::ModelArgs,
    context: &NumericContext,
) -> (
    Model,
    Vec<PrefillObservationDeclaration>,
    PreparedLayeredObservationPaths,
) {
    let architecture = KimiArchitecture::new(config.clone(), context).unwrap();
    let declarations = tensor_row_declarations(<KimiArchitecture as LayeredArchitecture<NumericBackend, KimiState>>::
        prefill_observation_declarations(&architecture, None).unwrap());
    assert!(declarations.len() >= 10 + 4 * config.num_hidden_layers as usize);
    for index in 0..config.num_hidden_layers as usize {
        let path = <KimiArchitecture as LayeredArchitecture<NumericBackend, KimiState>>::unit_path(
            &architecture,
            0,
            index, None)
        .unwrap();
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert!(declarations
                .iter()
                .any(|d| d.path() == format!("{path}.{suffix}")));
        }
    }
    let mut runtime = ResidentRuntime::new(architecture, context).unwrap();
    load_recurrent_parameters(&mut runtime, config);
    assert_loaded_parameters(&runtime, context);
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
    state: &mut KimiState,
    context: &NumericContext,
    rows: &mut DeclaredRows<'_>,
    paths: &PreparedLayeredObservationPaths,
    demand: OutputDemand,
) -> Option<NumericTensor> {
    let tokens = NumericTensor::token_ids(ids);
    let input = eredu_architectures::decoder::LayeredInput {
        tokens: &tokens,
        mask: None,
    };
    assert_eq!(
        <KimiArchitecture as LayeredArchitecture<NumericBackend, KimiState>>::inference_input_shape(
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

fn compare_equations(value: serde_json::Value, paged: bool, split_kv: bool) {
    let mut config = kimi_linear::model_args_from_config_value(&value).unwrap();
    config.split_kv_b = split_kv;
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

#[derive(Clone, Copy, Debug)]
pub(super) enum Shape {
    Fixed,
    Compressed,
    Mixed,
}

pub(super) fn configuration(
    routed: bool,
    shape: Shape,
    query_rank: Option<i32>,
    kernel: i32,
) -> serde_json::Value {
    let (kda, mla) = match shape {
        Shape::Fixed => (vec![1, 2], vec![]),
        Shape::Compressed => (vec![], vec![1, 2]),
        Shape::Mixed => (vec![1], vec![2]),
    };
    serde_json::json!({
        "model_type":"kimi_linear", "vocab_size":16, "hidden_size":8,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":2,
        "intermediate_size":11, "head_dim":4, "model_max_length":64,
        "linear_attn_config":{"kda_layers":kda,"full_attn_layers":mla,
            "num_heads":2,"head_dim":4,"short_conv_kernel_size":kernel},
        "num_experts":4, "moe_intermediate_size":5, "kv_lora_rank":3,
        "q_lora_rank":query_rank, "qk_nope_head_dim":2,"qk_rope_head_dim":2,"v_head_dim":3,
        "mla_use_nope":true,"num_experts_per_token":2,"num_shared_experts":1,
        "routed_scaling_factor":1.3,"first_k_dense_replace":if routed {1} else {2},
        "num_expert_group":2,"topk_group":1,"tie_word_embeddings":false
    })
}

fn shape_matrix(routed: bool) {
    for (shape, query, paged, split_kv) in [
        (Shape::Fixed, None, false, false),
        (Shape::Compressed, None, false, false),
        (Shape::Compressed, Some(3), true, true),
        (Shape::Mixed, None, false, true),
        (Shape::Mixed, Some(3), true, false),
    ] {
        compare_equations(configuration(routed, shape, query, 3), paged, split_kv);
    }
}

#[test]
fn kimi_linear_dense_rows_preserve_all_real_hooks_and_complete_heterogeneous_state() {
    shape_matrix(false);
}

#[test]
fn kimi_linear_routed_rows_preserve_causal_recurrence_and_blockwise_attention() {
    shape_matrix(true);
}

#[test]
fn kimi_linear_width_one_has_recurrence_without_fabricated_convolution_history() {
    for routed in [false, true] {
        for shape in [Shape::Fixed, Shape::Mixed] {
            compare_equations(configuration(routed, shape, Some(3), 1), true, false);
        }
    }
    // Even callers of the pure layout constructor cannot turn invalid negative
    // history into the valid width-one omission.
    let mut invalid =
        kimi_linear::model_args_from_config_value(&configuration(false, Shape::Fixed, None, 1))
            .unwrap();
    invalid.kda_config.short_conv_kernel_size = 0;
    assert!(matches!(
        kimi_linear::state_layout(&invalid),
        Err(kimi_linear::ConfigError::Invalid(_))
    ));
}

#[test]
fn kimi_linear_prepared_paths_bind_actual_sources_and_physical_readout() {
    for routed in [false, true] {
        for shape in [Shape::Fixed, Shape::Compressed, Shape::Mixed] {
            let value = configuration(routed, shape, Some(3), 3);
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
            let config = kimi_linear::model_args_from_config_value(&value).unwrap();
            let context = NumericContext::default();
            let (_, declarations, paths) = make_model(&config, &context);
            let (_, _, independent_paths) = make_model(&config, &context);
            let before = sources.target().source_diagnostics().unwrap();
            assert!(declarations.len() >= 18);
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
}

#[test]
fn kimi_linear_body_rows_keep_positions_before_state_only_and_last_position_readout() {
    for routed in [false, true] {
        for shape in [Shape::Fixed, Shape::Compressed, Shape::Mixed] {
            let config = kimi_linear::model_args_from_config_value(&configuration(
                routed,
                shape,
                Some(3),
                3,
            ))
            .unwrap();
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
}

#[test]
fn kimi_linear_target_rows_preserve_existing_nope_and_prediction_policy() {
    use eredu_runtime::inspection::ObservationHookSite;
    let context = NumericContext::default();
    let value = configuration(true, Shape::Mixed, Some(3), 3);
    let config = kimi_linear::model_args_from_config_value(&value).unwrap();
    let architecture = KimiArchitecture::new(config, &context).unwrap();
    let hooks =
        <KimiArchitecture as LayeredArchitecture<NumericBackend, KimiState>>::observation_hooks(
            &architecture,
        );
    for site in [
        ObservationHookSite::Input,
        ObservationHookSite::Unit,
        ObservationHookSite::Readout,
    ] {
        assert!(hooks.supports(site));
    }
    let declarations = tensor_row_declarations(<KimiArchitecture as LayeredArchitecture<NumericBackend, KimiState>>::prefill_observation_declarations(&architecture, None).unwrap());
    assert!(declarations.len() >= 18);
    assert!(declarations.iter().all(|d| !d.path().starts_with("mtp.")
        && !d.path().starts_with("vision.")
        && !d.path().starts_with("model.layers.2.")));
    for (field, invalid) in [
        ("num_nextn_predict_layers", serde_json::json!(1)),
        ("mla_use_nope", serde_json::json!(false)),
    ] {
        let mut invalid_config = value.clone();
        invalid_config[field] = invalid;
        assert!(matches!(
            kimi_linear::model_args_from_config_value(&invalid_config),
            Err(kimi_linear::ConfigError::Invalid(_))
        ));
    }
}

#[test]
fn kimi_linear_wider_convolution_still_rejects_a_missing_declared_state_role() {
    let config =
        kimi_linear::model_args_from_config_value(&configuration(false, Shape::Fixed, None, 3))
            .unwrap();
    let context = NumericContext::default();
    let (mut model, _, _) = make_model(&config, &context);
    let role = StateTensorRole::Convolution { slot: 0 };
    let mut state = NumericHybridLayerState::new(
        kimi_linear::state_layout(&config)
            .unwrap()
            .layer(0)
            .unwrap(),
    );
    assert!(matches!(state.fixed.remove(&role), Some(None)));
    let before = state.clone();
    let input = NumericTensor::new(
        vec![1, 2, config.hidden_size],
        vec![0.5; (2 * config.hidden_size) as usize],
    );
    let kimi_linear::TokenMixer::Kda(mixer) = &mut model.units_mut()[0][0].mixer else {
        panic!("actual KDA unit");
    };
    let error = mixer.forward(&input, &mut state, &context).unwrap_err();
    assert_eq!(
        error.to_string(),
        StateError::UnknownComponent { role }.to_string()
    );
    assert_eq!(
        state.fixed.keys().collect::<Vec<_>>(),
        before.fixed.keys().collect::<Vec<_>>()
    );
    assert!(state.fixed.values().all(Option::is_none));
    assert_eq!(state.fixed_offset, before.fixed_offset);
    assert_eq!(state.resets, before.resets);
    assert!(state.attention.is_none() && state.compressed.is_none() && state.pooling.is_none());
}
