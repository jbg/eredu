use super::*;
use eredu_core::capture::*;
use eredu_core::{ObservationMechanisms, ObservationSupportStatus};
use eredu_runtime::layered::{
    PrefillReadoutStage as Stage, PreparedCaptureSelectionError as SelectionError,
};
use eredu_runtime::{LayeredArchitecture, PreparedLayeredObservationError};

fn fixture() -> (
    tempfile::TempDir,
    PreparedModelSources,
    eredu_architectures::prepared_sources::PreparedModelDiscovery,
    SharedLayeredObservationPaths,
) {
    let (artifact, _) = prepared_adapter::payload_fixture_config(&model_config(), 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let (_, paths) = state_and_paths(&sources, &context, 0);
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
                CaptureTransformKind::Slice,
            ],
            ..Default::default()
        },
    );
    (artifact, sources, discovery, paths)
}
fn admission(
    discovery: &eredu_architectures::prepared_sources::PreparedModelDiscovery,
    paths: &[&str],
    preview: bool,
    active: bool,
) -> SharedCapturePlan {
    let discovery = discovery.capture().unwrap();
    let mut plan = CapturePlan::none();
    plan.selections = paths
        .iter()
        .enumerate()
        .map(|(i, path)| CaptureSelection {
            id: i.to_string(),
            path: (*path).into(),
            schedule: CaptureSchedule {
                prefill: active,
                ..Default::default()
            },
            slices: vec![],
            transform: if preview {
                CaptureTransform::Preview { max_elements: 0 }
            } else {
                CaptureTransform::FullTensor
            },
        })
        .collect();
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    plan.limits.per_step = unlimited;
    plan.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 5,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: 4,
            },
        )
        .unwrap(),
    )
}

#[test]
fn actual_dense_declarations_join_readout_before_exact_candidate_binding() {
    let (_artifact, sources, discovery, paths) = fixture();
    let before = sources.target().source_diagnostics().unwrap();
    for (path, stage) in [
        ("readout.embedding", Stage::BeforeReadout),
        ("model.layers.1.output", Stage::BeforeReadout),
        ("model.layers.0.feed_forward.units", Stage::BeforeReadout),
        ("model.layers.0.feed_forward.units.effective", Stage::BeforeReadout),
        ("model.layers.0.feed_forward.write", Stage::BeforeReadout),
        ("model.layers.0.feed_forward.write.effective", Stage::BeforeReadout),
        ("readout.residual", Stage::ReadoutInput),
        ("readout.normalized.effective", Stage::ReadoutInput),
        ("readout.projection_input", Stage::ReadoutInput),
        ("readout.linear", Stage::VocabularyScores),
        (
            eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
            Stage::VocabularyScores,
        ),
    ] {
        let source = admission(&discovery, &[path], false, true);
        let selected = discovery
            .prepare_capture_selection(&source, &paths)
            .unwrap();
        assert!(selected.source().same_storage(&source));
        assert!(selected.paths().same_storage(&paths));
        assert_eq!(
            selected.declaration(0).unwrap().unwrap().readout_stage(),
            stage
        );
        let mut geometry = request(4);
        geometry.prefill_chunk_positions = 2;
        geometry.output = selected.physical_output(OutputDemand::LastPosition);
        assert_eq!(
            geometry.output,
            if stage == Stage::BeforeReadout {
                OutputDemand::LastPosition
            } else {
                OutputDemand::Sequence
            }
        );
        let bound = selected.bind_geometry(geometry).unwrap();
        assert_eq!(bound.geometry(), geometry);
        assert!(std::ptr::eq(bound.selection(), &selected));
        if stage != Stage::BeforeReadout {
            geometry.output = OutputDemand::LastPosition;
            assert!(matches!(
                selected.bind_geometry(geometry),
                Err(SelectionError::Readout)
            ));
        }
    }
    // Zero Preview still requires the hook/factory and its original quota.
    let source = admission(&discovery, &["readout.projection_input"], true, true);
    let selected = discovery
        .prepare_capture_selection(&source, &paths)
        .unwrap();
    assert_eq!(
        selected.physical_output(OutputDemand::LastPosition),
        OutputDemand::Sequence
    );
    assert!(selected.declaration(0).unwrap().is_some());
    let mixed = admission(
        &discovery,
        &["readout.embedding", "model.logits"],
        false,
        true,
    );
    assert_eq!(
        discovery
            .prepare_capture_selection(&mixed, &paths)
            .unwrap()
            .physical_output(OutputDemand::LastPosition),
        OutputDemand::Sequence
    );
    let after = sources.target().source_diagnostics().unwrap();
    assert_eq!(before.physical_reads, after.physical_reads);
    // Binding constructs a full fixed row mapper even for Preview(0). Its
    // declared live controls must fit alongside the retained owner/bound view;
    // source aliases alone are insufficient to cover that construction/move peak.
    let bound = selected.bind_geometry(request(4)).unwrap();
    let assembly =
        CapturePrefillRowAssembly::prepare(source.admission(), 0, bound.geometry()).unwrap();
    let live = std::mem::size_of_val(&selected)
        + std::mem::size_of_val(&bound)
        + std::mem::size_of_val(&assembly);
    let controls = eredu_runtime::layered::PreparedCaptureSelection::control_peak_bytes().unwrap();
    assert!(controls >= u64::try_from(live.checked_mul(3).unwrap()).unwrap());
    assert_eq!(assembly.logical_geometry().elements(), 0);
    assert!(bound.selection().source().same_storage(&source));
    assert!(bound.selection().paths().same_storage(&paths));
    // No declaration collection/rebind runs here. Its transient Vec/String
    // storage belongs to the original loading/preparation owner, not this fact.
}

#[test]
fn selected_sources_reject_equal_independent_owners_and_changed_candidate_coordinates() {
    let (_artifact, sources, discovery, paths) = fixture();
    let source = admission(&discovery, &["model.layers.0.output"], false, true);
    let selected = discovery
        .prepare_capture_selection(&source, &paths)
        .unwrap();
    selected
        .validate_sources(&source.clone(), &paths.clone())
        .unwrap();
    let independent = admission(&discovery, &["model.layers.0.output"], false, true);
    assert_eq!(
        source.admission().identity(),
        independent.admission().identity()
    );
    assert!(matches!(
        selected.validate_sources(&independent, &paths),
        Err(SelectionError::Identity)
    ));
    let context = WorkspaceContext::new(Facts::default());
    let (_, independent_paths) = state_and_paths(&sources, &context, 0);
    assert!(matches!(
        selected.validate_sources(&source, &independent_paths),
        Err(SelectionError::Identity)
    ));
    let original = request(4);
    for field in 0..6 {
        let mut changed = original;
        match field {
            0 => changed.batch_size += 1,
            1 => changed.cached_positions += 1,
            2 => changed.input_positions += 1,
            3 => changed.max_output_tokens += 1,
            4 => changed.prefill_chunk_positions = 0,
            _ => changed.cached_positions = u64::MAX,
        }
        assert!(selected.bind_geometry(changed).is_err());
    }
    let mut candidate = original;
    candidate.prefill_chunk_positions = 1;
    candidate.output = OutputDemand::LastPosition;
    selected.bind_geometry(candidate).unwrap();
    // Candidate chunk size is bound only once selected, not inferred from shape.
    candidate.prefill_chunk_positions = 3;
    selected.bind_geometry(candidate).unwrap();
}

#[test]
fn undeclared_internal_hooks_fail_only_when_selected_and_catalog_checks_remain_required() {
    let (_artifact, _sources, discovery, paths) = fixture();
    let catalog = discovery.capture().unwrap();
    let point =
        catalog
            .catalog
            .points
            .iter()
            .find(|p| {
                p.prefill
                    && p.axes.as_ref().is_some_and(|axes| axes.len() == 3)
                    && paths.prefill_observation(&p.path).is_none()
                    && catalog.support.points.iter().any(|s| {
                        s.path == p.path && s.prefill == ObservationSupportStatus::Supported
                    })
            })
            .unwrap();
    let active = admission(&discovery, &[&point.path], false, true);
    assert!(matches!(
        discovery.prepare_capture_selection(&active, &paths),
        Err(SelectionError::Undeclared { index: 0 })
    ));
    let inactive = admission(&discovery, &[&point.path], false, false);
    let selected = discovery
        .prepare_capture_selection(&inactive, &paths)
        .unwrap();
    assert!(selected.declaration(0).unwrap().is_none());
    selected.bind_geometry(request(4)).unwrap();
}

#[test]
fn raw_path_binding_cannot_replace_catalog_or_declared_axis_validation() {
    let (_artifact, _sources, discovery, paths) = fixture();
    let source = admission(&discovery, &["readout.embedding"], false, true);
    let mut changed = discovery.capture().unwrap();
    let index = changed
        .catalog
        .points
        .iter()
        .position(|p| p.path == "readout.embedding")
        .unwrap();
    changed.catalog.points[index]
        .meaning
        .push_str(" changed semantics");
    let forged = SharedCapturePlan::new(
        source
            .admission()
            .plan()
            .clone()
            .admit_with_text_origin(
                &changed.catalog,
                &changed.support,
                &changed.support.capture,
                source.admission().request(),
                source.admission().text_origin().unwrap(),
            )
            .unwrap(),
    );
    // The low-level binder advertises declarations only; production architecture
    // entry rejects changed selected semantics using the retained original catalog.
    paths.prepare_capture_selection(&forged).unwrap();
    assert!(matches!(
        discovery.prepare_capture_selection(&forged, &paths),
        Err(SelectionError::Capture(_))
    ));
    let axes = changed.catalog.points[index].axes.as_mut().unwrap();
    axes.swap(0, 1);
    let wrong_axis = SharedCapturePlan::new(
        source
            .admission()
            .plan()
            .clone()
            .admit_with_text_origin(
                &changed.catalog,
                &changed.support,
                &changed.support.capture,
                source.admission().request(),
                source.admission().text_origin().unwrap(),
            )
            .unwrap(),
    );
    assert!(matches!(
        paths.prepare_capture_selection(&wrong_axis),
        Err(SelectionError::Axes { index: 0 })
    ));
}

// Same actual block builder with no declaration opt-in: the outer shared type
// must not infer arbitrary factory semantics. No numerical fixture is changed.
struct UndeclaredFactory;
impl<B: NeuralBackend, C: decoder::Config> decoder::BlockFactory<B, C> for UndeclaredFactory {
    type FeedForward = decoder::Mlp<B>;
    fn build(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<decoder::TransformerBlock<B, Self::FeedForward>, Error> {
        <decoder::DenseBlockFactory as decoder::BlockFactory<B, C>>::build(config, layer, context)
    }
    fn parameter_groups(
        block: &decoder::TransformerBlock<B, Self::FeedForward>,
        config: &C,
        layer: usize,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, eredu_runtime::ParallelPlanError> {
        <decoder::DenseBlockFactory as decoder::BlockFactory<B, C>>::parameter_groups(
            block, config, layer,
        )
    }
}
#[test]
fn cold_rebind_checks_causal_declarations_even_when_group_and_unit_paths_match() {
    let (_artifact, _sources, _discovery, paths) = fixture();
    let context = WorkspaceContext::new(Facts::default());
    let args = llama::model_args_from_config_value(&model_config()).unwrap();
    let architecture =
        decoder::LayeredModel::<WorkspaceBackend, _, UndeclaredFactory>::new(args, &context)
            .unwrap();
    let runtime =
        ResidentRuntime::<_, WorkspaceBackend, State>::new(architecture, &context).unwrap();
    let unknown = runtime.prepare_observation_paths().unwrap();
    assert_eq!(unknown.source().unit_paths(0, 0), paths.unit_paths(0, 0));
    assert!(unknown
        .source()
        .prefill_observation("model.layers.0.output")
        .is_none());
    assert!(unknown
        .source()
        .prefill_observation("readout.embedding")
        .is_some());
    assert!(matches!(
        runtime.bind_observation_paths(&paths, None),
        Err(PreparedLayeredObservationError::SemanticMismatch)
    ));
}

#[derive(Default)]
struct Rows(BTreeMap<String, NumericTensor>);
impl ActivationObserver<NumericTensor, Error> for Rows {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.0.insert(path.into(), value.clone());
        Ok(())
    }
}
#[test]
fn declared_dense_rows_match_actual_nonzero_full_and_chunked_equations_with_cached_origin() {
    let args = llama::model_args_from_config_value(&model_config()).unwrap();
    let context = NumericContext::default();
    let mut model = ResidentRuntime::new(
        llama::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let paths = model.prepare_observation_paths().unwrap();
    let create = || {
        DeviceState::<NumericBackend, _>::create(llama::state_layout(&args).unwrap(), |_, p| {
            Ok::<_, Error>(NumericHybridLayerState::new(p))
        })
        .unwrap()
    };
    let mut full_state = create();
    let prefix = NumericTensor::token_ids(&[4, 2]);
    model
        .forward(
            decoder::LayeredInput {
                tokens: &prefix,
                mask: None,
            },
            &mut full_state,
            &context,
        )
        .unwrap();
    let mut chunk_state = full_state.clone();
    let mut full = Rows::default();
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            decoder::LayeredInput {
                tokens: &NumericTensor::token_ids(&[1, 3, 5, 2, 6]),
                mask: None,
            },
            &mut full_state,
            &context,
            &mut full,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
    eredu_runtime::observe_model_logits(&mut full, &scores.unwrap()).unwrap();
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = Rows::default();
        let (scores, _) = model
            .forward_with_prepared_observer_and_context_with_readout(
                decoder::LayeredInput {
                    tokens: &NumericTensor::token_ids(ids),
                    mask: None,
                },
                &mut chunk_state,
                &context,
                &mut rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut rows, &scores.unwrap()).unwrap();
        for (path, value) in rows.0 {
            if paths.source().prefill_observation(&path).is_some() {
                assembled.entry(path).or_default().extend(value.data);
            }
        }
    }
    assert!(assembled.len() >= 10);
    for (path, data) in assembled {
        let expected = &full.0[&path];
        assert!(data.iter().any(|x| x.abs() > 1e-6), "nonzero {path}");
        assert_tensor_close(
            &NumericTensor::new(expected.shape.clone(), data),
            expected,
            &path,
        );
    }
}

#[path = "causal/qwen_rows.rs"]
mod qwen_rows;

#[path = "causal/custom_rows.rs"]
mod custom_rows;

#[path = "causal/qwen_hybrid_rows.rs"]
mod qwen_hybrid_rows;

#[path = "causal/lfm2_rows.rs"]
mod lfm2_rows;

#[path = "causal/nemotron_rows.rs"]
mod nemotron_rows;

#[path = "causal/muse_rows.rs"]
mod muse_rows;

#[path = "causal/deepseek_v3_rows.rs"]
mod deepseek_v3_rows;

#[path = "causal/deepseek_v4_rows.rs"]
mod deepseek_v4_rows;

#[path = "causal/kimi_linear_rows.rs"]
mod kimi_linear_rows;

#[path = "causal/replicated_rows.rs"]
mod replicated_rows;

#[path = "causal/gemma4_rows.rs"]
mod gemma4_rows;

#[path = "causal/inkling_rows.rs"]
mod inkling_rows;

#[path = "causal/qwen_mrope_zero.rs"]
mod qwen_mrope_zero;

#[path = "causal/conditional_qwen_rows.rs"]
mod conditional_qwen_rows;
