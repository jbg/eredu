//! Actual selected fixed/compressed owners, not reconstructed direct models.
use super::*;
use eredu_architectures::prepared_execution::*;
use eredu_runtime::inspection::{ObservationHookSite, ObservationHookSupport};
use eredu_runtime::layered::PrefillObservationDeclaration;
use eredu_runtime::{ReplicatedTextArchitecture, ReplicatedTextStateAccess as Access};

type SelectedState = DeviceState<NumericBackend, NumericHybridLayerState>;
type SelectedSession<A> =
    eredu_runtime::ReplicatedTextSession<A, NumericBackend, NumericReplicatedMechanisms>;

#[derive(Clone, Copy, Debug)]
enum Trial {
    Full,
    Split,
    Body(OutputDemand),
    Binding,
}

#[derive(Default)]
struct Rows {
    values: BTreeMap<String, NumericTensor>,
    sequence: bool,
}
impl ActivationObserver<NumericTensor, Error> for Rows {
    fn requires_sequence_readout(&self) -> bool {
        self.sequence
    }
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        assert!(
            self.values.insert(path.into(), value.clone()).is_none(),
            "actual callback occurred twice: {path}"
        );
        Ok(())
    }
    fn observe_generated(
        &mut self,
        _: &str,
        _: &NumericTensor,
        _: &GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<NumericTensor, Error>,
    ) -> Result<(), Error> {
        // These preserve-checkpoint F32 fixtures have no generated projection
        // input. Any missing declared callback still fails completeness below.
        Ok(())
    }
}

fn assert_nonzero(value: &NumericTensor, label: &str) {
    assert!(!value.data.is_empty(), "empty {label}");
    assert!(value.data.iter().all(|v| v.is_finite()), "finite {label}");
    assert!(value.data.iter().any(|v| v.abs() > 1e-9), "nonzero {label}");
}

fn assert_state(actual: &SelectedState, expected: &SelectedState) {
    assert_eq!(actual.layout(), expected.layout());
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (index, (a, b)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_eq!(a.position(), b.position(), "layer {index}");
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        for (role, a) in &a.fixed {
            let b = &b.fixed[role];
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(a), Some(b)) = (a, b) {
                assert_tensor_close(a, b, &format!("layer {index} {role:?}"));
            }
        }
        assert_eq!(a.attention.is_some(), b.attention.is_some());
        if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.window, b.window);
            assert_eq!(a.attention_history.is_some(), b.attention_history.is_some());
            if let (Some((ak, av)), Some((bk, bv))) = (&a.attention_history, &b.attention_history) {
                assert_tensor_close(ak, bk, "attention-owned keys");
                assert_tensor_close(av, bv, "attention-owned values");
            }
            for (name, a, b) in [("keys", &a.keys, &b.keys), ("values", &a.values, &b.values)] {
                assert_eq!(a.is_some(), b.is_some());
                if let (Some(a), Some(b)) = (a, b) {
                    assert_tensor_close(a, b, &format!("layer {index} {name}"));
                }
            }
        }
        assert_eq!(a.compressed.is_some(), b.compressed.is_some());
        if let (Some(a), Some(b)) = (&a.compressed, &b.compressed) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.block_size, b.block_size);
            assert_eq!(a.state.is_some(), b.state.is_some());
            if let (Some(a), Some(b)) = (&a.state, &b.state) {
                assert_tensor_close(&a.latent, &b.latent, "compressed latent");
                assert_tensor_close(&a.rotary, &b.rotary, "compressed positional channels");
            }
        }
        // None of these selected families uses a pooling cache. Do not hide an
        // unexpected mutable representation behind an incomplete state check.
        assert!(a.pooling.is_none() && b.pooling.is_none());
    }
}

fn snapshot<A>(session: &SelectedSession<A>, positions: i32) -> SelectedState
where
    A: ReplicatedTextArchitecture<NumericBackend, SelectedState, Error = Error>,
{
    let state = session
        .inspect_runtime_state(|state| Ok(state.clone()))
        .unwrap();
    for layer in state.as_ref() {
        assert_eq!(layer.resets, 0);
        let stateful =
            layer.attention.is_some() || layer.compressed.is_some() || !layer.fixed.is_empty();
        assert_eq!(layer.position(), if stateful { positions } else { 0 });
        for (role, value) in &layer.fixed {
            assert_nonzero(
                value.as_ref().expect("declared fixed slot populated"),
                &format!("{role:?}"),
            );
        }
        if let Some(cache) = &layer.attention {
            assert_eq!(cache.offset, positions);
            assert_nonzero(cache.keys.as_ref().unwrap(), "KV keys");
            assert_nonzero(cache.values.as_ref().unwrap(), "KV values");
        }
        if let Some(cache) = &layer.compressed {
            assert_eq!(cache.offset, positions);
            assert_eq!(cache.block_size, None);
            let retained = cache.state.as_ref().unwrap();
            assert_nonzero(&retained.latent, "compressed latent");
            // Validate the actual retained positional payload when present.
            if !retained.rotary.data.is_empty() {
                assert_nonzero(&retained.rotary, "compressed positional channels");
            }
        }
        assert!(layer.pooling.is_none());
    }
    state
}

fn assert_loaded<A>(session: &SelectedSession<A>)
where
    A: ReplicatedTextArchitecture<NumericBackend, SelectedState, Error = Error>,
{
    let mut count = 0;
    assert!(session
        .visit_retained_values(&mut |value| {
            count += 1;
            assert_nonzero(value, "actual retained selected parameter");
        })
        .unwrap());
    assert!(count > 0);
}

fn assert_callbacks(rows: &Rows, units: &[String], positions: i32) {
    for path in [
        "readout.embedding",
        "readout.embedding.effective",
        "readout.normalized",
        "readout.projection_input",
        "readout.linear",
        "model.logits",
    ] {
        let value = rows
            .values
            .get(path)
            .unwrap_or_else(|| panic!("missing actual {path}"));
        assert_eq!(value.shape[..2], [1, positions]);
        assert_nonzero(value, path);
    }
    for unit in units {
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert_nonzero(&rows.values[&format!("{unit}.{suffix}")], suffix);
        }
        // An outer traversal callback alone does not prove internal Unit support.
        assert!(
            rows.values
                .keys()
                .any(|path| path.starts_with(&format!("{unit}."))
                    && !["input", "input.effective", "output", "output.effective"]
                        .iter()
                        .any(|suffix| path == &format!("{unit}.{suffix}"))),
            "missing instrumented internal unit callbacks for {unit}"
        );
    }
}

fn declared_rows(
    rows: &Rows,
    declarations: &[PrefillObservationDeclaration],
    positions: i32,
) -> BTreeMap<String, NumericTensor> {
    declarations
        .iter()
        .map(|declaration| {
            let path = declaration.path();
            let value = rows
                .values
                .get(path)
                .unwrap_or_else(|| panic!("missing declared {path}"));
            assert_eq!(declaration.sequence_axis(), 1);
            assert_eq!(value.shape.len(), 3);
            assert_eq!(value.shape[..2], [1, positions], "{path}");
            assert_nonzero(value, path);
            (path.into(), value.clone())
        })
        .collect()
}

struct Trace {
    prefix: Option<SelectedState>,
    rows: BTreeMap<String, NumericTensor>,
    continuation: Option<SelectedState>,
    outputs: Vec<(usize, Option<NumericTensor>)>,
    decodes: Vec<(
        NumericTensor,
        BTreeMap<String, NumericTensor>,
        SelectedState,
    )>,
}
impl Trace {
    fn empty() -> Self {
        Self {
            prefix: None,
            rows: BTreeMap::new(),
            continuation: None,
            outputs: vec![],
            decodes: vec![],
        }
    }
}

fn capture_source(
    discovery: &eredu_architectures::prepared_sources::PreparedModelDiscovery,
    path: &str,
    preview: bool,
) -> SharedCapturePlan {
    let discovery = discovery.capture().unwrap();
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "actual".into(),
        path: path.into(),
        schedule: CaptureSchedule {
            prefill: true,
            ..Default::default()
        },
        slices: vec![],
        transform: if preview {
            CaptureTransform::Preview { max_elements: 0 }
        } else {
            CaptureTransform::FullTensor
        },
    });
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
                cached_positions: 2,
            },
        )
        .unwrap(),
    )
}

fn binding(
    discovery: &eredu_architectures::prepared_sources::PreparedModelDiscovery,
    paths: &SharedLayeredObservationPaths,
    declarations: &[PrefillObservationDeclaration],
) {
    assert!(!declarations.is_empty());
    for declaration in declarations {
        for preview in [false, true] {
            let source = capture_source(discovery, declaration.path(), preview);
            let selected = discovery.prepare_capture_selection(&source, paths).unwrap();
            assert!(selected.source().same_storage(&source));
            assert!(selected.paths().same_storage(paths));
            assert_eq!(selected.declaration(0).unwrap(), Some(declaration));
            selected
                .validate_sources(&source.clone(), &paths.clone())
                .unwrap();
            let other = capture_source(discovery, declaration.path(), preview);
            assert!(matches!(
                selected.validate_sources(&other, paths),
                Err(SelectionError::Identity)
            ));
            for requested in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
                let physical = selected.physical_output(requested);
                assert_eq!(
                    physical,
                    if declaration.readout_stage() == Stage::BeforeReadout {
                        requested
                    } else {
                        OutputDemand::Sequence
                    }
                );
                let mut geometry = InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 2,
                    input_positions: 5,
                    max_output_tokens: 4,
                    prefill_chunk_positions: 2,
                    output: physical,
                };
                let bound = selected.bind_geometry(geometry).unwrap();
                assert_eq!(bound.geometry(), geometry);
                assert!(std::ptr::eq(bound.selection(), &selected));
                if physical == OutputDemand::Sequence {
                    geometry.output = requested;
                    assert!(matches!(
                        selected.bind_geometry(geometry),
                        Err(SelectionError::Readout)
                    ));
                }
            }
        }
    }
}

struct Visitor<'a> {
    context: &'a NumericContext,
    discovery: &'a eredu_architectures::prepared_sources::PreparedModelDiscovery,
    expected: Access,
    trial: Trial,
    started: bool,
}
impl ReplicatedTextArchitectureVisitor<NumericBackend, SelectedState> for Visitor<'_> {
    type Output = Trace;
    type Error = String;
    fn construction_started(&mut self) {
        self.started = true;
    }
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        checkpoint: RetainedCheckpointSource,
    ) -> Result<Trace, String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, SelectedState, Error = Error> + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        assert!(self.started);
        assert_eq!(prepared.requirements().state_access(), self.expected);
        let resident = prepared.selected().residency().is_fully_resident();
        let layout = prepared.requirements().state_layout().clone();
        let mut modules = prepared.into_modules();
        let architecture = modules.take_architecture();
        assert_eq!(
            architecture.observation_hooks(),
            ObservationHookSupport::internal(true, true, true)
        );
        for site in [
            ObservationHookSite::Input,
            ObservationHookSite::Unit,
            ObservationHookSite::Readout,
        ] {
            assert!(architecture.observation_hooks().supports(site));
        }
        assert!(!architecture
            .observation_hooks()
            .supports(ObservationHookSite::RoutedUnits));
        assert!(!architecture
            .observation_hooks()
            .supports(ObservationHookSite::Publication));
        let declarations = architecture.prefill_observation_declarations().unwrap();
        let units = (0..architecture.group_unit_count(0).unwrap())
            .map(|i| architecture.unit_path(0, i).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(declarations.len(), 10 + 4 * units.len());
        let mechanisms = if resident {
            NumericReplicatedMechanisms::with_bound_checkpoint(checkpoint)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(checkpoint)
        };
        // Consume the actual original modules/contract, exactly as the shared
        // construct_selected_text_session helper does. No direct model is built.
        let mut session = eredu_runtime::construct_replicated_text_session::<A, NumericBackend, _>(
            architecture,
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            self.context,
        )
        .map_err(|e| e.to_string())?;
        let paths = session.shared_observation_paths().unwrap().clone();
        session.validate_prepared_observation_paths(&paths).unwrap();
        for declaration in &declarations {
            assert_eq!(
                paths.prefill_observation(declaration.path()),
                Some(declaration)
            );
        }
        assert_eq!(paths.group_count(), 1);
        assert_eq!(paths.unit_count(0), Some(units.len()));
        for (i, unit) in units.iter().enumerate() {
            let (input, output) = paths.unit_paths(0, i).unwrap();
            assert_eq!(input, format!("{unit}.input"));
            assert_eq!(output, format!("{unit}.output"));
        }
        assert_loaded(&session);
        if matches!(self.trial, Trial::Binding) {
            binding(self.discovery, &paths, &declarations);
            return Ok(Trace::empty());
        }
        // Keep the original prepared token from the first invocation. Ordinary
        // Noop traversal can invalidate it before the observed continuation.
        let mut prefix_rows = Rows {
            sequence: true,
            ..Default::default()
        };
        session
            .prefill_with_observer(
                &NumericTensor::token_ids(&[4, 2]),
                None,
                self.context,
                &mut prefix_rows,
            )
            .map_err(|e| e.to_string())?;
        assert_callbacks(&prefix_rows, &units, 2);
        assert_eq!(
            declared_rows(&prefix_rows, &declarations, 2).len(),
            declarations.len()
        );
        drop(prefix_rows);
        let mut trace = Trace::empty();
        trace.prefix = Some(snapshot(&session, 2));
        assert_eq!(trace.prefix.as_ref().unwrap().layout(), &layout);
        let body = matches!(self.trial, Trial::Body(_));
        let active = declarations
            .iter()
            .filter(|d| !body || d.readout_stage() == Stage::BeforeReadout)
            .cloned()
            .collect::<Vec<_>>();
        let chunks: &[&[usize]] = if matches!(self.trial, Trial::Full) {
            &[&[1, 3, 5, 2, 6]]
        } else {
            &[&[1, 3], &[5], &[2, 6]]
        };
        let demand = match self.trial {
            Trial::Body(demand) => demand,
            _ => OutputDemand::Sequence,
        };
        let mut pieces = BTreeMap::<String, Vec<NumericTensor>>::new();
        let mut end = 0;
        for ids in chunks {
            let tokens = NumericTensor::token_ids(ids);
            let mut rows = Rows {
                sequence: !body,
                ..Default::default()
            };
            let output = session
                .prefill_input_with_readout(
                    A::text_input(&tokens, None),
                    None,
                    demand,
                    self.context,
                    &mut rows,
                )
                .map_err(|e| e.to_string())?;
            let values = declared_rows(&rows, &active, ids.len() as i32);
            if !body {
                assert_callbacks(&rows, &units, ids.len() as i32);
            }
            if demand == OutputDemand::StateOnly {
                assert!(output.is_none());
                assert!(!rows.values.contains_key("readout.linear"));
                assert!(!rows.values.contains_key("model.logits"));
            } else {
                assert_nonzero(output.as_ref().unwrap(), "actual selected output");
                assert_tensor_close(
                    output.as_ref().unwrap(),
                    &rows.values["model.logits"],
                    "returned score tensor is the actual publication callback",
                );
            }
            for (path, value) in values {
                pieces.entry(path).or_default().push(value);
            }
            end += ids.len();
            trace.outputs.push((end, output));
            // Intermediate checkpoints assert frontier/nonzero invariants.
            // Full cross-run state equality is checked at prefix/final/decode.
            snapshot(&session, 2 + end as i32);
            assert_loaded(&session);
        }
        trace.rows = pieces
            .into_iter()
            .map(|(path, values)| {
                (
                    path,
                    NumericTensor::concatenate(&values, 1, self.context).unwrap(),
                )
            })
            .collect();
        trace.continuation = Some(snapshot(&session, 7));
        for (i, token) in [7, 3, 1].into_iter().enumerate() {
            let mut rows = Rows {
                sequence: true,
                ..Default::default()
            };
            let output = session
                .decode_with_observer(&NumericTensor::token_ids(&[token]), self.context, &mut rows)
                .map_err(|e| e.to_string())?;
            assert_callbacks(&rows, &units, 1);
            trace.decodes.push((
                output,
                declared_rows(&rows, &declarations, 1),
                snapshot(&session, 8 + i as i32),
            ));
            assert_loaded(&session);
        }
        session.validate_prepared_observation_paths(&paths).unwrap();
        Ok(trace)
    }
}

struct Assembler;
impl PreparedExecutableAssembler<()> for Assembler {
    type Executable = Trace;
    type Output = Trace;
    type Error = String;
    fn floating_state_dtype(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, String> {
        Ok(eredu_runtime::StateStorageDtype::F32)
    }
    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), String> {
        Err("ordinary selected fixture has no communication resources".into())
    }
    fn finish(self, mut parts: PreparedExecutableParts<Trace, ()>) -> Result<Trace, String> {
        assert!(parts.take_communication().is_none());
        assert!(parts.take_processor().is_none());
        assert_eq!(parts.floating_state_bytes().get(), 4);
        Ok(parts.into_executable())
    }
}

fn execute(
    config: &serde_json::Value,
    expected: Access,
    residency: eredu_core::ResidencyPlan,
    trial: Trial,
) -> Trace {
    execute_with_parameters(config, expected, residency, trial, |_, _| None)
}

fn execute_with_parameters(
    config: &serde_json::Value,
    expected: Access,
    residency: eredu_core::ResidencyPlan,
    trial: Trial,
    parameters: fn(&str, &[i32]) -> Option<NumericTensor>,
) -> Trace {
    let (artifact, values) =
        prepared_adapter::payload_fixture_config_with(config, 1.0, |name, shape| {
            parameters(name, shape).or_else(|| {
                // Preserve the corrected Nemotron raw source for existing callers.
                (config["model_type"] == "nemotron_h" && name.ends_with(".A_log")).then(|| {
                    NumericTensor::new(
                        shape.to_vec(),
                        vec![-0.7; shape.iter().product::<i32>() as usize],
                    )
                })
            })
        });
    // The actual checkpoint includes every raw conv/scan parameter as well as
    // factory-created linears. No zero placeholder is used as numerical evidence.
    assert!(!values.is_empty());
    for (name, (_, bits)) in &values {
        assert!(
            bits.iter().all(|v| f32::from_bits(*v).is_finite()),
            "{name}"
        );
        assert!(
            bits.iter().any(|v| f32::from_bits(*v).abs() > 1e-9),
            "{name}"
        );
    }
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let plan = prepared_adapter::plan(None).with_residency(residency);
    let sources = prepared_adapter::prepare(
        &inspection,
        &plan,
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
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let visitor = Visitor {
        context: &context,
        discovery: &discovery,
        expected,
        trial,
        started: false,
    };
    let routes =
        PreparedExecutionRoutes::new().with_replicated(ReplicatedRoute::<NumericBackend, _>::new(
            &context,
            &context,
            SharedReplicatedTextVisitor::<NumericReplicatedStateProfiles, _>::new(visitor),
        ));
    construct_prepared_execution(sources, None, routes, Assembler)
        .unwrap_or_else(|e| panic!("{} {expected:?} {trial:?}: {e}", config["model_type"]))
}

fn compare(
    config: &serde_json::Value,
    access: Access,
    residency: eredu_core::ResidencyPlan,
    trial: Trial,
) {
    compare_with_parameters(config, access, residency, trial, |_, _| None)
}

fn compare_with_parameters(
    config: &serde_json::Value,
    access: Access,
    residency: eredu_core::ResidencyPlan,
    trial: Trial,
    parameters: fn(&str, &[i32]) -> Option<NumericTensor>,
) {
    let full = execute_with_parameters(config, access, residency.clone(), Trial::Full, parameters);
    let split = execute_with_parameters(config, access, residency, trial, parameters);
    assert_state(
        full.prefix.as_ref().unwrap(),
        split.prefix.as_ref().unwrap(),
    );
    assert_state(
        full.continuation.as_ref().unwrap(),
        split.continuation.as_ref().unwrap(),
    );
    assert!(!split.rows.is_empty());
    for (path, value) in &split.rows {
        assert_tensor_close(value, &full.rows[path], path);
    }
    if matches!(trial, Trial::Split) {
        assert_eq!(full.rows.len(), split.rows.len());
    }
    let scores = &full.rows["model.logits"];
    for (end, output) in split.outputs {
        if let Some(output) = output {
            if matches!(trial, Trial::Body(OutputDemand::LastPosition)) {
                assert_tensor_close(
                    &output,
                    &scores.axis_slice(1, end - 1, end),
                    "selected last row",
                );
            }
        } else {
            assert!(matches!(trial, Trial::Body(OutputDemand::StateOnly)));
        }
    }
    assert_eq!(full.decodes.len(), 3);
    assert_eq!(split.decodes.len(), 3);
    for ((a, ar, ast), (b, br, bst)) in full.decodes.iter().zip(&split.decodes) {
        assert_tensor_close(a, b, "same next decode");
        assert_eq!(ar.len(), br.len());
        for (path, value) in ar {
            assert_tensor_close(value, &br[path], path);
        }
        assert_state(ast, bst);
    }
}

fn residencies() -> [eredu_core::ResidencyPlan; 3] {
    [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: None,
            host_budget_bytes: None,
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 20,
            host_budget_bytes: 1 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ]
}
fn fixed_cases() -> Vec<(serde_json::Value, Access)> {
    let base = heterogeneous_replicated_configs();
    let mut cases = vec![];
    for (layers, access) in [
        (["conv", "full_attention"], Access::AttentionWithFixed),
        (["full_attention", "full_attention"], Access::KeyValue),
        (["conv", "conv"], Access::Fixed),
    ] {
        let mut config = base[0].clone();
        config["layer_types"] = serde_json::json!(layers);
        cases.push((config, access));
    }
    for (pattern, access) in [
        ("M*", Access::AttentionWithFixed),
        ("**", Access::KeyValue),
        ("MM", Access::Fixed),
        ("--", Access::Stateless),
    ] {
        let mut config = base[2].clone();
        config["hybrid_override_pattern"] = pattern.into();
        cases.push((config, access));
    }
    for config in [&base[3], &base[4]] {
        cases.push((config.clone(), Access::AttentionWithFixed));
    }
    cases
}
fn v3(query_rank: Option<i32>) -> serde_json::Value {
    let mut value = serde_json::json!({
        "model_type":"deepseek_v3", "hidden_size":8, "intermediate_size":10,
        "moe_intermediate_size":4, "num_hidden_layers":2, "num_attention_heads":2,
        "vocab_size":16, "max_position_embeddings":64, "q_lora_rank":query_rank,
        "kv_lora_rank":3, "qk_nope_head_dim":2, "qk_rope_head_dim":2, "v_head_dim":3,
        "first_k_dense_replace":2, "n_routed_experts":4, "n_shared_experts":1,
        "num_experts_per_tok":2, "n_group":2, "topk_group":1, "routed_scaling_factor":1.3,
        "norm_topk_prob":true, "num_nextn_predict_layers":0, "tie_word_embeddings":false
    });
    if query_rank.is_some() {
        value["rope_scaling"] = serde_json::json!({
            "type":"yarn", "factor":2.0, "original_max_position_embeddings":4,
            "beta_fast":32.0, "beta_slow":1.0, "mscale":1.0, "mscale_all_dim":0.5
        });
    }
    value
}

#[test]
fn selected_fixed_profiles_preserve_all_real_rows_and_continued_state() {
    for (config, access) in fixed_cases() {
        for residency in residencies() {
            compare(&config, access, residency, Trial::Split);
        }
    }
}
#[test]
fn selected_compressed_v3_preserves_all_real_rows_and_continued_state() {
    for rank in [None, Some(3)] {
        for residency in residencies() {
            compare(
                &v3(rank),
                Access::CompressedAttention,
                residency,
                Trial::Split,
            );
        }
    }
}
#[test]
fn selected_shells_bind_actual_discovery_sources_and_physical_readout() {
    let base = heterogeneous_replicated_configs();
    for (config, access) in [
        (&base[0], Access::AttentionWithFixed),
        (&base[2], Access::Fixed),
        (&base[3], Access::AttentionWithFixed),
        (&v3(Some(3)), Access::CompressedAttention),
    ] {
        for residency in residencies() {
            execute(config, access, residency, Trial::Binding);
        }
    }
}
#[test]
fn selected_shell_body_rows_precede_state_only_and_last_position_readout() {
    let base = heterogeneous_replicated_configs();
    for (config, access) in [
        (&base[0], Access::AttentionWithFixed),
        (&base[2], Access::Fixed),
        (&base[3], Access::AttentionWithFixed),
        (&v3(Some(3)), Access::CompressedAttention),
    ] {
        for residency in residencies() {
            for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
                compare(config, access, residency.clone(), Trial::Body(demand));
            }
        }
    }
}
#[path = "replicated_rows/lfm2_width_one.rs"]
mod lfm2_width_one;

#[path = "replicated_rows/kimi_linear_rows.rs"]
mod kimi_linear_rows;

#[path = "replicated_rows/qwen_width_one.rs"]
mod qwen_width_one;
