//! Actual artifact admission through retained ordinary preparation and construction.
use super::*;
use eredu_architectures::{prepared_execution::*, PreparationMechanismProvider};
use eredu_runtime::*;
use std::cell::Cell;

pub(super) struct Mechanisms {
    pub(super) rows: bool,
    pub(super) dtype_supported: bool,
    pub(super) raw_media: bool,
    pub(super) prediction: bool,
    pub(super) prediction_state: bool,
    pub(super) communication: bool,
    pub(super) row_queries: Cell<usize>,
}
impl Default for Mechanisms {
    fn default() -> Self {
        Self {
            rows: true,
            dtype_supported: true,
            raw_media: false,
            prediction: false,
            prediction_state: true,
            communication: true,
            row_queries: Cell::new(0),
        }
    }
}
impl PreparationMechanismProvider for Mechanisms {
    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<StateStorageDtype> {
        self.dtype_supported
            .then(|| {
                ReplicatedTextMechanismSupport::floating_state_dtype(
                    &NumericMechanismSupport::default(),
                    source,
                )
            })
            .flatten()
    }
    fn row_lookup_storage(&self) -> Option<AddressableStorageCapabilities> {
        self.row_queries.set(self.row_queries.get() + 1);
        self.rows.then(|| cold::Support.storage().unwrap())
    }
    fn row_lookup_workspace(
        &self,
        descriptor: &RowLookupDescriptor,
    ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
        self.row_queries.set(self.row_queries.get() + 1);
        cold::Support.workspace(descriptor)
    }
    fn recipe_materialization_workspace<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        &self,
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &C,
    ) -> Result<u64, String> {
        recipe
            .peak_materialization_bytes(source)
            .map_err(|e| e.to_string())
    }
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        super::super::prepared_adapter::NumericPreparationProvider { addressable: true }
            .preparation_capabilities()
    }
    fn supports_grouped_operation(&self, _: GroupedOperationRequirement) -> bool {
        true
    }
    fn replicated_text_capabilities(
        &self,
        requirements: &ReplicatedTextRequirements,
        request: &ReplicatedTextSelectionRequest,
    ) -> BackendMechanismCapabilities {
        super::super::prepared_adapter::NumericPreparationProvider { addressable: true }
            .replicated_text_capabilities(requirements, request)
    }
    fn state_capabilities(
        &self,
        requirements: &StateRealizationRequirements,
        policy: &CacheResidencyPolicy,
    ) -> StateMechanismCapabilities {
        if !self.prediction_state {
            return StateMechanismCapabilities::new([]);
        }
        synthesize_state_capabilities(
            requirements,
            policy,
            &NumericMechanismSupport {
                persistent_session: true,
                ..Default::default()
            },
        )
    }
    fn processor_capabilities(&self) -> MediaPrimitiveCapabilities {
        if self.raw_media {
            use eredu_core::InputModality::{Image, Text, Video};
            use ProcessorPrimitive::*;
            return MediaPrimitiveCapabilities::new(
                [Image, Video],
                [Text, Image, Video],
                [Text, Image, Video],
                [
                    TensorU32,
                    TensorF32,
                    TensorI32,
                    MetadataInspection,
                    RgbResizeBicubic,
                    RgbNormalize,
                    VideoSampling,
                ],
                i32::MAX as u64,
            );
        }
        super::super::prepared_adapter::NumericPreparationProvider { addressable: true }
            .processor_capabilities()
    }
    fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities {
        use SpeculativeMechanism::*;
        SpeculativeMechanismCapabilities::new(
            [
                TensorOperations,
                NeuralOperations,
                GroupedNeuralOperations,
                HyperNeuralOperations,
                PayloadMaterialization,
                LogitsProcessing,
                Sampling,
                Randomness,
                StateStorage,
                StorageResidency,
                ExactCompletion,
                Observation,
                QueueBinding,
                Communication,
                Agreement,
                Publication,
            ]
            .into_iter()
            .filter(|_| self.prediction),
        )
    }
    fn communication_capabilities(&self) -> CommunicationCapabilities {
        if !self.communication {
            return CommunicationCapabilities::new([]).unwrap();
        }
        super::super::prepared_adapter::NumericPreparationProvider { addressable: true }
            .communication_capabilities()
    }
}

pub(super) type Trajectory = (Vec<NumericTensor>, State);
pub(super) struct Assembler;
impl PreparedExecutableAssembler<()> for Assembler {
    type Executable = Trajectory;
    type Output = Trajectory;
    type Error = String;
    fn floating_state_dtype(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<StateStorageDtype, String> {
        Ok(StateStorageDtype::F32)
    }
    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), String> {
        Err("no partition resources in replicated fixture".into())
    }
    fn finish(
        self,
        mut parts: PreparedExecutableParts<Trajectory, ()>,
    ) -> Result<Trajectory, String> {
        assert!(parts.take_communication().is_none());
        assert!(parts.take_processor().is_none());
        assert_eq!(parts.floating_state_bytes().get(), 4);
        Ok(parts.into_executable())
    }
}
pub(super) struct WrongProfile;
impl eredu_architectures::routed_text::RoutedTextArchitectureVisitor<NumericBackend, State>
    for WrongProfile
{
    type Output = Trajectory;
    type Error = String;
    fn visit<A>(
        self,
        _: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        _: SharedCheckpointSource,
    ) -> Result<Trajectory, String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        Err("ordinary target entered a non-stream state profile".into())
    }
}
impl eredu_architectures::routed_text::Relu2RoutedTextArchitectureVisitor<NumericBackend, State>
    for WrongProfile
{
    type Output = Trajectory;
    type Error = String;
    fn visit<A>(
        self,
        _: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        _: SharedCheckpointSource,
    ) -> Result<Trajectory, String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        Err("ordinary target entered the relu2 state profile".into())
    }
}
struct Binding<'a> {
    ctx: &'a NumericContext,
    residency: LayerWeightResidency,
}
impl eredu_architectures::routed_text::RoutedTextArchitectureVisitor<NumericBackend, State>
    for Binding<'_>
{
    type Output = Trajectory;
    type Error = String;
    fn visit<A>(
        self,
        prepared: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        source: SharedCheckpointSource,
    ) -> Result<Trajectory, String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        let ctx = self.ctx;
        assert!(
            source
                .source_keys()
                .iter()
                .all(|key| !key.starts_with("mtp.")),
            "ordinary target binding must not receive prediction parameters"
        );
        let observed = source.clone();
        let mechanisms = if self.residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(source)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(source)
        };
        macro_rules! exercise {
            ($session:expr, $facts:expr) => {{
                let (_, _, model_type, residency) = $facts.into_parts();
                assert_eq!(model_type, "qwen4_exp_text");
                assert_eq!(residency, self.residency);
                let mut session = $session;
                let ids = [3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
                let mut last = None;
                for part in [&ids[..3], &ids[3..8], &ids[8..]] {
                    let tokens =
                        NumericTensor::from_i32_slice(part, &[1, part.len() as i32], ctx).unwrap();
                    last = Some(session.prefill(&tokens, None, ctx).unwrap());
                }
                let mut outputs = vec![last.unwrap()];
                for id in 0..16 {
                    let token = NumericTensor::from_i32_slice(&[id], &[1, 1], ctx).unwrap();
                    outputs.push(session.decode(&token, ctx).unwrap());
                }
                let saved = session.checkpoint(ctx).unwrap();
                let token = NumericTensor::from_i32_slice(&[5], &[1, 1], ctx).unwrap();
                let first = session.decode(&token, ctx).unwrap();
                let charged = observed.source_diagnostics().unwrap().physical_reads;
                session.rollback(saved.clone(), ctx).unwrap();
                assert_eq!(
                    observed.source_diagnostics().unwrap().physical_reads,
                    charged
                );
                session::assert_checkpoint(&saved, &session.checkpoint(ctx).unwrap());
                assert_tensor_exact(
                    &first,
                    &session.decode(&token, ctx).unwrap(),
                    "ordinary registry rollback",
                );
                Ok::<_, String>((outputs, saved))
            }};
        }
        construct_selected_routed_session(
            prepared,
            mechanisms,
            ctx,
            |_, _, rows| {
                let rows = rows.expect("lexical row authority retained by ordinary construction");
                let providers = rows
                    .prepared()
                    .bind(
                        rows.prepared()
                            .entries()
                            .iter()
                            .map(|(id, entry)| {
                                (id.clone(), super::super::row_bank::SourceRows::new(entry))
                            })
                            .collect(),
                    )
                    .unwrap();
                Ok::<_, String>(
                    (
                        BTreeMap::<
                            RoutedBankId,
                            (NumericGroupedBankMechanism, NumericIndexedMovement),
                        >::new(),
                        Some(providers),
                    ),
                )
            },
            (),
            |_, session, facts| exercise!(session, facts),
            |_, session, facts| exercise!(session, facts),
        )
        .map_err(|e| e.to_string())
    }
}
fn run(path: &std::path::Path, residency: LayerWeightResidency) -> Trajectory {
    let request = load_policy::request(residency);
    run_with_request(path, residency, &request)
}
fn run_with_request(
    path: &std::path::Path,
    residency: LayerWeightResidency,
    request: &NormalizedLoadRequest,
) -> Trajectory {
    let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
    let mechanisms = Mechanisms::default();
    let selected =
        eredu_architectures::select_preparation(&inspection, request, &mechanisms).unwrap();
    assert!(mechanisms.row_queries.get() > 0);
    let admission =
        eredu_core::ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
    let before_bind_queries = mechanisms.row_queries.get();
    let sources =
        eredu_architectures::prepared_sources::prepare_model_sources(admission, selected).unwrap();
    assert_eq!(mechanisms.row_queries.get(), before_bind_queries);
    assert!(!sources.execution_identity().is_empty());
    assert!(Arc::ptr_eq(sources.primary(), sources.complete()));
    assert!(Arc::ptr_eq(sources.primary(), sources.target()));
    let reads = sources
        .primary()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    assert_eq!(reads, if path.is_dir() { 3 } else { 0 });
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let routes = PreparedExecutionRoutes::new().with_routed(RoutedRoute::<
        NumericBackend,
        State,
        State,
        State,
        _,
        _,
        _,
        _,
    >::new(
        &context,
        &context,
        WrongProfile,
        WrongProfile,
        WrongProfile,
        Binding {
            ctx: &context,
            residency,
        },
    ));
    construct_prepared_execution(sources, None, routes, Assembler).unwrap()
}
#[test]
fn qwen4_registry_constructs_both_artifacts_across_residency() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    let reference = run(
        &fixtures.safetensors_path,
        LayerWeightResidency::FullyResident,
    );
    assert_eq!(reference.0.len(), 17);
    let expected: serde_json::Value =
        serde_json::from_str(eredu_evaluation::qwen4_exp::DENSE_TRAJECTORY_JSON).unwrap();
    for (actual, expected) in reference
        .0
        .iter()
        .zip(expected["logits"].as_array().unwrap())
    {
        let expected = expected.as_array().unwrap();
        for (actual, expected) in actual.data[actual.data.len() - 16..].iter().zip(expected) {
            let expected = expected.as_f64().unwrap() as f32;
            assert!((actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs());
        }
    }
    for path in [&fixtures.safetensors_path, &fixtures.gguf_path] {
        for residency in [
            LayerWeightResidency::FullyResident,
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
            )),
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
            ),
        ] {
            let actual = run(path, residency);
            for (actual, expected) in actual.0.iter().zip(&reference.0) {
                assert_tensor_close(actual, expected, "ordinary registry trajectory");
            }
            assert_eq!(actual.1.layout(), reference.1.layout());
            for (actual, expected) in actual.1.as_ref().iter().zip(reference.1.as_ref()) {
                let actual: Vec<_> = RuntimeLayerState::retained_values(actual).collect();
                let expected: Vec<_> = RuntimeLayerState::retained_values(expected).collect();
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert_tensor_close(actual, expected, "ordinary registry retained state");
                    assert_eq!(actual.exact_i32, expected.exact_i32);
                }
            }
        }
    }
}
#[test]
fn qwen4_registry_defaults_match_explicit_execution_for_both_artifacts() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    for path in [&fixtures.safetensors_path, &fixtures.gguf_path] {
        let explicit = run(path, LayerWeightResidency::FullyResident);
        let default = run_with_request(
            path,
            LayerWeightResidency::FullyResident,
            &NormalizedLoadRequest::default(),
        );
        assert_eq!(default.0.len(), 17);
        for (actual, expected) in default.0.iter().zip(&explicit.0) {
            assert_tensor_close(actual, expected, "default ordinary trajectory");
        }
        for (actual, expected) in default.1.as_ref().iter().zip(explicit.1.as_ref()) {
            assert_eq!(actual.position(), expected.position());
            // Policies can choose different page sizes and reserve spare lanes
            // while retaining identical logical streams for this invocation.
            let active = |state: &NumericHybridLayerState| {
                state
                    .streams
                    .iter()
                    .filter(|(_, _, stream)| !stream.is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let actual_streams = active(actual);
            let expected_streams = active(expected);
            assert_eq!(actual_streams.len(), expected_streams.len());
            for ((slot, lane, actual), (expected_slot, expected_lane, expected)) in
                actual_streams.iter().zip(&expected_streams)
            {
                assert_eq!((slot, lane), (expected_slot, expected_lane));
                assert_eq!(actual.specification(), expected.specification());
                assert_eq!(actual.len(), expected.len());
                if !actual.is_empty() {
                    let joined = |stream: &eredu_runtime::ResidentAppendStream<NumericTensor>| {
                        NumericTensor::concatenate(
                            &stream.retained_values().cloned().collect::<Vec<_>>(),
                            0,
                            &NumericContext::default(),
                        )
                        .unwrap()
                    };
                    let actual = joined(actual);
                    let expected = joined(expected);
                    assert_tensor_close(&actual, &expected, "default ordinary stream");
                    assert_eq!(actual.exact_i32, expected.exact_i32);
                }
            }
            let mut actual = actual.clone();
            let mut expected = expected.clone();
            actual.streams.clear();
            expected.streams.clear();
            let actual: Vec<_> = RuntimeLayerState::retained_values(&actual).collect();
            let expected: Vec<_> = RuntimeLayerState::retained_values(&expected).collect();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_tensor_close(actual, expected, "default ordinary retained state");
                assert_eq!(actual.exact_i32, expected.exact_i32);
            }
        }
    }
}

#[test]
fn qwen4_registry_default_target_ignores_declared_prediction_weights() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    let request = NormalizedLoadRequest::default();
    let reference = run_with_request(
        &fixtures.safetensors_path,
        LayerWeightResidency::FullyResident,
        &request,
    );
    let path = fixtures.safetensors_path.clone();
    // Release old prepared file identities before replacing the miniature artifact.
    drop(fixtures);
    eredu_evaluation::qwen4_exp::add_prediction_weights(&path).unwrap();
    let inspection = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
    assert!(inspection
        .tensors()
        .descriptors()
        .any(|tensor| tensor.name.starts_with("mtp.")));
    let selected =
        eredu_architectures::select_preparation(&inspection, &request, &Mechanisms::default())
            .unwrap();
    assert!(selected.prediction_extension().is_none());
    assert!(selected.prediction_realization().is_none());
    assert!(selected
        .text_realization()
        .requirements()
        .auxiliary_parameters()
        .is_empty());
    let actual = run_with_request(&path, LayerWeightResidency::FullyResident, &request);
    for (actual, expected) in actual.0.iter().zip(&reference.0) {
        assert_tensor_exact(actual, expected, "default target with unused MTP weights");
    }
    session::assert_checkpoint(&actual.1, &reference.1);
}

#[test]
fn qwen4_registry_default_bounds_recheck_row_facts_without_payload_access() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    for path in [&fixtures.safetensors_path, &fixtures.gguf_path] {
        let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
        let supported = Mechanisms::default();
        let request = NormalizedLoadRequest::default();
        let payload = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.clone()
        };
        let hidden = payload.with_extension("unavailable");
        std::fs::rename(&payload, &hidden).unwrap();
        let cold = eredu_architectures::select_preparation(&inspection, &request, &supported);
        let report = eredu_architectures::model_inspection::inspect_selected_model(
            inspection.clone(),
            &request,
            &supported,
            eredu_core::MediaFeatureAvailability {
                image: false,
                audio: false,
            },
        );
        std::fs::rename(&hidden, &payload).unwrap();
        cold.expect("ordinary selection must consume retained headers without reopening payloads");
        let inspected = report
            .selected()
            .unwrap_or_else(|| panic!("ordinary metadata report failed: {:?}", report.report()));
        assert!(
            matches!(inspected.memory_materialization_workspace(), eredu_core::Observed::Available { value, .. } if value.ordinary_recipe_peak_bytes > 0)
        );
        let missing_rows = Mechanisms {
            rows: false,
            ..Default::default()
        };
        assert!(
            eredu_architectures::select_preparation(&inspection, &request, &missing_rows).is_err()
        );
        assert!(missing_rows.row_queries.get() > 0);
        let missing_dtype = Mechanisms {
            dtype_supported: false,
            ..Default::default()
        };
        assert!(
            eredu_architectures::select_preparation(&inspection, &request, &missing_dtype).is_err()
        );
        eredu_architectures::select_preparation(&inspection, &request, &supported).unwrap();

        let embedded = request.with_drafting(DraftingLoadRequest::embedded(1).unwrap());
        let error = eredu_architectures::select_preparation(&inspection, &embedded, &supported);
        use eredu_architectures::qwen4_exp::prepared::{PreparationError, TargetLoadError};
        assert!(matches!(error,
            Err(eredu_architectures::PreparationSelectionError::Qwen4TargetLoad(error))
                if matches!(error.as_ref(),
                    TargetLoadError::Preparation(PreparationError::MissingPrediction))
        ));
    }
}

#[test]
fn qwen4_registry_rejects_selection_from_a_different_inspection() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    for path in [&fixtures.safetensors_path, &fixtures.gguf_path] {
        let original = eredu_architectures::configuration::inspect_artifact(path).unwrap();
        let independent = eredu_architectures::configuration::inspect_artifact(path).unwrap();
        let selected = eredu_architectures::select_preparation(
            &original,
            &load_policy::request(LayerWeightResidency::FullyResident),
            &Mechanisms::default(),
        )
        .unwrap();
        let wrong = eredu_core::ModelPreparationPlan::from_retained_admission(
            independent,
            selected.admission(),
        )
        .unwrap();
        assert!(matches!(
            eredu_architectures::prepared_sources::prepare_model_sources(wrong, selected),
            Err(
                eredu_architectures::prepared_sources::PreparedModelSourcesError::InvalidSelection(
                    _
                )
            )
        ));
    }
}

#[test]
fn qwen4_registry_keys_retained_geometry_by_normalized_request() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    for path in [&fixtures.safetensors_path, &fixtures.gguf_path] {
        let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
        let mechanisms = Mechanisms::default();
        let request = load_policy::request(LayerWeightResidency::FullyResident);
        let first =
            eredu_architectures::select_preparation(&inspection, &request, &mechanisms).unwrap();
        let policy = eredu_evaluation::qwen4_exp::bounded_policy();
        let changed = request.clone().with_bounded_execution(
            BoundedExecutionPolicy::new(
                InvocationLimits::new(2, 32, 64).unwrap(),
                policy.selection(),
                policy.rows(),
                policy.append(),
            )
            .unwrap(),
        );
        let second =
            eredu_architectures::select_preparation(&inspection, &changed, &mechanisms).unwrap();
        assert_ne!(
            first
                .text_realization()
                .requirements()
                .architecture_identity(),
            second
                .text_realization()
                .requirements()
                .architecture_identity()
        );
        let restored =
            eredu_architectures::select_preparation(&inspection, &request, &mechanisms).unwrap();
        assert_eq!(
            first.text_realization().requirements(),
            restored.text_realization().requirements()
        );
        for selected in [first, second] {
            let identity = selected
                .text_realization()
                .requirements()
                .architecture_identity()
                .to_owned();
            let admission = eredu_core::ModelPreparationPlan::from_retained_admission(
                inspection.clone(),
                selected.admission(),
            )
            .unwrap();
            let sources =
                eredu_architectures::prepared_sources::prepare_model_sources(admission, selected)
                    .unwrap();
            assert_eq!(sources.execution_identity(), identity);
        }
    }
}
