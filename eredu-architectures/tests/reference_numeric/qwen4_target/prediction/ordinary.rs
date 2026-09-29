//! Public ordinary preparation retains the same target and prediction authorities.
use super::*;
use eredu_architectures::{prepared_sources::prepare_model_sources, select_preparation};
use eredu_runtime::{DraftingLoadRequest, NormalizedLoadRequest};

fn artifact() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    let path = fixtures.safetensors_path.clone();
    drop(fixtures);
    eredu_evaluation::qwen4_exp::add_prediction_weights(&path).unwrap();
    (directory, path)
}

#[test]
fn qwen4_ordinary_prediction_selects_from_retained_headers_and_preserves_source_roles() {
    let (_directory, path) = artifact();
    let inspection = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
    let request =
        NormalizedLoadRequest::default().with_drafting(DraftingLoadRequest::embedded(1).unwrap());
    let mechanisms = super::super::registry::Mechanisms {
        prediction: true,
        ..Default::default()
    };
    let payload = path.join("model.safetensors");
    let hidden = path.join("unavailable.safetensors");
    std::fs::rename(&payload, &hidden).unwrap();
    let selected = select_preparation(&inspection, &request, &mechanisms);
    std::fs::rename(&hidden, &payload).unwrap();
    let selected = selected.expect("prediction planning consumes the admitted headers");
    // This family retains typed prediction authority rather than the older
    // generic prediction-extension enum.
    assert!(selected.prediction_extension().is_none());
    assert!(selected.prediction_realization().is_some());
    assert!(selected
        .text_realization()
        .requirements()
        .auxiliary_parameters()
        .iter()
        .any(|parameter| parameter.name().starts_with("mtp.")));
    let topology = selected.embedded_prediction_topology().unwrap().unwrap();
    assert_eq!(topology.proposal_capacity, 1);
    assert_eq!(topology.state.len(), 1);
    assert_eq!(topology.state[0].layer, 0);
    assert_eq!(topology.state[0].processed_token_offset, 0);
    assert!(matches!(&topology.state[0].policy,
        eredu_core::cache::LayerCachePolicy::KeyValueWithState {
            num_key_value_heads, head_dim, ..
        } if num_key_value_heads.get() == 2 && head_dim.get() == 8
    ));
    assert!(!topology
        .nodes_for_scope(eredu_core::speculative::SpeculativeCaptureScope::Prediction { depth: 0 })
        .collect::<Vec<_>>()
        .is_empty());
    let features = topology.target_features.entries();
    assert_eq!(features.len(), 1);
    assert_eq!(&features[0].shape()[2..], &[2, 32]);
    topology
        .target_features
        .instantiate(vec![vec![1, 19, 2, 32]])
        .unwrap();
    assert!(topology
        .target_features
        .instantiate(vec![vec![1, 19, 32]])
        .is_err());

    // Header planning remains valid while a payload is unavailable, but changing
    // the admitted file identity must invalidate subsequent source preparation.
    let stale_admission = eredu_core::ModelPreparationPlan::from_retained_admission(
        inspection.clone(),
        selected.admission(),
    )
    .unwrap();
    assert!(matches!(
        prepare_model_sources(stale_admission, selected),
        Err(eredu_architectures::prepared_sources::PreparedModelSourcesError::InvalidSelection(_))
    ));
    let inspection = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
    let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
    let inspected = eredu_architectures::model_inspection::inspect_selected_model(
        inspection.clone(),
        &request,
        &mechanisms,
        eredu_core::MediaFeatureAvailability {
            image: false,
            audio: false,
        },
    );
    assert_eq!(
        inspected.report().resources.embedded_draft_layers.value(),
        Some(&1)
    );
    assert_eq!(
        inspected.report().resources.embedded_draft_capacity.value(),
        Some(&1)
    );
    let admission = eredu_core::ModelPreparationPlan::from_retained_admission(
        inspection.clone(),
        selected.admission(),
    )
    .unwrap();
    let sources = prepare_model_sources(admission, selected).unwrap();
    let discovery = sources.prepare_discovery(Default::default(), Default::default());
    assert_eq!(
        discovery.embedded_prediction_topology().unwrap(),
        Some(&topology)
    );
    let extension = sources
        .extension()
        .expect("selected prediction source role");
    assert!(sources.prediction_extension().is_none());
    assert!(Arc::ptr_eq(sources.primary(), sources.complete()));
    assert!(sources
        .target()
        .source_keys()
        .iter()
        .all(|key| !key.starts_with("mtp.")));
    assert!(extension
        .source_keys()
        .iter()
        .any(|key| key.starts_with("mtp.")));
    for (source, name) in [
        (sources.target(), "model.embed_tokens.weight"),
        (sources.target(), "lm_head.weight"),
        (extension, "mtp.fc_hidden.weight"),
    ] {
        assert_eq!(
            source.source_metadata(name).unwrap(),
            sources.primary().source_metadata(name).unwrap()
        );
        assert_eq!(
            source.source_provenance(name).unwrap(),
            sources.primary().source_provenance(name).unwrap()
        );
    }
    assert!(sources
        .target()
        .source_metadata("mtp.fc_hidden.weight")
        .is_err());
    assert!(sources
        .target()
        .acquire_lease("mtp.fc_hidden.weight".into())
        .is_err());

    // Explicit drafting needs actual mechanisms; target-only defaults continue
    // to admit the same MTP-bearing artifact on a target-only backend.
    assert!(select_preparation(
        &inspection,
        &request,
        &super::super::registry::Mechanisms::default()
    )
    .is_err());
    let missing_state = super::super::registry::Mechanisms {
        prediction: true,
        prediction_state: false,
        ..Default::default()
    };
    assert!(matches!(
        select_preparation(&inspection, &request, &missing_state),
        Err(eredu_architectures::PreparationSelectionError::Qwen4TargetSelection(_))
    ));
    let restored = select_preparation(&inspection, &request, &mechanisms).unwrap();
    assert!(restored.prediction_realization().is_some());
    let default = select_preparation(
        &inspection,
        &NormalizedLoadRequest::default(),
        &super::super::registry::Mechanisms::default(),
    )
    .unwrap();
    assert!(default.prediction_extension().is_none());
    assert!(default.prediction_realization().is_none());
}

use super::executor::{Materializer, SourceBinding};
use eredu_architectures::{prediction_extension::*, prepared_execution::*, routed_text::*};
use eredu_runtime::*;

type Providers = RoutedBankProviders<PlannedResidentBank>;
type Session<A> = ReplicatedTextSession<
    A,
    NumericBackend,
    NumericReplicatedMechanisms,
    RoutedReplicatedTextExecution<
        ParameterProviders<
            Providers,
            Option<RowLookupProviders<BoundedRowLookup<super::super::super::row_bank::SourceRows>>>,
        >,
    >,
>;
struct Invoker<'a, A>
where
    A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
        + RoutedLayeredArchitecture<NumericBackend, State>
        + 'static,
    A::StaticModules: Clone,
{
    session: &'a mut Session<A>,
    ctx: &'a NumericContext,
}
impl<A> PredictionOperationInvoker<A, NumericBackend, State> for Invoker<'_, A>
where
    A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
        + RoutedLayeredArchitecture<NumericBackend, State>
        + 'static,
    A::StaticModules: Clone,
{
    type Error = Error;
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Error>
    where
        O: PredictionTargetOperation<A, NumericBackend, State>,
    {
        self.session
            .apply_prediction_target_operation(operation, self.ctx)
            .map_err(Error::backend_source)
    }
    fn invalid(message: String) -> Error {
        Error::backend(message)
    }
}
struct Binding<'a> {
    ctx: &'a NumericContext,
    residency: LayerWeightResidency,
}
impl RoutedTextArchitectureVisitor<NumericBackend, State> for Binding<'_> {
    type Output = super::super::registry::Trajectory;
    type Error = String;
    fn visit<A>(
        self,
        _: PreparedRoutedTextArchitecture<A>,
        _: SharedCheckpointSource,
    ) -> Result<Self::Output, String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        Err("explicit prediction request reached target-only binding".into())
    }

    fn visit_prediction<A, W>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        prediction: W,
        target_source: SharedCheckpointSource,
        provider_source: SharedCheckpointSource,
        binding: PredictionBinding,
    ) -> Result<Self::Output, RoutedTextDispatchError<String>>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
        W: PreparedRoutedPrediction<NumericBackend, A>,
    {
        let ctx = self.ctx;
        assert!(target_source
            .source_keys()
            .iter()
            .all(|key| !key.starts_with("mtp.")));
        assert!(provider_source
            .source_metadata("mtp.fc_hidden.weight")
            .is_ok());
        let target_layout = prepared.text().selected().state().layout().clone();
        let banks = prediction.banks();
        assert_eq!(banks.len(), 1);
        let mut source_binding = SourceBinding {
            source: prediction.source().clone(),
            context: ctx,
            residency: self.residency,
            roles: vec![],
        };
        let extension = prediction
            .materialize::<Materializer>(&mut source_binding, |_, selected| {
                assert_ne!(selected.layout(), &target_layout);
                assert!(!selected.append_streams().is_empty());
                super::installed::realize(selected)
            })
            .map_err(|e| RoutedTextDispatchError::Backend(e.to_string()))?;
        assert_eq!(source_binding.roles.len(), 2);
        let rows = prepared.row_lookups().unwrap().prepared();
        let rows = rows
            .bind(
                rows.entries()
                    .iter()
                    .map(|(id, entry)| {
                        (
                            id.clone(),
                            super::super::super::row_bank::SourceRows::new(entry),
                        )
                    })
                    .collect(),
            )
            .unwrap();
        let mechanisms = if self.residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(target_source)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(target_source)
        };
        let (mut session, providers) = prepared
            .construct_resident_session::<NumericBackend, _, _>(mechanisms, Some(rows), &banks, ctx)
            .map_err(RoutedTextDispatchError::Backend)?;
        let mut extension = W::with_provider::<Materializer, _>(extension, providers.unwrap());
        let mut lane = extension.new_state();
        assert_eq!(extension.depth(), 1);
        let prompt = NumericTensor::token_ids(&[
            3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        ]);
        let (logits, mut capture) = session
            .prefill_prediction_target(&prompt, None, ctx)
            .unwrap();
        // The target capture keeps the complete residual width required by MTP.
        assert_eq!(capture.shape.len(), 4);
        assert_eq!(capture.shape[2], 2);
        let mut outputs = vec![logits.axis_slice(1, 18, 19)];
        capture = capture.axis_slice(1, 18, 19);
        for id in 0..16 {
            let token = NumericTensor::token_ids(&[id]);
            let untouched = session.checkpoint(ctx).unwrap();
            let saved_lane = extension.snapshot(&lane, ctx).unwrap().unwrap();
            let draft = extension
                .logits::<State, _>(
                    &mut Invoker {
                        session: &mut session,
                        ctx,
                    },
                    &capture,
                    &token,
                    0,
                    &mut lane,
                )
                .unwrap();
            assert!(draft.0.data.iter().all(|v| v.is_finite()));
            assert!(draft.0.data.iter().any(|v| v.abs() > 1e-5));
            super::super::session::assert_checkpoint(&untouched, &session.checkpoint(ctx).unwrap());
            // Reject a tentative lane, restore it, and replay the exact proposal.
            lane = saved_lane;
            let replay = extension
                .logits::<State, _>(
                    &mut Invoker {
                        session: &mut session,
                        ctx,
                    },
                    &capture,
                    &token,
                    0,
                    &mut lane,
                )
                .unwrap();
            assert_tensor_exact(&draft.0, &replay.0, "ordinary prediction rollback logits");
            assert_tensor_exact(&draft.1, &replay.1, "ordinary prediction rollback capture");
            outputs.push(draft.0);
            let next = session.decode_prediction_target(&token, ctx).unwrap();
            outputs.push(next.0);
            capture = next.1;
        }
        let _ = binding.selected();
        Ok((outputs, session.checkpoint(ctx).unwrap()))
    }
}

pub(super) fn run(
    path: &std::path::Path,
    prediction_source: Option<&std::path::Path>,
    residency: LayerWeightResidency,
) -> super::super::registry::Trajectory {
    let request = NormalizedLoadRequest::default()
        .with_weight_residency(WeightResidency::with_layers(residency))
        .with_drafting(DraftingLoadRequest::embedded(1).unwrap());
    let request = if let Some(path) = prediction_source {
        request.with_prediction_source(path.to_path_buf())
    } else {
        request
    };
    let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
    let mechanisms = super::super::registry::Mechanisms {
        prediction: true,
        ..Default::default()
    };
    let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
    let admission =
        eredu_core::ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
    let sources = prepare_model_sources(admission, selected).unwrap();
    let ctx = NumericContext {
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
        &ctx,
        &ctx,
        super::super::registry::WrongProfile,
        super::super::registry::WrongProfile,
        super::super::registry::WrongProfile,
        Binding {
            ctx: &ctx,
            residency,
        },
    ));
    construct_prepared_execution(sources, None, routes, super::super::registry::Assembler).unwrap()
}

#[test]
fn qwen4_ordinary_prediction_constructs_and_replays_across_weight_residency() {
    let (_directory, path) = artifact();
    let baseline = run(&path, None, LayerWeightResidency::FullyResident);
    assert_eq!(baseline.0.len(), 33);
    let golden: serde_json::Value =
        serde_json::from_str(eredu_evaluation::qwen4_exp::DENSE_TRAJECTORY_JSON).unwrap();
    for (actual, expected) in baseline
        .0
        .iter()
        .step_by(2)
        .zip(golden["logits"].as_array().unwrap())
    {
        for (actual, expected) in actual.data.iter().zip(expected.as_array().unwrap()) {
            let expected = expected.as_f64().unwrap() as f32;
            assert!(
                (actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs(),
                "prediction-enabled target differs from independent ordinary trajectory"
            );
        }
    }
    for residency in [
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let actual = run(&path, None, residency);
        for (actual, expected) in actual.0.iter().zip(&baseline.0) {
            assert_tensor_exact(actual, expected, "ordinary prediction residency trajectory");
        }
        super::super::session::assert_checkpoint(&actual.1, &baseline.1);
    }
}

#[test]
fn qwen4_ordinary_gguf_prediction_companion_matches_safetensors_and_residencies() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let reference = run(
        &fixtures.safetensors_path,
        None,
        LayerWeightResidency::FullyResident,
    );
    let baseline = run(
        &fixtures.gguf_path,
        Some(&fixtures.safetensors_path),
        LayerWeightResidency::FullyResident,
    );
    assert_eq!(baseline.0.len(), 33);
    for (actual, expected) in baseline.0.iter().zip(&reference.0) {
        assert_tensor_close(
            actual,
            expected,
            "ordinary GGUF prediction companion trajectory",
        );
    }
    for residency in [
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let actual = run(
            &fixtures.gguf_path,
            Some(&fixtures.safetensors_path),
            residency,
        );
        for (actual, expected) in actual.0.iter().zip(&baseline.0) {
            assert_tensor_exact(
                actual,
                expected,
                "ordinary GGUF prediction residency trajectory",
            );
        }
        super::super::session::assert_checkpoint(&actual.1, &baseline.1);
    }
}

#[test]
fn qwen4_ordinary_gguf_prediction_companion_retains_sources_and_rejects_stale_admission() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let prediction: SharedCheckpointSource = Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(&fixtures.safetensors_path).unwrap(),
    );
    let mechanisms = super::super::registry::Mechanisms {
        prediction: true,
        ..Default::default()
    };
    let request = NormalizedLoadRequest::default()
        .with_drafting(DraftingLoadRequest::embedded(1).unwrap())
        .with_prediction_source(fixtures.safetensors_path.clone());
    let inspection =
        eredu_architectures::configuration::inspect_artifact(&fixtures.gguf_path).unwrap();
    let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
    let topology = selected.embedded_prediction_topology().unwrap().unwrap();
    assert_eq!(topology.proposal_capacity, 1);
    let admission = eredu_core::ModelPreparationPlan::from_retained_admission(
        inspection.clone(),
        selected.admission(),
    )
    .unwrap();
    let sources = prepare_model_sources(admission, selected).unwrap();
    let extension = sources
        .extension()
        .expect("retained separate prediction source");
    assert!(sources
        .target()
        .source_keys()
        .iter()
        .all(|name| !name.starts_with("mtp.")));
    assert!(extension
        .source_keys()
        .iter()
        .all(|name| name.starts_with("mtp.")));
    for name in [
        "mtp.fc_hidden.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
    ] {
        assert_eq!(
            extension.source_metadata(name).unwrap(),
            prediction.source_metadata(name).unwrap()
        );
        assert_eq!(
            extension.source_provenance(name).unwrap(),
            prediction.source_provenance(name).unwrap()
        );
    }
    // Vocabulary stays in the GGUF target, even though the supplied complete
    // SafeTensors artifact also contains its own embedding and output matrices.
    for (name, physical) in [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("lm_head.weight", "output.weight"),
    ] {
        let provenance = sources.target().source_provenance(name).unwrap();
        assert_eq!(
            provenance,
            fixtures.gguf.artifact().source_provenance(name).unwrap()
        );
        assert_eq!(provenance.physical_tensor, physical);
        assert_eq!(provenance.backing_shard.as_ref(), Some(&fixtures.gguf_path));
        assert!(extension.source_metadata(name).is_err());
    }
    assert!(sources
        .target()
        .source_metadata("mtp.fc_hidden.weight")
        .is_err());
    assert!(sources
        .target()
        .acquire_lease("mtp.fc_hidden.weight".into())
        .is_err());
    drop(sources);
    drop(prediction);

    // The target file is untouched. Replacing only the admitted companion must
    // reject acquisition through the retained selection before native execution.
    let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
    let admission =
        eredu_core::ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
    let payload = fixtures.safetensors_path.join("model.safetensors");
    let replaced = fixtures.safetensors_path.join("replacement.safetensors");
    std::fs::copy(&payload, &replaced).unwrap();
    std::fs::rename(&replaced, &payload).unwrap();
    use eredu_architectures::prepared_sources::PreparedModelSourcesError;
    use eredu_checkpoint::store::StoreError;
    use eredu_core::artifact::ArtifactError;
    match prepare_model_sources(admission, selected) {
        Ok(sources) => {
            // Header-only source construction may defer the filesystem guard
            // until the first lease. The changed payload must never be read.
            let source = sources.extension().unwrap();
            assert!(matches!(
                source.acquire_lease("mtp.fc_hidden.weight".into()),
                Err(StoreError::AdmittedFileChanged { .. })
            ));
            assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
        }
        Err(PreparedModelSourcesError::Artifact(ArtifactError::CheckpointStore(
            StoreError::AdmittedFileChanged { .. },
        ))) => {}
        Err(error) => panic!("unexpected changed companion rejection: {error:?}"),
    }
}

#[test]
fn qwen4_ordinary_gguf_prediction_requires_matching_companion_geometry() {
    use eredu_architectures::qwen4_exp::prepared::{PreparationError, TargetLoadError};
    use eredu_architectures::PreparationSelectionError;

    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let inspection =
        eredu_architectures::configuration::inspect_artifact(&fixtures.gguf_path).unwrap();
    let mechanisms = super::super::registry::Mechanisms {
        prediction: true,
        ..Default::default()
    };
    let request =
        NormalizedLoadRequest::default().with_drafting(DraftingLoadRequest::embedded(1).unwrap());
    assert!(matches!(
        select_preparation(&inspection, &request, &mechanisms),
        Err(PreparationSelectionError::Qwen4TargetLoad(error))
            if matches!(error.as_ref(), TargetLoadError::Preparation(PreparationError::MissingPrediction))
    ));
    eredu_evaluation::qwen4_exp::set_context_length(&fixtures.safetensors_path, 256).unwrap();
    let request = request.with_prediction_source(fixtures.safetensors_path);
    assert!(matches!(
        select_preparation(&inspection, &request, &mechanisms),
        Err(PreparationSelectionError::Qwen4TargetLoad(error))
            if matches!(error.as_ref(), TargetLoadError::Preparation(PreparationError::PredictionMismatch { field: "max_positions" }))
    ));
}

#[test]
fn qwen4_ordinary_gguf_prediction_accepts_prediction_only_safetensors_artifact() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let payload = fixtures.safetensors_path.join("model.safetensors");
    let bytes = std::fs::read(&payload).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    safetensors::tensor::serialize_to_file(
        tensors
            .tensors()
            .into_iter()
            .filter(|(name, _)| name.starts_with("mtp.")),
        None,
        &payload,
    )
    .unwrap();
    let request = NormalizedLoadRequest::default()
        .with_drafting(DraftingLoadRequest::embedded(1).unwrap())
        .with_prediction_source(fixtures.safetensors_path);
    let mechanisms = super::super::registry::Mechanisms {
        prediction: true,
        ..Default::default()
    };
    let inspection =
        eredu_architectures::configuration::inspect_artifact(&fixtures.gguf_path).unwrap();
    let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
    let admission =
        eredu_core::ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
    let sources = prepare_model_sources(admission, selected).unwrap();
    let prediction = sources.extension().unwrap();
    assert!(prediction.source_metadata("mtp.fc_hidden.weight").is_ok());
    assert!(prediction
        .source_metadata("model.embed_tokens.weight")
        .is_err());
    assert!(sources
        .target()
        .source_metadata("model.embed_tokens.weight")
        .is_ok());
}
