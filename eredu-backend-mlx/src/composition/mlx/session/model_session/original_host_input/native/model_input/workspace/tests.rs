use super::super::super::super::tests::{admitted, cold_config, settle, source};
use super::*;
use crate::memory_fixture::LedgerFixture;
use eredu_core::{Completion, OutputDemand};
use eredu_nn::workspace::WorkspaceMechanisms;

fn family(conditional: bool) {
    if !crate::tests::support::native_process::enter("original-media-workspace") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    assert!(
        safemlx::metal::is_available().unwrap(),
        "explicit native Metal diagnostic prerequisite"
    );
    let artifact = tempfile::tempdir().unwrap();
    let hidden = if conditional {
        crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
            artifact.path(),
            false,
        );
        16
    } else {
        crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
            artifact.path(),
            false,
            false,
        );
        64
    };
    for mode in 0..4 {
        let backend = admitted::backend(&pool);
        backend.stream().synchronize().unwrap();
        safemlx::memory::clear_cache();
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        let baseline = pool.fixture_host_charge().unwrap();
        let i = source(&pool, hidden);
        let equal = source(&pool, hidden);
        assert_eq!(i.content_digest(), equal.content_digest());
        assert!(!i.same_source(&equal));
        let stream = backend.stream().clone();
        let config = workspace_config(&backend, artifact.path(), mode);
        let a = config
            .prepared_sources()
            .plan_original_media_semantics(&i)
            .unwrap()
            .compile(&pool)
            .unwrap();
        let other = config
            .prepared_sources()
            .plan_original_media_semantics(&equal)
            .unwrap()
            .compile(&pool)
            .unwrap();
        let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
        let b = materializer
            .model_input_plan(&a)
            .unwrap()
            .materialize(&pool)
            .unwrap();
        let foreign = materializer
            .model_input_plan(&other)
            .unwrap()
            .materialize(&pool)
            .unwrap();
        let model = backend.prepare_model_borrowed(&config).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        runtime
            .session()
            .payload
            .model
            .erased()
            .prepare_completed_media_binding_fixture()
            .unwrap();
        let prompt = b
            .bind(&runtime, a)
            .unwrap()
            .with_prefill_chunk_positions(2.try_into().unwrap());
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 9,
            max_output_tokens: 3,
            prefill_chunk_positions: 2,
            output: OutputDemand::LastPosition,
        };
        // Equal bytes from a different actual I/B owner do not enter the mapper.
        let ordinary = NativeMemoryOwner::acquire_typed(&pool).unwrap();
        let context = WorkspaceContext::new(
            runtime
                .session()
                .payload
                .model
                .workspace_mechanisms()
                .unwrap(),
        );
        let semantics = prompt.original_media.as_ref().unwrap().semantics();
        crate::tensor::reset_workspace_slot_projections();
        let failure = OriginalMediaWorkspaceInput::project(
            foreign.0.storage().prepared().unwrap(),
            semantics,
            &context,
            &ordinary.unquoted_lease().unwrap(),
        )
        .unwrap_err();
        assert_eq!(crate::tensor::workspace_slot_projections(), 0);
        drop(ordinary);
        assert!(pool.unquoted_owner_count().unwrap() > 0);
        drop(failure);
        drop(context);
        drop(foreign);
        drop(other);
        drop(equal);

        let before = runtime.session().payload.model.erased().state_snapshot();
        crate::tensor::reset_prepared_rotary_calls();
        crate::tensor::reset_workspace_slot_projections();
        let report = prompt
            .quote_original_media_workspace(&runtime, geometry)
            .unwrap_or_else(|error| panic!("workspace mode {mode}: {error:?}"));
        assert_eq!(crate::tensor::workspace_slot_projections(), i.slot_count());
        assert_eq!(
            crate::tensor::prepared_rotary_calls(),
            0,
            "metadata trace executed native encoder"
        );
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before
        );
        assert_eq!(report.intervals().len(), 8);
        let first = &report.intervals()[0];
        assert!(first.cut_operation().unwrap() > 0);
        assert!(
            first.operations() > first.cut_operation().unwrap(),
            "decoder/rollback omitted from encoder interval"
        );
        assert!(first.compact_root_count() > 0);
        assert!(first.compact_backing_bytes().unwrap() > 0);
        assert!(first.closing_root_count() >= first.compact_root_count());
        let cut_new = first
            .encoder_cut_tensor_bytes()
            .expect("native encoder tensor bound");
        let full_new = first
            .tensor_bytes()
            .expect("native complete first-span tensor bound");
        let cut_union = first
            .encoder_cut_interval_bytes()
            .expect("encoder cut source/compact union");
        let full_union = first
            .tensor_interval_bytes()
            .expect("full first-span source/compact/checkpoint union");
        assert!(cut_new > 0);
        assert!(
            full_new > cut_new,
            "decoder/checkpoint allocation missing from full trace"
        );
        assert!(
            full_union > cut_union,
            "first decoder overlap omitted from numerical union"
        );
        assert!(full_union >= first.closing_backing_bytes().unwrap());
        assert!(
            report.intervals().iter().all(|span| {
                span.tensor_bytes().is_some_and(|bytes| bytes > 0)
                    && span
                        .tensor_interval_bytes()
                        .is_some_and(|bytes| bytes >= span.closing_backing_bytes().unwrap())
            }),
            "selected native numerical term is unpriced"
        );
        assert_eq!(first.prepared_rotary_operations(), 1);
        assert!(report.intervals()[1..]
            .iter()
            .all(|span| span.cut_operation().is_none() && span.prepared_rotary_operations() == 0));
        // First span is text-only; all future image/deepstack roots still appear
        // at its cut, and every following interval carries complete closing roots.
        assert!(report
            .intervals()
            .iter()
            .all(|span| span.closing_root_count() > 0));
        if mode == 0 {
            missing_fact_and_error_custody(&prompt, &runtime, geometry, &pool);
        }
        // The cold report retains its ordinary source exclusion. It must retire
        // before the same complete source can enter bounded execution.
        assert!(pool.unquoted_owner_count().unwrap() > 0);
        drop(report);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            safemlx::reclaim_allocation_owners();
            pool.unquoted_owner_count().unwrap() == 0
        });
        let (_, roots) = crate::tests::support::media_completion::observe(None, || {
            admitted::collect(&mut runtime, prompt.clone(), 1, true)
        });
        assert_eq!(
            roots.len(),
            10,
            "five actual decoder spans each complete their retained media roots"
        );
        let first = &roots[1];
        assert!(
            first.after && !first.backing.is_empty(),
            "first interval omitted future media roots"
        );
        assert!(first.backing.iter().all(Option::is_some));
        for pair in roots.chunks_exact(2) {
            assert!(!pair[0].after && pair[1].after);
            assert_eq!(pair[0].shapes, pair[1].shapes);
            assert_eq!(
                pair[1].backing, first.backing,
                "one completed encoder backing is retained across all decoder spans"
            );
            assert!(pair[1].ready.iter().all(|ready| *ready));
        }
        crate::tensor::reset_workspace_slot_projections();
        let busy = prompt
            .quote_original_media_workspace(&runtime, geometry)
            .unwrap_err();
        assert!(matches!(
            busy.cause,
            Cause::Boundary(WorkingMemoryError::ReservedWorkActive)
        ));
        assert_eq!(crate::tensor::workspace_slot_projections(), 0);
        drop(busy);
        runtime.reset().unwrap();
        let before = runtime.session().payload.model.erased().state_snapshot();
        let stale = prompt
            .quote_original_media_workspace(&runtime, geometry)
            .unwrap_err();
        assert!(
            matches!(
                stale.cause,
                // The retained input still owns the completed encoder's original
                // bank after reset. Ordinary diagnostic exclusion precedes source
                // revision validation and must reject without projecting any slot.
                Cause::Boundary(WorkingMemoryError::ReservedWorkActive)
            ),
            "stale source quote: {stale:?}"
        );
        assert_eq!(crate::tensor::workspace_slot_projections(), 0);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before
        );
        drop(stale);
        drop(prompt);
        drop(i);
        drop(runtime);
        drop(config);
        stream.synchronize().unwrap();
        settle(&pool, 0, baseline);
    }
}
// Exercise direct foreground and ordinary background disk strategies through
// their actual selected source, allocation facts and canonical numerical path.
fn workspace_config(
    backend: &MlxBackend<'_>,
    path: &std::path::Path,
    mode: usize,
) -> crate::composition::mlx::loading::MlxModelConfig {
    if mode != 2 {
        return cold_config(backend, path, if mode == 3 { 2 } else { mode });
    }
    use eredu_core::ModelLoadingBackend;
    let inspection = eredu_core::inspect_artifact(path, backend.configuration_resolver()).unwrap();
    let residency = eredu_runtime::WeightResidency::dense_disk_stream(
        eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 26, 0, 0, 0).unwrap(),
    );
    let request = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
    );
    eredu_core::prepare_inspected_model_config(backend, inspection, request).unwrap()
}

#[test]
fn original_vl_media_equation_intervals_keep_future_roots_and_actual_source_custody() {
    family(false);
}
#[test]
fn original_conditional_media_equation_intervals_keep_future_roots_and_actual_source_custody() {
    family(true);
}

#[derive(Debug)]
struct MissingEncoderFact;
impl std::fmt::Display for MissingEncoderFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("injected prepared rotary fact failure")
    }
}
impl std::error::Error for MissingEncoderFact {}
#[derive(Debug)]
struct IncompleteFacts {
    actual: crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
    fail: bool,
}
impl eredu_nn::workspace::WorkspaceMechanisms for IncompleteFacts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        self.actual.memory_topology()
    }
    fn output_placement(
        &self,
        operation: eredu_nn::workspace::WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        self.actual.output_placement(operation, output)
    }
    fn scratch_placement(
        &self,
        operation: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        self.actual.scratch_placement(operation)
    }
    fn allocation_host_control_bytes(
        &self,
        operation: eredu_nn::workspace::WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        self.actual.allocation_host_control_bytes(operation, output)
    }
    fn scratch_host_control_bytes(
        &self,
        operation: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, eredu_nn::Error> {
        self.actual.scratch_host_control_bytes(operation)
    }
    fn output_representation(
        &self,
        operation: eredu_nn::workspace::WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<eredu_nn::workspace::WorkspaceRepresentation> {
        self.actual.output_representation(operation, output)
    }
    fn operation_bound(
        &self,
        operation: &eredu_nn::workspace::WorkspaceOperation,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationBound>, eredu_nn::Error> {
        if matches!(
            operation.kind,
            eredu_nn::workspace::WorkspaceOperationKind::PreparedMultiAxisRotary(_)
        ) {
            return if self.fail {
                Err(eredu_nn::Error::backend_retained_source(MissingEncoderFact))
            } else {
                Ok(None)
            };
        }
        self.actual.operation_bound(operation)
    }
    fn host_workspace_bound(
        &self,
        operation: &eredu_nn::workspace::WorkspaceOperation,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostBound>, eredu_nn::Error> {
        self.actual.host_workspace_bound(operation)
    }
    fn projection_input_observation_mechanism(
        &self,
        format: &eredu_nn::LinearFormatSpec,
    ) -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>, eredu_nn::Error> {
        self.actual.projection_input_observation_mechanism(format)
    }
    fn grouped_observation_schedule(
        &self,
        bank: &eredu_nn::workspace::WorkspaceGroupedBank,
        tokens: u32,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceGroupedObservationSchedule>, eredu_nn::Error>
    {
        self.actual.grouped_observation_schedule(bank, tokens)
    }
}
fn missing_fact_and_error_custody(
    prompt: &MlxModelInput,
    runtime: &ModelRuntime<MlxBackend<'_>>,
    geometry: eredu_core::InferenceGeometry,
    pool: &MemoryLedger,
) {
    let baseline = pool.unquoted_owner_count().unwrap();
    let trace = |fail| {
        let ordinary = NativeMemoryOwner::acquire_typed(pool).unwrap();
        let executable = &runtime.session().payload.model;
        let context = WorkspaceContext::new(IncompleteFacts {
            actual: executable.workspace_mechanisms().unwrap(),
            fail,
        });
        executable
            .erased()
            .install_workspace_parameter_representations(&context)
            .unwrap();
        let Some(input::OriginalMediaPacket::Original(packet)) = prompt.original_media.as_ref()
        else {
            panic!("actual original packet")
        };
        let source = OriginalMediaWorkspaceInput::project(
            packet.body.prepared().unwrap(),
            packet.semantics.clone(),
            &context,
            &ordinary.unquoted_lease().unwrap(),
        )
        .unwrap();
        let current = executable
            .erased()
            .current_media_semantic_binding()
            .unwrap();
        let state = executable
            .erased()
            .project_resident_workspace(1.try_into().unwrap(), &context)
            .unwrap();
        let result = executable
            .inference_blueprint()
            .unwrap()
            .quote_original_media_ordinary(source, &current, geometry, &state, &context, None);
        drop(state);
        drop(context);
        drop(ordinary);
        result
    };
    let report = trace(false).unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), baseline + 1);
    assert_eq!(report.intervals().len(), 8);
    assert_eq!(
        report.intervals()[0]
            .unpriced_tensor_operations()
            .filter(|operation| matches!(
                operation.kind,
                eredu_nn::workspace::WorkspaceOperationKind::PreparedMultiAxisRotary(_)
            ))
            .count(),
        1
    );
    assert!(report.intervals()[0].tensor_bytes().is_none());
    assert!(report.intervals()[0].tensor_interval_bytes().is_none());
    assert!(report.equations().first_gap().is_some());
    drop(report);
    assert_eq!(pool.unquoted_owner_count().unwrap(), baseline);
    let failure = trace(true).unwrap_err();
    assert_eq!(pool.unquoted_owner_count().unwrap(), baseline + 1);
    let mut cause: &(dyn std::error::Error + 'static) = &failure;
    loop {
        if cause.is::<MissingEncoderFact>() {
            break;
        }
        cause = cause
            .source()
            .expect("original encoder fact cause chain was erased");
    }
    drop(failure);
    assert_eq!(pool.unquoted_owner_count().unwrap(), baseline);
}
