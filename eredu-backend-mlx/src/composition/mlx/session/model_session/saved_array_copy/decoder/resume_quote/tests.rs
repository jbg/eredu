use super::super::super::tests::{
    AllTokens, Runtime, accounting, advance, cause, config, reclaim, runtime, settle, tokens, words,
};
use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{TextGenerationDriver, TextGenerationInput, TextSamplingStrategy, TokenFilter};
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceOperationKind};
use eredu_runtime::{ConfiguredTextSampler, RuntimeStateComponents};

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct ColdGuard;
impl ColdGuard {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
    fn assert_cold(&self) {
        assert_eq!(
            HOUSEKEEPING.get(),
            0,
            "saved quote entered native housekeeping"
        );
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

fn changed_allowance(original: TextGenerationConfig, count: Option<usize>) -> TextGenerationConfig {
    let mut sampling = original.sampling();
    sampling.max_new_tokens = count;
    let changed = TextGenerationConfig::new(sampling).with_seed(original.seed());
    match original.strategy() {
        TextSamplingStrategy::Standard => changed,
        TextSamplingStrategy::MirostatV2 { tau, eta } => {
            changed.with_mirostat_v2(tau, eta).unwrap()
        }
    }
}
fn source_config(adaptive: bool) -> TextGenerationConfig {
    let config = config(Some(u64::MAX));
    if adaptive {
        config.with_mirostat_v2(5.0, 0.3).unwrap()
    } else {
        config
    }
}
fn controller() -> TextControllerWorkspace<'static> {
    TextControllerWorkspace {
        filter: (&TokenFilter::All).into(),
        additional_host_bytes: 0,
    }
}
fn history(sampler: &ConfiguredTextSampler) -> &[u32] {
    match sampler {
        ConfiguredTextSampler::Standard(sampler) => sampler.generated_tokens(),
        ConfiguredTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}
pub(super) fn saved_source(runtime: &mut Runtime, adaptive: bool) -> CopiedTextComponents {
    let mut driver = TextGenerationDriver::new(runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            source_config(adaptive),
            AllTokens,
        )
        .unwrap();
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        PreparedTextComponentsCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    // The installed branch now has a different cache frontier and PRNG/history.
    let third = advance(&mut driver, &mut state);
    assert_eq!(saved.sampling.frontier(), 6);
    assert_eq!(saved.sampling.next_prediction, 2);
    assert_eq!(history(saved.sampling.sampler.as_sampler()).len(), 2);
    assert!(
        history(saved.sampling.sampler.as_sampler())
            .iter()
            .any(|word| *word != 0)
    );
    drop((third, outputs, state));
    drop(driver);
    reclaim();
    saved
}
#[derive(Debug, PartialEq)]
struct SourceValues {
    sampler: String,
    history_pointer: usize,
    capacity: usize,
    key: Vec<u32>,
    pending: Vec<u32>,
    decoder: Vec<Vec<f32>>,
}
fn values(saved: &CopiedTextComponents) -> SourceValues {
    let mut decoder = Vec::new();
    saved
        .decoder
        .native
        .prepare_copy()
        .unwrap()
        .visit_operands(&mut |array| {
            decoder.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap());
        });
    SourceValues {
        sampler: format!("{:?}", saved.sampling.sampler.as_sampler()),
        history_pointer: history(saved.sampling.sampler.as_sampler()).as_ptr() as usize,
        capacity: saved.sampling.sampler.as_sampler().history_capacity(),
        key: words(saved.sampling.arrays.key.as_ref().unwrap()),
        pending: words(saved.sampling.arrays.pending.as_ref().unwrap()),
        decoder,
    }
}

#[test]
fn saved_resume_quote_uses_copied_saved_frontier_and_populated_controls_coldly() {
    for adaptive in [false, true] {
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let saved = saved_source(&mut runtime, adaptive);
        let source = values(&saved);
        assert!(source.decoder.iter().flatten().any(|value| *value != 0.0));
        let current = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap();
        assert_eq!(current.admission().unwrap().position(), 7);
        let current_revision = current.revision().clone();
        let payload_count = runtime.session().payload.active_owner_count();
        let before = (accounting(&pool), paths::snapshot());
        let requested = changed_allowance(source_config(adaptive), Some(11));
        let cold = ColdGuard::new();
        let quote =
            PreparedSavedTextResumeQuote::prepare(&runtime, &saved, requested, controller())
                .unwrap();
        assert!(std::ptr::eq(quote.source, &saved));
        assert_eq!(quote.geometry().cached_positions, 6);
        assert_eq!(quote.geometry().input_positions, 1);
        assert_eq!(quote.geometry().prefill_chunk_positions, 1);
        assert_eq!(quote.geometry().max_output_tokens, 11);
        assert_eq!(quote.generation.equations.completed_spans(), 12);
        assert_eq!(quote.generation.sampling.steps, 11);
        assert!(quote.output_width() > 0);
        assert_eq!(quote.config, requested);
        assert_eq!(
            quote.controller,
            TextControllerContract::from_workspace(controller(), quote.output_width()).unwrap()
        );
        assert!(
            quote
                .copied_state
                .as_ref()
                .iter()
                .all(|layer| layer.position() == 6)
        );
        assert_eq!(quote.copied_key.as_ref().unwrap().layout().shape(), &[2]);
        assert_eq!(
            quote.copied_key.as_ref().unwrap().layout().dtype(),
            WorkspaceDtype::Uint32
        );
        assert_eq!(
            quote
                .copied_input
                .as_ref()
                .expect("text input copy")
                .layout()
                .shape(),
            &[1, 1]
        );
        assert_eq!(
            quote
                .copied_input
                .as_ref()
                .expect("text input copy")
                .layout()
                .dtype(),
            WorkspaceDtype::Uint32
        );
        assert!(
            quote
                .sampling_copy
                .operations
                .iter()
                .all(|op| !matches!(op.kind, WorkspaceOperationKind::Sampling(_)))
        );
        assert!(matches!(
            quote.sampling_copy.operations[0].kind,
            WorkspaceOperationKind::Contiguous
        ));
        assert!(matches!(
            quote.sampling_copy.operations[1].kind,
            WorkspaceOperationKind::DeepCopy
        ));
        assert!(quote.decoder_host_peak > 0 && quote.pending_host.bytes().unwrap() > 0);
        let expected_host = saved
            .sampling
            .sampler
            .borrow_funded()
            .prepare_resume(requested)
            .unwrap()
            .required_host_bytes();
        assert_eq!(quote.sampler_host_peak, expected_host);
        assert!(quote.generation.sampling.host_peak_bytes.unwrap() >= expected_host);
        assert!(quote.generation.sampling.final_history_bytes > source.capacity as u64 * 4);
        let decoder_full = full_span(
            &quote.decoder_copy,
            "test dense copy",
            WorkspaceReportMetadata::ordinary(),
        )
        .unwrap()
        .bytes()
        .unwrap();
        let sampling_full = full_span(
            &quote.sampling_copy,
            "test sampling copy",
            WorkspaceReportMetadata::ordinary(),
        )
        .unwrap()
        .bytes()
        .unwrap();
        assert!(decoder_full >= quote.decoder_copy.total_bytes.unwrap());
        assert!(sampling_full >= quote.sampling_copy.total_bytes.unwrap());
        let workspace = quote.full().execution_workspace.as_ref().unwrap();
        assert!(workspace.peak_bytes().unwrap().unwrap() > decoder_full + sampling_full);
        assert_eq!(
            workspace.activations.bytes(),
            Some(
                quote.pending_host.bytes().unwrap()
                    + quote.generation.equations.transient().bytes().unwrap()
            )
        );
        assert_eq!(quote.full().assumptions.requested_positions, 18);
        assert!(quote.generation.equations.first_gap().is_none());
        assert!(quote.generation.sampling.first_gap.is_none());
        assert_eq!((accounting(&pool), paths::snapshot()), before);
        assert_eq!(
            runtime.session().payload.active_owner_count(),
            payload_count
        );
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap()
                .revision(),
            &current_revision
        );
        drop(quote);
        cold.assert_cold();
        drop(cold);
        assert_eq!((accounting(&pool), paths::snapshot()), before);
        assert_eq!(values(&saved), source);
        drop((saved, current, runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn recopy_of_saved_pair_quotes_after_original_component_retirement_without_old_receipts() {
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let saved = saved_source(&mut runtime, false);
    let old = saved.decoder.retirement_probe();
    let actual_decoder_alias = saved.decoder.clone();
    let original_values = values(&saved);
    let copied = PreparedTextComponentsCopy::prepare_saved(&runtime, &saved)
        .unwrap()
        .copy(&mut runtime, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    let after_old_retirement = pool.used_bytes().unwrap() - saved.bytes();
    drop(saved);
    assert!(!old(), "the actual frozen decoder alias remains active");
    drop(actual_decoder_alias);
    settle(&pool, after_old_retirement);
    assert!(old());
    let before = (accounting(&pool), paths::snapshot());
    let first = PreparedSavedTextResumeQuote::prepare(
        &runtime,
        &copied,
        changed_allowance(source_config(false), Some(3)),
        controller(),
    )
    .unwrap();
    let second = PreparedSavedTextResumeQuote::prepare(
        &runtime,
        &copied,
        changed_allowance(source_config(false), Some(3)),
        controller(),
    )
    .unwrap();
    assert_eq!(first.geometry(), second.geometry());
    assert_eq!(
        first.full().execution_workspace,
        second.full().execution_workspace
    );
    assert_eq!(first.sampler_host_peak, second.sampler_host_peak);
    assert_eq!((accounting(&pool), paths::snapshot()), before);
    drop((first, second));
    let copied_values = values(&copied);
    assert_eq!(copied_values.sampler, original_values.sampler);
    assert_eq!(copied_values.decoder, original_values.decoder);
    assert_eq!(copied_values.key, original_values.key);
    assert_eq!(copied_values.pending, original_values.pending);
    assert_ne!(
        copied_values.history_pointer,
        original_values.history_pointer
    );
    drop((copied, runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn foreign_origin_policy_missing_pending_and_overflow_reject_without_copy_or_charge() {
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    // Load both actual executables before an active finite account exists.
    let (mut runtime, artifact) = runtime(&pool);
    let (other_runtime, other_artifact) = super::super::super::tests::runtime(&pool);
    let mut saved = saved_source(&mut runtime, false);
    let requested = changed_allowance(source_config(false), Some(3));
    let before = (accounting(&pool), paths::snapshot());
    let lease = runtime
        .session()
        .authority
        .borrow_mut()
        .begin_submission()
        .unwrap();
    let error = PreparedSavedTextResumeQuote::prepare(&runtime, &saved, requested, controller())
        .err()
        .unwrap();
    assert_eq!(
        cause::<eredu_core::SessionAuthorityError>(&error),
        Some(&eredu_core::SessionAuthorityError::Busy)
    );
    assert_eq!((accounting(&pool), paths::snapshot()), before);
    // This exclusion lease never entered native work or held a funding scope.
    // Resolve it explicitly before any later successful inspection or teardown.
    assert!(lease.resolve());
    drop(lease);
    runtime.session().authority.borrow().require_idle().unwrap();
    let error =
        PreparedSavedTextResumeQuote::prepare(&other_runtime, &saved, requested, controller())
            .err()
            .unwrap();
    assert!(
        error
            .to_string()
            .contains("control state belongs to a different executable"),
        "{error:?}"
    );
    for count in [None, Some(0)] {
        let error = PreparedSavedTextResumeQuote::prepare(
            &runtime,
            &saved,
            changed_allowance(requested, count),
            controller(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            cause::<WorkingMemoryError>(&error),
            Some(
                WorkingMemoryError::UnknownBound
                    | WorkingMemoryError::PreparationConfigurationMismatch
            )
        ));
    }
    let mut settings = requested.sampling();
    settings.temperature = 0.25;
    let error = PreparedSavedTextResumeQuote::prepare(
        &runtime,
        &saved,
        TextGenerationConfig::new(settings),
        controller(),
    )
    .err()
    .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::PreparationConfigurationMismatch)
    );
    settings = requested.sampling();
    settings.repetition_penalty = 1.5;
    let error = PreparedSavedTextResumeQuote::prepare(
        &runtime,
        &saved,
        TextGenerationConfig::new(settings),
        controller(),
    )
    .err()
    .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::PreparationConfigurationMismatch)
    );
    let pending = saved.sampling.arrays.pending.take().unwrap();
    let error = PreparedSavedTextResumeQuote::prepare(&runtime, &saved, requested, controller())
        .err()
        .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    saved.sampling.arrays.pending = Some(pending);
    let ordinal = saved.sampling.next_prediction;
    saved.sampling.next_prediction = u64::MAX - 1;
    let error = PreparedSavedTextResumeQuote::prepare(&runtime, &saved, requested, controller())
        .err()
        .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::Overflow)
    );
    saved.sampling.next_prediction = ordinal;
    let frontier = saved.sampling.arrays.source.frontier;
    saved.sampling.arrays.source.frontier = u64::MAX;
    let error = PreparedSavedTextResumeQuote::prepare(&runtime, &saved, requested, controller())
        .err()
        .unwrap();
    assert!(matches!(
        cause::<eredu_core::CapabilityError>(&error),
        Some(eredu_core::CapabilityError::ArithmeticOverflow { .. })
    ));
    saved.sampling.arrays.source.frontier = frontier + 1;
    let error = PreparedSavedTextResumeQuote::prepare(&runtime, &saved, requested, controller())
        .err()
        .unwrap();
    assert!(error.to_string().contains("frontier differs"), "{error:?}");
    saved.sampling.arrays.source.frontier = frontier;
    assert_eq!((accounting(&pool), paths::snapshot()), before);
    drop((saved, runtime, artifact, other_runtime, other_artifact));
    settle(&pool, 0);
}

#[test]
fn actual_pending_program_with_missing_native_facts_keeps_full_copy_unknown() {
    use eredu_nn::workspace::{
        WorkspaceHostBound, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
    };
    #[derive(Debug)]
    struct MissingFacts;
    impl WorkspaceMechanisms for MissingFacts {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
            Ok(None)
        }
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
            Ok(None)
        }
    }
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let saved = saved_source(&mut runtime, false);
    let original = values(&saved);
    let before = (accounting(&pool), paths::snapshot());
    let pending = saved.sampling.arrays.pending.as_ref().unwrap();
    let context = WorkspaceContext::new(MissingFacts);
    let mut projection = ExistingArrayProjection::new(&context);
    let source = projection.project(pending).unwrap();
    context.begin_state_span([&source]).unwrap();
    let plan = PreparedPendingPrompt::new(pending).unwrap();
    assert!(plan.host_plan().initialization_peak_bytes() > 0);
    let copied = plan.trace(&mut projection).unwrap();
    let report = context.report(&[source, copied]).unwrap();
    assert!(projection.is_complete());
    assert!(!report.unpriced_operations.is_empty());
    assert!(!report.unpriced_host_operations.is_empty());
    assert_eq!(report.total_bytes, None);
    let bound = full_span(
        &report,
        "actual pending program",
        WorkspaceReportMetadata::ordinary(),
    )
    .unwrap();
    assert!(matches!(bound, WorkspaceBound::Unknown { .. }));
    let host = WorkspaceBound::bounded(
        plan.host_plan().initialization_peak_bytes(),
        "actual closed host plan",
    );
    let unrelated = || WorkspaceBound::bounded(0, "outside this pending-program fixture");
    let outside = ExecutionWorkspaceEstimate {
        geometry: resume_geometry(&saved, source_config(false)).unwrap(),
        activations: host,
        attention: unrelated(),
        vocabulary: unrelated(),
        state_update: bound,
        materialization: unrelated(),
        retained: unrelated(),
    };
    assert_eq!(outside.peak_bytes().unwrap(), None);
    assert_eq!((accounting(&pool), paths::snapshot()), before);
    assert_eq!(values(&saved), original);
    drop((projection, plan));
    drop((saved, runtime, artifact));
    settle(&pool, 0);
}
