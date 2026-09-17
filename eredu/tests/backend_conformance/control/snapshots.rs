use super::*;
use eredu_core::PendingTextInput;
use eredu_runtime::{
    capture::CaptureSession,
    execution_control::{SamplingCopyPolicy, TextSnapshotBackend},
};

#[path = "snapshots/host_authority.rs"]
mod host_authority;

#[path = "snapshots/policy.rs"]
mod policy;

// This semantic fixture has no persistent model tensors: all model continuation
// state is its pending canonical input. Native storage conformance runs separately
// against actual dense, convolution/recurrent and MoE fixtures.
pub(crate) struct Native(String);
impl NativeTextStateBackend for MockBackend {
    type NativeTextState = Native;
    fn estimate_native_text_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Native,
        _: u64,
    ) -> Result<Option<u64>, MockError> {
        super::provider_errors::check("growth")?;
        Self::validate_native_text_state(runtime, saved)?;
        Ok(Some(0))
    }
    fn native_text_state_support(_: &ModelRuntime<Self>) -> ControlSupport {
        ControlSupport::Supported
    }
    fn estimate_native_text_state(
        runtime: &ModelRuntime<Self>,
        saved: Option<&Native>,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        policy::cold_estimate::ordinary_hook();
        super::provider_errors::check("estimate")?;
        runtime.session().authority.require_idle()?;
        if let Some(saved) = saved {
            Self::validate_native_text_state(runtime, saved)?;
        }
        let bytes = 64 + runtime.session().intervention_identity.len() as u64;
        Ok(Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        }))
    }
    fn capture_native_text_state(runtime: &mut ModelRuntime<Self>) -> Result<Native, MockError> {
        super::provider_errors::check("capture")?;
        runtime.session().authority.require_idle()?;
        crate::host_authority::copy("native capture")?;
        Ok(Native(runtime.session().intervention_identity.clone()))
    }
    fn copy_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        saved: &Native,
    ) -> Result<Native, MockError> {
        super::provider_errors::check("copy")?;
        Self::validate_native_text_state(runtime, saved)?;
        crate::host_authority::copy("native copy")?;
        Ok(Native(saved.0.clone()))
    }
    fn validate_native_text_state(
        runtime: &ModelRuntime<Self>,
        saved: &Native,
    ) -> Result<(), MockError> {
        runtime.session().authority.require_idle()?;
        if saved.0 != runtime.session().intervention_identity {
            return Err(MockError::Capture("foreign snapshot".into()));
        }
        Ok(())
    }
    fn exchange_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        saved: &mut Native,
    ) -> Result<(), MockError> {
        Self::validate_native_text_state(runtime, saved)
    }
}
use observed_mock::Sampling;
// This semantic fixture has no managed component proof. Bounded copying is
// rejected before either payload is cloned; full unquoted snapshots keep their
// existing enclosing host authority through copying and installation.
pub(crate) struct SavedSampling {
    sampling: Sampling,
    pending: Option<PendingTextInput<Vec<u32>, MockToken>>,
}

fn copy_sampling_parts(
    runtime: &mut ModelRuntime<MockBackend>,
    sampling: &Sampling,
    pending: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
) -> Result<(Sampling, Option<PendingTextInput<Vec<u32>, MockToken>>), MockError> {
    let pending = MockBackend::copy_pending_input(runtime, pending)?;
    let sampling = MockBackend::copy_sampling_state(runtime, sampling)?;
    Ok((sampling, pending))
}

// Only backend capture/copy hooks assemble this immutable pair. No allocating
// Clone or mutable component extraction is exposed by its associated type.
pub(crate) struct SavedComponents {
    native: Native,
    sampling: SavedSampling,
}

impl TextSnapshotBackend for MockBackend {
    type SamplingState = Sampling;
    type SavedSamplingState = SavedSampling;
    type SavedTextComponents = SavedComponents;
    fn original_saved_components_preparation_bytes(
        _: &ModelRuntime<Self>,
        _: &Sampling,
        _: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        // Scoped admission-refusal probe only: it constructs no native planner
        // and never enables the unpriced paired copy hook below.
        Ok(policy::cold_estimate::preparation_bytes())
    }
    fn original_snapshot_estimates(
        runtime: &ModelRuntime<Self>,
        _: &Sampling,
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
    ) -> Option<[SnapshotEstimate; 3]> {
        policy::cold_estimate::estimate(runtime, input)
    }
    fn capture_saved_components(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Sampling,
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
        policy: SamplingCopyPolicy,
    ) -> Result<SavedComponents, MockError> {
        if !matches!(policy, SamplingCopyPolicy::Unquoted) {
            return Err(MockError::Capture(
                "bounded saved text components are unpriced in this fixture".into(),
            ));
        }
        // Keep the established sampler/input then native-copy order. The
        // shared driver already holds host authority through partial failure.
        let sampling = Self::capture_saved_sampling(runtime, sampling, input, policy)?;
        let native = Self::capture_native_text_state(runtime)?;
        Ok(SavedComponents { native, sampling })
    }
    fn copy_saved_components(
        runtime: &mut ModelRuntime<Self>,
        saved: &SavedComponents,
        policy: SamplingCopyPolicy,
    ) -> Result<SavedComponents, MockError> {
        if !matches!(policy, SamplingCopyPolicy::Unquoted) {
            return Err(MockError::Capture(
                "bounded saved text components are unpriced in this fixture".into(),
            ));
        }
        Self::validate_saved_components(runtime, saved)?;
        let sampling = Self::copy_saved_sampling(runtime, &saved.sampling, policy)?;
        let native = Self::copy_native_text_state(runtime, &saved.native)?;
        Ok(SavedComponents { native, sampling })
    }
    fn saved_sampling(saved: &SavedComponents) -> &SavedSampling {
        &saved.sampling
    }
    fn validate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &SavedComponents,
    ) -> Result<(), MockError> {
        Self::validate_native_text_state(runtime, &saved.native)
    }
    fn estimate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &SavedComponents,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        Self::validate_saved_components(runtime, saved)?;
        let native = Self::estimate_native_text_state(runtime, Some(&saved.native))?;
        let sampling = Self::estimate_saved_sampling(runtime, &saved.sampling)?;
        Ok(native.zip(sampling).and_then(|(native, sampling)| {
            Some(SnapshotEstimate {
                retained_bytes: native.retained_bytes.checked_add(sampling.retained_bytes)?,
                copy_bytes: native.copy_bytes.checked_add(sampling.copy_bytes)?,
            })
        }))
    }
    fn estimate_saved_native_growth(
        runtime: &ModelRuntime<Self>,
        saved: &SavedComponents,
        input_tokens: u64,
    ) -> Result<Option<u64>, MockError> {
        Self::validate_saved_components(runtime, saved)?;
        Self::estimate_native_text_growth(runtime, &saved.native, input_tokens)
    }
    fn prepare_saved_components_resume(
        runtime: &mut ModelRuntime<Self>,
        saved: &SavedComponents,
    ) -> Result<
        (
            Native,
            Sampling,
            Option<PendingTextInput<Vec<u32>, MockToken>>,
        ),
        MockError,
    > {
        Self::validate_saved_components(runtime, saved)?;
        let (sampling, pending) = Self::prepare_saved_sampling_resume(runtime, &saved.sampling)?;
        let native = Self::copy_native_text_state(runtime, &saved.native)?;
        Ok((native, sampling, pending))
    }
    fn capture_saved_sampling(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Sampling,
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
        policy: SamplingCopyPolicy,
    ) -> Result<SavedSampling, MockError> {
        if !matches!(policy, SamplingCopyPolicy::Unquoted) {
            return Err(MockError::Capture(
                "bounded saved sampling is unpriced in this fixture".into(),
            ));
        }
        let (sampling, pending) = copy_sampling_parts(runtime, sampling, input)?;
        Ok(SavedSampling { sampling, pending })
    }
    fn copy_saved_sampling(
        runtime: &mut ModelRuntime<Self>,
        saved: &SavedSampling,
        policy: SamplingCopyPolicy,
    ) -> Result<SavedSampling, MockError> {
        if !matches!(policy, SamplingCopyPolicy::Unquoted) {
            return Err(MockError::Capture(
                "bounded saved sampling is unpriced in this fixture".into(),
            ));
        }
        let (sampling, pending) = copy_sampling_parts(
            runtime,
            &saved.sampling,
            saved.pending.as_ref().map(PendingTextInput::as_ref),
        )?;
        Ok(SavedSampling { sampling, pending })
    }
    fn saved_sampling_prediction(saved: &SavedSampling) -> u64 {
        Self::sampling_prediction(&saved.sampling)
    }
    fn estimate_saved_sampling(
        runtime: &ModelRuntime<Self>,
        saved: &SavedSampling,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        let sampling = Self::estimate_sampling_state(runtime, &saved.sampling)?;
        let input = Self::estimate_pending_input(
            runtime,
            saved.pending.as_ref().map(PendingTextInput::as_ref),
        )?;
        Ok(sampling.zip(input).and_then(|(sampling, input)| {
            Some(SnapshotEstimate {
                retained_bytes: sampling.retained_bytes.checked_add(input.retained_bytes)?,
                copy_bytes: sampling.copy_bytes.checked_add(input.copy_bytes)?,
            })
        }))
    }
    fn saved_input_tokens(saved: &SavedSampling, predictions: u64) -> Option<u64> {
        Self::continuation_input_tokens(
            saved.pending.as_ref().map(PendingTextInput::as_ref),
            predictions,
        )
    }
    fn estimate_saved_sampling_growth(
        runtime: &ModelRuntime<Self>,
        saved: &SavedSampling,
        predictions: u64,
    ) -> Result<Option<u64>, MockError> {
        Self::estimate_sampling_growth(runtime, &saved.sampling, predictions)
    }
    fn prepare_saved_sampling_resume(
        runtime: &mut ModelRuntime<Self>,
        saved: &SavedSampling,
    ) -> Result<(Sampling, Option<PendingTextInput<Vec<u32>, MockToken>>), MockError> {
        copy_sampling_parts(
            runtime,
            &saved.sampling,
            saved.pending.as_ref().map(PendingTextInput::as_ref),
        )
    }

    fn continuation_input_tokens(
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
        predictions: u64,
    ) -> Option<u64> {
        if predictions == 0 {
            return Some(0);
        }
        match input? {
            PendingTextInput::Prefill(prompt) => (prompt.len() as u64).checked_add(predictions - 1),
            PendingTextInput::Decode(_) => Some(predictions),
        }
    }
    fn estimate_sampling_growth(
        _: &ModelRuntime<Self>,
        _: &Sampling,
        _: u64,
    ) -> Result<Option<u64>, MockError> {
        Ok(Some(0))
    }
    fn sampling_state(state: &observed_mock::State) -> &Sampling {
        &state.sampling
    }
    fn install_sampling_state(state: &mut observed_mock::State, sampling: Sampling) {
        state.sampling = sampling;
    }
    fn assemble_generation_state(
        sampling: Sampling,
        capture: Option<CaptureSession>,
    ) -> observed_mock::State {
        observed_mock::State { sampling, capture }
    }
    fn sampling_prediction(sampling: &Sampling) -> u64 {
        sampling.prediction
    }
    fn estimate_sampling_state(
        _: &ModelRuntime<Self>,
        _: &Sampling,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        policy::cold_estimate::ordinary_hook();
        Ok(Some(SnapshotEstimate {
            retained_bytes: 64,
            copy_bytes: 64,
        }))
    }
    fn copy_sampling_state(
        _: &mut ModelRuntime<Self>,
        sampling: &Sampling,
    ) -> Result<Sampling, MockError> {
        crate::host_authority::copy("sampling copy")?;
        Ok(sampling.clone())
    }
    fn estimate_pending_input(
        _: &ModelRuntime<Self>,
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        policy::cold_estimate::ordinary_hook();
        let bytes = match input {
            Some(PendingTextInput::Prefill(ids)) => 24 + ids.len() as u64 * 4,
            Some(PendingTextInput::Decode(_)) => 4,
            None => 0,
        };
        Ok(Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        }))
    }
    fn copy_pending_input(
        _: &mut ModelRuntime<Self>,
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
    ) -> Result<Option<PendingTextInput<Vec<u32>, MockToken>>, MockError> {
        crate::host_authority::copy("input copy")?;
        Ok(input.map(|input| match input {
            PendingTextInput::Prefill(ids) => PendingTextInput::Prefill(ids.clone()),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(token.clone()),
        }))
    }
    fn capture_run(state: &observed_mock::State) -> Option<&CaptureSession> {
        state.capture.as_ref()
    }
    fn capture_run_mut(state: &mut observed_mock::State) -> Option<&mut CaptureSession> {
        state.capture.as_mut()
    }
    fn estimate_child_capture(
        _: &ModelRuntime<Self>,
        shape: &[u64],
        _: &eredu_core::capture::CaptureSelection,
        _: &eredu_core::capture::ResolvedCaptureSlice,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        observed_mock::cost(shape)
    }
    fn child_intervention_estimator(
        _: &ModelRuntime<Self>,
    ) -> Result<
        std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>,
        eredu_core::capture::CaptureError,
    > {
        Ok(std::sync::Arc::new(observed_mock::Estimates))
    }
}

pub(super) fn snapshot_setup() -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
    u32,
) {
    let request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        tools: vec![serde_json::json!({"type":"function", "function":{
            "name":"weather", "description":"Weather", "parameters":{"type":"object", "properties":{}}
        }})],
        tool_choice: eredu::runtime::chat::ToolChoice::None,
        add_generation_prompt: true,
        ..Default::default()
    };
    let mut probe = unicode_model_with_vocabulary(None, 512);
    let chat = probe.prepare_chat(request()).unwrap();
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    assert!(first + 3 < 512);
    let mut model = unicode_model_with_vocabulary(Some(first), 512);
    let chat = model.prepare_chat(request()).unwrap();
    (
        model,
        chat,
        PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(8),
                ..Default::default()
            },
            seed: 17,
            ..Default::default()
        },
        first,
    )
}
fn snapshot_limits() -> SnapshotLimits {
    SnapshotLimits {
        max_snapshots: 3,
        max_branches: 0,
        retained_bytes: 4_000_000,
        cumulative_copy_bytes: 32_000_000,
    }
}

#[test]
fn snapshot_configuration_identity_retains_inference_policy() {
    use eredu_core::TextInferencePolicy;
    let mut identities = Vec::new();
    for policy in [
        TextInferencePolicy::default(),
        TextInferencePolicy {
            prefill_chunk_positions: std::num::NonZeroU64::new(3),
            ..Default::default()
        },
        TextInferencePolicy {
            managed_memory_capacity_bytes: Some(16 << 20),
            ..Default::default()
        },
        TextInferencePolicy {
            managed_memory_capacity_bytes: Some(32 << 20),
            ..Default::default()
        },
        TextInferencePolicy::default(),
    ] {
        let (mut model, chat, mut settings, _) = snapshot_setup();
        settings.inference = policy;
        let prepared = model
            .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
            .unwrap();
        let mut session = model
            .start_controlled_chat(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        session.enable_snapshots(snapshot_limits()).unwrap();
        let saved = session.snapshot(|_| ControlFlow::Continue(())).unwrap();
        identities.push(saved.metadata().configuration_identity);
    }
    assert_eq!(identities[0], identities[4]);
    for index in 0..4 {
        for other in 0..index {
            assert_ne!(identities[index], identities[other]);
        }
    }
}

#[test]
fn facade_complete_snapshots_restore_partial_unicode_terminal_state_and_cumulative_budgets() {
    for mode in 0..4 {
        let (mut model, chat, settings, first) = snapshot_setup();
        let mut plan = if mode & 1 == 0 {
            CapturePlan::none()
        } else {
            observed_mock::plan()
        };
        plan.limits = observed_mock::plan().limits;
        let prepared = if mode & 2 == 0 {
            model
                .prepare_observed_chat(&chat, settings, plan, limits())
                .unwrap()
        } else {
            model
                .prepare_intervened_chat(
                    &chat,
                    settings,
                    plan,
                    observed_mock::intervention_plan(1.0),
                    limits(),
                )
                .unwrap()
        };
        let mut records = vec![];
        let mut session = model
            .start_controlled_chat(prepared, &[], Default::default(), |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        session.enable_snapshots(snapshot_limits()).unwrap();
        assert_eq!(session.capabilities().snapshot, ControlSupport::Supported);
        let initial = session
            .snapshot(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        session
            .step(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        let saved = session
            .snapshot(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(saved.token_ids(), [first]);
        assert_eq!(saved.metadata().output.next_prediction, 1);
        let from = records.len();
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        let baseline = semantic(&records[from..]);
        assert_eq!(session.status(), GenerationStatus::Completed);
        let history = session.token_ids().to_vec();
        let bytes = session.emitted_bytes();
        let usage = session.snapshot_usage().unwrap();
        session
            .restore(&saved, |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.output_checkpoint().epoch, 1);
        assert_eq!(session.next_prediction(), 1);
        assert!(session.emitted_bytes() > bytes);
        assert!(
            session.snapshot_usage().unwrap().cumulative_copy_bytes > usage.cumulative_copy_bytes
        );
        let from = records.len();
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.token_ids(), history);
        assert_eq!(semantic(&records[from..]), baseline);
        let terminal = session
            .snapshot(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert!(session
            .snapshot(|_| panic!("limited snapshot emitted"))
            .is_err());
        assert!(session.enable_snapshots(snapshot_limits()).is_err());
        session
            .restore(&initial, |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.status(), GenerationStatus::Prepared);
        assert!(session.token_ids().is_empty());
        session
            .restore(&terminal, |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.status(), GenerationStatus::Completed);
        assert_eq!(session.token_ids(), history);
        assert!(session
            .step(|_| panic!("terminal snapshot revived"))
            .is_err());
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.sequence, index as u64);
        }
        assert!(records.iter().any(|record| matches!(&record.generation.event,
            ObservedGenerationEvent::Restored { output, .. } if output == &saved.metadata().output)));
        drop((initial, saved, terminal));
        assert_eq!(session.snapshot_usage().unwrap().retained_bytes, 0);
    }
}

#[test]
fn opaque_grammar_estimates_fail_before_snapshot_retention() {
    let (mut model, chat, settings, _) = setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut session = model
        .start_controlled_chat(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(session.enable_snapshots(snapshot_limits()).is_err());
    assert!(session.snapshot_usage().is_none());
    assert!(matches!(
        session.capabilities().snapshot,
        ControlSupport::Unsupported { .. }
    ));
    session.step(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.next_prediction(), 1);
}

fn collect(
    records: &mut Vec<ControlledGenerationRecord>,
) -> impl FnMut(ControlledGenerationRecord) -> ControlFlow<()> + '_ {
    |record| {
        records.push(record);
        ControlFlow::Continue(())
    }
}

#[test]
fn facade_branches_keep_partial_text_identity_siblings_and_budgets_isolated() {
    use eredu::api::{GenerationBranchOptions, SamplingOverride};
    for mode in 0..4 {
        let (mut model, chat, settings, first) = snapshot_setup();
        let mut plan = if mode & 1 == 0 {
            CapturePlan::none()
        } else {
            observed_mock::plan()
        };
        plan.limits = observed_mock::plan().limits;
        let prepared = if mode & 2 == 0 {
            model
                .prepare_observed_chat(&chat, settings, plan, limits())
                .unwrap()
        } else {
            model
                .prepare_intervened_chat(
                    &chat,
                    settings,
                    plan,
                    observed_mock::intervention_plan(1.0),
                    limits(),
                )
                .unwrap()
        };
        let options = || GenerationBranchOptions {
            trace_limits: TraceLimits {
                per_record_bytes: 16384,
                total_bytes: 65536,
            },
            capture_limits: Some(observed_mock::plan().limits),
            sampling: None,
            intervention: None,
        };
        let mut records = vec![];
        let mut run = model
            .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
            .unwrap();
        let parent = run.output_checkpoint().run_id;
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 3,
            max_branches: 2,
            retained_bytes: 64_000_000,
            cumulative_copy_bytes: 256_000_000,
        })
        .unwrap();
        let initial = run.snapshot(collect(&mut records)).unwrap();
        assert_eq!(run.capabilities().fork, ControlSupport::Supported);
        run.step(collect(&mut records)).unwrap();
        let partial = run.snapshot(collect(&mut records)).unwrap();
        let mut left = run
            .fork(&partial, options(), collect(&mut records))
            .unwrap();
        let mut right = run
            .fork(&partial, options(), collect(&mut records))
            .unwrap();
        let left_id = left.run_id().to_owned();
        let right_id = right.run_id().to_owned();
        assert_ne!(left_id, right_id);
        assert_ne!(parent, left_id);
        assert_eq!(left.token_ids(), [first]);
        assert_eq!(run.token_ids(), [first]);
        assert_eq!(run.snapshot_usage().unwrap().branches, 2);
        assert!(run
            .fork(&partial, options(), collect(&mut records))
            .is_err());
        let before = records.len();
        run.run(collect(&mut records)).unwrap();
        let baseline_text = semantic(&records[before..]);
        let baseline_ids = run.token_ids().to_vec();
        let parent_bytes = run.emitted_bytes();
        run.exchange(&mut left, collect(&mut records)).unwrap();
        assert_eq!(left.run_id(), parent);
        assert_eq!(run.output_checkpoint().run_id, left_id);
        run.step(collect(&mut records)).unwrap();
        // Swap to the sibling while both the parent and left continuation remain stable.
        run.exchange(&mut right, collect(&mut records)).unwrap();
        assert_eq!(run.output_checkpoint().run_id, right_id);
        assert_eq!(run.token_ids(), [first]);
        let before = records.len();
        run.run(collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), baseline_ids);
        assert_eq!(semantic(&records[before..]), baseline_text);
        assert!(run.restore(&partial, collect(&mut records)).is_err());
        run.exchange(&mut left, collect(&mut records)).unwrap();
        assert_eq!(run.output_checkpoint().run_id, parent);
        assert_eq!(run.token_ids(), baseline_ids);
        assert!(run.emitted_bytes() > parent_bytes);
        assert_eq!(partial.token_ids(), [first]);
        drop(left);
        drop(right);
        assert_eq!(run.snapshot_usage().unwrap().branches, 0);
        // A modified child starts before the replaced decision, keeping the
        // parent's exact continuation intact. Reseeding is explicit provenance.
        let mut changed = options();
        changed.sampling = Some(SamplingOverride {
            temperature: Some(0.5),
            reseed: Some(711),
        });
        if mode & 2 != 0 {
            changed.intervention = Some(observed_mock::intervention_plan(2.0));
        }
        let mut slot = run.fork(&initial, changed, collect(&mut records)).unwrap();
        let changed_id = slot.run_id().to_owned();
        run.exchange(&mut slot, collect(&mut records)).unwrap();
        assert_eq!(run.sampling_state().unwrap().temperature, 0.5);
        run.force_next_token(first + 3).unwrap();
        run.step(collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), [first + 3]);
        assert!(records.iter().any(|r| r.generation.run_id == changed_id
            && matches!(
                r.generation.event,
                ObservedGenerationEvent::Token { forced: true, .. }
            )));
        run.cancel(collect(&mut records)).unwrap();
        // Cancelling a child does not cancel or strand the parent in its slot.
        run.exchange(&mut slot, collect(&mut records)).unwrap();
        assert_eq!(run.output_checkpoint().run_id, parent);
        assert_eq!(run.token_ids(), baseline_ids);
        drop(slot);
        assert_eq!(run.snapshot_usage().unwrap().branches, 0);
        let mut sequences = std::collections::HashMap::new();
        for record in &records {
            let previous = sequences.insert(&record.generation.run_id, record.sequence);
            assert!(previous.is_none_or(|previous| record.sequence > previous));
            if let ObservedGenerationEvent::BranchStarted {
                lineage,
                prompt_token_ids,
                inherited_token_ids,
                inherited_semantics,
            } = &record.generation.event
            {
                assert_eq!(record.sequence, 0);
                assert_eq!(record.generation.run_id, lineage.run_id);
                assert_eq!(lineage.parent.output.run_id, parent);
                assert_eq!(prompt_token_ids, initial.prompt_token_ids());
                assert!(inherited_token_ids.is_empty() || inherited_token_ids == &[first]);
                assert!(inherited_semantics.is_empty()); // Incomplete UTF-8 was never flushed.
                if let Some(parent_plan) = &lineage.parent.intervention_plan_id {
                    assert_ne!(
                        record.generation.intervention_plan_id.as_ref(),
                        Some(parent_plan)
                    );
                }
            }
        }
    }
}

#[test]
fn facade_snapshot_preserves_pending_choice_and_child_stream_includes_semantic_prefix() {
    use eredu::api::GenerationBranchOptions;
    let (mut model, chat, settings, first) = snapshot_setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut records = vec![];
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 3,
        max_branches: 1,
        retained_bytes: 32_000_000,
        cumulative_copy_bytes: 128_000_000,
    })
    .unwrap();
    run.force_next_token(first).unwrap();
    let forced = run.snapshot(collect(&mut records)).unwrap();
    assert_eq!(forced.metadata().pending_forced_token, Some(first));
    run.step(collect(&mut records)).unwrap();
    assert_eq!(run.pending_forced_token(), None);
    run.restore(&forced, collect(&mut records)).unwrap();
    assert_eq!(run.pending_forced_token(), Some(first));
    run.step(collect(&mut records)).unwrap();
    run.step(collect(&mut records)).unwrap();
    let text = run.snapshot(collect(&mut records)).unwrap();
    let mut branch = run
        .fork(
            &text,
            GenerationBranchOptions {
                trace_limits: TraceLimits {
                    per_record_bytes: 16384,
                    total_bytes: 32768,
                },
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            collect(&mut records),
        )
        .unwrap();
    let branch_id = branch.run_id().to_owned();
    let inherited = records
        .iter()
        .find_map(|record| {
            if record.generation.run_id != branch_id {
                return None;
            }
            match &record.generation.event {
                ObservedGenerationEvent::BranchStarted {
                    inherited_semantics,
                    ..
                } => Some(inherited_semantics.clone()),
                _ => None,
            }
        })
        .unwrap();
    assert_eq!(
        inherited,
        vec![eredu_core::generation::SemanticEvent::TextDelta("é".into())]
    );
    let before = records.len();
    run.run(collect(&mut records)).unwrap();
    let baseline = semantic(&records[before..]);
    run.exchange(&mut branch, collect(&mut records)).unwrap();
    let before = records.len();
    run.run(collect(&mut records)).unwrap();
    assert_eq!(semantic(&records[before..]), baseline);
    assert_eq!(text.token_ids(), [first, first + 1]);
}

#[test]
fn facade_fork_rejects_snapshot_from_an_earlier_driver_on_the_same_loaded_model() {
    use eredu::api::GenerationBranchOptions;
    let (mut model, chat, settings, _) = snapshot_setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let snapshot = {
        let mut run = model
            .start_controlled_chat(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        run.enable_snapshots(snapshot_limits()).unwrap();
        run.snapshot(|_| ControlFlow::Continue(())).unwrap()
    };
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_branches: 1,
        ..snapshot_limits()
    })
    .unwrap();
    let usage = run.snapshot_usage().unwrap();
    let result = run.fork(
        &snapshot,
        GenerationBranchOptions {
            trace_limits: TraceLimits {
                per_record_bytes: 4096,
                total_bytes: 8192,
            },
            capture_limits: None,
            sampling: None,
            intervention: None,
        },
        |_| ControlFlow::Continue(()),
    );
    assert!(matches!(
        result,
        Err(eredu::api::ControlledGenerationError::Snapshot(
            eredu_runtime::execution_control::TextSnapshotError::IncompatibleRun
        ))
    ));
    assert_eq!(run.snapshot_usage().unwrap(), usage);
    assert_eq!(run.status(), GenerationStatus::Prepared);
    run.step(|_| ControlFlow::Continue(())).unwrap();
}

#[test]
fn facade_restore_preserves_stop_lookbehind_across_a_saved_token_boundary() {
    let (mut model, chat, settings, first) = snapshot_setup();
    let alternate = first + 3;
    let stop = format!("é{}", model.decode(&[alternate], false).unwrap());
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut records = vec![];
    let mut run = model
        .start_controlled_chat(prepared, &[stop], Default::default(), collect(&mut records))
        .unwrap();
    run.enable_snapshots(snapshot_limits()).unwrap();
    run.step(collect(&mut records)).unwrap();
    run.step(collect(&mut records)).unwrap();
    assert!(
        semantic(&records).is_empty(),
        "stop-prefix text should still be buffered"
    );
    let saved = run.snapshot(collect(&mut records)).unwrap();
    let mut expected = None;
    for _ in 0..2 {
        run.force_next_token(alternate).unwrap();
        let before = records.len();
        run.step(collect(&mut records)).unwrap();
        assert_eq!(run.finish_reason(), Some(FinishReason::StopSequence));
        let current = semantic(&records[before..]);
        if let Some(expected) = &expected {
            assert_eq!(&current, expected);
        } else {
            expected = Some(current);
        }
        run.restore(&saved, collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), [first, first + 1]);
    }
}

#[test]
fn explicit_intervention_removal_retains_its_empty_override_provenance() {
    use eredu::api::GenerationBranchOptions;
    let (mut model, chat, settings, _) = snapshot_setup();
    let prepared = model
        .prepare_intervened_chat(
            &chat,
            settings,
            observed_mock::plan(),
            observed_mock::intervention_plan(1.0),
            limits(),
        )
        .unwrap();
    let mut records = vec![];
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 32_000_000,
        cumulative_copy_bytes: 128_000_000,
    })
    .unwrap();
    let saved = run.snapshot(collect(&mut records)).unwrap();
    let empty = eredu_core::intervention::InterventionPlan {
        schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
        operations: vec![],
    };
    let mut branch = run
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: TraceLimits {
                    per_record_bytes: 16384,
                    total_bytes: 65536,
                },
                capture_limits: Some(observed_mock::plan().limits),
                sampling: None,
                intervention: Some(empty.clone()),
            },
            collect(&mut records),
        )
        .unwrap();
    let start = records
        .iter()
        .find(|record| {
            matches!(
                record.generation.event,
                ObservedGenerationEvent::BranchStarted { .. }
            )
        })
        .unwrap();
    let ObservedGenerationEvent::BranchStarted { lineage, .. } = &start.generation.event else {
        unreachable!()
    };
    assert_eq!(lineage.intervention_override.as_ref(), Some(&empty));
    assert!(lineage.parent.intervention_plan_id.is_some());
    assert!(start.generation.intervention_plan_id.is_none());
    run.exchange(&mut branch, collect(&mut records)).unwrap();
    run.step(collect(&mut records)).unwrap();
}
