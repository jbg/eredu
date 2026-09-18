//! Complete local numerical producer; no native capability is inferred from it.
//! Immutable selected parameters and original input use actual host-source owners.
use super::*;
use crate::{input::host::*, prefill::*};
use eredu_core::*;
use std::{
    alloc::Layout,
    error::Error as _,
    num::NonZeroU8,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
mod backend;
mod report;
mod sequence;
use backend::*;
use sequence::*;

const EOS: [u32; 2] = [99, 99];
const OUTPUTS: usize = 3;
const TEXT: [u32; 3] = [2, 3, 1];
const IMAGE: [f32; 4] = [0.5, -1., 2., 1.5];
const WEIGHTS: [f32; 4] = [2., -0.5, 0.25, 1.5];

fn source(pool: &WorkingMemoryPool, model: bool) -> OriginalPreparedHostInput {
    let data = if model { &WEIGHTS } else { &IMAGE };
    let text = HostInputPart {
        modality: InputModality::Text,
        kind: InputPayloadKind::TokenIds,
        payload: HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::U32(&TEXT),
        },
        metadata: &[],
        extents: &[],
    };
    let numerical = HostInputPart {
        modality: InputModality::Image,
        kind: InputPayloadKind::Tensor,
        payload: HostTensorView {
            shape: &[2, 2],
            values: HostTensorValues::F32(data),
        },
        metadata: &[],
        extents: &[],
    };
    let parts = [text, numerical];
    pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
        .unwrap()
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: OUTPUTS as u64,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    }
}
fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        resolve_generation_config(
            None,
            GenerationConfigOverrides {
                max_new_tokens: Some(OUTPUTS),
                temperature: Some(0.),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(7)
}
fn arc_bytes<T>() -> usize {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .unwrap()
        .0
        .pad_to_align()
        .size()
}

// No telemetry allocation occurs inside a quote or operation. This test observer
// is caller setup and is excluded from the managed model/source recipe.
#[derive(Debug, Default)]
struct Facts {
    quotes: usize,
    report_count: usize,
    first_report: Option<(u64, u64, u64, u64)>,
    last_report: Option<(u64, u64, u64, u64)>,
    accepted_q: u64,
    prepared: usize,
    encoded: usize,
    spans: [usize; 5],
    span_count: usize,
    projections: usize,
    predictions: usize,
    last_state: Option<NumericalState>,
    compact_nonzero: bool,
    issued: [Option<TextStepContext>; OUTPUTS],
}
struct Fixture {
    pool: WorkingMemoryPool,
    sources: Sources,
    execution: InferenceExecutionIdentity,
    facts: Mutex<Facts>,
    // Existing selected-backend metadata, constructed and priced at setup.
    state_layout: StateMemoryLayout,
    budget: Option<u64>,
    manual_prefill: bool,
    revision: u64,
    current_revision: u64,
    current_frontier: u64,
    cancel: GenerationCancellationToken,
    cancel_after: AtomicUsize,
}
impl Fixture {
    fn report(
        &self,
        g: InferenceGeometry,
        core_controls: usize,
    ) -> Result<StateReport<'static>, WorkingMemoryError> {
        // Actual recurrent state, compact two-row encoder result and one-time
        // encoder scratch. The compact rows survive the first decoder interval.
        let logical = estimate_runtime_state_facts(
            &self.state_layout,
            self.request().input,
            g.max_output_tokens,
            g.batch_size,
            NonZeroU8::new(4).unwrap(),
        )
        .map_err(|error| match error {
            AdmissionPolicyError::ArithmeticOverflow { .. } => WorkingMemoryError::Overflow,
            _ => WorkingMemoryError::IdentityMismatch,
        })?;
        assert_eq!(
            logical.fixed_state_bytes,
            std::mem::size_of::<([f32; 2], [[f32; 2]; 4])>() as u64
        );
        assert_eq!(logical.context_state_bytes, 0);
        assert_eq!(
            logical.multimodal_embedding_bytes,
            std::mem::size_of::<[[f32; 2]; 2]>() as u64
        );
        assert_eq!(
            logical.media_execution_workspace_bytes,
            std::mem::size_of::<([[f32; 2]; 2], [f32; 2])>() as u64
        );
        assert_eq!(
            logical.completeness,
            EstimationCompleteness::PersistentStateOnly
        );
        // The complete native-free recurrence supplies execution/backing separately.
        let mut report = StateReport {
            report_recipe: Some(report::recipe()),
            geometry: g,
            fixed_state_bytes: logical.fixed_state_bytes,
            bytes_per_position_per_batch: logical.bytes_per_position_per_batch,
            context_state_bytes: logical.context_state_bytes,
            selected_state_bytes: std::mem::size_of::<([f32; 2], [[f32; 2]; 4])>() as u64,
            multimodal_embedding_bytes: logical.multimodal_embedding_bytes,
            media_execution_workspace_bytes: logical.media_execution_workspace_bytes,
            components: [
                std::mem::size_of::<[[f32; 2]; 2]>() as u64,
                0,
                std::mem::size_of::<[f32; 4]>() as u64,
                0,
                0,
                0,
            ],
            dtype_bytes: NonZeroU8::new(4).unwrap(),
            allocation_granularity: 1,
            sliding_windows: diagnostics::ReportWindows::Explicit(&[2, 4]),
            descriptions: [
                "two recurrent values and the actual four-row 2/4-window history",
                "one input row and two window sums",
                "no attention operation in this recurrence",
                "four actual output scores",
                "in-place fixed state with no rollback allocation",
                "original immutable borrowed sources; no transfer",
                "closed diagnostic, request, sequence and shared-driver controls",
            ],
        };
        let fixed = [
            core_controls,
            report::controls(),
            arc_bytes::<Accepted>(),
            std::mem::size_of::<Accepted>(),
            arc_bytes::<OutputPayload>(),
            std::mem::size_of::<OutputPayload>(),
            std::mem::size_of::<FixedSequence>(),
            std::mem::size_of::<Box<FixedSequence>>(),
            std::mem::size_of::<RetainedGenerationSequence>(),
            std::mem::size_of::<Result<RetainedGenerationSequence, BackendFailure>>(),
            GenerationSequenceAdmissionError::rejected_sequence_retention_peak_bytes().unwrap(),
            BackendFailure::source_retention_peak_bytes::<RetainedSequenceConstructionError>()
                .unwrap(),
            std::mem::size_of::<State>(),
            std::mem::size_of::<Prompt>(),
            std::mem::size_of::<Token>(),
            std::mem::size_of::<Submission<Token, Done>>(),
            std::mem::size_of::<Done>(),
            // Synchronous completion keeps len <= 1, but the shared worker
            // allocates len + 1 before pruning. By prediction three both the
            // previous and replacement Vec can have capacity two.
            Layout::array::<Done>(2).unwrap().size(), // prior allocation
            Layout::array::<Done>(2).unwrap().size(), // replacement allocation
            std::mem::size_of::<PrefillDriver<[f32; 4], Done>>(),
            std::mem::size_of::<Executor<'_>>(),
            std::mem::size_of::<PrefillProgress<[f32; 4]>>(),
            std::mem::size_of::<Submission<Option<[f32; 4]>, Done>>(),
            // Core next() creates one private cancellation token under Q;
            // the persistent prefill token is the existing caller token.
            arc_bytes::<std::sync::atomic::AtomicBool>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .unwrap();
        report.components[5] = report
            .control_bytes::<Sources>()?
            .checked_add(u64::try_from(fixed).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(report)
    }
    fn config(&self) -> TextGenerationConfig {
        let source_residence = self.pool.0.existing
            + self.sources.input.original_bytes()
            + self.sources.selected_model.original_bytes();
        config().with_inference_policy(TextInferencePolicy {
            prefill_chunk_positions: std::num::NonZeroU64::new(2),
            managed_memory_capacity_bytes: self.budget.map(|q| source_residence + q),
            submission_tracking_capacity_bytes: None, // This synchronous scalar mechanism creates no registry.
            graph_metadata_capacity_bytes: None, // No native/lazy graph exists in the selected mechanism.
        })
    }
    fn request(&self) -> AdmissionRequest {
        let text = self.tokens().unwrap().len() as u64;
        let media = (self.image().unwrap().len() / 2) as u64;
        AdmissionRequest {
            input: InputTokenCount::prepared(
                text,
                media,
                text + media,
                std::mem::size_of::<([[f32; 2]; 2], [f32; 2])>() as u64,
                ObservationKind::Exact,
            ),
            max_output_tokens: OUTPUTS as u64,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: self.budget,
            require_complete_estimate: true,
        }
    }
}
// Actual ten-float recurrent/ring storage, represented as one fixed component.
// Sliding recurrence windows are explicit fixture semantics, not invented KV.
fn selected_state_layout() -> (StateMemoryLayout, usize) {
    use eredu_core::cache::{
        LayerCachePolicy, StateResidencyClass, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    };
    let shape = vec![StateTensorDimension::fixed(10).unwrap()];
    let shape_bytes = Layout::array::<StateTensorDimension>(shape.capacity())
        .unwrap()
        .size();
    let tensors = vec![StateTensorPolicy::new_with_residency(
        StateTensorRole::Recurrent,
        shape,
        StateTensorDtype::Float32,
        StateResidencyClass::LayerScopedOffloadable,
    )
    .unwrap()];
    let tensor_bytes = Layout::array::<StateTensorPolicy>(tensors.capacity())
        .unwrap()
        .size();
    let policies = vec![LayerCachePolicy::fixed_only(tensors).unwrap()];
    // LayerSchedule owns an exact boxed slice, not the Vec's former capacity.
    let policy_bytes = Layout::array::<LayerCachePolicy>(policies.len())
        .unwrap()
        .size();
    let offsets = vec![0i32];
    let offset_bytes = Layout::array::<i32>(offsets.capacity()).unwrap().size();
    let layout = StateMemoryLayout::new(
        eredu_core::attention::LayerSchedule::new(1, policies).unwrap(),
        offsets,
        2,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    (
        layout,
        shape_bytes + tensor_bytes + policy_bytes + offset_bytes,
    )
}
fn setup(manual_prefill: bool, budget: Option<u64>) -> (ModelRuntime<Backend>, Arc<Fixture>) {
    let (state_layout, layout_bytes) = selected_state_layout();
    // Exact fixed selected-backend/header objects are setup residence; all source
    // buffers use their own real original accounts below. The baseline is never
    // treated as a credit against Q or as request work authority.
    let baseline = layout_bytes
        + arc_bytes::<Fixture>()
        + arc_bytes::<()>()
        + arc_bytes::<std::sync::atomic::AtomicBool>()
        + if manual_prefill { arc_bytes::<()>() } else { 0 };
    let pool = WorkingMemoryPool::new(1 << 24, baseline as u64).unwrap();
    let fixture = Arc::new(Fixture {
        sources: Sources {
            report: None,
            selected_model: source(&pool, true),
            input: source(&pool, false),
        },
        pool,
        execution: InferenceExecutionIdentity::default(),
        facts: Mutex::new(Facts::default()),
        state_layout,
        budget,
        manual_prefill,
        revision: 1,
        current_revision: 1,
        current_frontier: 0,
        cancel: GenerationCancellationToken::new(),
        cancel_after: AtomicUsize::new(0),
    });
    let runtime = ModelRuntime::prepare(Backend(fixture.clone()), ()).unwrap();
    (runtime, fixture)
}
fn prompt(fixture: &Fixture) -> TextGenerationInput<Prompt> {
    TextGenerationInput::OriginalPrepared(Prompt {
        input: fixture.sources.input.clone(),
        selected_model: fixture.sources.selected_model.clone(),
        execution: fixture.execution.clone(),
        revision: fixture.revision,
        bound: None,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct NumericalState {
    value: [f32; 2],
    history: [[f32; 2]; 4],
    compact: [[f32; 2]; 2],
    position: usize,
}
struct Oracle {
    outputs: [(NumericalState, [f32; 4], u32); OUTPUTS],
    prefixes: [NumericalState; 5],
}
// Independent scalar oracle, not a second chunk scheduler. It retains the full
// history and derives the selected ring representation only for comparisons.
fn independent() -> Oracle {
    let mut rows = [[0f32; 2]; 5];
    for (i, id) in TEXT.into_iter().enumerate() {
        rows[i] = [id as f32, 1.];
    }
    let mut compact = [[0.; 2]; 2];
    for i in 0..2 {
        compact[i] = [
            IMAGE[2 * i] * WEIGHTS[0] + IMAGE[2 * i + 1] * WEIGHTS[2],
            IMAGE[2 * i] * WEIGHTS[1] + IMAGE[2 * i + 1] * WEIGHTS[3],
        ];
        rows[i + 3] = compact[i];
    }
    let mut state = [0.25, -0.5];
    let mut all_states = [[0.; 2]; 5 + OUTPUTS];
    let mut position: usize = 0;
    let mut scalar = |row: [f32; 2]| {
        let sum2 = all_states[position.saturating_sub(2)..position]
            .iter()
            .rev()
            .map(|v| v[0])
            .sum::<f32>();
        let sum4 = all_states[position.saturating_sub(4)..position]
            .iter()
            .rev()
            .map(|v| v[1])
            .sum::<f32>();
        state = [
            0.75 * state[0] + row[0] + 0.0625 * sum2,
            0.5 * state[1] + row[1] + 0.125 * sum4,
        ];
        all_states[position] = state;
        position += 1;
        let mut history = [[0.; 2]; 4];
        for (i, value) in all_states[..position].iter().enumerate() {
            history[i % 4] = *value;
        }
        NumericalState {
            value: state,
            history,
            compact,
            position,
        }
    };
    let mut prefixes = [NumericalState::default(); 5];
    for (entry, row) in prefixes.iter_mut().zip(rows) {
        *entry = scalar(row);
    }
    let mut current = prefixes[4];
    let mut outputs = [(NumericalState::default(), [0.; 4], 0); OUTPUTS];
    for (i, entry) in outputs.iter_mut().enumerate() {
        let value = current.value;
        let scores = [
            value[0],
            value[1],
            value[0] - value[1],
            -value[0] - value[1],
        ];
        let mut best = 0;
        for j in 1..4 {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        *entry = (current, scores, best as u32);
        if i + 1 < OUTPUTS {
            current = scalar([best as f32 * 0.5, -(best as f32)]);
        }
    }
    Oracle { outputs, prefixes }
}

impl Fixture {
    fn tokens(&self) -> Result<&[u32], WorkingMemoryError> {
        let slot = self
            .sources
            .input
            .slot(0)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match (slot.shape, slot.values) {
            ([1, 3], HostTensorValues::U32(values)) if values.len() == 3 => Ok(values),
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    fn image(&self) -> Result<&[f32], WorkingMemoryError> {
        let slot = self
            .sources
            .input
            .slot(1)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match (slot.shape, slot.values) {
            ([2, 2], HostTensorValues::F32(values)) if values.len() == 4 => Ok(values),
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    fn weights(&self) -> Result<&[f32], WorkingMemoryError> {
        let slot = self
            .sources
            .selected_model
            .slot(1)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match (slot.shape, slot.values) {
            ([2, 2], HostTensorValues::F32(values)) if values.len() == 4 => Ok(values),
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    fn validate_input(&self, input: &OriginalPreparedHostInput) -> Result<(), WorkingMemoryError> {
        input.validate_pool(&self.pool)?;
        if !input.same_source(&self.sources.input) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let text = self.tokens()?;
        let media = self.image()?;
        let weights = self.weights()?;
        if text.len() as u64 + (media.len() / 2) as u64 != geometry().input_positions
            || weights.len() != 4
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}

#[test]
fn complete_original_neutral_request_matches_full_numerics_and_manual_driver_at_exact_q() {
    for manual in [false, true] {
        let q = probe_q(manual);
        let (mut runtime, f) = setup(manual, Some(q));
        let baseline = f.pool.used_bytes().unwrap();
        let expected = independent();
        let input = prompt(&f);
        let claim = GenerationSequenceRequest::new(OUTPUTS, &EOS);
        let mut delivered = [(NumericalState::default(), [0.; 4], 0); OUTPUTS];
        let (ids, receipts) = if manual {
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut run = driver
                .start_input_with_sequence(input, f.config(), AllController, None, claim)
                .unwrap();
            assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
            assert_eq!(f.facts.lock().unwrap().predictions, 0);
            let mut sequence = driver
                .take_prepared_sequence(&mut run)
                .unwrap()
                .unwrap()
                .prepare_storage()
                .unwrap();
            let mut receipts = [None, None, None];
            for i in 0..OUTPUTS {
                let token = driver.advance(&mut run).unwrap().unwrap().into_output();
                assert!(driver.take_completed_delivery(&mut run).unwrap().is_none());
                delivered[i] = (token.snapshot, token.scores, token.id);
                sequence
                    .commit(token.id, TokenTerminalSignals::default())
                    .unwrap();
                receipts[i] = Some(token.receipt.clone());
            }
            assert!(driver.advance(&mut run).unwrap().is_none());
            drop(run);
            (sequence.into_token_ids(), receipts)
        } else {
            let mut run = TextGeneration::from_input_with_sequence(
                &mut runtime,
                input,
                f.config(),
                TokenFilter::All,
                None,
                claim,
            )
            .unwrap();
            assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
            assert_eq!(f.facts.lock().unwrap().predictions, 0);
            let mut sequence = run
                .take_prepared_sequence()
                .unwrap()
                .prepare_storage()
                .unwrap();
            let mut receipts = [None, None, None];
            for i in 0..OUTPUTS {
                let token = run.next().unwrap().unwrap();
                delivered[i] = (token.snapshot, token.scores, token.id);
                sequence
                    .commit(token.id, TokenTerminalSignals::default())
                    .unwrap();
                receipts[i] = Some(token.receipt.clone());
            }
            assert!(run.next().is_none());
            drop(run);
            (sequence.into_token_ids(), receipts)
        };
        assert_eq!(delivered, expected.outputs);
        assert!(delivered[0].0.value.iter().all(|v| *v != 0.));
        assert_eq!(ids.as_slice(), &expected.outputs.map(|v| v.2));
        let facts = f.facts.lock().unwrap();
        assert_eq!(facts.encoded, 1);
        assert_eq!(facts.report_count, 5);
        assert_eq!(facts.first_report, Some((56, 16, 40, 56)));
        assert_eq!(facts.last_report, Some((16, 0, 16, 56)));
        assert_eq!(&facts.spans[..facts.span_count], &[2, 2, 1]);
        assert_eq!(facts.projections, 1);
        assert_eq!(facts.predictions, OUTPUTS);
        assert_eq!(
            facts
                .issued
                .iter()
                .map(|c| c.as_ref().unwrap().attempt())
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        drop(facts);
        let alias = ids.clone();
        drop(ids);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
        drop(alias);
        // Scalar receipts can escape; they do not retain any request allocation.
        assert_eq!(f.pool.used_bytes().unwrap(), baseline);
        drop(receipts);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    }
}

#[test]
fn complete_original_neutral_q_minus_one_rejects_before_any_request_constructor_or_model_work() {
    let q = probe_q(false);
    let (mut runtime, f) = setup(false, Some(q - 1));
    let before = f.pool.used_bytes().unwrap();
    let issued = f.pool.0.usage.lock().unwrap().next_funding;
    let error = TextGeneration::from_input_with_sequence(
        &mut runtime,
        prompt(&f),
        f.config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .err()
    .unwrap();
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<PreparedRequestRejection>(),
        Some(&PreparedRequestRejection::CapacityExceeded)
    );
    assert_eq!(f.pool.used_bytes().unwrap(), before);
    assert_eq!(f.pool.0.usage.lock().unwrap().next_funding, issued);
    let facts = f.facts.lock().unwrap();
    assert_eq!(facts.quotes, 2); // Existing 2 -> 1 shared retry policy.
    assert_eq!(facts.report_count, 0);
    assert_eq!(
        (
            facts.prepared,
            facts.encoded,
            facts.span_count,
            facts.predictions
        ),
        (0, 0, 0, 0)
    );
}

#[test]
fn complete_original_diagnostic_failures_hold_each_real_prefix_through_neutral_error_retirement() {
    let mut previous = 0;
    for at in 0..8 {
        let (mut runtime, f) = setup(false, None);
        let baseline = f.pool.used_bytes().unwrap();
        let q = probe_q(false);
        diagnostics::fail_destination_for_test(at);
        let error = TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .err()
        .unwrap();
        let concrete = error
            .source()
            .unwrap()
            .downcast_ref::<ConstructionFailure<Sources>>()
            .unwrap();
        assert_eq!(concrete.destination(), Some(at));
        let prefix = concrete.prefix_bytes();
        assert_eq!(prefix == 0, at == 0);
        if at != 0 {
            assert!(prefix > previous);
        }
        previous = prefix;
        assert!(concrete
            .source()
            .unwrap()
            .source()
            .unwrap()
            .is::<std::collections::TryReserveError>());
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
        assert!(f.pool.0.usage.lock().unwrap().pending_original.is_none());
        assert_eq!(f.facts.lock().unwrap().prepared, 0);
        // A retained partial error does not keep construction Busy. The same
        // untouched immutable source can receive a distinct new request grant.
        let fresh = TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .unwrap();
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + 2 * q);
        drop(fresh);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
        drop(error);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    }
}

#[derive(Clone, Copy)]
struct AllController;
impl TokenFilterController for AllController {
    type Error = std::convert::Infallible;
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwned
    }
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: TextFilterWorkspace::Exact(&TokenFilter::All),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[test]
fn complete_original_source_selection_domain_and_stale_state_reject_before_candidate_work() {
    for change in 0..5 {
        let (runtime, mut f) = setup(false, None);
        let TextGenerationInput::OriginalPrepared(mut input) = prompt(&f) else {
            unreachable!()
        };
        drop(runtime);
        match change {
            0 => input.input = source(&f.pool, false), // equal bytes, distinct actual I
            1 => input.selected_model = source(&f.pool, true), // equal parameters, another selection source
            2 => input.execution = InferenceExecutionIdentity::default(),
            3 => {
                let foreign = WorkingMemoryPool::new(1 << 24, 0).unwrap();
                input.input = source(&foreign, false);
            }
            _ => Arc::get_mut(&mut f).unwrap().current_revision += 1,
        }
        let mut runtime = ModelRuntime::prepare(Backend(f.clone()), ()).unwrap();
        let before = f.pool.used_bytes().unwrap();
        // Retain any original substitute through the observation so its own
        // legitimate source retirement cannot be mistaken for request activity.
        let keep = (input.input.clone(), input.selected_model.clone());
        let error = TextGeneration::from_input_with_sequence(
            &mut runtime,
            TextGenerationInput::OriginalPrepared(input),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .err()
        .unwrap();
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<PreparedRequestRejection>(),
            Some(&PreparedRequestRejection::IdentityMismatch)
        );
        assert_eq!(f.pool.used_bytes().unwrap(), before);
        let facts = f.facts.lock().unwrap();
        assert_eq!(
            (
                facts.quotes,
                facts.prepared,
                facts.encoded,
                facts.predictions
            ),
            (0, 0, 0, 0)
        );
        drop(facts);
        drop(keep);
    }
}

#[test]
fn complete_original_cancellation_holds_future_media_and_stops_all_later_work() {
    for completed_spans in 0..=2 {
        let (mut runtime, f) = setup(false, None);
        let baseline = f.pool.used_bytes().unwrap();
        let q = probe_q(false);
        if completed_spans == 0 {
            f.cancel.cancel();
        } else {
            f.cancel_after.store(completed_spans, Ordering::Release);
        }
        let mut run = TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .unwrap();
        assert!(run.next().is_none());
        assert!(run.next().is_none());
        let facts = f.facts.lock().unwrap();
        if completed_spans == 0 {
            assert_eq!((facts.encoded, facts.span_count), (0, 0));
            assert!(facts.last_state.is_none());
        } else {
            assert_eq!((facts.encoded, facts.span_count), (1, completed_spans));
            assert!(facts.spans[..completed_spans].iter().all(|n| *n == 2));
            assert!(facts.compact_nonzero); // First interval retains future rows.
            assert_eq!(
                facts.last_state,
                Some(independent().prefixes[2 * completed_spans - 1])
            );
        }
        assert_eq!((facts.projections, facts.predictions), (0, 0));
        drop(facts);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
        drop(run);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    }
}

#[test]
fn complete_original_terminal_aliases_share_one_charge_through_concurrent_last_retirement() {
    let (mut runtime, f) = setup(false, None);
    let baseline = f.pool.used_bytes().unwrap();
    let q = probe_q(false);
    let mut run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        prompt(&f),
        f.config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .unwrap();
    let mut sequence = run
        .take_prepared_sequence()
        .unwrap()
        .prepare_storage()
        .unwrap();
    let mut genuine_request = None;
    for _ in 0..OUTPUTS {
        let token = run.next().unwrap().unwrap();
        if genuine_request.is_none() {
            genuine_request = Some(token.owner.preparation.request().clone());
        }
        sequence
            .commit(token.id, TokenTerminalSignals::default())
            .unwrap();
    }
    drop(run);
    let genuine_request = genuine_request.unwrap();
    let request_aliases: [InferenceRequest; 8] = std::array::from_fn(|_| genuine_request.clone());
    let reservation_aliases: [WorkingMemoryReservation; 8] =
        std::array::from_fn(|_| genuine_request.memory_reservation().unwrap().clone());
    drop(genuine_request);
    let ids = sequence.into_token_ids();
    let aliases: [GenerationTokenIds; 8] = std::array::from_fn(|_| ids.clone());
    drop(ids);
    let pool = f.pool.clone();
    let source_bytes = f.sources.input.original_bytes() + f.sources.selected_model.original_bytes();
    drop(runtime);
    drop(f); // No caller/model alias remains; outputs own both real sources.
    assert_eq!(pool.used_bytes().unwrap(), baseline + q);
    // Test-owned thread synchronization is outside the managed evaluator. All
    // eight library owners are the same actual allocated output/source/Q body.
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        for alias in aliases {
            let barrier = &barrier;
            let pool = &pool;
            scope.spawn(move || {
                barrier.wait();
                assert_eq!(pool.used_bytes().unwrap(), baseline + q);
                assert_eq!(alias.len(), OUTPUTS);
                drop(alias);
            });
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), baseline - source_bytes + q);
    // Actual request and direct reservation aliases came from the same core
    // startup. Each owner family can retire concurrently without refunding Q
    // before the remaining family has destroyed its payload and Arc shell.
    std::thread::scope(|scope| {
        for request in request_aliases {
            let barrier = &barrier;
            let pool = &pool;
            scope.spawn(move || {
                barrier.wait();
                assert_eq!(pool.used_bytes().unwrap(), baseline - source_bytes + q);
                drop(request);
            });
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), baseline - source_bytes + q);
    std::thread::scope(|scope| {
        for reservation in reservation_aliases {
            let barrier = &barrier;
            let pool = &pool;
            scope.spawn(move || {
                barrier.wait();
                assert_eq!(pool.used_bytes().unwrap(), baseline - source_bytes + q);
                drop(reservation);
            });
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), baseline - source_bytes);
}

// Measure through the genuine generic startup path, including the actual private
// fixed-controller type used by TextGeneration. No guessed C layout or byte
// equality authenticates the concrete admission in the subsequently fresh pool.
fn probe_q(manual: bool) -> u64 {
    let (mut runtime, f) = setup(manual, None);
    let baseline = f.pool.used_bytes().unwrap();
    if manual {
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let run = driver
            .start_input_with_sequence(
                prompt(&f),
                f.config(),
                AllController,
                None,
                GenerationSequenceRequest::new(OUTPUTS, &EOS),
            )
            .unwrap();
        assert_eq!(f.facts.lock().unwrap().predictions, 0);
        drop(run);
    } else {
        let run = TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .unwrap();
        assert_eq!(f.facts.lock().unwrap().predictions, 0);
        drop(run);
    }
    assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    let q = f.facts.lock().unwrap().accepted_q;
    assert!(q > 0);
    q
}

struct UndeclaredController {
    shared: bool,
}
impl TokenFilterController for UndeclaredController {
    type Error = std::convert::Infallible;
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        if self.shared {
            TextControllerStorage::RunOwnedWithSharedFilters(&[])
        } else {
            TextControllerStorage::Unknown
        }
    }
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: TextFilterWorkspace::Exact(&TokenFilter::All),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("rejected before callback")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("rejected before callback")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("rejected before callback")
    }
}
#[test]
fn complete_original_unknown_or_shared_controller_rejects_before_candidate_or_callback() {
    for shared in [false, true] {
        let (mut runtime, f) = setup(true, None);
        let before = f.pool.used_bytes().unwrap();
        let issued = f.pool.0.usage.lock().unwrap().next_funding;
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let error = driver
            .start_input_with_sequence(
                prompt(&f),
                f.config(),
                UndeclaredController { shared },
                None,
                GenerationSequenceRequest::new(OUTPUTS, &EOS),
            )
            .err()
            .unwrap();
        match error {
            ControlledTextGenerationError::Preparation(error) => assert_eq!(
                error
                    .source()
                    .unwrap()
                    .downcast_ref::<PreparedRequestRejection>(),
                Some(&PreparedRequestRejection::MissingController)
            ),
            _ => panic!("fixed original controller rejection"),
        }
        assert_eq!(f.pool.used_bytes().unwrap(), before);
        assert_eq!(f.pool.0.usage.lock().unwrap().next_funding, issued);
        let facts = f.facts.lock().unwrap();
        assert_eq!(
            (
                facts.quotes,
                facts.prepared,
                facts.encoded,
                facts.predictions
            ),
            (0, 0, 0, 0)
        );
    }
    // The actual builtin mask controller has a valid complete declaration for
    // its own mechanism, but this scalar producer does not implement that
    // mechanism. It cannot inherit the All profile's zero payload term.
    let (mut runtime, f) = setup(false, None);
    let before = f.pool.used_bytes().unwrap();
    let error = TextGeneration::from_input_with_sequence(
        &mut runtime,
        prompt(&f),
        f.config(),
        TokenFilter::Allowed(vec![true; 4]),
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .err()
    .unwrap();
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<PreparedRequestRejection>(),
        Some(&PreparedRequestRejection::MissingController)
    );
    assert_eq!(f.pool.used_bytes().unwrap(), before);
    let facts = f.facts.lock().unwrap();
    assert_eq!(
        (
            facts.quotes,
            facts.prepared,
            facts.encoded,
            facts.predictions
        ),
        (0, 0, 0, 0)
    );
}

#[test]
fn complete_original_filled_diagnostic_poison_retains_exact_error_source_and_conservative_q() {
    let q = probe_q(false);
    let (mut runtime, f) = setup(false, None);
    let baseline = f.pool.used_bytes().unwrap();
    diagnostics::set_after_fill_for_test(diagnostics::AfterFill::Poison);
    let error = TextGeneration::from_input_with_sequence(
        &mut runtime,
        prompt(&f),
        f.config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .err()
    .unwrap();
    let concrete = error
        .source()
        .unwrap()
        .downcast_ref::<ConstructionFailure<Sources>>()
        .unwrap();
    assert_eq!(concrete.destination(), None);
    assert!(concrete.prefix_bytes() > 0);
    assert_eq!(
        concrete
            .source()
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>(),
        Some(&WorkingMemoryError::Poisoned)
    );
    {
        let usage = f.pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(
            usage.reserved + usage.registered + f.pool.0.existing,
            baseline + q
        );
        assert!(usage.pending_original.is_none());
    }
    drop(error);
    // Sticky lock failure is not a certificate of safe account cleanup.
    let usage = f.pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(
        usage.reserved + usage.registered + f.pool.0.existing,
        baseline + q
    );
    let facts = f.facts.lock().unwrap();
    assert_eq!(
        (facts.prepared, facts.encoded, facts.predictions),
        (0, 0, 0)
    );
}
#[test]
fn complete_original_filled_diagnostic_unwind_retires_buffers_before_same_q() {
    let (mut runtime, f) = setup(false, None);
    let baseline = f.pool.used_bytes().unwrap();
    diagnostics::set_after_fill_for_test(diagnostics::AfterFill::Unwind);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .map(drop)
    }));
    assert!(outcome.is_err());
    assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    assert!(f.pool.0.usage.lock().unwrap().pending_original.is_none());
    let fresh = TextGeneration::from_input_with_sequence(
        &mut runtime,
        prompt(&f),
        f.config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .unwrap();
    drop(fresh);
    assert_eq!(f.pool.used_bytes().unwrap(), baseline);
}

#[test]
fn original_post_q_window_plan_copies_before_source_borrow_ends_and_keeps_real_failures() {
    use eredu_core::{
        attention::{AttentionPolicy, LayerSchedule},
        cache::LayerCachePolicy,
    };
    for failure in (0..8).map(Some).chain(std::iter::once(None)) {
        let (_runtime, f) = setup(false, None);
        // Independent caller-owned diagnostic fixture. Its window set is the
        // actual recurrence's [2,4]; it supplies no alternate state/byte report.
        let policies = [4, 2, 4]
            .into_iter()
            .map(|window| {
                LayerCachePolicy::key_only(AttentionPolicy::sliding(window).unwrap(), 1, 1).unwrap()
            })
            .collect();
        let layout = StateMemoryLayout::new(
            LayerSchedule::new(3, policies).unwrap(),
            vec![0; 3],
            2,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let windows = estimate_runtime_state_facts(
            &layout,
            f.request().input,
            OUTPUTS as u64,
            1,
            NonZeroU8::new(4).unwrap(),
        )
        .unwrap()
        .sliding_windows;
        assert_eq!(windows.iter().collect::<Vec<_>>(), [2, 4]);
        let baseline = f.pool.used_bytes().unwrap();
        if let Some(at) = failure {
            diagnostics::fail_destination_for_test(at);
        }
        let result = admit(
            Recipe {
                pool: &f.pool,
                execution: &f.execution,
                selected_model: &f.sources.selected_model,
                input: &f.sources.input,
                initialized_revision: f.revision,
                current_revision: f.current_revision,
                current_frontier: f.current_frontier,
                maximum_context: 32,
                request: f.request(),
                geometry: geometry(),
                capacity: None,
            },
            f.sources.clone(),
            |g| {
                let mut report: StateReport<'_> = f.report(
                    g,
                    text_generation_control_bytes::<Backend, AllController>().unwrap(),
                )?;
                report.sliding_windows = diagnostics::ReportWindows::State(windows);
                Ok(report)
            },
        );
        // Success and owning errors cannot retain an unowned report/layout loan.
        drop(layout);
        let charged = f.pool.used_bytes().unwrap();
        assert!(charged > baseline);
        match result {
            Ok((sources, reservation)) => {
                assert!(failure.is_none());
                assert_eq!(
                    reservation
                        .admission()
                        .state
                        .assumptions
                        .sliding_window_bounds,
                    [2, 4]
                );
                assert!(sources.input.same_source(&f.sources.input));
                drop((sources, reservation));
            }
            Err(Failure::Construction(error)) => {
                let error = BackendFailure::from_error(error);
                let exact = error
                    .source()
                    .unwrap()
                    .downcast_ref::<ConstructionFailure<Sources>>()
                    .unwrap();
                assert_eq!(exact.prefix_bytes() == 0, failure == Some(0));
                assert_eq!(f.pool.used_bytes().unwrap(), charged);
                drop(error);
            }
            other => panic!("unexpected window construction result: {other:?}"),
        }
        assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    }
}

// Shared-core-issued identity for focused private lock tests. This executes the
// real original neutral consumer, rather than constructing context evidence.
pub(in crate::working_memory) fn issued_lock_context() -> TextStepContext {
    let (mut runtime, fixture) = setup(false, None);
    let input = prompt(&fixture);
    let mut generation = TextGeneration::from_input_with_sequence(
        &mut runtime,
        input,
        fixture.config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .unwrap();
    let _sequence = generation
        .take_prepared_sequence()
        .unwrap()
        .prepare_storage()
        .unwrap();
    let token = generation.next().unwrap().unwrap();
    token.owner.context.clone()
}
