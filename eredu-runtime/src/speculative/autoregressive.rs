//! Speculative transactions between two independently executable language models.
//!
//! Models retain their ordinary equations, residency and state geometry. This
//! mechanism owns only proposal isolation, verification and accepted-prefix replay.
use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill,
    SpeculativePrefillOutcome, Submission, execution_control::SnapshotEstimate,
};

mod occurrence;
mod occurrence_owners;
pub use occurrence::{
    AutoregressiveContinuation, AutoregressiveInvocation, AutoregressiveInvocationDomain,
    AutoregressiveOccurrenceClaim, AutoregressiveOccurrenceCursor, AutoregressiveOccurrenceError,
    AutoregressiveScheduleIdentity, AutoregressiveSchedulePlan, AutoregressiveSource,
};
use occurrence_owners::OccurrenceOwners;

/// Attribution of a model invocation within an independent-draft transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoregressivePass {
    /// Canonical target prompt.
    TargetPrefill,
    /// Canonical draft prompt.
    DraftPrefill,
    /// Tentative draft advancement.
    Proposal,
    /// Tentative target verification.
    Verification,
    /// Accepted target prefix replay following a rejection.
    TargetCommit,
    /// Canonical draft advancement to the accepted target frontier.
    DraftCommit,
}

/// Actual selected initial-prefill result. Target prefill returns its one
/// sampling distribution; draft prefill has no scores. Verification retains the
/// separate complete Output type and its existing sequence indexing.
pub struct AutoregressivePrefill<L> {
    /// Present only for target LastPosition demand.
    pub logits: Option<L>,
    /// Actual prompt positions consumed, independent of score row count.
    pub evaluated_tokens: usize,
}

/// Ordinary model and state operations required by independent-model drafting.
/// Checkpoints are immutable and cloning one must preserve branch isolation.
/// Methods retaining native work must retain all inputs through exact completion.
pub trait AutoregressiveMechanisms {
    /// An ordinary executable; no model-family dispatch occurs in this mechanism.
    type Model: ?Sized;
    /// Prepared input reusable by both tokenizer-compatible models.
    type Input;
    /// Complete ordinary mutable state.
    type State;
    /// Immutable reusable state checkpoint.
    type Checkpoint: Clone;
    /// Complete sequence logits.
    type Output;
    /// Tensor exposed by the same ordinary model observation hooks.
    type Activation: 'static;
    /// One sampling distribution.
    type Logits;
    /// Selected native placement.
    type Context<'a>: Copy;
    /// Exact verification completion.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional backend timing and resource measurements.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Projects a stable request ID through the existing batch assignment.
    /// No allocation or new admission is permitted during this projection.
    fn request_context<'a>(
        _request: eredu_core::generation::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<Self::Context<'a>, Self::Error>
    where
        Self: 'a,
    {
        Ok(context)
    }
    /// Existing table insertion ID retained by a selected request context.
    /// A finite occurrence table requires an ID; ordinary/single execution does
    /// not. This is a read-only projection and creates no request authority.
    fn occurrence_request(_context: Self::Context<'_>) -> Option<eredu_core::SpeculativeRequestId> {
        None
    }
    /// Original host destination for shared proposal/history bookkeeping.
    fn driver_buffer<T>(
        capacity: usize,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<T>, Self::Error> {
        Ok(eredu_core::SpeculativeBuffer::with_capacity(capacity))
    }
    /// Admits one concrete host metadata constructor. Managed mechanisms reject
    /// unknown storage before allocation; ordinary construction is unchanged.
    fn driver_host_metadata(
        _bytes: Option<usize>,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::HostPreparationAuthority, Self::Error> {
        Ok(eredu_core::HostPreparationAuthority::unmanaged())
    }
    /// Exact query for the same actual host destination producer.
    fn driver_buffer_bytes<T>(capacity: usize) -> Option<usize> {
        eredu_core::SpeculativeBuffer::<T>::retained_control_bytes(capacity)
    }
    /// Creates an exact canonical request identity.
    fn driver_identity(
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeRequestIdentity, Self::Error> {
        Ok(eredu_core::SpeculativeRequestIdentity::new())
    }
    /// Copies the actual canonical sequence before a shared transaction.
    fn copy_sequence(
        source: eredu_core::SpeculativeSequenceRef<'_>,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeSequence, eredu_core::SpeculativeDriverError<Self::Error>>
    {
        source
            .copy_ordinary()
            .map_err(eredu_core::SpeculativeDriverError::Preparation)
    }
    /// Complete provider copy request for controlled snapshot estimation.
    fn sequence_copy_bytes(source: &eredu_core::SpeculativeSequence) -> Option<u64> {
        match source {
            eredu_core::SpeculativeSequence::Ordinary(_) => source.snapshot_storage_bytes(),
            eredu_core::SpeculativeSequence::Retained(_) => None,
        }
    }
    /// Transfers an already retained neutral failure without allocating a new source.
    /// Success must preserve its existing kind, operation and source identity;
    /// it must not infer origin or reclassify an arbitrary error. Return
    /// `Err(error)` with the identical unmodified owner when no retained
    /// representation is available. The default preserves ordinary conversion.
    /// This hook does not allocate storage, grant admission, or settle work.
    fn take_retained_failure(
        error: Self::Error,
    ) -> Result<eredu_core::BackendFailure, Self::Error> {
        Err(error)
    }

    /// Coordinates the existing preparation boundary; native implementations
    /// delegate to the retained execution transport rather than checking locally.
    fn agree_text_preparation(
        _stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        use eredu_core::run_preparation::{
            TextPreparationOutcome as O, TextPreparationStatus as S,
        };
        Ok(match status {
            S::Ready => O::Ready,
            S::Cancelled => O::Cancelled,
            S::Failed => O::Rejected { rank: 0 },
        })
    }
    /// Keeps later lane cancellation/completion scheduling in the same agreement.
    /// Preserves an admitted scheduler destination through coordination.
    fn coordinate_speculative_buffer(
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<
        eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        eredu_core::BackendFailure,
    > {
        match local.try_into_ordinary() {
            Ok(local) => Self::coordinate_speculative_step(local, context).map(Into::into),
            Err(_) => Err(eredu_core::HostMetadataFundingError::Unavailable.into()),
        }
    }
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        _context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Ok(local)
    }

    /// Realizes one empty ordinary state using the model's retained selection.
    fn empty(
        model: &mut Self::Model,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Self::Error>;
    /// Executes selected LastPosition (target) or StateOnly (draft) prefill.
    /// Cancellation is returned only at an agreed, safely completed boundary.
    fn prefill(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        pass: AutoregressivePass,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'_>,
    ) -> Result<SpeculativePrefillOutcome<AutoregressivePrefill<Self::Logits>>, Self::Error>;
    /// Executes supplied token IDs with the ordinary decoder.
    fn decode(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::Output, Self::Error>;
    /// Read-only actual state frontier for an installed occurrence contract.
    /// Unknown state remains unqualified; ordinary execution does not call this.
    fn invocation_frontier(_state: &Self::State) -> Result<Option<u64>, Self::Error> {
        Ok(None)
    }
    /// Exact prepared plain-input length for a selected fresh-request contract.
    /// No token encoding or model work may occur in this inspection.
    fn invocation_input_positions(_input: &Self::Input) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }
    /// Retains the fixed preflight failure in the mechanism's error domain.
    fn occurrence_error(_cause: AutoregressiveOccurrenceError) -> Self::Error {
        Self::invalid("independent speculative occurrence preflight refused")
    }

    /// Funds additional request slots for a restored canonical future. This is
    /// separate from native role admission and never resets spent occurrences.
    /// Ordinary mechanisms with no installed schedule do not call this hook.
    fn prepare_continuation(
        _continuation: &AutoregressiveContinuation,
        _context: Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Prepares the exact source/pass/width before the existing decoder can
    /// allocate inputs or mutate state. An original implementation consumes its
    /// same-request occurrence cursor and installs the corresponding role here.
    /// Refusal prevents `decode`; a later decode error never refunds an attempt.
    /// The default preserves the ordinary mechanism and grants no native fit.
    fn before_invocation(
        _model: &Self::Model,
        _state: &Self::State,
        _claim: AutoregressiveOccurrenceClaim<'_>,
        _context: Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Keeps one source/role preparation alive around the actual shared worker.
    /// A mechanism can retain exclusive source loans through preparation and
    /// an unwind-safe role guard through `run`; refusal must occur before `run`.
    /// The once-only callback prevents this hook from duplicating decoder work.
    /// The ordinary uninstalled route does not call either invocation hook.
    fn with_invocation<T>(
        model: &mut Self::Model,
        state: &mut Self::State,
        _input: Option<&Self::Input>,
        claim: AutoregressiveOccurrenceClaim<'_>,
        context: Self::Context<'_>,
        run: impl FnOnce(&mut Self::Model, &mut Self::State) -> Result<T, Self::Error>,
    ) -> Result<T, Self::Error> {
        Self::before_invocation(model, state, claim, context)?;
        run(model, state)
    }

    /// Source-aware form of the same invocation loan. Implementations must
    /// authenticate and price observation before the once-only worker callback.
    fn with_observed_invocation<T>(
        model: &mut Self::Model,
        state: &mut Self::State,
        input: Option<&Self::Input>,
        claim: AutoregressiveOccurrenceClaim<'_>,
        context: Self::Context<'_>,
        _phase: eredu_core::speculative::SpeculativeActivationPhase,
        observer: Option<
            &mut dyn crate::inspection::SpeculativeActivationObserver<Self::Activation, Self::Error>,
        >,
        run: impl FnOnce(
            &mut Self::Model,
            &mut Self::State,
            Option<
                &mut dyn crate::inspection::SpeculativeActivationObserver<
                    Self::Activation,
                    Self::Error,
                >,
            >,
        ) -> Result<T, Self::Error>,
    ) -> Result<T, Self::Error> {
        if observer.is_some() {
            return Err(Self::invalid(
                "selected invocation has no observed source producer",
            ));
        }
        Self::with_invocation(model, state, input, claim, context, |model, state| {
            run(model, state, None)
        })
    }

    /// Executes the same prefill worker, retaining the collector through every
    /// exact physical span. The collector does not grant native role authority.
    fn prefill_with_observer(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        pass: AutoregressivePass,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'_>,
        observer: Option<
            &mut dyn crate::inspection::SpeculativeActivationObserver<Self::Activation, Self::Error>,
        >,
    ) -> Result<SpeculativePrefillOutcome<AutoregressivePrefill<Self::Logits>>, Self::Error> {
        if observer.is_some() {
            return Err(Self::invalid(
                "selected prefill has no observed source producer",
            ));
        }
        Self::prefill(model, input, state, pass, cancellation, context)
    }

    /// Executes the same cached-sequence worker with its source-bound collector.
    fn decode_with_observer(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
        observer: Option<
            &mut dyn crate::inspection::SpeculativeActivationObserver<Self::Activation, Self::Error>,
        >,
    ) -> Result<Self::Output, Self::Error> {
        if observer.is_some() {
            return Err(Self::invalid(
                "selected decode has no observed source producer",
            ));
        }
        Self::decode(model, tokens, state, pass, context)
    }

    /// Creates one admitted source-only collector before any model execution.
    fn activation_observer(
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        _request: eredu_core::SpeculativeRequestId,
        _context: Self::Context<'_>,
    ) -> Result<
        Option<
            Box<
                dyn crate::inspection::SpeculativeActivationObserver<Self::Activation, Self::Error>,
            >,
        >,
        eredu_core::speculative::SpeculativeControlError,
    > {
        if plan.is_empty() {
            Ok(None)
        } else {
            Err(
                eredu_core::speculative::SpeculativeControlError::Unsupported(
                    "selected independent source has no internal capture collector",
                ),
            )
        }
    }

    /// Retains immutable verification inputs before decoding can mutate a cache.
    /// An admitted implementation must fund its real storage and keep that
    /// custody with every escaping token owner. The default is ordinary storage.
    fn verification_tokens(
        tokens: &[u32],
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::GenerationTokenIds, Self::Error> {
        Ok(tokens.to_vec().into())
    }

    /// Captures complete state without permitting mutation through the checkpoint.
    fn checkpoint(state: &Self::State) -> Result<Self::Checkpoint, Self::Error>;
    /// Creates isolated state, settling any copies before publication.
    fn restore(
        saved: &Self::Checkpoint,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Self::Error>;
    /// Reports a complete logical bound for a durable checkpoint copy.
    fn estimate(saved: &Self::Checkpoint) -> Option<SnapshotEstimate>;
    /// Estimates installed state without allocating a checkpoint or issuing copies.
    fn estimate_state(state: &Self::State) -> Option<SnapshotEstimate>;
    /// Selects one sequence position for the shared sampler.
    fn logits(
        output: &Self::Output,
        position: usize,
        context: Self::Context<'_>,
    ) -> Result<Self::Logits, Self::Error>;
    /// Submits exact completion retaining verification and all mutable state use.
    fn completion(
        output: &Self::Output,
        context: Self::Context<'_>,
    ) -> Result<Self::Completion, Self::Error>;
    /// Constructs a typed native implementation error for invalid transactions.
    fn invalid(message: &'static str) -> Self::Error;
}

/// Both ordinary caches at the canonical frontier.
pub struct AutoregressiveCache<S> {
    /// Target verification cache.
    pub target: S,
    /// Draft cache advanced only after canonical commitment.
    pub draft: S,
}

/// Immutable joint rollback boundary.
pub struct AutoregressiveCheckpoint<C> {
    activations: Option<crate::capture::SpeculativeActivationCheckpoint>,
    target: C,
    draft: C,
}

/// Private draft branch. Cloning never shares mutable cache storage.
#[derive(Clone)]
pub struct AutoregressiveProposal<C> {
    depth: usize,
    saved: C,
}

/// Retained target outputs and their exact verification inputs.
pub struct AutoregressiveVerification<O> {
    output: O,
    tokens: eredu_core::GenerationTokenIds,
}

/// Independent drafting through the shared ordinary and controlled drivers.
pub struct AutoregressiveExecutor<'a, M: AutoregressiveMechanisms> {
    target: &'a mut M::Model,
    draft: &'a mut M::Model,
    capacity: std::num::NonZeroUsize,
    occurrences: OccurrenceOwners<'a>,
    observer:
        Option<Box<dyn crate::inspection::SpeculativeActivationObserver<M::Activation, M::Error>>>,
    execution_started: bool,
    capture_configured: bool,
}

impl<'a, M: AutoregressiveMechanisms> AutoregressiveExecutor<'a, M> {
    /// Binds ordinary executables after tokenizer and placement admission.
    pub fn new(
        target: &'a mut M::Model,
        draft: &'a mut M::Model,
        capacity: std::num::NonZeroUsize,
    ) -> Self {
        Self {
            target,
            draft,
            capacity,
            occurrences: OccurrenceOwners::Uninstalled,
            observer: None,
            execution_started: false,
            capture_configured: false,
        }
    }
    /// Borrows this exact selected independent-model execution to prepare its
    /// finite fresh-request occurrence envelope. Selection is retained by loan;
    /// neither an equation trace nor native admission is inferred from it.
    pub fn schedule_plan<'s>(
        &self,
        selected: &'s crate::SelectedSpeculativeRealization,
        input_positions: std::num::NonZeroU64,
        context_positions: std::num::NonZeroU64,
        config: &eredu_core::generation::SpeculativeConfig,
        options: eredu_core::generation::SpeculativeSchedulerOptions,
    ) -> Result<AutoregressiveSchedulePlan<'s>, AutoregressiveOccurrenceError> {
        AutoregressiveSchedulePlan::new(
            selected,
            self.capacity,
            input_positions,
            context_positions,
            config,
            options,
        )
    }

    /// Installs one fresh request's monotonic occurrence owner outside every
    /// cache/checkpoint. This accepts no native grant and changes no scheduler.
    /// A selected source-aware mechanism must supply actual frontier inspection.
    pub fn with_schedule(
        mut self,
        plan: AutoregressiveSchedulePlan<'a>,
    ) -> Result<Self, AutoregressiveOccurrenceError> {
        if !matches!(self.occurrences, OccurrenceOwners::Uninstalled)
            || plan
                .selected()
                .requirements()
                .strategy()
                .proposal_capacity()
                != self.capacity
        {
            return Err(AutoregressiveOccurrenceError::Selection);
        }
        self.occurrences = OccurrenceOwners::Single(std::cell::RefCell::new(plan.into_cursor()));
        Ok(self)
    }

    /// Installs the same monotonic cursor once for each actual batch request.
    /// Plans arrive in shared-table insertion order. Their selected sources
    /// remain borrowed; mutable cache copies never include this occurrence owner.
    pub fn with_schedules(
        mut self,
        plans: eredu_core::SpeculativeBuffer<AutoregressiveSchedulePlan<'a>>,
        context: M::Context<'_>,
    ) -> Result<Self, M::Error> {
        if !matches!(self.occurrences, OccurrenceOwners::Uninstalled) {
            return Err(M::occurrence_error(
                AutoregressiveOccurrenceError::Selection,
            ));
        }
        self.occurrences = OccurrenceOwners::prepare::<M>(plans, self.capacity, context)?;
        Ok(self)
    }

    /// Realizes an isolated pair of ordinary lane caches.
    pub fn new_cache(
        &mut self,
        context: M::Context<'_>,
    ) -> Result<AutoregressiveCache<M::State>, M::Error> {
        if let Some(cursor) = self
            .occurrences
            .select(M::occurrence_request(context))
            .map_err(M::occurrence_error)?
        {
            cursor
                .try_borrow_mut()
                .map_err(|_| M::occurrence_error(AutoregressiveOccurrenceError::Reentrant))?
                .begin_cache()
                .map_err(M::occurrence_error)?;
        }
        Ok(AutoregressiveCache {
            target: M::empty(self.target, AutoregressivePass::TargetPrefill, context)?,
            draft: M::empty(self.draft, AutoregressivePass::DraftPrefill, context)?,
        })
    }
}

impl<M: AutoregressiveMechanisms> SpeculativeExecutor for AutoregressiveExecutor<'_, M> {
    type Input = M::Input;
    type Cache = AutoregressiveCache<M::State>;
    type TargetState = M::Checkpoint;
    type DraftState = AutoregressiveProposal<M::Checkpoint>;

    fn copy_draft_state<'a>(
        &self,
        state: &Self::DraftState,
        _context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'a,
    {
        Ok(state.clone())
    }
    type CacheCheckpoint = AutoregressiveCheckpoint<M::Checkpoint>;
    type Verification = AutoregressiveVerification<M::Output>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn configure_activation_capture<'a>(
        &mut self,
        plan: eredu_core::speculative::AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        if self.execution_started || self.capture_configured {
            return Err(
                eredu_core::speculative::SpeculativeControlError::Unsupported(
                    "activation authority must be installed once before independent execution",
                ),
            );
        }
        self.observer = M::activation_observer(&plan, request, context)?;
        self.capture_configured = true;
        Ok(())
    }
    fn requires_activation_origin(&self) -> bool {
        self.observer.is_some()
    }
    fn set_activation_origin(
        &mut self,
        origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    ) {
        if let Some(observer) = &mut self.observer {
            observer.set_activation_origin(origin);
        }
    }
    fn take_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        self.observer
            .as_mut()
            .and_then(|observer| observer.take_activation_capture())
    }
    fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.observer
            .as_mut()
            .and_then(|observer| observer.take_activation_error())
    }
    fn validate_activation_readmission(
        &self,
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        discovery: Option<&eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        match &self.observer {
            Some(observer) => observer.validate_activation_readmission(plan, discovery),
            None => plan
                .validate(discovery.ok_or(
                    eredu_core::speculative::SpeculativeControlError::Unsupported(
                        "loaded execution has no internal activation discovery",
                    ),
                )?)
                .map_err(Into::into),
        }
    }
    fn readmit_activation_interventions(
        &mut self,
        plan: eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        self.observer
            .as_mut()
            .ok_or(
                eredu_core::speculative::SpeculativeControlError::Unsupported(
                    "internal re-admission requires capture authority from run creation",
                ),
            )?
            .readmit_activation_interventions(plan)
    }

    fn take_retained_failure(
        error: Self::Error,
    ) -> Result<eredu_core::BackendFailure, Self::Error> {
        M::take_retained_failure(error)
    }

    fn driver_buffer<T>(
        &self,
        capacity: usize,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<T>, Self::Error> {
        M::driver_buffer(capacity, context)
    }

    fn driver_host_metadata(
        &self,
        bytes: Option<usize>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::HostPreparationAuthority, Self::Error> {
        M::driver_host_metadata(bytes, context)
    }
    fn driver_buffer_bytes<T>(&self, capacity: usize) -> Option<usize> {
        M::driver_buffer_bytes::<T>(capacity)
    }
    fn request_context<'a>(
        &self,
        request: eredu_core::generation::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<Self::Context<'a>, Self::Error>
    where
        Self: 'a,
    {
        M::request_context(request, context)
    }
    fn driver_identity(
        &self,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeRequestIdentity, Self::Error> {
        M::driver_identity(context)
    }
    fn coordinate_speculative_buffer<'a>(
        &mut self,
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        eredu_core::BackendFailure,
    > {
        M::coordinate_speculative_buffer(local, context)
    }

    fn copy_sequence(
        &self,
        source: eredu_core::SpeculativeSequenceRef<'_>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeSequence, eredu_core::SpeculativeDriverError<Self::Error>>
    {
        M::copy_sequence(source, context)
    }
    fn sequence_copy_bytes(&self, source: &eredu_core::SpeculativeSequence) -> Option<u64> {
        M::sequence_copy_bytes(source)
    }

    fn max_proposals(&self) -> usize {
        self.capacity.get()
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn agree_text_preparation<'a>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        M::agree_text_preparation(stage, status, context)
    }
    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        M::coordinate_speculative_step(local, context)
    }

    fn prefill(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'_>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
        match self.prefill_cancellable(
            input,
            cache,
            &eredu_core::GenerationCancellationToken::new(),
            context,
        )? {
            SpeculativePrefillOutcome::Complete(value) => Ok(value),
            SpeculativePrefillOutcome::Cancelled { .. } => {
                Err(M::invalid("uncancellable prefill joined cancellation"))
            }
        }
    }

    fn prefill_cancellable(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'_>,
    ) -> Result<
        SpeculativePrefillOutcome<SpeculativePrefill<Self::TargetState, Self::Logits>>,
        Self::Error,
    > {
        self.execution_started = true;
        let target = match prefill_invocation::<M>(
            self.occurrences
                .select(M::occurrence_request(context))
                .map_err(M::occurrence_error)?,
            self.target,
            &input,
            &mut cache.target,
            AutoregressivePass::TargetPrefill,
            cancellation,
            self.observer.as_mut().map(|observer| {
                observer.as_mut()
                    as &mut dyn crate::inspection::SpeculativeActivationObserver<
                        M::Activation,
                        M::Error,
                    >
            }),
            context,
        )? {
            SpeculativePrefillOutcome::Complete(target) => target,
            SpeculativePrefillOutcome::Cancelled { evaluated_tokens } => {
                return Ok(SpeculativePrefillOutcome::Cancelled { evaluated_tokens });
            }
        };
        if target.evaluated_tokens == 0 {
            return Err(M::invalid(
                "independent drafting requires a nonempty prompt",
            ));
        }
        let logits = target
            .logits
            .ok_or_else(|| M::invalid("target prefill has no selected logits"))?;
        let draft = match prefill_invocation::<M>(
            self.occurrences
                .select(M::occurrence_request(context))
                .map_err(M::occurrence_error)?,
            self.draft,
            &input,
            &mut cache.draft,
            AutoregressivePass::DraftPrefill,
            cancellation,
            self.observer.as_mut().map(|observer| {
                observer.as_mut()
                    as &mut dyn crate::inspection::SpeculativeActivationObserver<
                        M::Activation,
                        M::Error,
                    >
            }),
            context,
        )? {
            SpeculativePrefillOutcome::Complete(draft) => draft,
            SpeculativePrefillOutcome::Cancelled { .. } => {
                return Ok(SpeculativePrefillOutcome::Cancelled {
                    evaluated_tokens: target.evaluated_tokens,
                });
            }
        };
        if draft.evaluated_tokens != target.evaluated_tokens || draft.logits.is_some() {
            return Err(M::invalid(
                "draft prefill does not match state-only prompt demand",
            ));
        }
        Ok(SpeculativePrefillOutcome::Complete(
            SpeculativePrefill::new(
                logits,
                M::checkpoint(&cache.draft)?,
                target.evaluated_tokens,
            ),
        ))
    }

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        _: u32,
        capacity: usize,
        _: Self::Context<'_>,
    ) -> Result<Self::DraftState, Self::Error> {
        if capacity == 0 || capacity > self.capacity.get() {
            return Err(M::invalid("invalid independent draft proposal capacity"));
        }
        Ok(AutoregressiveProposal {
            depth: 0,
            saved: state.clone(),
        })
    }

    fn proposal_logits(
        &mut self,
        proposal: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
        self.execution_started = true;
        let depth = proposal.depth;
        proposal.depth = depth
            .checked_add(1)
            .ok_or_else(|| M::invalid("proposal observation depth overflow"))?;
        let mut state = M::restore(&proposal.saved, context)?;
        let output = decode_invocation::<M>(
            self.occurrences
                .select(M::occurrence_request(context))
                .map_err(M::occurrence_error)?,
            self.draft,
            &[last_token],
            &mut state,
            AutoregressivePass::Proposal,
            eredu_core::speculative::SpeculativeActivationPhase::Proposal { depth },
            self.observer.as_mut().map(|observer| {
                observer.as_mut()
                    as &mut dyn crate::inspection::SpeculativeActivationObserver<
                        M::Activation,
                        M::Error,
                    >
            }),
            context,
        )?;
        proposal.saved = M::checkpoint(&state)?;
        M::logits(&output, 0, context)
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        Ok(AutoregressiveCheckpoint {
            activations: None,
            target: M::checkpoint(&cache.target)?,
            draft: M::checkpoint(&cache.draft)?,
        })
    }

    fn restore_checkpoint(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        context: Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        // Prepare both copies before replacing either installed state.
        let target = M::restore(&saved.target, context)?;
        let draft = M::restore(&saved.draft, context)?;
        *cache = AutoregressiveCache { target, draft };
        Ok(())
    }

    fn submit_verification(
        &mut self,
        tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
        if tokens.is_empty() || tokens.len() > self.capacity.get().saturating_add(1) {
            return Err(M::invalid("invalid independent draft verification width"));
        }
        self.execution_started = true;
        // Refusal cannot follow a partially submitted verification transaction.
        let retained_tokens = M::verification_tokens(tokens, context)?;
        let output = decode_invocation::<M>(
            self.occurrences
                .select(M::occurrence_request(context))
                .map_err(M::occurrence_error)?,
            self.target,
            tokens,
            &mut cache.target,
            AutoregressivePass::Verification,
            eredu_core::speculative::SpeculativeActivationPhase::Verification,
            self.observer.as_mut().map(|observer| {
                observer.as_mut()
                    as &mut dyn crate::inspection::SpeculativeActivationObserver<
                        M::Activation,
                        M::Error,
                    >
            }),
            context,
        )?;
        let completion = M::completion(&output, context)?;
        Ok(Submission {
            output: AutoregressiveVerification {
                output,
                tokens: retained_tokens,
            },
            completion,
        })
    }

    fn verification_logits(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
        if index >= output.tokens.len() {
            return Err(M::invalid("independent verification index is out of range"));
        }
        M::logits(&output.output, index, context)
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        _: Self::DraftState,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        verified: usize,
        context: Self::Context<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
        if verified == 0 || verified > output.tokens.len() {
            return Err(M::invalid("invalid independent verification commitment"));
        }
        self.execution_started = true;
        let tokens = &output.tokens[..verified];
        let replayed = if verified < output.tokens.len() {
            let mut state = M::restore(&saved.target, context)?;
            decode_invocation::<M>(
                self.occurrences
                    .select(M::occurrence_request(context))
                    .map_err(M::occurrence_error)?,
                self.target,
                tokens,
                &mut state,
                AutoregressivePass::TargetCommit,
                eredu_core::speculative::SpeculativeActivationPhase::TargetReplay,
                self.observer.as_mut().map(|observer| {
                    observer.as_mut()
                        as &mut dyn crate::inspection::SpeculativeActivationObserver<
                            M::Activation,
                            M::Error,
                        >
                }),
                context,
            )?;
            cache.target = state;
            verified
        } else {
            0
        };
        let mut draft = M::restore(&saved.draft, context)?;
        decode_invocation::<M>(
            self.occurrences
                .select(M::occurrence_request(context))
                .map_err(M::occurrence_error)?,
            self.draft,
            tokens,
            &mut draft,
            AutoregressivePass::DraftCommit,
            eredu_core::speculative::SpeculativeActivationPhase::PredictionReplay,
            self.observer.as_mut().map(|observer| {
                observer.as_mut()
                    as &mut dyn crate::inspection::SpeculativeActivationObserver<
                        M::Activation,
                        M::Error,
                    >
            }),
            context,
        )?;
        let seed = M::checkpoint(&draft)?;
        cache.draft = draft;
        Ok(SpeculativeCommit::new(seed, replayed))
    }

    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
    ) -> Option<SnapshotEstimate> {
        let capture = self
            .observer
            .as_ref()
            .map_or(Some(0), |observer| observer.activation_checkpoint_bytes())?
            .checked_add(std::mem::size_of::<Self::CacheCheckpoint>() as u64)?;
        [
            SnapshotEstimate {
                retained_bytes: capture,
                copy_bytes: capture,
            },
            M::estimate_state(&cache.target)?,
            M::estimate_state(&cache.draft)?,
            M::estimate(state)?,
        ]
        .into_iter()
        .try_fold(
            SnapshotEstimate {
                retained_bytes: 0,
                copy_bytes: 0,
            },
            |sum, item| {
                Some(SnapshotEstimate {
                    retained_bytes: sum.retained_bytes.checked_add(item.retained_bytes)?,
                    copy_bytes: sum.copy_bytes.checked_add(item.copy_bytes)?,
                })
            },
        )
    }

    fn control_snapshot(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        _: Self::Context<'_>,
    ) -> Result<
        Option<(Self::CacheCheckpoint, Self::TargetState)>,
        eredu_core::speculative::SpeculativeControlError,
    > {
        let activations = self
            .observer
            .as_ref()
            .map(|observer| observer.activation_checkpoint())
            .transpose()?;
        let mut checkpoint = self.checkpoint(cache).map_err(|error| {
            eredu_core::speculative::SpeculativeControlError::backend_with_retained(
                error,
                Self::take_retained_failure,
            )
        })?;
        checkpoint.activations = activations;
        Ok(Some((checkpoint, state.clone())))
    }

    fn prepare_control_continuation(
        &mut self,
        committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus,
        context: Self::Context<'_>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        let prepare = || -> Result<(), M::Error> {
            let Some(cursor) = self
                .occurrences
                .select(M::occurrence_request(context))
                .map_err(M::occurrence_error)?
            else {
                return Ok(());
            };
            let mut cursor = cursor
                .try_borrow_mut()
                .map_err(|_| M::occurrence_error(AutoregressiveOccurrenceError::Reentrant))?;
            let continuation = cursor
                .continuation(committed, status)
                .map_err(M::occurrence_error)?;
            M::prepare_continuation(&continuation, context)?;
            cursor.install_continuation(continuation);
            Ok(())
        };
        prepare().map_err(|cause| {
            eredu_core::speculative::SpeculativeControlError::backend_with_retained(
                cause,
                M::take_retained_failure,
            )
        })
    }

    fn restore_control_snapshot(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        state: &Self::TargetState,
        context: Self::Context<'_>,
    ) -> Result<Option<Self::TargetState>, eredu_core::speculative::SpeculativeControlError> {
        let prepared = match (self.observer.as_mut(), saved.activations.as_ref()) {
            (Some(observer), Some(saved)) => Some(observer.prepare_activation_restore(saved)?),
            (None, None) => None,
            _ => {
                return Err(eredu_core::speculative::SpeculativeControlError::Invalid(
                    "internal snapshot authority changed",
                ));
            }
        };
        let retain = |error| {
            eredu_core::speculative::SpeculativeControlError::backend_with_retained(
                error,
                M::take_retained_failure,
            )
        };
        let target = M::restore(&saved.target, context).map_err(retain)?;
        let draft = M::restore(&saved.draft, context).map_err(retain)?;
        *cache = AutoregressiveCache { target, draft };
        if let Some(prepared) = prepared {
            prepared.commit();
        }
        Ok(Some(state.clone()))
    }
}

// Single entry around the existing native/portable decoder worker. This is not
// a second speculative driver: proposal, verification and commit still select
// all tokens, passes, state and rollback in the methods above.
fn decode_invocation<M: AutoregressiveMechanisms>(
    occurrences: Option<&std::cell::RefCell<AutoregressiveOccurrenceCursor<'_>>>,
    model: &mut M::Model,
    tokens: &[u32],
    state: &mut M::State,
    pass: AutoregressivePass,
    phase: eredu_core::speculative::SpeculativeActivationPhase,
    observer: Option<
        &mut dyn crate::inspection::SpeculativeActivationObserver<M::Activation, M::Error>,
    >,
    context: M::Context<'_>,
) -> Result<M::Output, M::Error> {
    if let Some(occurrences) = occurrences {
        let invocation = AutoregressiveInvocation::decode(pass, tokens.len())
            .ok_or_else(|| M::invalid("invalid independent draft invocation geometry"))?;
        let claim = claim_invocation::<M>(occurrences, state, invocation)?;
        return M::with_observed_invocation(
            model,
            state,
            None,
            claim,
            context,
            phase,
            observer,
            |model, state, observer| {
                M::decode_with_observer(model, tokens, state, pass, context, observer)
            },
        );
    }
    M::decode_with_observer(model, tokens, state, pass, context, observer)
}

fn claim_invocation<'s, M: AutoregressiveMechanisms>(
    cursor: &std::cell::RefCell<AutoregressiveOccurrenceCursor<'s>>,
    state: &M::State,
    invocation: AutoregressiveInvocation,
) -> Result<AutoregressiveOccurrenceClaim<'s>, M::Error> {
    let frontier = M::invocation_frontier(state)?
        .ok_or_else(|| M::occurrence_error(AutoregressiveOccurrenceError::Geometry))?;
    cursor
        .try_borrow_mut()
        .map_err(|_| M::occurrence_error(AutoregressiveOccurrenceError::Reentrant))?
        .claim(frontier, invocation)
        .map_err(M::occurrence_error)
}
fn prefill_invocation<M: AutoregressiveMechanisms>(
    occurrences: Option<&std::cell::RefCell<AutoregressiveOccurrenceCursor<'_>>>,
    model: &mut M::Model,
    input: &M::Input,
    state: &mut M::State,
    pass: AutoregressivePass,
    cancellation: &eredu_core::GenerationCancellationToken,
    observer: Option<
        &mut dyn crate::inspection::SpeculativeActivationObserver<M::Activation, M::Error>,
    >,
    context: M::Context<'_>,
) -> Result<SpeculativePrefillOutcome<AutoregressivePrefill<M::Logits>>, M::Error> {
    if let Some(occurrences) = occurrences {
        let positions = M::invocation_input_positions(input)?
            .ok_or_else(|| M::occurrence_error(AutoregressiveOccurrenceError::Geometry))?;
        let invocation = AutoregressiveInvocation::prefill(pass, positions)
            .ok_or_else(|| M::occurrence_error(AutoregressiveOccurrenceError::Geometry))?;
        let claim = claim_invocation::<M>(occurrences, state, invocation)?;
        return M::with_observed_invocation(
            model,
            state,
            Some(input),
            claim,
            context,
            match pass {
                AutoregressivePass::TargetPrefill => {
                    eredu_core::speculative::SpeculativeActivationPhase::TargetPrefill
                }
                AutoregressivePass::DraftPrefill => {
                    eredu_core::speculative::SpeculativeActivationPhase::PredictionPrefill
                }
                _ => return Err(M::invalid("non-prefill observation pass")),
            },
            observer,
            |model, state, observer| {
                M::prefill_with_observer(model, input, state, pass, cancellation, context, observer)
            },
        );
    }
    M::prefill_with_observer(model, input, state, pass, cancellation, context, observer)
}
