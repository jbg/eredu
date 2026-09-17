//! One preparation and prefill authority per admitted request.
use super::control_mutex::{self, ControlMutex};
use super::{
    DecoderCopyAdmissionError, HostSlotStorageKey, InferenceExecutionIdentity, InferenceRequest,
    InitializedDenseDecoderSlots, RegisteredDenseDecoderInitialization, RunOwnedTextSampler,
    SamplerResumePlan, WorkingMemoryError, WorkingMemoryFundingRun, WorkingMemoryFundingScope,
    WorkingMemoryReservation, WorkingMemorySamplerScope, WorkingMemoryStorage,
};
use eredu_core::{InferenceGeometry, TextGenerationConfig};
use std::sync::Arc;

mod no_decoder;
mod run;
pub use run::{InferenceTextStep, InferenceTextStepReceipt};

#[derive(Debug)]
pub(super) enum RequestStart {
    Fresh,
    Preparing(Arc<TextPreparationAuthority>),
    Started(Option<Arc<TextPreparationAuthority>>),
}

#[derive(Debug)]
pub(super) struct TextPreparationAuthority {
    config: TextGenerationConfig,
    // Binding and stage claims share one lock: no construction can race a
    // supposedly cold binding of the original run and policy.
    state: ControlMutex<TextPreparationState>,
}

#[derive(Debug, Default)]
struct TextPreparationState {
    prompt: u8,
    sampling: u8,
    run: Option<run::RunProgress>,
}
const UNCLAIMED: u8 = 0;
const CLAIMED: u8 = 1;
const CONSTRUCTED: u8 = 2;
const READY: u8 = 3;
// Original sequence construction precedes the prompt in the same authority.
// Reuse its existing byte; no new separately allocated authority control.
const SEQUENCE_CLAIMED: u8 = 4;
const SEQUENCE_CONSTRUCTED: u8 = 5;

/// Shared retention for one admitted text preparation. Clones preserve the
/// request charge and stage claims; they cannot construct another prompt or
/// sampler. Only its bound request can enter prefill after preparation succeeds.
/// This binds startup configuration, not a new or complete workspace quote.
#[derive(Debug, Clone)]
pub struct InferenceTextPreparation {
    request: InferenceRequest,
}

/// Move-only completion authority for one prompt or sampler construction.
/// Dropping it after failure never makes the stage available for another attempt.
#[derive(Debug)]
#[must_use = "finish the stage only after its native construction succeeds"]
pub struct InferencePreparationStage {
    request: InferenceRequest,
    prompt: bool,
}

/// Finish-only remainder of a consumed sampler construction stage. This cannot
/// construct another sampler; native PRNG/recovery preparation must finish first.
#[derive(Debug)]
#[must_use = "finish only after the entire original sampling preparation succeeds"]
pub struct InferenceSamplerCompletion {
    stage: InferencePreparationStage,
}

impl InferenceSamplerCompletion {
    pub(in crate::working_memory) fn validate_scope_claim(
        &self,
        preparation: &InferenceTextPreparation,
    ) -> Result<(), WorkingMemoryError> {
        self.stage.validate_scope_claim(preparation, false)
    }

    /// Completes the original sampler preparation without issuing another copy
    /// or allocation permission.
    pub fn finish(self) -> Result<(), WorkingMemoryError> {
        self.stage.finish()
    }
}

/// Finish-only remainder of the original prompt construction. Table-backed
/// construction returns it after exact fill/publication; a no-decoder prompt
/// returns it after exact run and complete-source validation. Native/input
/// preparation must also succeed before finishing it.
#[derive(Debug)]
#[must_use = "finish only after all original prompt preparation succeeds"]
pub struct InferencePromptCompletion {
    stage: InferencePreparationStage,
}

impl InferencePromptCompletion {
    pub(in crate::working_memory) fn new(stage: InferencePreparationStage) -> Self {
        Self { stage }
    }

    pub(super) fn validate_pending_token_input(
        &self,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.stage.validate_dense_prompt_construction()
    }

    /// Protects the closed one-position incremental token container before its
    /// construction. Numerical token preparation remains separately admitted;
    /// the returned owner can only construct one Text/TokenIds part with empty
    /// metadata/extents, and cannot publish another preparation claim.
    pub fn prepare_pending_token_input<T>(
        self,
        funding: &WorkingMemoryFundingRun,
    ) -> Result<super::PreparedPendingTokenInputHost<T>, WorkingMemoryError> {
        let reservation = self.stage.validate_dense_prompt_construction()?;
        let geometry = self.stage.request.geometry();
        if geometry.batch_size != 1
            || geometry.input_positions != 1
            || geometry.prefill_chunk_positions != 1
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let request = self.stage.request.clone();
        super::pending_input::prepare(self, &reservation, funding, request, None)
    }

    /// Constructs one complete prepared text part for the exact admitted input
    /// geometry. This covers both a saved decode token and a saved prompt; the
    /// backend must authenticate its tensor geometry and retain native recovery.
    /// It supplies no token values, cache identity or new preparation claim.
    pub fn prepare_pending_text_input<T>(
        self,
        funding: &WorkingMemoryFundingRun,
        positions: std::num::NonZeroU64,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<super::PreparedPendingTokenInputHost<T>, WorkingMemoryError> {
        let reservation = self.stage.validate_dense_prompt_construction()?;
        let geometry = self.stage.request.geometry();
        if geometry.batch_size != 1
            || geometry.input_positions != positions.get()
            || geometry.prefill_chunk_positions == 0
            || geometry.prefill_chunk_positions > positions.get()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let request = self.stage.request.clone();
        super::pending_input::prepare(self, &reservation, funding, request, host.cloned())
    }

    /// Marks the original prompt construction complete. The existing owned
    /// input binding must still run before prefill becomes ready.
    pub fn finish(self) -> Result<(), WorkingMemoryError> {
        self.stage.finish()
    }
}

impl InferencePreparationStage {
    // Reservation equality alone is insufficient: require this reservation's
    // one actual preparation Arc, its bound run and the still-claimed stage.
    pub(in crate::working_memory) fn validate_scope_claim(
        &self,
        preparation: &InferenceTextPreparation,
        prompt: bool,
    ) -> Result<(), WorkingMemoryError> {
        self.request.validate_same_request(preparation.request())?;
        if self.prompt != prompt {
            return Err(WorkingMemoryError::InvocationPhaseMismatch);
        }
        preparation.validate_scope_preparation()?;
        let actual = self
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let expected = preparation
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !Arc::ptr_eq(actual, expected) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let state = actual
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if (if prompt { state.prompt } else { state.sampling }) != CLAIMED {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        Ok(())
    }

    /// Initializes a fixed dense destination from an actual registered or
    /// funded source under this request's fresh prompt claim. The run must
    /// belong to this exact reservation. Source health and the protected host
    /// hold are checked together before the one-buffer initializer runs.
    ///
    /// The returned native scope is separate from private host custody. Retain
    /// actual native source owners and execute only the enclosing fresh quote's
    /// admitted program through existing settlement/publication machinery.
    /// `complete_source` pins the provider's complete registered inventory; it
    /// does not by itself certify arbitrary nested source or constructor work.
    /// The prompt completion is returned only by exact fill and table publish.
    pub fn construct_dense_decoder<'a, S, D, K: HostSlotStorageKey>(
        self,
        source: RegisteredDenseDecoderInitialization<'a, S, D, K>,
        funding: &WorkingMemoryFundingRun,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<
        (
            InitializedDenseDecoderSlots<'a, S, D, K>,
            WorkingMemoryFundingScope,
        ),
        DecoderCopyAdmissionError,
    > {
        let reservation = self.validate_dense_prompt_construction()?;
        source.construct(self, &reservation, funding, complete_source)
    }

    /// Consumes one original prompt claim to construct the actual outer and
    /// every ordered child table. All sources and the complete P sum are checked
    /// atomically. Children never receive a prompt claim; only exact final outer
    /// publication returns the original finish-only completion.
    pub fn construct_dense_decoder_group<'a, OS, OD, CS, CD, K: HostSlotStorageKey>(
        self,
        source: super::RegisteredDenseDecoderTableGroup<'a, OS, OD, CS, CD, K>,
        funding: &WorkingMemoryFundingRun,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<
        (
            super::InitializedDenseDecoderTableGroup<'a, OS, OD, CS, CD, K>,
            WorkingMemoryFundingScope,
        ),
        DecoderCopyAdmissionError,
    > {
        let reservation = self.validate_dense_prompt_construction()?;
        source.construct(self, &reservation, funding, complete_source)
    }

    fn validate_dense_prompt_construction(
        &self,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        if !self.prompt {
            return Err(WorkingMemoryError::InvocationPhaseMismatch);
        }
        let authority = self
            .request
            .preparation
            .as_ref()
            .expect("preparation grant");
        {
            let state = authority
                .state
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            if state.prompt != CLAIMED {
                return Err(WorkingMemoryError::PreparationNotReady);
            }
        }
        self.request
            .memory_reservation()
            .cloned()
            .ok_or(WorkingMemoryError::UnknownBound)
    }

    /// Constructs the exact canonical sampler under a distinct host-only scope
    /// from this request's funding account. This closed custody cannot cover
    /// native PRNG or recovery work. Rejected unused custody settles only its
    /// dedicated host scope; native work scopes remain independent.
    ///
    /// Consuming this stage prevents repeated construction. The returned token
    /// can only finish preparation after the native caller completes its work.
    pub fn construct_sampler(
        self,
        host_scope: WorkingMemorySamplerScope,
    ) -> Result<(RunOwnedTextSampler, InferenceSamplerCompletion), WorkingMemoryError> {
        let (config, maximum, execution) = self.validate_sampler_construction(&host_scope)?;
        let sampler = RunOwnedTextSampler::construct(config, maximum, execution, host_scope)?;
        Ok((sampler, InferenceSamplerCompletion { stage: self }))
    }

    /// Copies an actual funded sampler into this fresh request's closed host
    /// custody, preserving history and adaptive state. The plan's new allowance
    /// and configuration must exactly match this consumed preparation stage.
    /// This does not copy native RNG/decoder state or grant a resumed step.
    pub fn construct_resumed_sampler(
        self,
        plan: SamplerResumePlan<'_>,
        host_scope: WorkingMemorySamplerScope,
    ) -> Result<(RunOwnedTextSampler, InferenceSamplerCompletion), WorkingMemoryError> {
        let (config, maximum, execution) = self.validate_sampler_construction(&host_scope)?;
        let sampler = plan.construct(config, maximum, execution, host_scope)?;
        Ok((sampler, InferenceSamplerCompletion { stage: self }))
    }

    fn validate_sampler_construction(
        &self,
        host_scope: &WorkingMemorySamplerScope,
    ) -> Result<(TextGenerationConfig, u64, InferenceExecutionIdentity), WorkingMemoryError> {
        if self.prompt {
            return Err(WorkingMemoryError::InvocationPhaseMismatch);
        }
        let authority = self
            .request
            .preparation
            .as_ref()
            .expect("preparation grant");
        {
            let state = authority
                .state
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            if state.sampling != CLAIMED {
                return Err(WorkingMemoryError::PreparationNotReady);
            }
        }
        let reservation = self
            .request
            .memory_reservation()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let execution = host_scope.validate_stage(reservation)?;
        Ok((
            authority.config,
            self.request.geometry().max_output_tokens,
            execution,
        ))
    }

    /// Publishes successful construction. Prompt construction still needs binding
    /// to its owned input before prefill can begin.
    pub fn finish(self) -> Result<(), WorkingMemoryError> {
        let authority = self
            .request
            .preparation
            .as_ref()
            .expect("preparation grant");
        let mut state = authority
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let (stage, next) = if self.prompt {
            (&mut state.prompt, CONSTRUCTED)
        } else {
            (&mut state.sampling, READY)
        };
        if *stage != CLAIMED {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        *stage = next;
        Ok(())
    }
}

impl InferenceRequest {
    /// Claims preparation before native prompt or sampler work. Identity errors
    /// leave the request fresh; a successful claim is never refunded on failure.
    pub fn prepare_text(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
    ) -> Result<InferenceTextPreparation, WorkingMemoryError> {
        self.validate(execution, geometry)?;
        if config
            .sampling()
            .max_new_tokens
            .and_then(|tokens| u64::try_from(tokens).ok())
            != Some(geometry.max_output_tokens)
        {
            return Err(WorkingMemoryError::PreparationConfigurationMismatch);
        }
        let mut start = self
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if !matches!(*start, RequestStart::Fresh) {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        let preparation = Arc::new(TextPreparationAuthority {
            config,
            state: ControlMutex::new(TextPreparationState::default()),
        });
        let mut request = self.clone();
        request.preparation = Some(Arc::clone(&preparation));
        *start = RequestStart::Preparing(preparation);
        Ok(InferenceTextPreparation { request })
    }

    pub(crate) fn begin_prefill(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.validate(execution, geometry)?;
        let mut start = self
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        match &*start {
            RequestStart::Fresh => {}
            RequestStart::Preparing(preparation) => {
                if self
                    .preparation
                    .as_ref()
                    .is_none_or(|bound| !Arc::ptr_eq(bound, preparation))
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                let state = preparation
                    .state
                    .lock()
                    .map_err(|_| WorkingMemoryError::Poisoned)?;
                if state.prompt != READY || state.sampling != READY {
                    return Err(WorkingMemoryError::PreparationNotReady);
                }
            }
            RequestStart::Started(_) => return Err(WorkingMemoryError::AlreadyStarted),
        }
        let canonical = match &*start {
            RequestStart::Preparing(authority) => Some(Arc::clone(authority)),
            RequestStart::Fresh => None,
            RequestStart::Started(_) => unreachable!("checked fresh prefill"),
        };
        *start = RequestStart::Started(canonical);
        Ok(())
    }
}

impl InferenceTextPreparation {
    pub(in crate::working_memory) fn validate_scope_preparation(
        &self,
    ) -> Result<(), WorkingMemoryError> {
        let start = self
            .request
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let actual = self
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match &*start {
            RequestStart::Preparing(expected) if Arc::ptr_eq(actual, expected) => {}
            RequestStart::Started(_) => return Err(WorkingMemoryError::AlreadyStarted),
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        }
        let state = actual
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.run.is_none() {
            return Err(WorkingMemoryError::TextRunUnbound);
        }
        Ok(())
    }

    // Memory construction only, before the actual core run binding. This exact
    // private Preparing Arc must be untouched; it cannot authorize a native Scope.
    pub(in crate::working_memory) fn validate_tracking_preparation(
        &self,
        capacity: Option<std::num::NonZeroU64>,
    ) -> Result<(), WorkingMemoryError> {
        let start = self
            .request
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let actual = self
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match &*start {
            RequestStart::Preparing(expected) if Arc::ptr_eq(actual, expected) => {}
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        }
        if actual
            .config
            .inference_policy()
            .submission_tracking_capacity_bytes
            != capacity
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let state = actual
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.prompt != UNCLAIMED || state.sampling != UNCLAIMED {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        Ok(())
    }

    // Memory construction only, before the actual core run binding. This exact
    // private Preparing Arc must be untouched; it cannot authorize a native Scope.
    pub(in crate::working_memory) fn validate_graph_preparation(
        &self,
        capacity: Option<std::num::NonZeroU64>,
    ) -> Result<(), WorkingMemoryError> {
        let start = self
            .request
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let actual = self
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match &*start {
            RequestStart::Preparing(expected) if Arc::ptr_eq(actual, expected) => {}
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        }
        if actual
            .config
            .inference_policy()
            .graph_metadata_capacity_bytes
            != capacity
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let state = actual
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.prompt != UNCLAIMED || state.sampling != UNCLAIMED {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        Ok(())
    }

    /// Retains the same charge while identifying this exact preparation path.
    pub fn request(&self) -> &InferenceRequest {
        &self.request
    }

    fn authority(&self) -> &TextPreparationAuthority {
        self.request
            .preparation
            .as_ref()
            .expect("preparation owns its grant")
    }

    /// Claims one native prompt construction before allocating its tensors.
    pub fn claim_prompt(&self) -> Result<InferencePreparationStage, WorkingMemoryError> {
        let mut state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.prompt == SEQUENCE_CONSTRUCTED {
            state.prompt = UNCLAIMED;
        }
        claim(&mut state.prompt)?;
        Ok(InferencePreparationStage {
            request: self.request.clone(),
            prompt: true,
        })
    }

    /// Publishes an owned prepared input after successful binding. An existing
    /// input may skip construction, but binding still consumes a single claim.
    pub fn bind_prompt(&self) -> Result<(), WorkingMemoryError> {
        let mut state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.prompt != UNCLAIMED
            && state.prompt != CONSTRUCTED
            && state.prompt != SEQUENCE_CONSTRUCTED
        {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        state.prompt = READY;
        Ok(())
    }

    /// Claims one sampler initialization with the exact startup configuration.
    /// A mismatched configuration does not consume the stage.
    pub fn claim_sampling(
        &self,
        config: TextGenerationConfig,
    ) -> Result<InferencePreparationStage, WorkingMemoryError> {
        if config != self.authority().config {
            return Err(WorkingMemoryError::PreparationConfigurationMismatch);
        }
        let mut state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.prompt == SEQUENCE_CLAIMED {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        claim(&mut state.sampling)?;
        Ok(InferencePreparationStage {
            request: self.request.clone(),
            prompt: false,
        })
    }
}

fn claim(stage: &mut u8) -> Result<(), WorkingMemoryError> {
    if *stage != UNCLAIMED {
        return Err(WorkingMemoryError::PreparationAlreadyStarted);
    }
    *stage = CLAIMED;
    Ok(())
}

// Actual loan/result controls used only by the original tracking extraction.
pub(super) fn tracking_validation_control_bytes() -> Option<usize> {
    validation_control_bytes()
}

// Actual loan/result controls used only by the original graph metadata extraction.
pub(super) fn graph_validation_control_bytes() -> Option<usize> {
    validation_control_bytes()
}

fn validation_control_bytes() -> Option<usize> {
    control_mutex::require_known_layout().ok()?;
    [
        std::mem::size_of::<std::num::NonZeroU64>(),
        control_mutex::operation_control_bytes::<RequestStart>(),
        control_mutex::operation_control_bytes::<TextPreparationState>(),
        std::mem::size_of::<Result<(), WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

// Fixed guard, attempted acquisition and poison/result controls. Two concurrent
// state loans are real in ordered supersession, so enumerate that second loan.
// A request that only uses one loan retains this same finite shared envelope.
pub(super) fn synchronization_control_bytes() -> Option<usize> {
    control_mutex::require_known_layout().ok()?;
    [
        control_mutex::operation_control_bytes::<RequestStart>(),
        control_mutex::operation_control_bytes::<TextPreparationState>(),
        control_mutex::operation_control_bytes::<TextPreparationState>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
