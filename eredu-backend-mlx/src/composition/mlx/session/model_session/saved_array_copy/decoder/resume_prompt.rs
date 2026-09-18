//! One original prompt claim constructs provisional saved-source resume values.

use super::*;
use crate::backend::runtime::cache::state::PublishedResidentDecoderState;
use crate::composition::mlx::session::model_session::{
    pending_prompt::PreparedPendingPrompt, text_quote::PendingSavedTextAdmission,
};
use eredu_runtime::working_memory::{
    InferencePendingPromptCompletion, InferencePromptCompletion, InferenceTextPreparation,
    PreparedPendingTokenInputHost, WorkingMemoryFundingRun,
};

// Both variants consume the same original prompt claim. Media retains completed
// immutable B; it has no pending-token numerical or part-container producer.
enum PendingPromptHost {
    Tokens(PreparedPendingTokenInputHost<Array>),
    Media(InferencePromptCompletion),
}
enum SavedPromptCompletion {
    Tokens(InferencePendingPromptCompletion),
    Media(InferencePromptCompletion),
}
impl SavedPromptCompletion {
    fn finish(self) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Tokens(completion) => completion.finish(),
            Self::Media(completion) => completion.finish(),
        }
    }
}

/// Fully settled, independently charged values for later sampling/installation.
/// This carries no submission lease, ready seal or historical inference grant.
/// All numerical/control payloads precede the input's fresh request custody.
pub(super) struct ProvisionalTextResumePrompt {
    state: MlxNativeTextState,
    key: Option<Array>,
    prompt: MlxModelInput,
}

impl ProvisionalTextResumePrompt {
    pub(super) fn key(&self) -> Option<&Array> {
        self.key.as_ref()
    }

    pub(super) fn prompt(&self) -> &MlxModelInput {
        &self.prompt
    }

    /// Move the settled parts and the input's original fresh request custody.
    /// The enclosing preparation still binds the prompt exactly once; this
    /// does not install state or issue readiness.
    pub(super) fn into_parts(self) -> (MlxNativeTextState, Option<Array>, MlxModelInput) {
        (self.state, self.key, self.prompt)
    }
}

/// The sole production entry accepts the closed actual-source admission owner.
/// A same-shaped generic reservation is not proof of the native copy envelope.
/// Its claim is consumed once; successful return has no active native lease.
/// The whole prompt driver retains H across this constructor and the later
/// binding checks. Sampling, installation and ready sealing remain later hooks.
pub(super) fn construct_saved_resume_prompt(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    admission: &PendingSavedTextAdmission,
) -> Result<ProvisionalTextResumePrompt, Error> {
    admission.with_funding(|funding| {
        construct_with_original(
            runtime,
            PromptAuthority {
                source: admission.source(),
                preparation: admission.preparation(),
                funding,
            },
            admission
                .host_preparation()
                .map(|host| OriginalPromptAuthority {
                    host,
                    quote: admission.quote(),
                    registered: admission.registered_source(),
                    source_native: admission.source_native(),
                }),
        )
    })
}

/// The lower-level core resume entry can be used without runtime's outer
/// failure wrapper. Keep this attempt's H with an escaping native prompt cause.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct ResumeFailure {
    #[source]
    cause: Error,
    _host: eredu_core::HostPreparationAuthority,
}
impl ResumeFailure {
    pub(super) fn new(cause: Error, host: &eredu_core::HostPreparationAuthority) -> Self {
        Self {
            cause,
            _host: host.clone(),
        }
    }

    /// Core deallocates the concrete failure shell before dropping its cause
    /// and final H. Each whole resume phase calls this once on failure.
    pub(super) fn retain(cause: Error, host: &eredu_core::HostPreparationAuthority) -> Error {
        Error::StorageSource(eredu_core::BackendFailure::from_error(Self::new(
            cause, host,
        )))
    }
}

// Construction is confined to the sealed entry above. Child tests may exercise
// the native worker with an actual full source quote and fresh reservation,
// without inventing core TextStepContext or a production admission constructor.
struct PromptAuthority<'a> {
    source: &'a CopiedTextComponentsOwner,
    preparation: &'a InferenceTextPreparation,
    funding: &'a WorkingMemoryFundingRun,
}

struct OriginalPromptAuthority<'a> {
    source_native: Option<&'a crate::backend::nn::workspace::ProjectedNativeStorage>,
    host: &'a eredu_core::HostPreparationAuthority,
    quote: &'a crate::composition::mlx::session::model_session::text_quote::TextExecutionQuoteOwner,
    registered: &'a WorkingMemoryStorage<StorageIdentity>,
}

#[cfg(test)]
fn construct(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    authority: PromptAuthority<'_>,
) -> Result<ProvisionalTextResumePrompt, Error> {
    construct_with_original(runtime, authority, None)
}

fn construct_with_original(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    authority: PromptAuthority<'_>,
    original: Option<OriginalPromptAuthority<'_>>,
) -> Result<ProvisionalTextResumePrompt, Error> {
    let source = authority.source;
    source.validate_resume_origin(runtime)?;
    let session = runtime.session();
    let pool = runtime.backend().memory_pool();
    let request = authority.preparation.request();
    let geometry = request.geometry();
    let terminal = geometry.input_positions == 0 && geometry.max_output_tokens == 0
        && geometry.prefill_chunk_positions == 0 && geometry.output == eredu_core::OutputDemand::StateOnly;
    let (positions, saved_chunk) = if terminal { (0, 0) } else { source.sampling.pending_geometry().map_err(memory)? };
    if geometry.batch_size != 1
        || geometry.cached_positions != source.sampling.frontier()
        || geometry.input_positions != positions
        || (!terminal && geometry.prefill_chunk_positions == 0)
        || geometry.prefill_chunk_positions > saved_chunk
        || (!terminal && geometry.max_output_tokens == 0)
        || !authority.funding.pool().same_domain(pool)
    {
        return Err(mismatch());
    }
    // Readout comes from the actual admitted child declaration, which may
    // change score interventions. The sealed reservation below authenticates
    // that complete geometry; the saved parent cannot prescribe child readout.
    request
        .memory_reservation()
        .ok_or_else(unknown)?
        .validate(
            session
                .payload
                .model
                .erased()
                .inference_execution_identity(),
            geometry,
        )
        .map_err(memory)?;
    let mut pending_plan = if terminal { None } else { source.sampling.prepare_pending_tokens().map_err(other)? };
    if let Some(pending) = &mut pending_plan {
        pending
            .select_chunk_positions(
                std::num::NonZeroU64::new(geometry.prefill_chunk_positions).ok_or_else(mismatch)?,
            )
            .map_err(other)?;
    } else if original.is_none() {
        return Err(unknown());
    }
    let plan = if original.is_some() {
        source
            .decoder
            .native
            .prepare_copy_fixed()
            .map_err(other)?
            .into_dense_fixed()
            .map_err(other)?
    } else {
        source.decoder.native.prepare_copy()?.into_dense()?
    };
    let origin = source.decoder.origin.clone();
    let prompt_identity = source.decoder.input.clone();
    let complete = if let Some(original) = &original {
        source.validate_resume_account(original.registered)?;
        original.registered.clone()
    } else {
        let mut complete = DecoderCopyOwner::Saved(source.decoder.clone()).complete_storage()?;
        for array in source
            .sampling
            .arrays
            .key
            .iter()
            .chain(source.sampling.arrays.pending.iter().filter(|_| !terminal))
        {
            complete.include_array(array)?;
        }
        complete.pin_registered(pool)?
    };
    let mut operands = Some(0usize);
    if plan.is_paged() {
        plan.visit_snapshot_operands(&mut |source| {
            if matches!(
                source,
                crate::backend::runtime::cache::state::SnapshotOperand::Array(_)
            ) {
                operands = operands.and_then(|n| n.checked_add(1));
            }
            Ok(())
        })
        .map_err(crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error)?;
    } else {
        plan.visit_operands(&mut |_| operands = operands.and_then(|n| n.checked_add(1)))
            .map_err(crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error)?;
    }

    let operands = operands.ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let key_count = usize::from(source.sampling.arrays.key.is_some());
    let mut retained_count = Some(0usize);
    plan.visit_snapshot_operands(&mut |source| {
        if matches!(
            source,
            crate::backend::runtime::cache::state::SnapshotOperand::Array(_)
        ) {
            retained_count = retained_count.and_then(|n| n.checked_add(1));
        }
        Ok(())
    })
    .map_err(crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error)?;
    let retained_count = retained_count.ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let root_count = recovery_root_count(
        operands,
        retained_count,
        key_count,
        pending_plan
            .as_ref()
            .map_or(0, PreparedPendingPrompt::retained_descriptor_count),
        pending_plan.is_some(),
    )
    .map_err(memory)?;
    if original.as_ref().is_some_and(|original| {
        original
            .quote
            .saved_copy_population()
            .is_none_or(|copy| copy.roots() != root_count)
    }) {
        return Err(mismatch());
    }
    let collector = RootCollectorPlan::from_root_count(root_count)?;
    let roots = collector.construct().map_err(other)?;
    let owned_stream = original
        .is_none()
        .then(|| runtime.backend().stream().clone());
    // With no native operands, the shared dense table constructor is host-only.
    // Do not open Scope/Graph/Record or infer a completion from an empty marker.
    let native = operands != 0 || retained_count != 0 || key_count != 0 || pending_plan.is_some();
    let lease = native
        .then(|| {
            session
                .authority
                .borrow_mut()
                .begin_submission()
                .map_err(other)
        })
        .transpose()?;
    let (stage, native_custody) = match &original {
        Some(original) => original.quote.claim_saved_prompt(authority.preparation)?,
        None => (authority.preparation.claim_prompt().map_err(memory)?, None),
    };
    let (slots, scope) = match &original {
        Some(original) => {
            plan.construct_prepared(stage, authority.funding, complete, pool, original.host)?
        }
        None => plan.construct(stage, authority.funding, complete, pool)?,
    };
    let (backend, session) = runtime.parts_mut();
    let stream = owned_stream.as_ref().unwrap_or_else(|| backend.stream());
    let funding = match &original {
        Some(original) => original.quote.funded_work(scope)?,
        None => text_funding::FundedWork::new(scope),
    };
    let (owner, mut operation) = if let Some(lease) = lease {
        let owner = SubmissionResources::with_purpose(
            lease,
            Rc::clone(&session.poison),
            SubmissionPurpose::SavedComponentsCopy(CopyRecovery {
                roots: Rc::clone(&roots),
                source: RefCell::new(Some(CopyRecoverySource::Resume(source.clone()))),
            }),
        );
        owner.payload.replace(Some(session.payload.clone()));
        owner.funding.replace(Some(funding.clone()));
        let recovery = match (&original, native_custody) {
            (Some(original), Some(custody)) => {
                original.quote.begin_saved_prompt(custody, owner.ticket())?
            }
            (None, None) => owner.recovery()?,
            _ => return Err(mismatch()),
        };
        let operation = SessionOperation {
            session,
            owner: owner.clone(),
            recovery: Some(recovery),
            handed_off: false,
            token_validations: Default::default(),
        };
        (Some(owner), Some(operation))
    } else {
        drop(native_custody);
        (None, None)
    };
    let copy = || {
        // Recovery owns the complete source before the first original clone.
        // Its graph bank is active for every source/result C handle.
        let mut failure = None;
        if plan.is_paged() {
            let native = original
                .as_ref()
                .and_then(|authority| authority.source_native)
                .ok_or_else(mismatch)?;
            for (identity, _, _) in native.iter() {
                if let Some(array) = native.native_array(identity) {
                    retain_source(array, &roots, true)?;
                } else if native.native_host(identity).is_none() {
                    return Err(mismatch());
                }
            }
        } else {
            plan.visit_retained_arrays(&mut |array| {
                if failure.is_none() {
                    failure = retain_source(array, &roots, original.is_some()).err();
                }
            })
            .map_err(crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error)?;
        }
        for array in source
            .sampling
            .arrays
            .key
            .iter()
            .chain(source.sampling.arrays.pending.iter().filter(|_| !terminal))
        {
            if failure.is_none() {
                failure = retain_source(array, &roots, original.is_some()).err();
            }
        }
        if let Some(cause) = failure {
            return Err(cause);
        }
        let paged = plan.is_paged();
        let mut observe = |array: &Array| {
            funding.retain(array);
            if let Some(cause) = funding.take_collection_failure() {
                return Err(cause);
            }
            completed(array, original.is_some())
        };
        let state = plan.copy_dense_retained_observed(slots, &stream, &roots, &mut observe)?;
        if !paged {
            let mut error = None;
            state
                .visit_operands(&mut |array| {
                    if error.is_none() {
                        error = observe(array).err();
                    }
                })
                .map_err(
                    crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
                )?;
            if let Some(error) = error {
                return Err(error);
            }
        }
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        tests::before_table_publication(state.slot_metadata(), authority.funding)?;
        // This publishes the host table only. Numerical completion is still
        // protected by this operation; binding is deferred until exact settle.
        let (state, completion) = state.publish_for_control().map_err(|error| {
            let (owner, cause) = error.into_parts();
            drop(owner); // every partial native descriptor remains in recovery
            other(cause)
        })?;
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        tests::after_table_publication(authority.funding)?;
        let host = match &pending_plan {
            Some(pending) => PendingPromptHost::Tokens(pending.prepare_host_with_authority(
                completion,
                authority.funding,
                original.as_ref().map(|original| original.host),
            )?),
            None => PendingPromptHost::Media(completion),
        };
        let key = source
            .sampling
            .arrays
            .key
            .as_ref()
            .map(|array| {
                let key = IsolatedArrayCopy::new(array).copy_retained(&stream, &roots)?;
                funding.retain(&key);
                completed(&key, original.is_some())?;
                Ok::<_, Error>(key)
            })
            .transpose()?;
        let (prompt, completion) = match (pending_plan, host) {
            (Some(pending), PendingPromptHost::Tokens(host)) => {
                let (prompt, completion) = pending.copy_into_prompt(host, &stream, &roots)?;
                let tokens = prompt.parts.first().ok_or_else(mismatch)?.payload().value();
                funding.retain(tokens);
                completed(tokens, original.is_some())?;
                (prompt, SavedPromptCompletion::Tokens(completion))
            }
            (None, PendingPromptHost::Media(completion)) if terminal => {
                let prompt = MlxModelInput {
                    parts: super::super::super::pending_prompt::ModelInputParts::Owned(Vec::new()),
                    controlled_attribution: None, prepared_capture: None, original_media: None,
                    placement_semantics: None, cache_identity: None, prefill_chunk_positions: None,
                    inference_request: Some(request.clone()), memory_owner: None, quote: None,
                };
                (prompt, SavedPromptCompletion::Media(completion))
            }
            (None, PendingPromptHost::Media(completion)) => {
                let media = source.sampling.pending_media().ok_or_else(mismatch)?;
                let prompt = media.packet().pending_prompt(
                    request.clone(),
                    std::num::NonZeroU64::new(geometry.prefill_chunk_positions)
                        .ok_or_else(mismatch)?,
                );
                (prompt, SavedPromptCompletion::Media(completion))
            }
            _ => return Err(mismatch()),
        };
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        tests::after_pending_construction(authority.funding)?;
        debug_assert!(roots.borrow().len() <= root_count);
        Ok::<_, Error>((state, key, prompt, completion))
    };
    let copied = match (&original, native) {
        (Some(original), true) => original.quote.with_saved_prompt_construction(copy),
        _ => copy(),
    };
    let (state, key, prompt, completion): (
        PublishedResidentDecoderState,
        Option<Array>,
        MlxModelInput,
        SavedPromptCompletion,
    ) = match operation.take() {
        Some(operation) => finish_saved_copy(operation, copied)?,
        None => copied?,
    };
    // The optional holder also carries the session lifetime. It is empty only
    // after the owning completion above has settled or returned its failure.
    drop(operation);
    #[cfg(all(
        test,
        target_vendor = "apple",
        feature = "metal",
        not(feature = "cuda")
    ))]
    tests::before_final_publication(authority.funding)?;
    let mut host = funding.prepare_inventory()?;
    state.visit_registered_child_metadata(&mut |metadata| {
        host.include_slot_metadata(metadata.clone())
            .map_err(Error::from)
    })?;
    funding.publish(host)?;
    if let Some(owner) = &owner {
        if let SubmissionPurpose::SavedComponentsCopy(recovery) = &owner.purpose {
            recovery.retire();
        }
    }
    funding.certify()?;
    // The lease is already resolved and every final numerical root has its
    // independent charge. No completion is inferred from marker destruction.
    let model = runtime.session().payload.model.erased();
    let state = match &original {
        Some(original) => model.bind_original_resident_control_state(
            &origin,
            state,
            prompt_identity,
            original.host,
        )?,
        None => MlxNativeTextState::from_ordinary_prepared(
            model.bind_prepared_resident_control_state(&origin, state, prompt_identity)?,
        ),
    };
    completion.finish().map_err(memory)?;
    Ok(ProvisionalTextResumePrompt { state, key, prompt })
}

/// Host-only contribution for the actual retained source views and returned
/// wrappers. Graph/Record/Scope and FundedWork already belong to fresh Q; the
/// destination table/erased state and pending part have separate owning plans.
pub(super) fn preparation_control_bytes(
    decoder: &PreparedResidentDecoderCopy<'_>,
    pending: Option<&PreparedPendingPrompt<'_>>,
    has_key: bool,
) -> Result<usize, super::resume_quote::ResumeSourceCause> {
    use std::mem::size_of;
    let mut operands = Some(0usize);
    let mut retained = Some(0usize);
    if decoder.is_paged() {
        crate::backend::runtime::cache::state::SnapshotArraySources::visit_operands(
            decoder,
            &mut |source| {
                if matches!(
                    source,
                    crate::backend::runtime::cache::state::SnapshotOperand::Array(_)
                ) {
                    operands = operands.and_then(|n| n.checked_add(1));
                    retained = retained.and_then(|n| n.checked_add(1));
                }
                Ok(())
            },
        )?;
    } else {
        decoder.visit_operands(&mut |_| operands = operands.and_then(|n| n.checked_add(1)))?;
        decoder
            .visit_retained_arrays(&mut |_| retained = retained.and_then(|n| n.checked_add(1)))?;
    }

    let roots = recovery_root_count(
        operands.ok_or(WorkingMemoryError::Overflow)?,
        retained.ok_or(WorkingMemoryError::Overflow)?,
        usize::from(has_key),
        pending.map_or(0, PreparedPendingPrompt::retained_descriptor_count),
        pending.is_some(),
    )?;
    let collector = RootCollectorPlan::from_root_count_fixed(roots)?;
    let parts = [
        usize::try_from(collector.retention_control_bytes())
            .map_err(|_| WorkingMemoryError::Overflow)?,
        super::resume_driver::MlxTextResumePreparation::preparation_control_bytes()
            .ok_or(WorkingMemoryError::Overflow)?,
        size_of::<ProvisionalTextResumePrompt>(),
        size_of::<PendingPromptHost>(),
        size_of::<SavedPromptCompletion>(),
        size_of::<Option<PreparedPendingPrompt<'_>>>(),
        size_of::<Option<SessionOperation<'_>>>(),
        size_of::<Option<SubmissionResourcesOwner>>(),
        size_of::<Option<eredu_core::SubmissionLease>>(),
        size_of::<Result<Option<eredu_core::SubmissionLease>, Error>>(),
        size_of::<Result<ProvisionalTextResumePrompt, Error>>(),
        size_of::<(MlxNativeTextState, Option<Array>, MlxModelInput)>(),
        size_of::<PromptAuthority<'_>>(),
        size_of::<OriginalPromptAuthority<'_>>(),
        size_of::<Option<OriginalPromptAuthority<'_>>>(),
        size_of::<(
            PublishedResidentDecoderState,
            Option<Array>,
            MlxModelInput,
            SavedPromptCompletion,
        )>(),
        size_of::<
            Result<
                (
                    PublishedResidentDecoderState,
                    Option<Array>,
                    MlxModelInput,
                    SavedPromptCompletion,
                ),
                Error,
            >,
        >(),
        size_of::<Result<RootCollectorPlan, WorkingMemoryError>>(),
        size_of::<ResumeFailure>(),
        size_of::<Box<ResumeFailure>>(),
        size_of::<Option<Error>>(),
        // One whole-phase closed error owner: prompt construction/binding,
        // sampling and installation are sequential and stop at first failure.
        eredu_core::BackendFailure::source_retention_peak_bytes::<ResumeFailure>()
            .ok_or(WorkingMemoryError::Overflow)?,
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
        .map_err(Into::into)
}

fn recovery_root_count(
    operands: usize,
    retained: usize,
    keys: usize,
    pending_descriptors: usize,
    pending_source: bool,
) -> Result<usize, WorkingMemoryError> {
    // Complete sources include uncopied compressed capacity stores. Each copy
    // retains its contiguous/deep results; the shared pending worker supplies
    // its actual extra reshape/cast descriptors.
    retained
        .checked_add(keys)
        .and_then(|n| n.checked_add(usize::from(pending_source)))
        .and_then(|sources| {
            operands
                .checked_add(keys)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(pending_descriptors))
                .and_then(|n| n.checked_add(sources))
        })
        .ok_or(WorkingMemoryError::Overflow)
}

fn retain_source(array: &Array, roots: &RefCell<Vec<Array>>, original: bool) -> Result<(), Error> {
    let retained = if original {
        array.try_clone_handle()?
    } else {
        array.clone()
    };
    let mut roots = roots.try_borrow_mut().map_err(|_| mismatch())?;
    if roots.len() == roots.capacity() {
        return Err(mismatch());
    }
    roots.push(retained);
    Ok(())
}
fn completed(array: &Array, original: bool) -> Result<(), Error> {
    if original {
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        array.completed_in_original_scope(&observer)?;
    } else {
        array.evaluated()?;
    }
    Ok(())
}

fn other(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(error))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
