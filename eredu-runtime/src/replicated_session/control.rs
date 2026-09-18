//! Opaque independent state slots for serial ordinary-generation branches.

use super::*;
use eredu_core::execution_control::SnapshotEstimate;
use std::sync::Arc;
mod branch;
pub use branch::{ControlBranchSource, ControlBranchPlacement, ControlExchangeResult};

/// Native mechanisms for complete, independently writable ordinary state copies.
/// Unlike rollback checkpoints, these copies must remain stable as any descendant
/// advances. Estimation is side-effect-free; completion stays with the existing
/// backend submission owner, not this portable driver.
pub trait ReplicatedTextSnapshotMechanisms<A, B>: ReplicatedTextSessionMechanisms<A, B>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, Self::State>,
    Self::State: RuntimeState<B>,
    Self::ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
    Self::BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
{
    /// Known logical cost for copying the exact state. None explicitly means
    /// unknown/unsupported; callers must reserve a known estimate before copying.
    fn estimate_snapshot_state(&self, state: &Self::State) -> Option<SnapshotEstimate>;

    /// Same logical copy cost from a strictly allocation-free immutable query.
    /// Ordinary estimates may construct host inventories and cannot serve this
    /// pre-grant path. Missing producer facts or unavailable loans stay unknown.
    fn estimate_original_snapshot_state(&self, _state: &Self::State) -> Option<SnapshotEstimate> {
        None
    }

    /// Additional retained native storage through the admitted input span.
    /// No allocation, execution, or mutation is permitted during estimation.
    fn estimate_snapshot_growth(
        &self,
        _state: &Self::State,
        _additional_input_tokens: u64,
    ) -> Option<u64> {
        None
    }

    /// Logical allocation and initialization bound for a fresh empty state with
    /// the same selected realization. This includes metadata and fixed storage;
    /// it must not be inferred solely from currently populated cache tensors.
    fn estimate_reset_state(&self, _state: &Self::State) -> Option<SnapshotEstimate> {
        None
    }

    /// Copies every native state component and completes its native work,
    /// preserving geometry and positions before distributed preparation agreement.
    /// Mutable storage must be isolated. On error the source remains unchanged;
    /// all unresolved native resources stay with the existing recovery owner.
    fn copy_snapshot_state(
        &mut self,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::State, Self::Error>;
}

/// Exact executable and parameter-snapshot identity at a resolved control boundary.
///
/// Clones retain only an identity allocation, never model parameters, state,
/// requests or completion resources. This token proves neither source inventory
/// completeness nor a byte bound, and grants no execution or allocation authority.
/// An enclosing saved-source owner must bind it to the actual state being copied.
#[derive(Clone, Debug)]
pub struct ReplicatedTextControlOrigin {
    owner: Arc<()>,
    // Captured state provenance is separate from executable/parameter identity.
    // A fixed origin loan never initializes a missing revision allocation.
    captured: Option<CapturedControlRevision>,
}
#[derive(Clone, Debug)]
struct CapturedControlRevision {
    execution: crate::working_memory::InferenceExecutionIdentity,
    revision: crate::working_memory::InferenceStateRevision,
}
impl ReplicatedTextControlOrigin {
    /// Compares the retained executable/parameter identity only. Equality
    /// grants no source inventory, state compatibility or execution authority.
    pub fn same_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }
}

/// Fixed binding failure for an already constructed control slot. No diagnostic
/// String or native error is created before the caller's host admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PreparedControlBindingError {
    /// The shared runtime is fenced, unresolved, or inside a transaction.
    #[error(transparent)]
    Boundary(#[from] RuntimeInspectionBoundary),
    /// Exact executable identity differs.
    #[error("control state belongs to a different executable")]
    ForeignOrigin,
    /// The selected execution has no stateful control geometry.
    #[error("ordinary control requires a selected stateful text execution")]
    MissingSelectedState,
    /// The actual constructed state has different geometry.
    #[error("realized state layout differs from selection")]
    StateLayout,
}
impl PreparedControlBindingError {
    fn into_legacy<A: std::fmt::Display, P: std::fmt::Display, M: std::fmt::Display>(
        self,
    ) -> ReplicatedTextSessionError<A, P, M> {
        match self {
            Self::Boundary(error) => error.into_legacy(),
            Self::ForeignOrigin => ReplicatedTextSessionError::Contract(
                "control state belongs to a different executable".into(),
            ),
            Self::MissingSelectedState => ReplicatedTextSessionError::Contract(
                "ordinary control requires a selected stateful text execution".into(),
            ),
            Self::StateLayout => ReplicatedTextSessionError::Contract(
                "realized state layout differs from selection".into(),
            ),
        }
    }
}

/// Fixed local exchange failures plus the original distributed mechanism
/// cause. Agreement runs through the same strategy/phase worker as ordinary
/// exchange; its native resources remain the caller's separately admitted work.
#[derive(Debug, thiserror::Error)]
pub enum PreparedControlExchangeError<A: std::fmt::Display, P: std::fmt::Display> {
    /// Local resolved-boundary, owner, or geometry failure.
    #[error(transparent)]
    Binding(#[from] PreparedControlBindingError),
    /// Exact media source or copied-state frontier differs.
    #[error(transparent)]
    Media(#[from] crate::working_memory::WorkingMemoryError),
    /// The actual metadata account refused before the new revision was allocated.
    #[error(transparent)]
    Metadata(#[from] eredu_nn::workspace::HostMetadataFundingError),
    /// This selected strategy has no bounded control agreement.
    #[error("partitioned cache control requires the selected bounded failure agreement")]
    MissingAgreement,
    /// A remote participant refused this exact phase.
    #[error("another rank failed distributed cache control at {phase:?}")]
    Remote {
        /// The shared control phase.
        phase: crate::DistributedExecutionPhase,
    },
    /// The actual strategy error, without formatting or loss of its source.
    #[error(transparent)]
    Agreement(ReplicatedTextSessionError<A, P, std::convert::Infallible>),
}

/// Native state plus complete portable execution metadata. This is an opaque
/// in-process slot, not a serialized snapshot or a complete generation snapshot.
/// Slots share an exact executable owner and can be exchanged serially while
/// keeping weights resident. Copying a slot requires the native copy mechanism.
pub struct ReplicatedTextControlState<S> {
    owner: Arc<()>,
    state: S,
    prompt_input_identity: Option<SharedPreparedInputCacheIdentity>,
    captured: Option<CapturedControlRevision>,
}

impl ReplicatedTextControlState<()> {
    /// Adds the same logical control header and borrowed input identity used by
    /// ordinary snapshot estimation. This neither allocates a state nor binds
    /// it to an executable; actual copy admission remains a separate operation.
    pub fn logical_snapshot_estimate(
        native: SnapshotEstimate,
        input: Option<&PreparedInputCacheIdentity>,
    ) -> Option<SnapshotEstimate> {
        let metadata = u64::try_from(std::mem::size_of::<ReplicatedTextControlState<()>>())
            .ok()?
            .checked_add(match input {
                Some(input) => input.logical_metadata_bytes()?,
                None => 0,
            })?;
        Some(SnapshotEstimate {
            retained_bytes: native.retained_bytes.checked_add(metadata)?,
            copy_bytes: native.copy_bytes.checked_add(metadata)?,
        })
    }
}

impl<S> ReplicatedTextControlState<S> {
    /// Borrows the exact saved prompt-identity owner for retained storage inventory.
    pub fn shared_prompt_input_identity(&self) -> Option<&SharedPreparedInputCacheIdentity> {
        self.prompt_input_identity.as_ref()
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSnapshotMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Retains the current exact control identity without copying state or issuing
    /// work. The boundary must be resolved, unfenced and outside a transaction;
    /// backend completion and source custody remain the caller's responsibility.
    pub fn control_state_origin(
        &self,
    ) -> Result<
        ReplicatedTextControlOrigin,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.control_state_origin_fixed()
            .map_err(PreparedControlBindingError::into_legacy)
    }

    /// Same resolved-boundary loan without allocation or diagnostic formatting.
    /// The shared owner changes only when parameter publication invalidates
    /// snapshots; this does not copy a state or issue execution authority.
    pub fn control_state_origin_fixed(
        &self,
    ) -> Result<ReplicatedTextControlOrigin, PreparedControlBindingError> {
        self.inspect_runtime_execution_fixed(|_, _, _| Ok::<(), std::convert::Infallible>(()))
            .map(|_| ())?;
        Ok(ReplicatedTextControlOrigin {
            owner: Arc::clone(&self.control_identity),
            captured: self
                .state
                .inference_retention()
                .established_revision()
                .map(|revision| CapturedControlRevision {
                    execution: self.prefill_identity.clone(),
                    revision: revision.clone(),
                }),
        })
    }

    /// Cold exact-origin validation before independently admitted construction.
    /// Revalidate when binding the completed state: parameter publication can
    /// invalidate this origin between inspection and construction.
    pub fn validate_control_state_origin(
        &self,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.validate_control_state_origin_fixed(origin)
            .map_err(PreparedControlBindingError::into_legacy)
    }

    /// Same exact origin and resolved-boundary check without allocating an error.
    /// No state is inspected or copied and no authority is issued.
    pub fn validate_control_state_origin_fixed(
        &self,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<(), PreparedControlBindingError> {
        self.inspect_runtime_execution_fixed(|_, _, _| Ok::<(), std::convert::Infallible>(()))
            .map(|_| ())?;
        if !Arc::ptr_eq(&self.control_identity, &origin.owner) {
            return Err(PreparedControlBindingError::ForeignOrigin);
        }
        Ok(())
    }

    /// Wraps an already constructed state for a later ordinary control exchange.
    /// Origin and selected geometry are checked again before moving the exact
    /// state and shared prompt identity into the slot. No state/table copy,
    /// retention inheritance, native work, agreement, exchange or admission is
    /// performed. In particular, the state's existing fresh retention is kept.
    ///
    /// The caller owns construction funding, complete source binding and native
    /// recovery. Inputs are consumed on error; keep any required recovery and
    /// host custody outside this call until their destruction is safe. A valid
    /// origin alone cannot certify that arbitrary supplied state came from it.
    pub fn bind_prepared_control_state(
        &self,
        origin: &ReplicatedTextControlOrigin,
        state: M::State,
        prompt_input_identity: Option<SharedPreparedInputCacheIdentity>,
    ) -> Result<
        ReplicatedTextControlState<M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.bind_prepared_control_state_fixed(origin, state, prompt_input_identity)
            .map_err(PreparedControlBindingError::into_legacy)
    }

    /// Same binding worker with fixed errors. This performs no allocation,
    /// native work, identity creation, admission, exchange, or completion.
    /// The caller keeps the construction account through consumed error inputs.
    pub fn bind_prepared_control_state_fixed(
        &self,
        origin: &ReplicatedTextControlOrigin,
        state: M::State,
        prompt_input_identity: Option<SharedPreparedInputCacheIdentity>,
    ) -> Result<ReplicatedTextControlState<M::State>, PreparedControlBindingError> {
        self.validate_control_state_origin_fixed(origin)?;
        let selected = self
            .selected_state
            .state()
            .ok_or(PreparedControlBindingError::MissingSelectedState)?;
        if !realized_state_layout_matches::<B, _>(&state, selected) {
            return Err(PreparedControlBindingError::StateLayout);
        }
        Ok(ReplicatedTextControlState {
            owner: Arc::clone(&origin.owner),
            state,
            prompt_input_identity,
            captured: origin.captured.clone(),
        })
    }

    /// Estimates an independent copy of the currently installed state, without
    /// allocating native storage, submitting work, evaluating or resetting state.
    pub fn estimate_control_state(&self) -> Option<SnapshotEstimate> {
        self.estimate_control_state_parts(&self.state, self.committed_prompt_input_identity())
    }

    /// Logical copy cost without host/native allocation or diagnostic creation.
    /// The mechanism must implement its original query explicitly. The enclosing
    /// driver contributes the same retained prompt identity and control header.
    pub fn estimate_original_control_state(&self) -> Option<SnapshotEstimate> {
        self.with_control_metadata(
            self.mechanisms
                .estimate_original_snapshot_state(&self.state)?,
            self.committed_prompt_input_identity(),
        )
    }

    /// Prices a provisional empty state for an enclosing parameter transaction.
    /// The installed state is retained by exchange, without copying its tensors.
    pub fn estimate_parameter_reset_state(&self) -> Option<SnapshotEstimate> {
        let native = self.mechanisms.estimate_reset_state(&self.state)?;
        let metadata = std::mem::size_of::<ReplicatedTextControlState<()>>() as u64;
        Some(SnapshotEstimate {
            retained_bytes: native.retained_bytes.checked_add(metadata)?,
            copy_bytes: native.copy_bytes.checked_add(metadata)?,
        })
    }

    /// Prepares an empty state without changing installed state or doing a
    /// collective exchange. The enclosing parameter coordinator owns admission,
    /// preparation agreement, publication, rollback and terminal fencing.
    /// Call only after reserving `estimate_parameter_reset_state`.
    pub fn prepare_parameter_reset_state(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        ReplicatedTextControlState<M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        let selected = self.selected_state.state().ok_or_else(|| {
            ReplicatedTextSessionError::Contract(
                "parameter state reset requires selected state".into(),
            )
        })?;
        let state = self
            .mechanisms
            .realize_state(selected, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        self.validate_control_geometry(&state)?;
        Ok(ReplicatedTextControlState {
            owner: Arc::clone(&self.control_identity),
            state,
            prompt_input_identity: None,
            captured: None,
        })
    }

    /// Exchanges complete state under an enclosing parameter transaction, with
    /// no nested collective or tensor copy. A second exchange restores the exact
    /// original cache and prompt identity. Commit epochs are never rewound.
    /// Invalidate snapshots only after every peer confirms publication.
    pub fn exchange_parameter_reset_state(
        &mut self,
        slot: &mut ReplicatedTextControlState<M::State>,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.validate_control_state(slot)?;
        crate::working_memory::exchange_inference_state(&mut self.state, &mut slot.state);
        slot.captured = None;
        std::mem::swap(
            &mut self.committed_prompt_input_identity,
            &mut slot.prompt_input_identity,
        );
        Ok(())
    }

    /// Estimates another independent copy of an existing compatible slot.
    pub fn estimate_control_state_copy(
        &self,
        saved: &ReplicatedTextControlState<M::State>,
    ) -> Result<
        Option<SnapshotEstimate>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.validate_control_state(saved)?;
        Ok(self.estimate_control_state_parts(
            &saved.state,
            saved.prompt_input_identity.as_ref().map(AsRef::as_ref),
        ))
    }

    /// Estimates future storage from a compatible saved slot. Architecture
    /// geometry remains in the typed state; native mechanisms price its storage.
    pub fn estimate_control_state_growth(
        &self,
        saved: &ReplicatedTextControlState<M::State>,
        additional_input_tokens: u64,
    ) -> Result<Option<u64>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.validate_control_state(saved)?;
        Ok(self
            .mechanisms
            .estimate_snapshot_growth(&saved.state, additional_input_tokens))
    }

    fn estimate_control_state_parts(
        &self,
        state: &M::State,
        input: Option<&PreparedInputCacheIdentity>,
    ) -> Option<SnapshotEstimate> {
        self.with_control_metadata(self.mechanisms.estimate_snapshot_state(state)?, input)
    }

    fn with_control_metadata(
        &self,
        native: SnapshotEstimate,
        input: Option<&PreparedInputCacheIdentity>,
    ) -> Option<SnapshotEstimate> {
        ReplicatedTextControlState::logical_snapshot_estimate(native, input)
    }

    /// Copies the installed state after the caller reserves its estimated costs.
    /// Native completion must be established before exposing the returned slot.
    pub fn capture_control_state(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        ReplicatedTextControlState<M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        let provisional = self
            .mechanisms
            .copy_snapshot_state(&self.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)
            .and_then(|state| {
                self.validate_control_geometry(&state)?;
                Ok(state)
            });
        let state = self.agree_control_preparation(
            provisional,
            crate::DistributedExecutionPhase::ControlCapturePreparation,
            context,
        )?;
        let mut state = state;
        state.inherit_inference_retention(&self.state);
        Ok(ReplicatedTextControlState {
            owner: Arc::clone(&self.control_identity),
            state,
            prompt_input_identity: self.committed_prompt_input_identity.clone(),
            captured: self
                .state
                .inference_retention()
                .established_revision()
                .map(|revision| CapturedControlRevision {
                    execution: self.prefill_identity.clone(),
                    revision: revision.clone(),
                }),
        })
    }

    /// Makes a reusable snapshot or child state from an existing saved slot,
    /// without replaying input, loading weights or changing the installed state.
    pub fn copy_control_state(
        &mut self,
        saved: &ReplicatedTextControlState<M::State>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        ReplicatedTextControlState<M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let provisional = self.validate_control_state(saved).and_then(|()| {
            let state = self
                .mechanisms
                .copy_snapshot_state(&saved.state, context)
                .map_err(ReplicatedTextSessionError::Mechanism)?;
            self.validate_control_geometry(&state)?;
            Ok(state)
        });
        let state = self.agree_control_preparation(
            provisional,
            crate::DistributedExecutionPhase::ControlCopyPreparation,
            context,
        )?;
        let mut state = state;
        state.inherit_inference_retention(&saved.state);
        Ok(ReplicatedTextControlState {
            owner: Arc::clone(&self.control_identity),
            state,
            prompt_input_identity: saved.prompt_input_identity.clone(),
            captured: saved.captured.clone(),
        })
    }

    /// Distributed callers advance the same operation and branch on every rank.
    /// All provisional copies finish before publication; failure leaves installed
    /// state untouched and fences the distributed session through its shared owner.
    fn agree_control_preparation<T>(
        &mut self,
        provisional: Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        phase: crate::DistributedExecutionPhase,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        self.require_cache_control_agreement()?;
        self.agree_control_result(provisional, phase, context, widen_infallible, |phase| {
            ReplicatedTextSessionError::Contract(format!(
                "another rank failed distributed cache control at {phase:?}"
            ))
        })
    }

    // Both diagnostic policies use the same vote, fence, and error precedence.
    fn agree_control_result<T, E>(
        &mut self,
        provisional: Result<T, E>,
        phase: crate::DistributedExecutionPhase,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        native_error: impl FnOnce(
            ReplicatedTextSessionError<A::Error, M::PolicyError, std::convert::Infallible>,
        ) -> E,
        remote_error: impl FnOnce(crate::DistributedExecutionPhase) -> E,
    ) -> Result<T, E> {
        if !D::PARTITIONED_SESSION {
            return provisional;
        }
        let agreement = self
            .agree_execution_phase(phase, provisional.is_ok(), context)
            .inspect_err(|_| {
                self.control_fence.get_or_insert(phase);
            });
        match (provisional, agreement) {
            (Ok(value), Ok(true)) => Ok(value),
            (Ok(_), Ok(false)) => {
                self.control_fence.get_or_insert(phase);
                Err(remote_error(phase))
            }
            (Ok(_), Err(error)) => Err(native_error(error)),
            (Err(error), _) => {
                self.control_fence = Some(phase);
                Err(error)
            }
        }
    }

    fn validate_control_geometry(
        &self,
        state: &M::State,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        match self.selected_state.state() {
            Some(selected) => validate_realized_state(state, selected),
            None => Err(ReplicatedTextSessionError::Contract(
                "ordinary control requires a selected stateful text execution".into(),
            )),
        }
    }

    /// Exact executable identity and geometry checks performed before mutation.
    pub fn validate_control_state(
        &self,
        saved: &ReplicatedTextControlState<M::State>,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.validate_control_state_fixed(saved)
            .map_err(PreparedControlBindingError::into_legacy)
    }

    /// Exact ordinary validation with fixed local diagnostics. This preserves
    /// the resolved-boundary, executable-owner, then selected-layout order.
    pub fn validate_control_state_fixed(
        &self,
        saved: &ReplicatedTextControlState<M::State>,
    ) -> Result<(), PreparedControlBindingError> {
        RuntimeInspectionBoundary::resolved(self.control_fence, self.last_commit_outcome)?;
        if !Arc::ptr_eq(&self.control_identity, &saved.owner) {
            return Err(PreparedControlBindingError::ForeignOrigin);
        }
        let selected = self
            .selected_state
            .state()
            .ok_or(PreparedControlBindingError::MissingSelectedState)?;
        if !realized_state_layout_matches::<B, _>(&saved.state, selected) {
            return Err(PreparedControlBindingError::StateLayout);
        }
        Ok(())
    }

    /// The same exchange and distributed vote with fixed local failures. State
    /// is moved only after the shared preparation phase accepts every rank.
    /// This is not a completion, allocation, or execution authority.
    pub fn exchange_control_state_fixed(
        &mut self,
        slot: &mut ReplicatedTextControlState<M::State>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), PreparedControlExchangeError<A::Error, M::PolicyError>> {
        self.exchange_control_state_prepared_fixed(slot, context, None, None, None)
            .map(drop)
    }

    /// The same validated exchange, with the actual constructor account and an
    /// optional captured-media source. The new revision is funded before the
    /// existing vote/state move; only a successful exchange returns a receipt.
    pub fn exchange_control_state_prepared_fixed(
        &mut self,
        slot: &mut ReplicatedTextControlState<M::State>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        metadata: Option<&eredu_nn::workspace::HostMetadataFunding>,
        media: Option<&crate::working_memory::MediaSessionBinding>,
        branch: Option<(&crate::working_memory::PendingTextBranchExchange, &[ControlBranchSource; 2])>,
    ) -> Result<
        ControlExchangeResult,
        PreparedControlExchangeError<A::Error, M::PolicyError>,
    > {
        let validation = self
            .validate_control_state_fixed(slot)
            .map_err(PreparedControlExchangeError::Binding)
            .and_then(|()| {
                if let Some(metadata) = metadata {
                    metadata
                        .reserve_metadata(std::mem::size_of::<(
                            [Option<crate::working_memory::InferenceStateRevision>; 2],
                            Option<ControlBranchSource>,
                            Option<ControlBranchPlacement>,
                            Option<[ControlBranchPlacement; 2]>,
                            ControlExchangeResult,
                            Option<crate::working_memory::CopiedMediaStateBinding>,
                            Result<
                                ControlExchangeResult,
                                PreparedControlExchangeError<A::Error, M::PolicyError>,
                            >,
                            Option<&crate::working_memory::MediaSessionBinding>,
                        )>())
                        .map_err(PreparedControlExchangeError::Metadata)?;
                }
                if let Some((pending, sources)) = branch {
                    if metadata.is_none() || media.is_some() { return Err(crate::working_memory::WorkingMemoryError::IdentityMismatch.into()); }
                    self.validate_branch_sources(slot, pending, sources)?;
                }
                if let Some(source) = media {
                    if metadata.is_none()
                        || !Arc::ptr_eq(&self.control_identity, &source.control)
                        || !slot.captured.as_ref().is_some_and(|captured| {
                            source.matches_snapshot(
                                &self.prefill_identity,
                                &captured.revision,
                                source.frontier,
                            ) && source.matches_snapshot(
                                &captured.execution,
                                &captured.revision,
                                source.frontier,
                            )
                        })
                        || self
                            .mechanisms
                            .original_prefill_state_frontier(&slot.state)?
                            != Some(source.frontier)
                    {
                        return Err(
                            crate::working_memory::WorkingMemoryError::IdentityMismatch.into()
                        );
                    }
                }
                let installed = metadata.map(crate::working_memory::InferenceStateRevision::prepare_metadata)
                    .transpose().map_err(PreparedControlExchangeError::Metadata)?;
                let displaced = if metadata.is_some() {
                    Some(crate::working_memory::InferenceStateRevision::prepare_metadata(metadata.expect("validated original funding"))
                        .map_err(PreparedControlExchangeError::Metadata)?)
                } else { None };
                let source = if metadata.is_some() && branch.is_none() { Some(self.control_branch_source(&self.state)?) } else { None };
                Ok((installed, displaced, source))
            });
        RuntimeInspectionBoundary::resolved(self.control_fence, self.last_commit_outcome)
            .map_err(PreparedControlBindingError::from)?;
        if D::PARTITIONED_SESSION && !D::DISTRIBUTED_PHASE_AGREEMENT {
            return Err(PreparedControlExchangeError::MissingAgreement);
        }
        let (revision, displaced, displaced_source) = self.agree_control_result(
            validation,
            crate::DistributedExecutionPhase::ControlExchangePreparation,
            context,
            PreparedControlExchangeError::Agreement,
            |phase| PreparedControlExchangeError::Remote { phase },
        )?;
        let placements = branch.map(|(_, sources)| branch::placements(sources,
            displaced.as_ref().expect("branch displaced revision").clone(),
            revision.as_ref().expect("branch installed revision").clone()));
        let displaced_placement = displaced_source.map(|source| branch::displaced(source,
            displaced.as_ref().expect("funded displaced revision").clone()));
        self.exchange_control_payload(slot);
        if let Some(revision) = displaced { slot.state.inference_retention_mut().install_exchanged_revision(revision); }
        if let Some(revision) = revision {
            self.state
                .inference_retention_mut()
                .install_exchanged_revision(revision);
        }
        Ok(ControlExchangeResult { placements, displaced: displaced_placement, media: media.map(|source| {
            crate::working_memory::CopiedMediaStateBinding::new(
                source,
                crate::working_memory::MediaSessionBinding {
                    execution: self.prefill_identity.clone(),
                    revision: self.state.inference_retention().revision().clone(),
                    control: Arc::clone(&self.control_identity),
                    // Checked on the exact slot before the infallible swap above.
                    frontier: source.frontier,
                },
            )
        }) })
    }

    /// Atomically exchanges complete state at an already completed boundary.
    /// The old installed state is returned in `slot`; no array data is copied.
    /// This supports serial branch switching under one native completion owner.
    pub fn exchange_control_state(
        &mut self,
        slot: &mut ReplicatedTextControlState<M::State>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let validation = self.validate_control_state(slot);
        self.agree_control_preparation(
            validation,
            crate::DistributedExecutionPhase::ControlExchangePreparation,
            context,
        )?;
        self.exchange_control_payload(slot);
        Ok(())
    }

    fn exchange_control_payload(&mut self, slot: &mut ReplicatedTextControlState<M::State>) {
        crate::working_memory::exchange_inference_state(&mut self.state, &mut slot.state);
        slot.captured = None;
        std::mem::swap(
            &mut self.committed_prompt_input_identity,
            &mut slot.prompt_input_identity,
        );
        // Commit epochs and outcomes belong to the live submission owner. A
        // branch switch must never reuse a completed transaction identity.
    }
}
