//! Exact selected Embedded request and monotonic invocation custody.
use super::{OriginalSpeculativeNumericalSources, OriginalSpeculativeNumericalPreparation};
use crate::{
    backend::error::Error,
    composition::mlx::{model::retain_planning_error, model::Executable},
};
use eredu_core::speculative::{
    SpeculativeActivationOrigin, SpeculativeActivationPhase, SpeculativePrefillSpan,
};
use eredu_core::SpeculativeRequestId;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{
    speculative::embedded_occurrence::{
        EmbeddedOccurrenceClaim, EmbeddedOccurrenceCursor, EmbeddedSchedulePlan,
    },
    working_memory::{OriginalSpeculativeRequest, WorkingMemoryError, WorkingMemoryPool},
};
use std::{
    cell::{Cell, RefCell},
    mem::{size_of, size_of_val},
};

/// This owner stays outside model/cache snapshots. Every failed or discarded
/// attempt remains spent; source projection and native completion are separate.
pub(crate) struct OriginalEmbeddedSources<'a> {
    cursor: RefCell<EmbeddedOccurrenceCursor<'a>>,
    scheduler_request: Cell<Option<SpeculativeRequestId>>,
    numerical: OriginalSpeculativeNumericalSources,
}
impl std::fmt::Debug for OriginalEmbeddedSources<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalEmbeddedSources")
    }
}
impl<'a> OriginalEmbeddedSources<'a> {
    pub(crate) fn prepare(
        target: &Executable,
        schedule: EmbeddedSchedulePlan<'a>,
        pool: &WorkingMemoryPool,
        capacity: u64,
        funding: HostMetadataFunding,
    ) -> Result<Self, Error> {
        let source = OriginalSpeculativeNumericalPreparation::prepare(target, pool, funding)?;
        Self::prepare_from_source(source, schedule, capacity)
    }

    /// The selected schedule is borrowed only after the executable's immutable
    /// source has been captured. Request issuance remains owned by this cursor.
    pub(crate) fn prepare_from_source(
        source: OriginalSpeculativeNumericalPreparation,
        schedule: EmbeddedSchedulePlan<'a>,
        capacity: u64,
    ) -> Result<Self, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<EmbeddedSchedulePlan<'a>>(),
            size_of::<(
                OriginalSpeculativeNumericalPreparation,
                u64,
            )>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        source.metadata_funding()
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let request = OriginalSpeculativeRequest::prepare_embedded(
            source.pool(),
            source.execution_identity(),
            &schedule,
            capacity,
        )
        .map_err(|cause| retain_planning_error(cause, source.metadata_funding().clone()))?;
        let numerical = source.bind(request)?;
        Ok(Self {
            cursor: RefCell::new(schedule.into_cursor()),
            scheduler_request: Cell::new(None),
            numerical,
        })
    }

    pub(crate) fn with_intervention_declaration(
        mut self,
        declaration: Option<crate::composition::mlx::session::OriginalInterventionDeclaration>,
    ) -> Result<Self, Error> {
        self.numerical = self.numerical.with_intervention_declaration(declaration)?;
        Ok(self)
    }

    pub(crate) fn numerical_sources(&self) -> &OriginalSpeculativeNumericalSources {
        &self.numerical
    }

    /// Extend the same request's finite future without resetting attempted work.
    /// The cursor stays exclusively borrowed until paid slot replacement and
    /// exact limit installation both complete; snapshots never own this cursor.
    pub(crate) fn prepare_continuation(
        &self, committed: usize, status: eredu_core::generation::SpeculativeRequestStatus,
    ) -> Result<(), Error> {
        use eredu_runtime::speculative::embedded_occurrence::{EmbeddedContinuation, EmbeddedOccurrenceError};
        let funding=self.numerical.metadata_funding();
        let parts=[size_of::<(&Self,usize,eredu_core::generation::SpeculativeRequestStatus)>(),
            size_of::<EmbeddedContinuation>(),size_of::<Result<EmbeddedContinuation,EmbeddedOccurrenceError>>(),
            size_of::<std::cell::RefMut<'_,EmbeddedOccurrenceCursor<'_>>>(),
            size_of::<Result<(),Error>>(),size_of::<Result<(),EmbeddedOccurrenceError>>(),
            size_of::<Result<(),eredu_runtime::working_memory::SpeculativeContinuationError>>(),
            size_of::<super::SpeculativeExecutionStreams<'_>>(),
            size_of::<Result<(),eredu_core::speculative::SpeculativeControlError>>()];
        funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let mut cursor=self.cursor.try_borrow_mut().map_err(|_|self.numerical.retain_startup_error(
            WorkingMemoryError::AccountConstructionBusy))?;
        let continuation=cursor.continuation(committed,status)
            .map_err(|cause|self.numerical.retain_startup_error(cause))?;
        self.numerical.request().prepare_embedded_continuation(&continuation,funding)
            .map_err(|cause|self.numerical.retain_startup_error(cause))?;
        cursor.install_continuation(continuation)
            .map_err(|cause|self.numerical.retain_startup_error(cause))
    }

    /// Claims once before source-dependent quoting/admission. Exact request
    /// geometry is normalized by runtime; native code does no family dispatch.
    pub(crate) fn claim(
        &self,
        phase: SpeculativeActivationPhase,
        positions: usize,
        span: Option<SpeculativePrefillSpan>,
        origin: SpeculativeActivationOrigin,
    ) -> Result<EmbeddedOccurrenceClaim<'a>, Error> {
        let parts = [
            size_of::<EmbeddedOccurrenceClaim<'a>>(),
            size_of::<Result<EmbeddedOccurrenceClaim<'a>, Error>>(),
            size_of::<std::cell::RefMut<'_, EmbeddedOccurrenceCursor<'a>>>(),
            size_of::<(
                SpeculativeActivationPhase,
                usize,
                Option<SpeculativePrefillSpan>,
                SpeculativeActivationOrigin,
            )>(),
        ];
        self.numerical
            .metadata_funding()
            .reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        HostMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if self
            .scheduler_request
            .get()
            .is_some_and(|previous| previous != origin.request)
        {
            return Err(self
                .numerical
                .retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        let mut cursor = self.cursor.try_borrow_mut().map_err(|_| {
            self.numerical
                .retain_startup_error(WorkingMemoryError::AccountConstructionBusy)
        })?;
        let invocation = cursor
            .plan()
            .scheduler_invocation(phase, positions, span, origin)
            .map_err(|cause| self.numerical.retain_startup_error(cause))?;
        self.scheduler_request.set(Some(origin.request));
        cursor
            .claim(invocation)
            .map_err(|cause| self.numerical.retain_startup_error(cause))
    }
}


/// Borrowed request access erases only the cursor's selected-source lifetime.
/// The concrete owner keeps all issuance and source state; this interface lends
/// claims, never manufactured model or native identities.
pub(crate) trait EmbeddedInvocationSource: std::fmt::Debug {
    fn numerical_sources(&self) -> &OriginalSpeculativeNumericalSources;
    fn prepare_continuation(&self, committed:usize,
        status:eredu_core::generation::SpeculativeRequestStatus)->Result<(),Error>;
    fn claim(
        &self,
        phase: SpeculativeActivationPhase,
        positions: usize,
        span: Option<SpeculativePrefillSpan>,
        origin: SpeculativeActivationOrigin,
    ) -> Result<EmbeddedOccurrenceClaim<'_>, Error>;
}
impl EmbeddedInvocationSource for OriginalEmbeddedSources<'_> {
    fn prepare_continuation(&self,committed:usize,
        status:eredu_core::generation::SpeculativeRequestStatus)->Result<(),Error> {
        self.prepare_continuation(committed,status)
    }
    fn numerical_sources(&self) -> &OriginalSpeculativeNumericalSources {
        self.numerical_sources()
    }
    fn claim(
        &self,
        phase: SpeculativeActivationPhase,
        positions: usize,
        span: Option<SpeculativePrefillSpan>,
        origin: SpeculativeActivationOrigin,
    ) -> Result<EmbeddedOccurrenceClaim<'_>, Error> {
        self.claim(phase, positions, span, origin)
    }
}

/// The actual scoped scalar producer, lent only for one invocation. It validates
/// the prepared token source and native scope when used by the shared equation.
pub(crate) trait EmbeddedNumericalInvocation: std::fmt::Debug {
    fn sources(&self) -> &OriginalSpeculativeNumericalSources;
    fn validate_scope(&self, stream:&safemlx::Stream)->Result<(), Error>;
    fn token(&self, token:u32, stream:&safemlx::Stream)
        ->Result<crate::MlxTensor, Error>;
    fn complete_state<'values>(
        &self,
        _point: eredu_architectures::speculative_execution::PredictionCompletionPoint,
        _values: &mut dyn Iterator<Item=&'values crate::MlxTensor>,
        _stream: &safemlx::Stream,
    ) -> Result<(),Error> {
        Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound))
    }
    /// The actual quoted immediate row, retaining its own pending destination.
    /// A scalar-only binding cannot authorize a logit row as a side effect.
    fn logits_row(
        &self, _value: &safemlx::Array, _row: usize, _stream: &safemlx::Stream,
    ) -> Result<super::sampling::logits::IndependentLogits, Error> {
        Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound))
    }

}
