//! Exact model-internal callbacks retained from the shared partition traversal.
use super::super::role::{OriginalParallelControlOwner, SpeculativeModelRole};
use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use eredu_runtime::replicated_session::{
    ParallelControlCallbackVisitor, SessionModelControlPlan, SessionTransactionControlOccurrence,
};
use eredu_runtime::working_memory::{OriginalSpeculativeRequest, OriginalSpeculativeRole};
use safemlx::OriginalBufferBudget;

/// Source description and one-use activation for one recorded model span.
pub(crate) struct SpeculativeModelControlQuote {
    source: PreparedSpeculativeControl,
    plan: RefCell<Option<SessionModelControlPlan>>,
    capacity: AgreementCapacity,
    runtime: ModelRuntimeFunding,
}
pub(crate) struct ModelRuntimeFunding {
    source: PreparedSpeculativeControl,
    funding: HostMetadataFunding,
}
impl ModelRuntimeFunding {
    pub(crate) fn validate(
        &self,
        source: &PreparedSpeculativeControl,
    ) -> Result<&HostMetadataFunding, Error> {
        if !self.source.same_control_source(source) {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(&self.funding)
    }
}
impl SpeculativeModelControlQuote {
    pub(crate) fn metadata_funding(&self) -> &HostMetadataFunding {
        &self.source.request().funding
    }
    pub(crate) fn backing_bytes(&self) -> usize {
        self.capacity.backing
    }
    fn activation_frames() -> Option<usize> {
        let parts = [
            size_of::<(
                &Self,
                &OriginalSpeculativeRequest,
                &SpeculativeModelRole,
                &OriginalBufferBudget,
                &OriginalScopeObserver,
                &HostMetadataFunding,
            )>(),
            size_of::<Result<OriginalParallelControlOwner, Error>>(),
            size_of::<SpeculativeModelRole>(),
            size_of::<&ModelRuntimeFunding>(),
            size_of::<Result<&HostMetadataFunding, Error>>(),
            size_of::<Option<SessionModelControlPlan>>(),
            size_of::<
                Result<
                    std::cell::RefMut<'_, Option<SessionModelControlPlan>>,
                    std::cell::BorrowMutError,
                >,
            >(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn activate(
        &self,
        request: &OriginalSpeculativeRequest,
        role: &OriginalSpeculativeRole,
        budget: &OriginalBufferBudget,
        observer: &OriginalScopeObserver,
        funding: &HostMetadataFunding,
    ) -> Result<OriginalParallelControlOwner, Error> {
        self.activate_role(
            request,
            &SpeculativeModelRole::Autoregressive(role.clone()),
            budget,
            observer,
            funding,
        )
    }
    pub(crate) fn activate_role(
        &self,
        request: &OriginalSpeculativeRequest,
        role: &SpeculativeModelRole,
        budget: &OriginalBufferBudget,
        observer: &OriginalScopeObserver,
        funding: &HostMetadataFunding,
    ) -> Result<OriginalParallelControlOwner, Error> {
        let runtime = self.runtime.validate(&self.source)?;
        runtime.reserve_metadata(Self::activation_frames().ok_or_else(overflow)?)?;
        let plan = self
            .plan
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .take()
            .ok_or(Error::PrefillScopeUnavailable)?;
        OriginalParallelControlOwner::new_speculative_role(
            &self.source,
            request,
            role,
            budget,
            observer,
            plan,
            funding,
            &self.runtime,
        )
    }
}

pub(crate) struct SpeculativeModelControlVisitor<'a> {
    source: &'a PreparedSpeculativeControl,
    plan: SessionModelControlPlan,
    capacity: AgreementCapacity,
    metadata: usize,
    outer: usize,
    inner: usize,
}
impl<'a> SpeculativeModelControlVisitor<'a> {
    pub(crate) fn new(
        source: &'a PreparedSpeculativeControl,
        plan: SessionModelControlPlan,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        funding.reserve_metadata(size_of::<(Self, Result<Self, Error>)>())?;
        Ok(Self {
            source,
            plan,
            capacity: AgreementCapacity {
                graph: 0,
                records: 0,
                backing: 0,
            },
            metadata: 0,
            outer: 0,
            inner: 0,
        })
    }
    fn validate(
        &self,
        occurrence: SessionTransactionControlOccurrence,
        expected: usize,
    ) -> Result<eredu_core::CollectiveGroupId, Error> {
        let row = self
            .plan
            .occurrences()
            .get(expected)
            .ok_or(Error::PrefillScopeUnavailable)?;
        if occurrence.ordinal() != row.operation_ordinal()
            || occurrence.event() != row.event()
            || occurrence.operation() != CommunicationOperation::FailureAgreement
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(row.declaration().group)
    }
    pub(crate) fn finish(
        self,
        funding: &HostMetadataFunding,
    ) -> Result<SpeculativeModelControlQuote, Error> {
        funding.reserve_metadata(size_of::<(
            Self,
            SpeculativeModelControlQuote,
            ModelRuntimeFunding,
            Result<SpeculativeModelControlQuote, Error>,
            &HostMetadataFunding,
        )>())?;
        let count = self.plan.occurrences().len();
        if self.outer != count || self.inner != count {
            return Err(Error::PrefillScopeUnavailable);
        }
        let metadata = self
            .metadata
            .checked_add(
                OriginalParallelControlOwner::speculative_activation_control_bytes()
                    .ok_or_else(overflow)?,
            )
            .and_then(|n| n.checked_add(SpeculativeModelControlQuote::activation_frames()?))
            .ok_or_else(overflow)?;
        let runtime = ModelRuntimeFunding {
            source: self.source.clone(),
            funding: crate::backend::submission_recovery::addressable::prepare_wrapper_funding(
                metadata,
                &self.source.request().funding,
            )?,
        };
        Ok(SpeculativeModelControlQuote {
            source: self.source.clone(),
            plan: RefCell::new(Some(self.plan)),
            capacity: self.capacity,
            runtime,
        })
    }
}
impl ParallelControlCallbackVisitor<MlxNeuralBackend> for SpeculativeModelControlVisitor<'_> {
    type Error = Error;
    fn visit<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.validate(occurrence, self.outer)?;
        let metadata =
            OriginalParallelControlOwner::model_context_control_bytes::<T, E, F>(self.source)?;
        self.metadata = self.metadata.checked_add(metadata).ok_or_else(overflow)?;
        self.outer = self.outer.checked_add(1).ok_or_else(overflow)?;
        Ok(())
    }
    fn visit_group<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Error>
    where
        F: FnOnce(Option<&Group>) -> Result<T, E>,
    {
        let group = self.validate(occurrence, self.inner)?;
        let quote =
            OriginalParallelControlOwner::model_call_requirements::<T, E, F>(self.source, group)?;
        self.capacity = AgreementCapacity {
            graph: self
                .capacity
                .graph
                .checked_add(quote.capacity.graph)
                .ok_or_else(overflow)?,
            records: self
                .capacity
                .records
                .checked_add(quote.capacity.records)
                .ok_or_else(overflow)?,
            backing: self
                .capacity
                .backing
                .checked_add(quote.capacity.backing)
                .ok_or_else(overflow)?,
        };
        self.metadata = self
            .metadata
            .checked_add(quote.metadata)
            .ok_or_else(overflow)?;
        self.inner = self.inner.checked_add(1).ok_or_else(overflow)?;
        Ok(())
    }
}
