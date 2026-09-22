//! The ordinary payload-free I32 vote using retained original source workers.
use super::operation_storage::OriginalCommunicationCompletedOperation;
use super::*;
use crate::backend::runtime::distributed::completion::{
    MlxNeuralCommunicationCompletion, OriginalCommunicationBool, PreparedCommunicationScalar,
    prepared::ReadyCompletionResources,
};
use safemlx::{OriginalScopeObserver, PreparedInputRuntime, distributed::GroupWorkerOperation};
use std::mem::size_of_val;
mod chain;
mod inputs;
pub(super) mod quote;
pub(crate) use inputs::OriginalAgreementInputs;

/// Finite native operation, scalar destination and existing completion prepared
/// from the actual selected status leaf. No Graph or Record grant is issued here.
struct PreparedNativeAgreement<'a> {
    operation: OriginalCommunicationCompletedOperation<'a>,
    scalar: PreparedCommunicationScalar,
    ready: ReadyCompletionResources,
    runtime: &'a PreparedInputRuntime,
    backing: usize,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
pub(crate) struct PreparedOriginalAgreement<'a> {
    kind: PreparedAgreementKind<'a>,
}
enum PreparedAgreementKind<'a> {
    Native(PreparedNativeAgreement<'a>),
    Chain(chain::PreparedStatusAgreement<'a>),
}
impl PreparedOriginalAgreement<'_> {
    pub(crate) fn graph_capacity(&self) -> usize {
        match &self.kind {
            PreparedAgreementKind::Native(value) => value.graph_capacity(),
            PreparedAgreementKind::Chain(value) => value.capacity().graph,
        }
    }
    pub(crate) fn record_capacity(&self) -> usize {
        match &self.kind {
            PreparedAgreementKind::Native(value) => value.record_capacity(),
            PreparedAgreementKind::Chain(value) => value.capacity().records,
        }
    }
    pub(crate) fn backing_capacity(&self) -> usize {
        match &self.kind {
            PreparedAgreementKind::Native(value) => value.backing_capacity(),
            PreparedAgreementKind::Chain(value) => value.capacity().backing,
        }
    }
    pub(crate) fn runtime(&self) -> &PreparedInputRuntime {
        match &self.kind {
            PreparedAgreementKind::Native(value) => value.runtime(),
            PreparedAgreementKind::Chain(value) => value.runtime(),
        }
    }
    pub(crate) fn submit(
        self,
        source: &OriginalCommunicationSource<'_>,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<(OriginalCommunicationBool, MlxNeuralCommunicationCompletion), Error> {
        match self.kind {
            PreparedAgreementKind::Native(value) => value.submit(source, observer, stream),
            PreparedAgreementKind::Chain(value) => value.submit(source, observer, stream),
        }
    }
}
/// Maximum of the two actual immutable protocol branches, for one vote.
/// This grants neither an occurrence nor a native allocation domain.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AgreementCapacity {
    pub(crate) graph: usize,
    pub(crate) records: usize,
    pub(crate) backing: usize,
}
impl AgreementCapacity {
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            graph: self.graph.max(other.graph),
            records: self.records.max(other.records),
            backing: self.backing.max(other.backing),
        }
    }
    pub(crate) fn covers(self, actual: Self) -> bool {
        actual.graph <= self.graph
            && actual.records <= self.records
            && actual.backing <= self.backing
    }
}
impl OriginalAgreementInputs {
    pub(crate) fn prepare<'a>(
        &'a self,
        source: &'a OriginalCommunicationSource<'_>,
        success: bool,
    ) -> Result<PreparedOriginalAgreement<'a>, Error> {
        self.prepare_for_group(source, self.group(), success)
    }
    pub(crate) fn prepare_for_group<'a>(
        &'a self,
        source: &'a OriginalCommunicationSource<'_>,
        group: CollectiveGroupId,
        success: bool,
    ) -> Result<PreparedOriginalAgreement<'a>, Error> {
        source
            .funding()
            .reserve_metadata(Self::preparation_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if !self.same_source(source) {
            return Err(failure(Cause::Identity, source.source(), source.funding()));
        }
        source.validate()?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(group, CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        if !selected.requirement().exact_completion() {
            return Err(failure(Cause::Resource, source.source(), source.funding()));
        }
        if let Some(value) =
            chain::PreparedStatusAgreement::prepare(self, source, selected.order(), success)?
        {
            return Ok(PreparedOriginalAgreement {
                kind: PreparedAgreementKind::Chain(value),
            });
        }
        let (operation, backing) = self.prepare_operation(source, group, success)?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(group, CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        let ready = operation.prepare_resources(source, Some(selected.order()))?;
        let scalar = PreparedCommunicationScalar::prepare_agreement(source, selected.order())?;
        Ok(PreparedOriginalAgreement {
            kind: PreparedAgreementKind::Native(PreparedNativeAgreement {
                operation,
                scalar,
                ready,
                runtime: self.runtime(),
                backing,
                source: source.source().clone(),
                funding: source.funding().clone(),
            }),
        })
    }
}
impl OriginalAgreementInputs {
    fn prepare_operation<'a>(
        &'a self,
        source: &'a OriginalCommunicationSource<'_>,
        group: CollectiveGroupId,
        success: bool,
    ) -> Result<(OriginalCommunicationCompletedOperation<'a>, usize), Error> {
        source
            .funding()
            .reserve_metadata(Self::operation_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if !self.same_source(source) {
            return Err(failure(Cause::Identity, source.source(), source.funding()));
        }
        source.validate()?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(group, CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        if !selected.requirement().exact_completion() {
            return Err(failure(Cause::Resource, source.source(), source.funding()));
        }
        // The protocol itself creates a single private I32 status word. The
        // payload-free declaration grants no logical tensor or dtype allowance.
        let input = self.value(success);
        let operation = source.group_cpu_operation_storage(
            selected.order(),
            input,
            GroupWorkerOperation::Sum,
        )?;
        if operation.native().constructor().output_geometry() != (1, 1) {
            return Err(failure(Cause::Output, source.source(), source.funding()));
        }
        let backing = operation.backing_storage(self.runtime())?.capacity();
        let operation = operation.with_completion()?;
        Ok((operation, backing))
    }
    pub(crate) fn capacity(
        &self,
        source: &OriginalCommunicationSource<'_>,
        success: bool,
    ) -> Result<AgreementCapacity, Error> {
        self.capacity_for_group(source, self.group(), success)
    }
    pub(crate) fn capacity_for_group(
        &self,
        source: &OriginalCommunicationSource<'_>,
        group: CollectiveGroupId,
        success: bool,
    ) -> Result<AgreementCapacity, Error> {
        source
            .funding()
            .reserve_metadata(Self::capacity_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if !self.same_source(source) {
            return Err(failure(Cause::Identity, source.source(), source.funding()));
        }
        source.validate()?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(group, CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        if !selected.requirement().exact_completion() {
            return Err(failure(Cause::Resource, source.source(), source.funding()));
        }
        if let Some(value) =
            chain::PreparedStatusAgreement::prepare(self, source, selected.order(), success)?
        {
            return Ok(value.capacity());
        }
        let (operation, backing) = self.prepare_operation(source, group, success)?;
        Ok(AgreementCapacity {
            graph: operation.graph_capacity(),
            records: operation.record_capacity(),
            backing,
        })
    }
}
impl PreparedNativeAgreement<'_> {
    pub(crate) fn graph_capacity(&self) -> usize {
        self.operation.graph_capacity()
    }
    pub(crate) fn record_capacity(&self) -> usize {
        self.operation.record_capacity()
    }
    pub(crate) fn backing_capacity(&self) -> usize {
        self.backing
    }
    pub(crate) fn runtime(&self) -> &PreparedInputRuntime {
        self.runtime
    }
    /// The enclosing phase supplies its actual role and native stream. The
    /// original accepted handoff is never re-admitted after construction.
    pub(crate) fn submit(
        self,
        source: &OriginalCommunicationSource<'_>,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<(OriginalCommunicationBool, MlxNeuralCommunicationCompletion), Error> {
        self.funding
            .reserve_metadata(Self::submit_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if !self.source.same_source(source.source()) {
            return Err(failure(Cause::Identity, &self.source, &self.funding));
        }
        let accepted = self
            .operation
            .construct_accepted(source, observer, stream)?;
        let (result, completion) = self.scalar.submit_accepted(accepted, self.ready)?;
        Ok((result, completion.into()))
    }
}
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
fn reserve(funding: &HostMetadataFunding, bytes: &[usize]) -> Result<(), Error> {
    funding
        .reserve_metadata(
            bytes
                .iter()
                .copied()
                .try_fold(std::mem::size_of_val(bytes), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)
}

impl OriginalAgreementInputs {
    pub(super) fn preparation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<PreparedOriginalAgreement<'_>>(),
            size_of::<Result<PreparedOriginalAgreement<'_>, Error>>(),
            size_of::<(
                &Self,
                &OriginalCommunicationSource<'_>,
                CollectiveGroupId,
                bool,
            )>(),
            CommunicationManifest::group_operation_control_bytes()?,
            failure_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl OriginalAgreementInputs {
    pub(super) fn operation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<(
                &Self,
                &OriginalCommunicationSource<'_>,
                CollectiveGroupId,
                bool,
            )>(),
            size_of::<Result<(OriginalCommunicationCompletedOperation<'_>, usize), Error>>(),
            CommunicationManifest::group_operation_control_bytes()?,
            failure_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl OriginalAgreementInputs {
    pub(super) fn capacity_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<AgreementCapacity>(),
            size_of::<Result<AgreementCapacity, Error>>(),
            size_of::<(
                &Self,
                &OriginalCommunicationSource<'_>,
                CollectiveGroupId,
                bool,
            )>(),
            size_of::<(OriginalCommunicationCompletedOperation<'_>, usize)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl PreparedNativeAgreement<'_> {
    pub(super) fn submit_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<(
                &OriginalCommunicationSource<'_>,
                &OriginalScopeObserver,
                &Stream,
            )>(),
            size_of::<Result<(OriginalCommunicationBool, MlxNeuralCommunicationCompletion), Error>>(
            ),
            failure_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

fn metadata_bytes(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(size_of_val(parts), usize::checked_add)
}

/// The exact selected vote's source query and actual execution populations.
/// Values describe one branch; neither metadata nor native capacity is granted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AgreementRequirements {
    pub(crate) capacity: AgreementCapacity,
    pub(crate) capacity_metadata: usize,
    pub(crate) execution_metadata: usize,
}
impl OriginalAgreementInputs {
    pub(crate) fn requirements_for_group(
        &self,
        source: &OriginalCommunicationSource<'_>,
        group: CollectiveGroupId,
    ) -> Result<AgreementRequirements, Error> {
        reserve(
            source.funding(),
            &[
                size_of::<AgreementRequirements>(),
                size_of::<Result<AgreementRequirements, Error>>(),
                size_of::<(&Self, &OriginalCommunicationSource<'_>, CollectiveGroupId)>(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        if !self.same_source(source) {
            return Err(failure(Cause::Identity, source.source(), source.funding()));
        }
        source.validate()?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(group, CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        if !selected.requirement().exact_completion() {
            return Err(failure(Cause::Resource, source.source(), source.funding()));
        }
        let actual = source
            .group(selected.order())
            .ok_or_else(|| failure(Cause::Identity, source.source(), source.funding()))?
            .0;
        let validation =
            OriginalCommunicationSource::validation_control_bytes().ok_or_else(overflow)?;
        if let Some(plan) = actual.selected_status_plan().map_err(|_| {
            failure(
                Cause::LogicalWorldTransport,
                source.source(),
                source.funding(),
            )
        })? {
            let mut value = self
                .status_quote(selected.order())
                .and_then(|quote| quote.requirements(actual, plan))
                .ok_or_else(|| failure(Cause::Resource, source.source(), source.funding()))?;
            value.capacity_metadata = value
                .capacity_metadata
                .checked_add(Self::capacity_control_bytes().ok_or_else(overflow)?)
                .and_then(|n| n.checked_add(validation))
                .ok_or_else(overflow)?;
            value.execution_metadata = value
                .execution_metadata
                .checked_add(Self::preparation_control_bytes().ok_or_else(overflow)?)
                .and_then(|n| n.checked_add(validation))
                .ok_or_else(overflow)?;
            return Ok(value);
        }
        let leaf = quote::LeafQuote::prepare(
            source,
            self.runtime(),
            selected.order(),
            GroupWorkerOperation::Sum,
            false,
        )?;
        let operation = Self::operation_control_bytes()
            .ok_or_else(overflow)?
            .checked_add(validation)
            .and_then(|n| n.checked_add(leaf.operation))
            .and_then(|n| n.checked_add(leaf.backing))
            .ok_or_else(overflow)?;
        let capacity_metadata = Self::capacity_control_bytes()
            .ok_or_else(overflow)?
            .checked_add(validation)
            .and_then(|n| n.checked_add(operation))
            .ok_or_else(overflow)?;
        let parts = [
            Self::preparation_control_bytes().ok_or_else(overflow)?,
            validation,
            operation,
            leaf.resources,
            PreparedCommunicationScalar::control_bytes().ok_or_else(overflow)?,
            validation,
            PreparedNativeAgreement::submit_control_bytes().ok_or_else(overflow)?,
            leaf.submit,
        ];
        let execution_metadata = parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        Ok(AgreementRequirements {
            capacity: leaf.capacity,
            capacity_metadata,
            execution_metadata,
        })
    }
}
