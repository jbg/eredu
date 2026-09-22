//! Finite host owners for the existing distributed completion/recovery worker.
//! Original submission consumes an actual enclosing role and its prepared Eval
//! traversal. Native communicator census and result readouts remain separate.
use super::destinations::Destination;
use super::*;
use crate::backend::{
    runtime::distributed::topology::OriginalCommunicationSource,
    submission_recovery::PreparedRecovery,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::RetainedCommunicationSource;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
};

/// Only immutable source/account custody may enter the native Scope owner.
#[derive(Clone, Debug)]
pub(super) struct ResourceCustody {
    pub(super) source: RetainedCommunicationSource,
    pub(super) funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error(
        "distributed completion host destination does not match its prepared population or source"
    )]
    Identity,
    #[error("distributed completion host destination capacity failed: {0}")]
    Capacity(#[source] TryReserveError),
    #[error("distributed completion host payload funding failed: {0}")]
    Funding(#[source] HostMetadataFundingError),
    #[error("distributed completion recovery preparation failed: {0}")]
    Recovery(#[source] safemlx::SubmissionScopeOwnerCause),
    #[error("distributed completion housekeeping registration failed: {0}")]
    Housekeeping(#[source] safemlx::HousekeepingRegistrationCause),
    #[error("distributed completion native event failed: {0}")]
    Native(#[source] safemlx::error::Exception),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    custody: ResourceCustody,
}
pub(super) fn error(cause: Cause, custody: &ResourceCustody) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::new(
            eredu_core::BackendFailureKind::Other,
            Failure {
                cause,
                custody: custody.clone(),
            },
        ),
        false,
    )
}
pub(super) fn error_control_bytes() -> Option<usize> {
    let controls = [
        size_of::<Failure>(),
        size_of::<Cause>(),
        size_of::<ResourceCustody>(),
        size_of::<Error>(),
        size_of::<(&ResourceCustody, Cause)>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}

/// Host populations supplied by the actual communication worker. These counts
/// grant no native allocation credit and cannot open a distributed execution gate.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompletionResourceLayout<'a> {
    pub(crate) arrays: usize,
    pub(crate) counts: &'a [usize],
    pub(crate) groups: usize,
    pub(crate) routes: usize,
    pub(crate) streams: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum CompletionResourceKind {
    Communication,
    Distributed,
}
#[derive(Debug)]
pub(super) enum OrphanDestination {
    Communication(Destination<MlxCommunicationCompletion>),
    Distributed(Destination<generic::DistributedCompletionOrphan>),
}
impl OrphanDestination {
    fn bytes(kind: CompletionResourceKind) -> Option<usize> {
        match kind {
            CompletionResourceKind::Communication => {
                Destination::<MlxCommunicationCompletion>::control_bytes()
            }
            CompletionResourceKind::Distributed => {
                Destination::<generic::DistributedCompletionOrphan>::control_bytes()
            }
        }
    }
    fn new(kind: CompletionResourceKind, custody: &ResourceCustody) -> Self {
        match kind {
            CompletionResourceKind::Communication => {
                Self::Communication(Destination::new(Some(custody.clone())))
            }
            CompletionResourceKind::Distributed => {
                Self::Distributed(Destination::new(Some(custody.clone())))
            }
        }
    }
}
/// All destinations exist before any graph or submission. Every push moves an
/// already-owned value; no native handle clone occurs in this preparation.
pub(crate) struct PreparedCompletionResources<'source, 'native> {
    source: &'source OriginalCommunicationSource<'native>,
    arrays: Vec<Array>,
    counts: Vec<Vec<usize>>,
    count_filled: Vec<bool>,
    groups: Vec<Group>,
    routes: Vec<CommunicationRouteRealization>,
    streams: Vec<Stream>,
    limits: [usize; 4],
    original: Option<safemlx::OperationEvalTraversalLayout>,
    consumers: Option<consumers::PreparedConsumers>,
    registry: Destination<std::rc::Weak<NativeResources>>,
    orphan: OrphanDestination,
    custody: ResourceCustody,
}
/// Same preallocated Recovery; still unsubmitted. The actual native producer
/// must supply its record/graph quotas and exact event before constructing a
/// public completion. No ordinary submission fallback is provided here.
pub(crate) struct ReadyCompletionResources {
    pub(super) recovery: PreparedCompletionRecovery,
    pub(super) orphan: OrphanDestination,
    consumers: Option<consumers::PreparedConsumers>,
    custody: ResourceCustody,
}
pub(super) enum PreparedCompletionRecovery {
    Owned(PreparedRecovery<NativeOwner, ResourceCustody>),
    Original {
        resources: NativeOwner,
        pending: crate::backend::submission_recovery::observed::PreparedObservedRecovery<
            NativeOwner,
            ResourceCustody,
        >,
        housekeeping: safemlx::PreparedThreadRuntimeHousekeeping<ResourceCustody>,
        traversal: safemlx::OperationEvalTraversalLayout,
    },
}
impl PreparedCompletionRecovery {
    #[cfg(test)]
    fn allocation_identity(&self) -> usize {
        match self {
            Self::Owned(value) => value.allocation_identity(),
            Self::Original { resources, .. } => std::ptr::from_ref(resources) as usize,
        }
    }
}
impl<'source, 'native> PreparedCompletionResources<'source, 'native> {
    pub(crate) fn control_bytes(
        layout: CompletionResourceLayout<'_>,
        kind: CompletionResourceKind,
    ) -> Option<usize> {
        Self::controls(layout, kind, None)
    }
    /// Fixed recovery/retention and exact collection cells used by
    /// `prepare_original`, without constructing resources or debiting custody.
    pub(crate) fn original_control_bytes(
        layout: CompletionResourceLayout<'_>,
        traversal: &safemlx::OperationEvalTraversalLayout,
    ) -> Option<usize> {
        Self::controls(
            layout,
            CompletionResourceKind::Communication,
            Some(traversal),
        )
    }
    fn controls(
        layout: CompletionResourceLayout<'_>,
        kind: CompletionResourceKind,
        original: Option<&safemlx::OperationEvalTraversalLayout>,
    ) -> Option<usize> {
        let fixed = [
            size_of::<Self>(),
            size_of::<ReadyCompletionResources>(),
            size_of::<NativeResources>(), // Rc payload
            2 * size_of::<usize>(),       // Rc strong/weak header
            size_of::<NativeResources>(), // constructor transport
            size_of::<NativeOwner>(),
            size_of::<Option<Rc<NativeResources>>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Option<Destination<std::rc::Weak<NativeResources>>>>(),
            vector_controls::<Array>()?,
            vector_controls::<Vec<usize>>()?,
            vector_controls::<bool>()?,
            vector_controls::<usize>()?,
            vector_controls::<Group>()?,
            vector_controls::<CommunicationRouteRealization>()?,
            vector_controls::<Stream>()?,
            size_of::<Result<Self, Error>>(),
            size_of::<Result<ReadyCompletionResources, Error>>(),
            size_of::<CompletionResourceLayout<'_>>(),
            size_of::<CompletionResourceKind>(),
            size_of::<ResourceCustody>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Option<usize>>(),
            size_of::<(&Self, &OriginalCommunicationSource<'_>, usize)>(),
            size_of::<(Array, Group, CommunicationRouteRealization, Stream)>(),
            size_of::<(&[usize], &mut [usize], usize, bool)>(),
            size_of::<std::slice::Iter<'_, usize>>(),
            size_of::<[usize; 4]>(),
            size_of::<
                std::cell::RefMut<'_, destinations::Destinations<std::rc::Weak<NativeResources>>>,
            >(),
            size_of::<destinations::Iter<'_, std::rc::Weak<NativeResources>>>(),
            Destination::<std::rc::Weak<NativeResources>>::control_bytes()?,
            OrphanDestination::bytes(kind)?,
            recovery_controls(original)?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
            Layout::array::<Array>(layout.arrays).ok()?.size(),
            Layout::array::<Vec<usize>>(layout.counts.len())
                .ok()?
                .size(),
            Layout::array::<bool>(layout.counts.len()).ok()?.size(),
            Layout::array::<Group>(layout.groups).ok()?.size(),
            Layout::array::<CommunicationRouteRealization>(layout.routes)
                .ok()?
                .size(),
            Layout::array::<Stream>(layout.streams).ok()?.size(),
        ];
        let bytes = fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)?;
        layout.counts.iter().try_fold(bytes, |sum, &length| {
            sum.checked_add(Layout::array::<usize>(length).ok()?.size())
        })
    }
    pub(crate) fn prepare(
        source: &'source OriginalCommunicationSource<'native>,
        layout: CompletionResourceLayout<'_>,
        kind: CompletionResourceKind,
    ) -> Result<Self, Error> {
        Self::prepare_with(source, layout, kind, None)
    }
    /// The layout is the existing actual event traversal recipe; it grants no
    /// Graph/Record allowance. The caller must supply its current admitted role.
    pub(crate) fn prepare_original(
        source: &'source OriginalCommunicationSource<'native>,
        layout: CompletionResourceLayout<'_>,
        traversal: safemlx::OperationEvalTraversalLayout,
    ) -> Result<Self, Error> {
        Self::prepare_with(
            source,
            layout,
            CompletionResourceKind::Communication,
            Some(traversal),
        )
    }
    /// The caller includes these exact native WaitRecord requests in its same
    /// admitted role. This helper pays only the finite host/recovery destinations.
    pub(crate) fn prepare_original_with_consumers(
        source: &'source OriginalCommunicationSource<'native>,
        layout: CompletionResourceLayout<'_>,
        traversal: safemlx::OperationEvalTraversalLayout,
        waits: safemlx::OperationWaitRecordLayout,
    ) -> Result<Self, Error> {
        let mut prepared = Self::prepare_original(source, layout, traversal)?;
        prepared.consumers = Some(consumers::PreparedConsumers::prepare(source, waits)?);
        Ok(prepared)
    }
    fn prepare_with(
        source: &'source OriginalCommunicationSource<'native>,
        layout: CompletionResourceLayout<'_>,
        kind: CompletionResourceKind,
        original: Option<safemlx::OperationEvalTraversalLayout>,
    ) -> Result<Self, Error> {
        source.validate()?;
        source
            .funding()
            .reserve_metadata(Self::controls(layout, kind, original.as_ref()).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let custody = ResourceCustody {
            source: source.source().clone(),
            funding: source.funding().clone(),
        };
        // Provisional native/value destinations precede custody on every exit.
        let arrays = vector(layout.arrays, &custody)?;
        let mut counts = vector(layout.counts.len(), &custody)?;
        let mut count_filled = vector(layout.counts.len(), &custody)?;
        for &length in layout.counts {
            let mut values = vector(length, &custody)?;
            values.resize(length, 0);
            counts.push(values);
            count_filled.push(false);
        }
        let groups = vector(layout.groups, &custody)?;
        let routes = vector(layout.routes, &custody)?;
        let streams = vector(layout.streams, &custody)?;
        let registry = Destination::new(Some(custody.clone()));
        let orphan = OrphanDestination::new(kind, &custody);
        Ok(Self {
            source,
            arrays,
            counts,
            count_filled,
            groups,
            routes,
            streams,
            limits: [layout.arrays, layout.groups, layout.routes, layout.streams],
            original,
            consumers: None,
            registry,
            orphan,
            custody,
        })
    }
    pub(crate) fn push_array(&mut self, value: Array) -> Result<(), Error> {
        if self.arrays.len() == self.limits[0] {
            return Err(error(Cause::Identity, &self.custody));
        }
        self.arrays.push(value);
        Ok(())
    }
    pub(crate) fn set_counts(&mut self, index: usize, values: &[usize]) -> Result<(), Error> {
        let Some(target) = self.counts.get_mut(index) else {
            return Err(error(Cause::Identity, &self.custody));
        };
        if self.count_filled[index] || target.len() != values.len() {
            return Err(error(Cause::Identity, &self.custody));
        }
        target.copy_from_slice(values);
        self.count_filled[index] = true;
        Ok(())
    }
    pub(crate) fn push_group(&mut self, value: Group, order: usize) -> Result<(), Error> {
        self.source.validate()?;
        if self.groups.len() == self.limits[1] || !self.source.matches_group(order, &value) {
            return Err(error(Cause::Identity, &self.custody));
        }
        self.groups.push(value);
        Ok(())
    }
    pub(crate) fn push_control_world(&mut self, value: Group) -> Result<(), Error> {
        self.source.validate()?;
        if self.groups.len() == self.limits[1] || !self.source.matches_world(&value) {
            return Err(error(Cause::Identity, &self.custody));
        }
        self.groups.push(value);
        Ok(())
    }
    pub(crate) fn push_route(
        &mut self,
        value: CommunicationRouteRealization,
        order: usize,
    ) -> Result<(), Error> {
        self.source.validate()?;
        if self.routes.len() == self.limits[2] || !self.source.matches_route(order, &value) {
            return Err(error(Cause::Identity, &self.custody));
        }
        self.routes.push(value);
        Ok(())
    }
    /// The selected native operation owns the stream's qualified provenance;
    /// this method only moves its already-owned handle into the finite retention.
    pub(crate) fn push_stream(&mut self, value: Stream) -> Result<(), Error> {
        if self.streams.len() == self.limits[3] {
            return Err(error(Cause::Identity, &self.custody));
        }
        self.streams.push(value);
        Ok(())
    }
    pub(crate) fn finish(self) -> Result<ReadyCompletionResources, Error> {
        self.source.validate()?;
        if [
            self.arrays.len(),
            self.groups.len(),
            self.routes.len(),
            self.streams.len(),
        ] != self.limits
            || self.count_filled.iter().any(|filled| !filled)
        {
            return Err(error(Cause::Identity, &self.custody));
        }
        let Self {
            source: _,
            arrays,
            counts,
            count_filled,
            groups,
            routes,
            streams,
            limits: _,
            original,
            consumers,
            registry,
            orphan,
            custody,
        } = self;
        drop(count_filled);
        let resources = NativeResources::from_owned(
            arrays,
            counts,
            groups,
            routes,
            streams,
            Some(custody.clone()),
            Some(registry),
        );
        let recovery = match original {
            Some(traversal) => PreparedCompletionRecovery::Original {
                resources,
                pending:
                    crate::backend::submission_recovery::observed::PreparedObservedRecovery::new(
                        custody.clone(),
                    ),
                housekeeping: safemlx::PreparedThreadRuntimeHousekeeping::new(
                    original::reap_original_communication,
                    custody.clone(),
                ),
                traversal,
            },
            None => PreparedCompletionRecovery::Owned(
                PreparedRecovery::new(resources, custody.clone()).map_err(|failed| {
                    let cause = error(Cause::Recovery(failed.cause), &custody);
                    drop(failed.retention);
                    drop(failed.custody);
                    cause
                })?,
            ),
        };
        Ok(ReadyCompletionResources {
            recovery,
            orphan,
            consumers,
            custody,
        })
    }
}
fn vector_controls<T>() -> Option<usize> {
    let parts = [
        size_of::<Vec<T>>(),
        size_of::<Result<Vec<T>, Error>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<(&ResourceCustody, usize)>(),
        size_of::<Layout>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn vector<T>(capacity: usize, custody: &ResourceCustody) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|cause| error(Cause::Capacity(cause), custody))?;
    Ok(values)
}

/// Every original event enters the existing finite native root/Eval worker.
/// No independent Scope, ordinary Event or housekeeping fallback is created.
impl ReadyCompletionResources {
    fn submit_adapter_control_bytes<'a, I>() -> Option<usize>
    where
        I: IntoIterator<Item = &'a Array>,
        I::IntoIter: ExactSizeIterator,
    {
        let parts = [
            size_of::<I>(),
            size_of::<I::IntoIter>(),
            size_of::<(&Self, &safemlx::OriginalScopeObserver, &Stream)>(),
            size_of::<usize>(),
            size_of::<Result<OriginalCommunicationCompletion, Error>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn submit_event_control_bytes(closure: usize) -> Option<usize> {
        let parts = [
            closure,
            size_of::<(&Self, &safemlx::OriginalScopeObserver, usize)>(),
            size_of::<Result<safemlx::OperationEvent, safemlx::error::Exception>>(),
            size_of::<Result<OriginalCommunicationCompletion, Error>>(),
            error_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Actual accepted one-root slice adapter, its native traversal closure,
    /// and the shared event/recovery publisher. The retained traversal and
    /// resources are quoted separately; this method allocates nothing.
    pub(crate) fn original_one_root_submit_control_bytes() -> Option<usize> {
        Self::accepted_entry_control_bytes()?.checked_add(Self::one_root_submit_control_bytes()?)
    }
    /// The existing checked-source one-root submission, including its validation.
    /// This has no constructor-minted accepted-entry charge.
    pub(crate) fn original_checked_one_root_submit_control_bytes() -> Option<usize> {
        OriginalCommunicationSource::validation_control_bytes()?
            .checked_add(Self::one_root_submit_control_bytes()?)
    }
    fn one_root_submit_control_bytes() -> Option<usize> {
        fn result_size<A, B, C, D, R>(_: impl FnOnce(A, B, C, D) -> R) -> usize {
            size_of::<R>()
        }
        let closure = result_size(traversal_submitter::<std::slice::Iter<'_, Array>>);
        Self::submit_adapter_control_bytes::<&[Array]>()?
            .checked_add(Self::submit_event_control_bytes(closure)?)
    }
    pub(crate) fn submit_original<'a, I>(
        self,
        source: &OriginalCommunicationSource<'_>,
        observer: &safemlx::OriginalScopeObserver,
        stream: &Stream,
        outputs: I,
    ) -> Result<OriginalCommunicationCompletion, Error>
    where
        I: IntoIterator<Item = &'a Array>,
        I::IntoIter: ExactSizeIterator,
    {
        source.validate()?;
        if !self.custody.source.same_source(source.source()) {
            return Err(error(Cause::Identity, &self.custody));
        }
        self.submit_for_source(observer, stream, outputs)
    }
    // Shared engine; the two callers supply fresh checked source or an exact
    // constructor-minted accepted source. This helper is private to this module.
    fn submit_for_source<'a, I>(
        self,
        observer: &safemlx::OriginalScopeObserver,
        stream: &Stream,
        outputs: I,
    ) -> Result<OriginalCommunicationCompletion, Error>
    where
        I: IntoIterator<Item = &'a Array>,
        I::IntoIter: ExactSizeIterator,
    {
        // Borrowed roots enter the existing iterator-based native traversal.
        // Price the actual adapter and iterator frames before constructing it;
        // neither this adapter nor the event needs a second Array owner.
        self.custody
            .funding
            .reserve_metadata(Self::submit_adapter_control_bytes::<I>().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let outputs = outputs.into_iter();
        safemlx::OperationEvent::validate_traversal_context(observer)
            .map_err(|cause| error(Cause::Native(cause), &self.custody))?;
        let PreparedCompletionRecovery::Original { traversal, .. } = &self.recovery else {
            return Err(error(Cause::Identity, &self.custody));
        };
        let traversal = *traversal;
        self.submit_event(
            observer,
            outputs.len(),
            traversal_submitter(outputs, observer, stream, &traversal),
        )
    }
    fn submit_event<F>(
        self,
        observer: &safemlx::OriginalScopeObserver,
        roots: usize,
        submit: F,
    ) -> Result<OriginalCommunicationCompletion, Error>
    where
        F: FnOnce() -> Result<safemlx::OperationEvent, safemlx::error::Exception>,
    {
        self.custody
            .funding
            .reserve_metadata(
                Self::submit_event_control_bytes(size_of::<F>()).ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let Self {
            recovery,
            orphan,
            consumers,
            custody,
        } = self;
        let PreparedCompletionRecovery::Original {
            resources,
            pending,
            housekeeping,
            traversal,
        } = recovery
        else {
            return Err(error(Cause::Identity, &custody));
        };
        let OrphanDestination::Communication(destination) = orphan else {
            return Err(error(Cause::Identity, &custody));
        };
        if roots != traversal.roots() {
            return Err(error(Cause::Identity, &custody));
        }
        let housekeeping = housekeeping.try_register().map_err(|failed| {
            let (cause, prepared) = failed.into_parts();
            let retained = error(Cause::Housekeeping(cause), &custody);
            drop(prepared);
            retained
        })?;
        *resources.housekeeping.borrow_mut() = Some(housekeeping);
        let mut recovery = CompletionRecovery::original(pending, resources, observer.clone());
        // The worker retains every actual output in the enclosing Graph and
        // submits its exact finite Eval traversal into the same Record owner.
        let event = submit();
        recovery.seal();
        recovery.retention().host_failed.set(event.is_err());
        let event = Rc::new(event.map_err(|cause| error(Cause::Native(cause), &custody))?);
        *recovery.retention().original_event.borrow_mut() = Some(event.clone());
        check_native_status(&recovery).map_err(|cause| error(Cause::Native(cause), &custody))?;
        Ok(OriginalCommunicationCompletion(
            MlxCommunicationCompletion::from_original(
                NativeEvent::Original(event),
                recovery,
                destination,
                roots,
                consumers,
            ),
        ))
    }
}
fn traversal_submitter<'values, 'native, I>(
    outputs: I,
    observer: &'native safemlx::OriginalScopeObserver,
    stream: &'native Stream,
    traversal: &'native safemlx::OperationEvalTraversalLayout,
) -> impl FnOnce() -> Result<safemlx::OperationEvent, safemlx::error::Exception> + use<'values, 'native, I>
where
    I: Iterator<Item = &'values Array> + ExactSizeIterator,
{
    move || {
        safemlx::transforms::async_eval_with_original_prepared_traversal(
            outputs, observer, stream, traversal,
        )
    }
}
fn recovery_controls(traversal: Option<&safemlx::OperationEvalTraversalLayout>) -> Option<usize> {
    let Some(traversal) = traversal else {
        return usize::try_from(PreparedRecovery::<NativeOwner, ResourceCustody>::control_bytes()?)
            .ok();
    };
    use crate::backend::submission_recovery::observed::{
        PreparedObservedRecovery, operation::OperationRecovery,
    };
    let event = Layout::new::<[usize; 2]>()
        .extend(Layout::new::<safemlx::OperationEvent>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let parts = [
        size_of::<PreparedCompletionRecovery>(),
        size_of::<CompletionRecovery>(),
        size_of::<OriginalCommunicationCompletion>(),
        size_of::<MlxCommunicationCompletion>(),
        size_of::<NativeEvent>(),
        size_of::<Rc<safemlx::OperationEvent>>(),
        event,
        size_of::<Result<safemlx::OperationEvent, safemlx::error::Exception>>(),
        size_of::<Result<OriginalCommunicationCompletion, Error>>(),
        size_of::<(
            &OriginalCommunicationSource<'_>,
            &safemlx::OriginalScopeObserver,
            &Stream,
            &[Array],
        )>(),
        size_of::<std::cell::RefMut<'_, Option<Rc<safemlx::OperationEvent>>>>(),
        size_of::<std::cell::RefMut<'_, Option<safemlx::RegisteredThreadRuntimeHousekeeping>>>(),
        size_of::<safemlx::OperationEvalTraversalLayout>(),
        traversal.query_control_bytes()?,
        safemlx::OperationEvent::control_bytes()?,
        original::completion_controls()?,
        safemlx::PreparedThreadRuntimeHousekeeping::<ResourceCustody>::control_bytes()?,
        usize::try_from(
            PreparedObservedRecovery::<NativeOwner, ResourceCustody>::control_bytes::<
                safemlx::error::Exception,
            >()?,
        )
        .ok()?,
        usize::try_from(OperationRecovery::<NativeOwner, ResourceCustody>::control_bytes()?)
            .ok()?,
        size_of::<(&dyn original::CompletionState, Status, bool)>(),
        size_of::<Result<Status, safemlx::error::Exception>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

#[cfg(test)]
mod tests;

mod selected;

mod accepted;

mod nested;
