//! Ordered row activation of the accepted request's exact indexed sources.
use super::{request_sources::AddressableRequestSources, run_region};
use crate::{
    MlxTensor,
    backend::{
        error::Error,
        nn::workspace::AddressableInvocation,
        runtime::{
            execution::generic::{OriginalOperationRegistration, OriginalSelectedResidencyAccess},
            residency::{
                parameter_bank::{
                    IndexedBankSource, IndexedRequestInstallation, IndexedRequestSource,
                    IndexedResidencyFactory, OriginalIndexedResidencyFactory,
                },
                storage::native_storage::BankOwner,
            },
        },
    },
};
use eredu_nn::{
    TensorParallelGroupedOutput,
    workspace::{HostMetadataFunding, HostMetadataFundingError, WorkspaceContext},
};
use eredu_runtime::{
    expert::IndexedInvocationRequest,
    working_memory::{
        InferenceRequest, OriginalOperationMetadataCustody, OriginalTextControlGuard,
        WorkingMemoryError,
    },
};
use safemlx::{OriginalBufferBudget, OriginalScopeObserver, Stream};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
    rc::Rc,
};
fn identity() -> Error {
    super::mismatch("addressable request source")
}
fn source_failure(stage: &'static str, cause: Error) -> Error {
    match cause {
        Error::PrefillControl(cause) => Error::OriginalSourceContract { stage, cause },
        cause => cause,
    }
}
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(size_of_val(parts), usize::checked_add)
}
fn reserve(funding: &HostMetadataFunding, bytes: Option<usize>) -> Result<(), Error> {
    funding
        .reserve_metadata(bytes.ok_or_else(overflow)?)
        .map_err(Error::WorkspacePlanning)
}
fn shared_bytes<T>() -> Option<usize> {
    Some(
        Layout::new::<[usize; 2]>()
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}
enum Authority {
    Text {
        request: InferenceRequest,
        bank: BankOwner,
        controls: OriginalTextControlGuard,
        registration: OriginalOperationRegistration,
    },
    Prepared {
        access: OriginalSelectedResidencyAccess,
        budget: OriginalBufferBudget,
        observer: OriginalScopeObserver,
        metadata: OriginalOperationMetadataCustody,
    },
}
impl Authority {
    fn access(&self) -> Result<OriginalSelectedResidencyAccess, Error> {
        match self {
            Self::Text { registration, .. } => registration.selected_residency_access(),
            Self::Prepared {
                access,
                observer,
                metadata,
                ..
            } => {
                access.validate_observer(observer)?;
                if !access.metadata_custody()?.same_account(metadata) {
                    return Err(identity());
                }
                Ok(access.clone())
            }
        }
    }
    fn budget(&self) -> Result<OriginalBufferBudget, Error> {
        match self {
            Self::Text { bank, controls, .. } => bank
                .try_borrow()
                .map_err(|_| identity())?
                .budget_for_controls(controls)
                .cloned()
                .map_err(Error::PrefillControl),
            Self::Prepared {
                budget, observer, ..
            } => {
                if !observer.same_scope(&OriginalScopeObserver::require_current()?) {
                    return Err(identity());
                }
                Ok(budget.clone())
            }
        }
    }
}
struct Request {
    sources: AddressableRequestSources,
    authority: Authority,
    failed: Cell<bool>,
    running: Cell<bool>,
    timeout: Option<std::time::Duration>,
    native_funding: HostMetadataFunding,
    funding: HostMetadataFunding,
}
pub(crate) struct AddressableRequestOwner(Option<Rc<Request>>);
impl Clone for AddressableRequestOwner {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for AddressableRequestOwner {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl std::fmt::Debug for AddressableRequestOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddressableRequestOwner")
            .field("occurrences", &self.inner().sources.occurrences().len())
            .finish_non_exhaustive()
    }
}
impl AddressableRequestOwner {
    fn construction_control_bytes() -> Option<usize> {
        sum(&[
            shared_bytes::<Request>()?,
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<AddressableRequestSources>(),
            size_of::<InferenceRequest>(),
            size_of::<BankOwner>(),
            size_of::<OriginalTextControlGuard>(),
            size_of::<OriginalOperationRegistration>(),
        ])
    }
    fn row_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<AddressableExecutionRow>(),
            size_of::<Result<AddressableExecutionRow, Error>>(),
            size_of::<(&Self, usize, usize, AddressableInvocation)>(),
            size_of::<Option<usize>>(),
            size_of::<std::ops::Range<usize>>(),
        ])
    }
    /// Actual provider ownership, activation and per-region dispatch population.
    /// Rows and repeats must be the same canonical reports retained by the source
    /// program. Numerical/source workers and their generic callers are separate.
    pub(crate) fn runtime_control_bytes(rows: &[(&AddressableInvocation, usize)]) -> Option<usize> {
        Self::runtime_control_bytes_for(rows, false)
    }
    pub(crate) fn prepared_runtime_control_bytes(
        rows: &[(&AddressableInvocation, usize)],
    ) -> Option<usize> {
        Self::runtime_control_bytes_for(rows, true)
    }
    fn prepared_extra_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<OriginalSelectedResidencyAccess>(),
            size_of::<OriginalBufferBudget>(),
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<&HostMetadataFunding>(),
            OriginalScopeObserver::control_bytes()?,
        ])
    }
    fn runtime_control_bytes_for(
        rows: &[(&AddressableInvocation, usize)],
        prepared: bool,
    ) -> Option<usize> {
        let mut bytes = 0usize;
        let mut any = false;
        for (row, repeats) in rows {
            let count = row.occurrences().len();
            if count == 0 || *repeats == 0 {
                continue;
            }
            any = true;
            let channels = (0..count)
                .filter(|&index| first_channel(row, index))
                .count();
            let one = Self::row_control_bytes()?
                .checked_add(AddressableExecutionRow::activation_control_bytes()?)?
                .checked_add(WorkspaceContext::metadata_vec_bytes::<
                    IndexedRequestInstallation,
                >(count)?)?
                .checked_add(IndexedRequestInstallation::control_bytes()?.checked_mul(channels)?)?
                .checked_add(Active::dispatch_control_bytes()?.checked_mul(count)?)?
                .checked_add(crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement::request_channel_control_bytes()?.checked_mul(count)?)?
                .checked_add(
                    Active::local_control_bytes()?.checked_mul(
                        row.occurrences()
                            .iter()
                            .filter(|(_, quote)| quote.is_local())
                            .count(),
                    )?,
                )?;
            let mut one = one;
            for (_, quote) in row
                .occurrences()
                .iter()
                .filter(|(_, quote)| quote.is_local())
            {
                one = one.checked_add(quote.specialization_control_bytes()?)?
                    .checked_add(IndexedBankSource::local_request_control_bytes()?)?
                    .checked_add(crate::backend::runtime::residency::parameter_bank::IndexedConstructorPartitions::select_count_control_bytes()?)?;
            }
            bytes = bytes.checked_add(one.checked_mul(*repeats)?)?;
        }
        if any {
            bytes = bytes.checked_add(Self::construction_control_bytes()?)?;
            if prepared {
                bytes = bytes.checked_add(Self::prepared_extra_control_bytes()?)?;
            }
        }
        Some(bytes)
    }
    fn inner(&self) -> &Request {
        self.0.as_deref().expect("live addressable request")
    }
    pub(crate) fn new(
        sources: AddressableRequestSources,
        request: &InferenceRequest,
        bank: BankOwner,
        controls: OriginalTextControlGuard,
        registration: OriginalOperationRegistration,
        timeout: Option<std::time::Duration>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        reserve(funding, Self::construction_control_bytes())?;
        controls
            .validate_reservation(request.memory_reservation())
            .map_err(Error::PrefillControl)?;
        bank.try_borrow()
            .map_err(|_| identity())?
            .budget_for_controls(&controls)
            .map_err(Error::PrefillControl)?;
        Ok(Self(Some(Rc::new(Request {
            sources,
            authority: Authority::Text {
                request: request.clone(),
                bank,
                controls,
                registration,
            },
            failed: Cell::new(false),
            running: Cell::new(false),
            timeout,
            native_funding: funding.clone(),
            funding: funding.clone(),
        }))))
    }
    /// Bind this accepted source program to the actual installed speculative
    /// operation. Its ordinary bank remains alive around during().
    pub(crate) fn new_prepared(
        sources: AddressableRequestSources,
        access: OriginalSelectedResidencyAccess,
        budget: OriginalBufferBudget,
        observer: &OriginalScopeObserver,
        timeout: Option<std::time::Duration>,
        funding: &HostMetadataFunding,
        native_funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        reserve(
            funding,
            Self::construction_control_bytes()
                .and_then(|n| n.checked_add(Self::prepared_extra_control_bytes()?)),
        )?;
        access.validate_observer(observer)?;
        let metadata = access.metadata_custody()?;
        Ok(Self(Some(Rc::new(Request {
            sources,
            authority: Authority::Prepared {
                access,
                budget,
                observer: observer.clone(),
                metadata,
            },
            failed: Cell::new(false),
            running: Cell::new(false),
            timeout,
            native_funding: native_funding.clone(),
            funding: funding.clone(),
        }))))
    }
    pub(crate) fn row(
        &self,
        row: usize,
        repetition: usize,
        invocation: AddressableInvocation,
    ) -> Result<AddressableExecutionRow, Error> {
        let owner = self.inner();
        reserve(&owner.funding, Self::row_control_bytes())?;
        let count = invocation.occurrences().len();
        let range = owner
            .sources
            .occurrence_range(row, repetition)
            .ok_or_else(identity)?;
        let (start, end) = (range.start, range.end);
        let entries = owner
            .sources
            .occurrences()
            .get(range)
            .ok_or_else(identity)?;
        if count == 0
            || entries.len() != count
            || entries
                .iter()
                .any(|entry| entry.row != row || entry.repetition != repetition)
        {
            return Err(identity());
        }
        for (entry, (operation, quote)) in entries.iter().zip(invocation.occurrences()) {
            if entry.operation != *operation || !entry.quote.same_quote(quote) {
                return Err(identity());
            }
        }
        Ok(AddressableExecutionRow {
            request: self.clone(),
            invocation,
            start,
            end,
            started: Cell::new(false),
        })
    }
    pub(crate) fn validate_request(&self, request: &InferenceRequest) -> Result<(), Error> {
        match &self.inner().authority {
            Authority::Text {
                request: actual, ..
            } => actual
                .validate_same_request(request)
                .map_err(Error::PrefillControl),
            Authority::Prepared { .. } => Err(identity()),
        }
    }
}
pub(crate) struct AddressableExecutionRow {
    request: AddressableRequestOwner,
    invocation: AddressableInvocation,
    start: usize,
    end: usize,
    started: Cell<bool>,
}
struct Active {
    local: Cell<Option<(usize, usize)>>,
    request: AddressableRequestOwner,
    first: usize,
    end: usize,
    source: crate::backend::nn::workspace::AddressableSources,
    next: Cell<usize>,
}
struct ActiveOwner(Option<Rc<Active>>);
impl Drop for ActiveOwner {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
struct Running<'a> {
    request: &'a Request,
    finished: bool,
}
impl Drop for Running<'_> {
    fn drop(&mut self) {
        if !self.finished || std::thread::panicking() {
            self.request.failed.set(true);
        }
        self.request.running.set(false);
    }
}
impl AddressableExecutionRow {
    fn activation_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<&mut dyn FnMut() -> Result<bool, Error>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Running<'_>>(),
            shared_bytes::<Active>()?,
            size_of::<ActiveOwner>(),
            size_of::<Rc<dyn IndexedRequestSource>>(),
            size_of::<Option<Error>>(),
            size_of::<Vec<IndexedRequestInstallation>>(),
        ])
    }
    /// The ordinary generic caller keeps its own output/error slots. This fixed
    /// borrowed worker owns only activation and channels; false records a typed
    /// caller failure without copying or erasing that caller's error.
    pub(crate) fn during(&self, run: &mut dyn FnMut() -> Result<bool, Error>) -> Result<(), Error> {
        let request = self.request.inner();
        if let Err(cause) = reserve(&request.funding, Self::activation_control_bytes()) {
            request.failed.set(true);
            return Err(cause);
        }
        if self.started.replace(true) || request.failed.get() || request.running.replace(true) {
            request.failed.set(true);
            return Err(identity());
        }
        let mut running = Running {
            request,
            finished: false,
        };
        let owner = ActiveOwner(Some(Rc::new(Active {
            local: Cell::new(None),
            request: self.request.clone(),
            source: self.invocation.source().clone(),
            first: self.start,
            end: self.end,
            next: Cell::new(self.start),
        })));
        let provider: Rc<dyn IndexedRequestSource> =
            owner.0.as_ref().expect("live active owner").clone();
        let count = self.invocation.occurrences().len();
        let mut installed = request.funding.metadata_vec(count).map_err(Error::Neural)?;
        // Several logical units can borrow one physical constructor channel.
        // Install that actual channel once, not once per region declaration.
        for (index, (_, quote)) in self.invocation.occurrences().iter().enumerate() {
            let source = quote.identity().source();
            if !first_channel(&self.invocation, index) {
                continue;
            }
            installed.push(source.install_request(&provider)?);
        }
        let result = run();
        if matches!(&result, Ok(true))
            && (request.failed.get()
                || owner.0.as_ref().expect("live active owner").next.get() != self.end)
        {
            return Err(identity());
        }
        if !matches!(&result, Ok(true)) {
            request.failed.set(true);
        }
        running.finished = true;
        // Channel weak references retire before the active allocation's final
        // strong owner. Failed native regions retain their own source/custody.
        drop(installed);
        drop(provider);
        drop(owner);
        drop(running);
        result.map(|_| ())
    }
}
fn first_channel(invocation: &AddressableInvocation, index: usize) -> bool {
    let source = invocation.occurrences()[index].1.identity().source();
    !invocation.occurrences()[..index]
        .iter()
        .any(|(_, old)| old.identity().source().same_binding(source))
}
impl Active {
    fn local_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<(&Self, &IndexedBankSource, usize)>(),
            size_of::<(&Self, bool, Result<(), Error>)>(),
            size_of::<&Self>(),
            size_of::<eredu_nn::workspace::WorkspaceAddressableRegionView<'_>>(),
            size_of::<OriginalSelectedResidencyAccess>(),
            size_of::<Option<(usize, usize)>>(),
            size_of::<Result<(), Error>>(),
            size_of::<(
                crate::backend::runtime::residency::parameter_bank::IndexedConstructorPartitions,
                Option<crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity>,
            )>(),
        ])
    }
    fn dispatch_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<(&Self, &IndexedBankSource, &Stream)>(),
            size_of::<IndexedInvocationRequest<'_, MlxTensor>>(),
            size_of::<usize>(),
            size_of::<crate::backend::nn::workspace::AddressableQuoteRef>(),
            size_of::<Result<crate::backend::nn::workspace::AddressableQuoteRef, eredu_nn::Error>>(
            ),
            size_of::<Result<TensorParallelGroupedOutput<MlxTensor>, Error>>(),
            size_of::<crate::backend::runtime::execution::generic::OriginalSelectedResidencyAccess>(
            ),
            size_of::<(OriginalBufferBudget, OriginalOperationMetadataCustody)>(),
            size_of::<(
                &Authority,
                Result<OriginalBufferBudget, Error>,
                Result<OriginalOperationMetadataCustody, Error>,
            )>(),
            size_of::<(&'static str, Error)>(),
            size_of::<
                &mut dyn FnMut(
                    IndexedResidencyFactory,
                )
                    -> Result<TensorParallelGroupedOutput<MlxTensor>, Error>,
            >(),
            size_of::<IndexedResidencyFactory>(),
        ])
    }
}
impl IndexedRequestSource for Active {
    fn funding(&self) -> &HostMetadataFunding {
        &self.request.inner().funding
    }
    fn enter_local(
        &self,
        bank: &IndexedBankSource,
        declaration: eredu_nn::workspace::WorkspaceAddressableRegionView<'_>,
        rows: usize,
    ) -> Result<(), Error> {
        let owner = self.request.inner();
        if let Err(cause) = reserve(&owner.funding, Self::local_control_bytes()) {
            owner.failed.set(true);
            return Err(cause);
        }
        let index = self.next.get();
        if owner.failed.get()
            || !owner.running.get()
            || self.local.get().is_some()
            || index < self.first
            || index >= self.end
        {
            owner.failed.set(true);
            return Err(identity());
        }
        let result = (|| {
            let occurrence = owner
                .sources
                .occurrences()
                .get(index)
                .ok_or_else(identity)?;
            if !occurrence.quote.is_local()
                || !occurrence.quote.identity().source().same_binding(bank)
                || occurrence.quote.declaration.as_view() != declaration
                || rows > declaration.chunks.rows
            {
                return Err(identity());
            }
            // Authentication precedes the callback and is still the original
            // registered parent. The addressable worker supplies its one child.
            let access = owner.authority.access()?;
            if rows == 0 {
                self.next.set(index + 1);
                let unused = owner.sources.take(index, &access)?;
                drop(unused);
            }
            self.local.set(Some((index, rows)));
            Ok(())
        })();
        if result.is_err() {
            self.local.set(None);
            owner.failed.set(true);
        }
        result
    }
    fn complete_local(&self, completed: bool) -> Result<(), Error> {
        let owner = self.request.inner();
        let local = self.local.take();
        if !completed {
            owner.failed.set(true);
            return Ok(());
        }
        if owner.failed.get()
            || !owner.running.get()
            || local.is_none_or(|(index, _)| self.next.get() != index + 1)
        {
            owner.failed.set(true);
            return Err(identity());
        }
        Ok(())
    }
    fn abort_local(&self) {
        self.local.set(None);
        self.request.inner().failed.set(true);
    }
    fn with_region(
        &self,
        bank: &IndexedBankSource,
        request: IndexedInvocationRequest<'_, MlxTensor>,
        stream: &Stream,
        run: &mut dyn FnMut(
            IndexedResidencyFactory,
        ) -> Result<TensorParallelGroupedOutput<MlxTensor>, Error>,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, Error> {
        let owner = self.request.inner();
        if let Err(error) = reserve(&owner.funding, Self::dispatch_control_bytes()) {
            owner.failed.set(true);
            return Err(error);
        }
        if owner.failed.get() || !owner.running.get() {
            return Err(identity());
        }
        let index = self.next.get();
        if index < self.first || index >= self.end {
            owner.failed.set(true);
            return Err(identity());
        }
        // This physical occurrence cannot be retried after any fallible step.
        self.next.set(index + 1);
        let result = (|| {
            let occurrence = owner
                .sources
                .occurrences()
                .get(index)
                .ok_or_else(identity)?;
            if !occurrence.quote.identity().source().same_binding(bank) {
                return Err(super::mismatch("addressable request bank binding"));
            }
            let selected = if occurrence.quote.is_local() {
                let (selected_index, rows) = self.local.get().ok_or_else(identity)?;
                if selected_index != index || rows == 0 || rows != request.declaration.chunks.rows {
                    return Err(identity());
                }
                occurrence
                    .quote
                    .select_rows(rows, &owner.funding)
                    .map_err(Error::Neural)?
            } else {
                if self.local.get().is_some() {
                    return Err(identity());
                }
                occurrence.quote.clone()
            };
            if selected.declaration.as_view() != request.declaration {
                return Err(super::mismatch("addressable request declaration"));
            }
            let access = owner
                .authority
                .access()
                .map_err(|cause| source_failure("addressable request selected residency", cause))?;
            let (constructors, reads) = owner
                .sources
                .take(index, &access)
                .map_err(|cause| source_failure("addressable request source extraction", cause))?;
            let constructors = if occurrence.quote.is_local() {
                constructors
                    .select_count(selected.residency.census().plan().len(), &owner.funding)?
            } else {
                constructors
            };
            run_region(
                selected,
                bank,
                request,
                &access,
                constructors,
                reads,
                self.source.runtime(),
                &owner
                    .authority
                    .budget()
                    .map_err(|cause| source_failure("addressable request native budget", cause))?,
                &access.metadata_custody()?,
                &owner.native_funding,
                owner.timeout,
                stream,
                &mut |factory| run(IndexedResidencyFactory::Original(factory)),
            )
        })();
        if result.is_err() {
            owner.failed.set(true);
        }
        result
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
