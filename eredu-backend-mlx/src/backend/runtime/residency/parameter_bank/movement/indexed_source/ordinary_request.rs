//! Ordered ordinary occurrences over actual bank channels and a paid host loan.
use super::*;
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_nn::{TensorParallelGroupedOutput, workspace::WorkspaceAddressableRegionView};
use eredu_runtime::expert::IndexedInvocationRequest;

/// A borrowed occurrence from the retained, source-qualified cold program.
pub(crate) struct OrdinaryIndexedOccurrence<'a> {
    pub(crate) identity: &'a IndexedBindingIdentity,
    pub(crate) residency: &'a OrdinaryIndexedResidencyFacts,
    pub(crate) declaration: WorkspaceAddressableRegionView<'a>,
    pub(crate) local: bool,
}
pub(crate) trait OrdinaryIndexedRequestProgram {
    fn len(&self) -> usize;
    fn occurrence(&self, index: usize) -> Option<OrdinaryIndexedOccurrence<'_>>;
}

struct Active {
    program: Rc<dyn OrdinaryIndexedRequestProgram>,
    funding: HostMetadataFunding,
    next: Cell<usize>,
    local: Cell<Option<(usize, usize)>>,
    running: Cell<bool>,
    failed: Cell<bool>,
    closed: Cell<bool>,
}
/// Retained through the existing ordinary numerical completion boundary. It
/// installs no original observer, native arena, or additional reservation.
pub(crate) struct OrdinaryIndexedRequestOwner {
    active: Rc<Active>,
    installed: Vec<IndexedRequestInstallation>,
}
/// Immutable loan from the actual installed ordinary request. It grants no
/// scope or role; the enclosing Work authenticates its physical observer.
#[derive(Clone)]
pub(crate) struct OrdinaryIndexedLocalSource {
    active: Rc<Active>,
}
impl OrdinaryIndexedLocalSource {
    pub(crate) fn entry_control_bytes() -> Option<usize> {
        sum(&[
            identity_control_bytes(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<(&Self, WorkspaceAddressableRegionView<'_>, usize)>(),
            size_of::<Option<OrdinaryIndexedOccurrence<'_>>>(),
            size_of::<Result<super::request::IndexedLocalRequest<'_>, Error>>(),
        ])
    }
    pub(crate) fn enter_local(
        &self,
        declaration: WorkspaceAddressableRegionView<'_>,
        rows: usize,
    ) -> Result<super::request::IndexedLocalRequest<'_>, Error> {
        self.active.live()?;
        self.active.reserve(Self::entry_control_bytes())?;
        let occurrence = self
            .active
            .program
            .occurrence(self.active.next.get())
            .ok_or_else(|| identity("ordinary indexed local source occurrence"))?;
        occurrence
            .identity
            .source()
            .enter_local_request(declaration, rows)
    }
    pub(crate) fn reserve_callback(&self, bytes: usize) -> Result<(), Error> {
        self.active.live()?;
        self.active.reserve(Some(bytes))
    }
}
fn identity(stage: &'static str) -> Error {
    Error::OriginalSourceContract {
        stage,
        cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
    }
}
fn identity_control_bytes() -> usize {
    size_of::<(&'static str, Error)>()
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
fn first_channel(program: &dyn OrdinaryIndexedRequestProgram, index: usize) -> Option<bool> {
    let occurrence = program.occurrence(index)?;
    for prior in 0..index {
        if program
            .occurrence(prior)?
            .identity
            .source()
            .same_binding(occurrence.identity.source())
        {
            return Some(false);
        }
    }
    Some(true)
}
impl OrdinaryIndexedRequestOwner {
    pub(crate) fn local_source(&self) -> OrdinaryIndexedLocalSource {
        OrdinaryIndexedLocalSource {
            active: self.active.clone(),
        }
    }
    fn construction_control_bytes() -> Option<usize> {
        sum(&[
            identity_control_bytes(),
            shared_bytes::<Active>()?,
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Rc<dyn IndexedRequestSource>>(),
            size_of::<Rc<dyn OrdinaryIndexedRequestProgram>>(),
            size_of::<(&dyn OrdinaryIndexedRequestProgram, usize, Option<bool>)>(),
            size_of::<(&mut Self, Result<(), Error>)>(),
        ])
    }
    /// Provider, channel and dispatch controls only. Actual invocation/chunk
    /// workers and manager/transfer contributors are quoted by their sources.
    pub(crate) fn runtime_control_bytes(
        program: &dyn OrdinaryIndexedRequestProgram,
    ) -> Option<usize> {
        let mut bytes = Self::construction_control_bytes()?.checked_add(
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<IndexedRequestInstallation>(
                program.len(),
            )?,
        )?;
        for index in 0..program.len() {
            let occurrence = program.occurrence(index)?;
            if first_channel(program, index)? {
                bytes = bytes.checked_add(IndexedRequestInstallation::control_bytes()?)?;
            }
            bytes = bytes.checked_add(Active::dispatch_control_bytes()?)?;
            if occurrence.local {
                bytes = bytes
                    .checked_add(Active::local_control_bytes()?)?
                    .checked_add(OrdinaryIndexedLocalSource::entry_control_bytes()?)?
                    .checked_add(IndexedBankSource::local_request_control_bytes()?)?;
            }
        }
        Some(bytes)
    }
    pub(crate) fn new(
        program: Rc<dyn OrdinaryIndexedRequestProgram>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        require_ordinary()?;
        funding
            .reserve_metadata(Self::construction_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let count = program.len();
        let active = Rc::new(Active {
            program,
            funding: funding.clone(),
            next: Cell::new(0),
            local: Cell::new(None),
            running: Cell::new(false),
            failed: Cell::new(false),
            closed: Cell::new(false),
        });
        let provider: Rc<dyn IndexedRequestSource> = active.clone();
        let mut installed = funding.metadata_vec(count).map_err(Error::Neural)?;
        for index in 0..count {
            let occurrence = active
                .program
                .occurrence(index)
                .ok_or_else(|| identity("ordinary indexed installation occurrence"))?;
            active.validate_binding(occurrence.identity, occurrence.identity.source())?;
            if first_channel(&*active.program, index)
                .ok_or_else(|| identity("ordinary indexed installation channel census"))?
            {
                installed.push(occurrence.identity.source().install_request(&provider)?);
            }
        }
        Ok(Self { active, installed })
    }
    pub(crate) fn finish(&mut self) -> Result<(), Error> {
        if self.active.closed.get()
            || self.active.failed.get()
            || self.active.running.get()
            || self.active.local.get().is_some()
            || self.active.next.get() != self.active.program.len()
        {
            self.active.failed.set(true);
            return Err(identity("ordinary indexed request completion cursor"));
        }
        self.active.closed.set(true);
        self.installed.clear();
        Ok(())
    }
}
impl Drop for OrdinaryIndexedRequestOwner {
    fn drop(&mut self) {
        self.active.closed.set(true);
        self.installed.clear();
    }
}
impl Active {
    fn local_control_bytes() -> Option<usize> {
        sum(&[
            identity_control_bytes(),
            size_of::<(&Self, &IndexedBankSource, usize)>(),
            size_of::<WorkspaceAddressableRegionView<'_>>(),
            size_of::<Option<(usize, usize)>>(),
            size_of::<(&Self, bool, Result<(), Error>)>(),
        ])
    }
    fn dispatch_control_bytes() -> Option<usize> {
        sum(&[
            identity_control_bytes(),
            size_of::<(&Self, &IndexedBankSource, &Stream)>(),
            size_of::<IndexedInvocationRequest<'_, MlxTensor>>(),
            size_of::<OrdinaryIndexedOccurrence<'_>>(),
            size_of::<WorkspaceAddressableRegionView<'_>>(),
            size_of::<AddressableChunkCensus>(),
            size_of::<Dispatch<'_>>(),
            size_of::<IndexedResidencyFactory>(),
            size_of::<Result<TensorParallelGroupedOutput<MlxTensor>, Error>>(),
            size_of::<
                &mut dyn FnMut(
                    IndexedResidencyFactory,
                )
                    -> Result<TensorParallelGroupedOutput<MlxTensor>, Error>,
            >(),
        ])
    }
    fn reserve(&self, bytes: Option<usize>) -> Result<(), Error> {
        self.funding
            .reserve_metadata(bytes.ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)
    }
    fn validate_binding(
        &self,
        source: &IndexedBindingIdentity,
        actual: &IndexedBankSource,
    ) -> Result<(), Error> {
        if !source.source().same_binding(actual) {
            return Err(identity("ordinary indexed binding identity"));
        }
        actual
            .with_workspace_source(&self.funding, |bank| {
                if bank.parameter_revision() != source.revision() {
                    return Err(identity("ordinary indexed parameter revision"));
                }
                Ok(())
            })
            .map_err(|cause| failed(Cause::Bank(cause), &actual.bank, &self.funding, None))?
    }
    fn live(&self) -> Result<(), Error> {
        require_ordinary()?;
        if self.failed.get() || self.closed.get() {
            Err(identity("ordinary indexed request is failed or closed"))
        } else {
            Ok(())
        }
    }
}
struct Dispatch<'a> {
    owner: &'a Active,
    finished: bool,
}
impl Drop for Dispatch<'_> {
    fn drop(&mut self) {
        self.owner.running.set(false);
        if !self.finished {
            self.owner.failed.set(true);
        }
    }
}
impl IndexedRequestSource for Active {
    fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    fn enter_local(
        &self,
        bank: &IndexedBankSource,
        declaration: WorkspaceAddressableRegionView<'_>,
        rows: usize,
    ) -> Result<(), Error> {
        let result = (|| {
            self.live()?;
            self.reserve(Self::local_control_bytes())?;
            if self.running.get() || self.local.get().is_some() {
                return Err(identity(
                    "ordinary indexed local entry overlaps an active occurrence",
                ));
            }
            let index = self.next.get();
            let occurrence = self
                .program
                .occurrence(index)
                .ok_or_else(|| identity("ordinary indexed local entry occurrence"))?;
            self.validate_binding(occurrence.identity, bank)?;
            if !occurrence.local {
                return Err(identity(
                    "ordinary indexed local entry selected a nonlocal occurrence",
                ));
            }
            if occurrence.declaration != declaration {
                return Err(identity("ordinary indexed local entry declaration"));
            }
            if rows > declaration.chunks.rows {
                return Err(identity("ordinary indexed local entry row limit"));
            }
            self.local.set(Some((index, rows)));
            if rows == 0 {
                self.next.set(index.checked_add(1).ok_or_else(overflow)?);
            }
            Ok(())
        })();
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }
    fn complete_local(&self, completed: bool) -> Result<(), Error> {
        let local = self.local.take();
        if !completed {
            self.failed.set(true);
            return Ok(());
        }
        if self.live().is_err()
            || self.running.get()
            || local.is_none_or(|(index, _)| index.checked_add(1) != Some(self.next.get()))
        {
            self.failed.set(true);
            return Err(identity("ordinary indexed local completion cursor"));
        }
        Ok(())
    }
    fn abort_local(&self) {
        self.local.set(None);
        self.failed.set(true);
    }
    fn with_region(
        &self,
        bank: &IndexedBankSource,
        request: IndexedInvocationRequest<'_, MlxTensor>,
        _stream: &Stream,
        run: &mut dyn FnMut(
            IndexedResidencyFactory,
        ) -> Result<TensorParallelGroupedOutput<MlxTensor>, Error>,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, Error> {
        self.live()?;
        if self.running.replace(true) {
            self.failed.set(true);
            return Err(identity(
                "ordinary indexed dispatch overlaps an active occurrence",
            ));
        }
        let mut guard = Dispatch {
            owner: self,
            finished: false,
        };
        self.reserve(Self::dispatch_control_bytes())?;
        let index = self.next.get();
        self.next.set(index.checked_add(1).ok_or_else(overflow)?);
        let occurrence = self
            .program
            .occurrence(index)
            .ok_or_else(|| identity("ordinary indexed dispatch occurrence"))?;
        self.validate_binding(occurrence.identity, bank)?;
        let mut declaration = occurrence.declaration;
        let first = occurrence
            .identity
            .first()
            .ok_or_else(|| identity("ordinary indexed dispatch chunk census"))?;
        occurrence.residency.validate(bank, first)?;
        let first = if occurrence.local {
            let (selected, rows) = self
                .local
                .get()
                .ok_or_else(|| identity("ordinary indexed dispatch has no local row loan"))?;
            if selected != index || rows == 0 {
                return Err(identity("ordinary indexed dispatch local row occurrence"));
            }
            declaration.chunks.rows = rows;
            AddressableChunkCensus::new(
                first.bank(),
                first.unit(),
                first
                    .plan()
                    .for_rows(rows)
                    .ok_or_else(|| identity("ordinary indexed dispatch local chunk plan"))?,
                0,
                first.access(),
            )
            .ok_or_else(|| identity("ordinary indexed dispatch local chunk census"))?
        } else {
            if self.local.get().is_some() {
                return Err(identity(
                    "ordinary indexed nonlocal dispatch has a local row loan",
                ));
            }
            first
        };
        if declaration != request.declaration {
            return Err(identity("ordinary indexed dispatch declaration"));
        }
        let pins = occurrence.residency.prepare_source_pins(&self.funding)?;
        let factory = OrdinaryIndexedResidencyFactory::new(bank, first, &self.funding, pins)?;
        let result = run(IndexedResidencyFactory::Ordinary(factory));
        guard.finished = result.is_ok();
        result
    }
}
