//! Ordinary invocation custody over the existing histogram, remap and cache workers.
use super::*;
use crate::backend::runtime::residency::manager::HostCopySourcePins;
use crate::backend::runtime::residency::parameter_bank::OrdinaryBankHostSource;
use eredu_runtime::expert::{IndexedInvocationRequest, PreparedIndexedDemandLoan};

struct OrdinaryBody {
    identity: SharedStorageOwner<SourceIdentity>,
    host: OrdinaryBankHostSource,
    stream: StreamCopyPlan<()>,
    discovered: Cell<bool>,
    remapped: Cell<bool>,
    copies: Cell<usize>,
    acquired: Cell<bool>,
    completed: Cell<bool>,
    closed: Cell<bool>,
}
#[derive(Clone)]
pub(crate) struct OrdinaryIndexedChunkSource(Rc<OrdinaryBody>);
impl OrdinaryIndexedChunkSource {
    fn failure(&self, cause: Cause) -> Error {
        let id = &self.0.identity;
        failed(cause, &id.bank, &id.funding, Some(id))
    }
    pub(crate) fn control_bytes(census: AddressableChunkCensus) -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<OrdinaryBody>(),
            Layout::new::<[usize; 2]>()
                .extend(Layout::new::<OrdinaryBody>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
            shared_bytes::<SourceIdentity>()?,
            Array::inspection_clone_handle_bytes(),
            Array::descriptor_comparison_control_bytes()?,
            StreamCopyPlan::<()>::capture_control_bytes().ok()?,
            OrdinaryBankHostSource::control_bytes()?,
            IndexedDemandSource::control_bytes()?,
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<(usize, u64)>(
                census.maximum_members(),
            )?,
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<i32>(census.members())?,
            safemlx::EvaluatedArray::iteration_control_bytes::<i32>()?.checked_mul(2)?,
            safemlx::ops::OrdinaryRecipeCall::Evaluate { inputs: 2 }
                .control_bytes()?
                .metadata_bytes(),
            safemlx::ops::OrdinaryRecipeCall::Evaluate { inputs: 1 }
                .control_bytes()?
                .metadata_bytes(),
            safemlx::ops::OrdinaryRecipeCall::BorrowedEvaluation
                .control_bytes()?
                .metadata_bytes()
                .checked_mul(2)?,
            size_of::<[Array; 2]>(),
            size_of::<Vec<(usize, u64)>>(),
            size_of::<Vec<i32>>(),
            size_of::<Result<IndexedDemandSource, Error>>(),
            size_of::<Result<MlxTensor, Error>>(),
            size_of::<(&Self, &MlxTensor, &Stream)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    fn new(
        binding: &IndexedBankSource,
        census: AddressableChunkCensus,
        input: &MlxTensor,
        stream: &Stream,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        require_ordinary()?;
        let fail = |cause| failed(cause, &binding.bank, funding, None);
        let host = OrdinaryBankHostSource::new(binding, census, funding)?;
        let shape = [
            i32::try_from(census.rows()).map_err(|_| fail(Cause::Overflow))?,
            i32::try_from(census.routes()).map_err(|_| fail(Cause::Overflow))?,
        ];
        if input.shape() != shape
            || !matches!(
                input.as_array().dtype(),
                Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
            )
        {
            return Err(fail(Cause::Geometry));
        }
        // Dynamic destination helpers debit their own exact grants below.
        let controls = Self::control_bytes(census)
            .ok_or_else(|| fail(Cause::Overflow))?
            .checked_sub(
                OrdinaryBankHostSource::control_bytes().ok_or_else(|| fail(Cause::Overflow))?,
            )
            .and_then(|n| {
                n.checked_sub(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                    (usize, u64),
                >(census.maximum_members())?)
            })
            .and_then(|n| {
                n.checked_sub(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                    i32,
                >(census.members())?)
            })
            .ok_or_else(|| fail(Cause::Overflow))?;
        funding
            .reserve_metadata(controls)
            .map_err(|cause| fail(Cause::Funding(cause)))?;
        let revision = binding
            .with_workspace_source(funding, |source| {
                Ok::<_, Error>(source.parameter_revision())
            })
            .map_err(|cause| fail(Cause::Bank(cause)))??;
        let stream = StreamCopyPlan::capture(stream).map_err(|_| fail(Cause::Identity))?;
        let identity = SharedStorageOwner::new(SourceIdentity {
            input: input.as_array().clone(),
            bank: binding.bank.clone(),
            census,
            bulk_target_bytes: binding.prefill_compact_bank_target_bytes(),
            parameter_revision: revision,
            funding: funding.clone(),
        });
        Ok(Self(Rc::new(OrdinaryBody {
            identity,
            host,
            stream,
            discovered: Cell::new(false),
            remapped: Cell::new(false),
            copies: Cell::new(0),
            acquired: Cell::new(false),
            completed: Cell::new(false),
            closed: Cell::new(false),
        })))
    }
    fn validate(&self, input: &MlxTensor, stream: &Stream) -> Result<(), Error> {
        require_ordinary()?;
        if self.0.closed.get() || !self.0.stream.matches_source(stream) {
            return Err(self.failure(Cause::Identity));
        }
        self.0.host.validate_bank(&self.0.identity.bank)?;
        if !self
            .0
            .identity
            .input
            .try_descriptor()
            .map_err(|cause| self.failure(Cause::Descriptor(cause)))?
            .same_descriptor(input.as_array())
            .map_err(|cause| self.failure(Cause::Descriptor(cause)))?
        {
            return Err(self.failure(Cause::Identity));
        }
        Ok(())
    }
    fn validate_demands(&self, demands: &IndexedDemandSource) -> Result<(), Error> {
        let source = demands
            .source()
            .and_then(|source| source.downcast_ref::<SourceIdentity>())
            .ok_or_else(|| self.failure(Cause::Identity))?;
        if !std::ptr::eq(source, &*self.0.identity)
            || !demands
                .funding()
                .is_some_and(|f| f.same_account(&self.0.identity.funding))
            || !self.0.discovered.get()
            || self.0.closed.get()
        {
            return Err(self.failure(Cause::Identity));
        }
        Ok(())
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn discover(
        &self,
        input: &MlxTensor,
        census: AddressableChunkCensus,
        stream: &Stream,
    ) -> Result<IndexedDemandSource, Error> {
        self.validate(input, stream)?;
        if census != self.0.identity.census || self.0.discovered.replace(true) {
            return Err(self.failure(Cause::Spent));
        }
        let result = indexed_numerical::discover(
            &mut indexed_numerical::Native(stream),
            input.as_array(),
            census.members(),
        )?;
        eval([&result.histogram, &result.invalid])
            .map_err(|cause| self.failure(Cause::Native(cause)))?;
        let invalid = result
            .invalid
            .evaluated()
            .map_err(|cause| self.failure(Cause::Native(cause)))?;
        if invalid.as_slice::<i32>()[0] != 0 {
            return Err(self.failure(Cause::Geometry));
        }
        let values = result
            .histogram
            .evaluated()
            .map_err(|cause| self.failure(Cause::Native(cause)))?;
        let mut demands = self
            .0
            .identity
            .funding
            .metadata_vec(census.maximum_members())
            .map_err(Error::Neural)?;
        for (id, &count) in values.as_slice::<i32>().iter().enumerate() {
            if count < 0 {
                return Err(self.failure(Cause::Geometry));
            }
            if count != 0 {
                if demands.len() == census.maximum_members() {
                    return Err(self.failure(Cause::Geometry));
                }
                demands.push((id, count as u64));
            }
        }
        Ok(IndexedDemandSource::from_prepared(
            demands,
            self.0.identity.clone().erase(),
            self.0.identity.funding.clone(),
        ))
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn remap(
        &self,
        input: &MlxTensor,
        mapping: &[(usize, usize)],
        demands: &IndexedDemandSource,
        stream: &Stream,
    ) -> Result<MlxTensor, Error> {
        self.validate(input, stream)?;
        self.validate_demands(demands)?;
        if self.0.remapped.replace(true)
            || mapping.len() != demands.demands().len()
            || mapping
                .iter()
                .zip(demands.demands())
                .enumerate()
                .any(|(compact, (&(id, to), &(expected, _)))| id != expected || to != compact)
        {
            return Err(self.failure(Cause::Geometry));
        }
        let span = mapping
            .last()
            .and_then(|(id, _)| id.checked_add(1))
            .ok_or_else(|| self.failure(Cause::Overflow))?;
        if span > self.0.identity.census.members() {
            return Err(self.failure(Cause::Geometry));
        }
        let mut lookup = self
            .0
            .identity
            .funding
            .metadata_vec(span)
            .map_err(Error::Neural)?;
        lookup.resize(span, -1i32);
        for &(from, to) in mapping {
            lookup[from] = i32::try_from(to).map_err(|_| self.failure(Cause::Overflow))?;
        }
        indexed_numerical::remap(
            &mut indexed_numerical::Native(stream),
            input.as_array(),
            lookup.as_slice(),
        )
        .map(MlxTensor::from_array)
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn copy_route_value(
        &self,
        value: &MlxTensor,
        demands: &IndexedDemandSource,
        stream: &Stream,
    ) -> Result<MlxTensor, Error> {
        require_ordinary()?;
        self.validate_demands(demands)?;
        let copies = self.0.copies.get();
        if copies >= 2 || !self.0.remapped.get() || !self.0.stream.matches_source(stream) {
            return Err(self.failure(Cause::Spent));
        }
        self.0.copies.set(copies + 1);
        self.0
            .identity
            .funding
            .reserve_metadata(
                Array::inspection_clone_handle_bytes()
                    .checked_add(size_of::<(&Self, &MlxTensor, &IndexedDemandSource, &Stream)>())
                    .ok_or_else(|| self.failure(Cause::Overflow))?,
            )
            .map_err(|cause| self.failure(Cause::Funding(cause)))?;
        Ok(value.clone())
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn with_demand_loan<
        R,
        E,
        F,
    >(
        &self,
        demands: &IndexedDemandSource,
        run: F,
    ) -> Result<Result<R, E>, Error>
    where
        F: FnOnce(Option<PreparedIndexedDemandLoan<'_>>) -> Result<R, E>,
    {
        self.validate_demands(demands)?;
        Ok(run(Some(PreparedIndexedDemandLoan::new(
            self,
            &self.0.identity.funding,
        ))))
    }
    pub(crate) fn acquire(
        &self,
        bank: &SharedAddressableParameterBank,
        request: ParameterBankAcquisition<'_>,
        demands: &IndexedDemandSource,
        funding: &HostMetadataFunding,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, Error> {
        require_ordinary()?;
        self.validate_demands(demands)?;
        if !funding.same_account(&self.0.identity.funding)
            || !self.0.stream.matches_source(stream)
            || !self.0.remapped.get()
            || self.0.copies.get() != 2
            || self.0.acquired.replace(true)
        {
            return Err(self.failure(Cause::Spent));
        }
        let access = match request.access() {
            ParameterBankAccess::Bulk => BankAccessClass::Bulk,
            ParameterBankAccess::Incremental => BankAccessClass::Incremental,
            _ => return Err(self.failure(Cause::Geometry)),
        };
        let entries = request.entries();
        if entries.len() != demands.demands().len() || entries.is_empty() {
            return Err(self.failure(Cause::Identity));
        }
        let id = &self.0.identity;
        id.bank
            .with_workspace_source(&id.funding, |source| {
                if source.parameter_revision() != id.parameter_revision {
                    return Err(self.failure(Cause::Identity));
                }
                let mut selected = 0usize;
                for (local, (key, _)) in source
                    .unit_members(id.census.bank(), id.census.unit())
                    .enumerate()
                {
                    if let Some(&(expected, count)) = demands.demands().get(selected) {
                        if local == expected {
                            if entries.get(selected) != Some(&(key, count)) {
                                return Err(self.failure(Cause::Identity));
                            }
                            selected += 1;
                        }
                    }
                }
                if selected != entries.len() {
                    return Err(self.failure(Cause::Identity));
                }
                Ok::<_, Error>(())
            })
            .map_err(|cause| self.failure(Cause::Bank(cause)))??;
        let mut acquired = self
            .0
            .host
            .acquire(bank, request.entries(), access, stream)?;
        acquired.ordinary_chunk = Some(self.clone());
        Ok(acquired)
    }
    pub(crate) fn complete(
        &self,
        bank: &SharedAddressableParameterBank,
        mut acquired: AcquiredParameterGroups,
        output: &MlxTensor,
        stream: &Stream,
    ) -> Result<(), Error> {
        require_ordinary()?;
        self.0.host.validate_bank(bank)?;
        if !self.0.acquired.get()
            || self.0.completed.get()
            || !self.0.stream.matches_source(stream)
            || !acquired
                .ordinary_chunk
                .as_ref()
                .is_some_and(|s| Rc::ptr_eq(&s.0, &self.0))
        {
            return Err(self.failure(Cause::Identity));
        }
        eval([output.as_array()]).map_err(|cause| self.failure(Cause::Native(cause)))?;
        acquired
            .transfer
            .synchronize()
            .map_err(|cause| self.failure(Cause::Residency(cause)))?;
        self.0.completed.set(true);
        Ok(())
    }
}

pub(crate) struct OrdinaryIndexedResidencyFactory {
    source_pins: Option<HostCopySourcePins>,
    first: OrdinaryBankHostSource,
    binding: IndexedBankSource,
    funding: HostMetadataFunding,
}
struct Invocation {
    // Drops through the shared deferred pin owner only after this invocation.
    _source_pins: Option<HostCopySourcePins>,
    first: OrdinaryBankHostSource,
    binding: IndexedBankSource,
    funding: HostMetadataFunding,
    stream: StreamCopyPlan<()>,
    next: Cell<usize>,
    closed: Cell<bool>,
}
#[derive(Clone)]
pub(crate) struct OrdinaryIndexedResidencyInvocation(Rc<Invocation>);
impl OrdinaryIndexedResidencyFactory {
    pub(crate) fn new(
        binding: &IndexedBankSource,
        first: AddressableChunkCensus,
        funding: &HostMetadataFunding,
        source_pins: Option<HostCopySourcePins>,
    ) -> Result<Self, Error> {
        if first.index() != 0 {
            return Err(failed(Cause::Geometry, &binding.bank, funding, None));
        }
        let first = OrdinaryBankHostSource::new(binding, first, funding)?;
        Ok(Self {
            source_pins,
            first,
            binding: binding.clone(),
            funding: funding.clone(),
        })
    }
    pub(crate) fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Invocation>(),
            size_of::<OrdinaryIndexedResidencyInvocation>(),
            size_of::<StreamCopyPlan<()>>(),
            Layout::new::<[usize; 2]>()
                .extend(Layout::new::<Invocation>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
            StreamCopyPlan::<()>::capture_control_bytes().ok()?,
            size_of::<IndexedInvocationRequest<'_, MlxTensor>>(),
            size_of::<eredu_nn::workspace::ExpertRegionInputShape>(),
            OrdinaryBankHostSource::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn validate_request(
        &self,
        request: &IndexedInvocationRequest<'_, MlxTensor>,
    ) -> Result<(), Error> {
        let fail = || failed(Cause::Identity, &self.binding.bank, &self.funding, None);
        let first = self.first.census();
        let declaration = request.declaration;
        declaration.validate().map_err(|_| fail())?;
        let shape = eredu_nn::workspace::ExpertRegionInputShape::inspect(
            request.input.shape(),
            request.routes.group_indices().shape(),
        )
        .map_err(|_| fail())?;
        if declaration.bank as usize != first.bank()
            || declaration.unit != first.unit()
            || declaration.chunks != first.plan().workspace_source()
            || declaration.prefill != (first.access() == ParameterBankAccess::Bulk)
            || usize::try_from(shape.rows).ok() != Some(first.total_rows())
            || usize::try_from(shape.routes).ok() != Some(first.routes())
            || shape.width != declaration.kernel.dimensions().0
            || request.routes.selected_scores().shape() != request.routes.group_indices().shape()
            || request.routes.coefficients().shape() != request.routes.group_indices().shape()
        {
            return Err(fail());
        }
        self.first.validate_bank(&self.binding.bank)
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn prepare(
        self,
        movement: &MlxIndexedMovement,
        stream: &Stream,
    ) -> Result<OrdinaryIndexedResidencyInvocation, Error> {
        require_ordinary()?;
        let fail = |cause| failed(cause, &self.binding.bank, &self.funding, None);
        if movement.original.is_some()
            || movement.invocation.is_some()
            || movement.ordinary.is_some()
            || movement.ordinary_invocation.is_some()
            || !movement
                .binding
                .as_ref()
                .is_some_and(|b| b.same_binding(&self.binding))
        {
            return Err(fail(Cause::Identity));
        }
        let bytes = Self::control_bytes()
            .and_then(|n| n.checked_sub(OrdinaryBankHostSource::control_bytes()?))
            .ok_or_else(|| fail(Cause::Overflow))?;
        self.funding
            .reserve_metadata(bytes)
            .map_err(|cause| fail(Cause::Funding(cause)))?;
        let stream = StreamCopyPlan::capture(stream).map_err(|_| fail(Cause::Identity))?;
        Ok(OrdinaryIndexedResidencyInvocation(Rc::new(Invocation {
            _source_pins: self.source_pins,
            first: self.first,
            binding: self.binding,
            funding: self.funding,
            stream,
            next: Cell::new(0),
            closed: Cell::new(false),
        })))
    }
}
struct Guard<'a, P> {
    owner: &'a mut P,
    movement: fn(&mut P) -> &mut MlxIndexedMovement,
    source: OrdinaryIndexedResidencyInvocation,
}
impl<P> Drop for Guard<'_, P> {
    fn drop(&mut self) {
        self.source.0.closed.set(true);
        let movement = (self.movement)(self.owner);
        if let Some(chunk) = movement.ordinary.take() {
            chunk.0.closed.set(true);
        }
        movement.ordinary_invocation.take();
    }
}
impl OrdinaryIndexedResidencyInvocation {
    fn failure(&self, cause: Cause) -> Error {
        failed(cause, &self.0.binding.bank, &self.0.funding, None)
    }
    pub(crate) fn owner_control_bytes<P, R, E, F>() -> Option<usize> {
        let frames = [
            size_of::<Guard<'_, P>>(),
            size_of::<F>(),
            size_of::<Result<R, E>>(),
            size_of::<Result<Result<R, E>, Error>>(),
            size_of::<(&mut P, fn(&mut P) -> &mut MlxIndexedMovement, &Stream)>(),
            size_of::<Option<OrdinaryIndexedChunkSource>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn run_with_owner<
        P,
        R,
        E,
        F,
    >(
        self,
        owner: &mut P,
        movement: fn(&mut P) -> &mut MlxIndexedMovement,
        stream: &Stream,
        run: F,
    ) -> Result<Result<R, E>, Error>
    where
        F: FnOnce(&mut P) -> Result<R, E>,
    {
        require_ordinary()?;
        if self.0.closed.get() || !self.0.stream.matches_source(stream) {
            return Err(self.failure(Cause::Identity));
        }
        self.0
            .funding
            .reserve_metadata(
                Self::owner_control_bytes::<P, R, E, F>()
                    .ok_or_else(|| self.failure(Cause::Overflow))?,
            )
            .map_err(|cause| self.failure(Cause::Funding(cause)))?;
        movement(owner).ordinary_invocation = Some(self.clone());
        let guard = Guard {
            owner,
            movement,
            source: self,
        };
        let result = run(guard.owner);
        if result.is_ok()
            && (guard.source.0.next.get() != guard.source.0.first.census().plan().len()
                || !(guard.movement)(guard.owner)
                    .ordinary
                    .as_ref()
                    .is_some_and(|c| c.0.completed.get()))
        {
            return Err(guard.source.failure(Cause::Spent));
        }
        Ok(result)
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn begin_chunk(
        &self,
        movement: &mut MlxIndexedMovement,
        input: &MlxTensor,
        census: AddressableChunkCensus,
        stream: &Stream,
    ) -> Result<(), Error> {
        require_ordinary()?;
        let first = self.0.first.census();
        if self.0.closed.get()
            || !self.0.stream.matches_source(stream)
            || census.index() != self.0.next.get()
            || census.plan() != first.plan()
            || census.bank() != first.bank()
            || census.unit() != first.unit()
            || census.access() != first.access()
            || !movement
                .binding
                .as_ref()
                .is_some_and(|b| b.same_binding(&self.0.binding))
        {
            return Err(self.failure(Cause::Identity));
        }
        if let Some(previous) = movement.ordinary.take() {
            previous.0.closed.set(true);
            if !previous.0.completed.get() {
                return Err(self.failure(Cause::Spent));
            }
        }
        self.0.next.set(
            census
                .index()
                .checked_add(1)
                .ok_or_else(|| self.failure(Cause::Overflow))?,
        );
        movement.ordinary = Some(OrdinaryIndexedChunkSource::new(
            &self.0.binding,
            census,
            input,
            stream,
            &self.0.funding,
        )?);
        Ok(())
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
