//! Explicit model-bound immutable bank sources and their paid quote directory.
use super::*;
use crate::backend::runtime::residency::parameter_bank::IndexedBankSource;
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::{alloc::Layout, cell::RefCell, ops::Deref, rc::Rc};

pub(crate) struct AddressableQuoteRef(Option<Rc<AddressableQuote>>, Option<Rc<LocalEnvelope>>);
pub(super) struct LocalEnvelope {
    pub(super) equation: SpeculativeNumericalRecipe,
    pub(super) numerical: SpeculativeNumericalRecipe,
    pub(super) capacity: BoundaryStageCapacity,
    pub(super) host_bytes: u64,
    pub(super) capture_publications: usize,
}
impl AddressableQuoteRef {
    pub(crate) fn is_local(&self) -> bool { self.1.is_some() }
    pub(crate) fn local_numerical(&self) -> Option<SpeculativeNumericalRecipe> { self.1.as_ref().map(|v| v.numerical) }
    pub(crate) fn native_capacity(&self) -> BoundaryStageCapacity { self.1.as_ref().map_or(self.capacity, |v| v.capacity) }
    pub(crate) fn host_capacity(&self) -> u64 { self.1.as_ref().map_or(self.host_bytes, |v| v.host_bytes) }
    pub(crate) fn publications(&self) -> usize { self.1.as_ref().map_or(self.capture_publications, |v| v.capture_publications) }
    pub(crate) fn select_rows(&self, rows: usize, funding: &HostMetadataFunding) -> Result<Self, Error> {
        let envelope = self.1.as_deref().ok_or(WorkspaceMetadataError::Unqualified)?;
        let value = self.deref().for_rows(rows, envelope, funding)?;
        funding.reserve_metadata(shared_bytes::<AddressableQuote>().ok_or(WorkspaceMetadataError::Overflow)?).map_err(WorkspaceMetadataError::Funding)?;
        Ok(Self(Some(Rc::new(value)), None))
    }
}
impl AddressableQuoteRef {
    pub(crate) fn same_quote(&self,other:&Self)->bool {
        match (&self.0,&other.0){(Some(a),Some(b))=>Rc::ptr_eq(a,b) && match (&self.1,&other.1) {(None,None)=>true,(Some(a),Some(b))=>Rc::ptr_eq(a,b),_=>false},_=>false}
    }
}
impl Clone for AddressableQuoteRef {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}
impl Deref for AddressableQuoteRef {
    type Target = AddressableQuote;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live addressable quote")
    }
}
impl Drop for AddressableQuoteRef {
    fn drop(&mut self) {
        if let Some(rows) = self.1.take() { drop(Rc::into_inner(rows)); }
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl std::fmt::Debug for AddressableQuoteRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}
pub(crate) struct AddressableSources(Option<Rc<Sources>>);
impl Clone for AddressableSources {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for AddressableSources {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl std::fmt::Debug for AddressableSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddressableSources")
            .field("banks", &self.inner().banks.len())
            .finish_non_exhaustive()
    }
}
struct Sources {
    banks: Vec<(u32, IndexedBankSource)>,
    quotes: RefCell<Vec<AddressableQuoteRef>>,
    mechanism: ResidentExecutionMechanisms,
    runtime: safemlx::PreparedInputRuntime,
    pool: Option<WorkingMemoryPool>,
    funding: HostMetadataFunding,
}
pub(super) fn shared_bytes<T>() -> Option<usize> {
    Some(
        Layout::new::<[usize; 2]>()
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}
impl AddressableSources {
    pub(crate) fn new<'a, I>(
        banks: I,
        mechanism: ResidentExecutionMechanisms,
        runtime: &safemlx::PreparedInputRuntime,
        pool: Option<&WorkingMemoryPool>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error>
    where
        I: ExactSizeIterator<Item = (u32, &'a IndexedBankSource)>,
    {
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())?;
        let invalid =
            || context.metadata_error(format_args!("addressable model source directory differs"));
        context.charge_metadata(
            shared_bytes::<Sources>()
                .and_then(|n| n.checked_add(size_of::<(Self, I, Result<Self, Error>)>()))
                .and_then(|n| {
                    n.checked_add(safemlx::PreparedInputRuntime::inspection_alias_control_bytes())
                })
                .ok_or_else(invalid)?,
        )?;
        let mut owned = context.metadata_vec(banks.len())?;
        for (key, source) in banks {
            if owned.iter().any(|(prior, _)| *prior == key) {
                return Err(invalid());
            }
            owned.push((key, source.clone()));
        }
        Ok(Self(Some(Rc::new(Sources {
            banks: owned,
            quotes: RefCell::new(context.metadata_vec(0)?),
            mechanism,
            runtime: runtime.inspection_alias(),
            pool: pool.cloned(),
            funding: funding.clone(),
        }))))
    }
    fn inner(&self) -> &Sources {
        self.0.as_deref().expect("live addressable sources")
    }
    pub(crate) fn funding(&self) -> &HostMetadataFunding {
        &self.inner().funding
    }
    pub(crate) fn runtime(&self) -> &safemlx::PreparedInputRuntime {
        &self.inner().runtime
    }
    pub(crate) fn bank(&self, id: u32) -> Option<&IndexedBankSource> {
        self.inner()
            .banks
            .iter()
            .find(|(key, _)| *key == id)
            .map(|(_, source)| source)
    }
    pub(crate) fn mechanism(&self) -> ResidentExecutionMechanisms {
        self.inner().mechanism
    }
    pub(crate) fn observation_layout(&self,source:WorkspaceAddressableRegionView<'_>,inputs:&[WorkspaceLayout],
        context:&WorkspaceContext)->Result<WorkspaceAddressableObservationLayout,Error>{
        if context.metadata_funding().is_none_or(|funding|!funding.same_account(self.funding())){
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let bank=self.bank(source.bank).ok_or(WorkspaceMetadataError::Unqualified)?;
        super::observation::inspect(bank,source,inputs,self.mechanism(),self.funding())
    }
    pub(crate) fn quote(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<AddressableQuoteRef, Error> {
        let inner = self.inner();
        let context =
            WorkspaceContext::new_with_metadata_funding(inner.mechanism, inner.funding.clone())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "addressable occurrence lacks its exact model-bound source"
            ))
        };
        context.charge_metadata(size_of::<(
            AddressableQuoteRef,
            Result<AddressableQuoteRef, Error>,
            &Self,
            WorkspaceOperationView<'_>,
        )>())?;
        if let Some(value) = inner
            .quotes
            .try_borrow()
            .map_err(|_| invalid())?
            .iter()
            .find(|quote| !quote.is_local() && quote.matches(operation))
        {
            return Ok(value.clone());
        }
        let WorkspaceOperationKindView::AddressableRegion(declaration) = operation.kind else {
            return Err(invalid());
        };
        let source = &inner
            .banks
            .iter()
            .find(|(key, _)| *key == declaration.as_view().bank)
            .ok_or_else(invalid)?
            .1;
        let quote = AddressableQuote::prepare(
            source,
            operation,
            inner.mechanism,
            &inner.runtime,
            inner.pool.as_ref(),
            &inner.funding,
        )?;
        let mut quotes = inner.quotes.try_borrow_mut().map_err(|_| invalid())?;
        context.reserve_metadata_vec(&mut quotes, 1)?;
        context.charge_metadata(shared_bytes::<AddressableQuote>().ok_or_else(invalid)?)?;
        let quote = AddressableQuoteRef(Some(Rc::new(quote)), None);
        quotes.push(quote.clone());
        Ok(quote)
    }
    /// Each completed receive-row count has an exact quote from the same
    /// physical bank. These are alternatives, not additional invocations.
    pub(crate) fn local_quote(&self, operation: WorkspaceOperationView<'_>) -> Result<AddressableQuoteRef, Error> {
        let context = WorkspaceContext::new_with_metadata_funding(MlxAddressableWorkspaceMechanisms::new(self.mechanism(), self.clone()), self.funding().clone())?;
        let invalid = || context.metadata_error(format_args!("local indexed source differs from its retained expert region"));
        context.charge_metadata(size_of::<(WorkspaceContext, WorkspaceOperationView<'_>,
            Result<AddressableQuoteRef, Error>, Vec<AddressableQuoteRef>, Vec<WorkspaceTensor>,
            WorkspaceTraceReport, Option<AddressableQuoteRef>, usize,
            SpeculativeNumericalRecipe, LocalEnvelope)>() )?;
        let WorkspaceOperationKindView::ExpertRegion(region) = operation.kind else { return Err(invalid()); };
        let view = region.as_view();
        view.validate()?;
        let shape = ExpertRegionInputShape::inspect(operation.inputs.get(0).ok_or_else(invalid)?.shape(),
            operation.inputs.get(1).ok_or_else(invalid)?.shape())?;
        if usize::try_from(shape.rows).ok() != Some(view.source_rows)
            || usize::try_from(shape.routes).ok() != Some(view.routes_per_row)
            || shape.width != view.kernel.dimensions().0 { return Err(invalid()); }
        let source = view.addressable.ok_or_else(invalid)?;
        let maximum = view.maximum_received_rows().ok_or_else(invalid)?;
        if maximum == 0 || source.chunks.rows != maximum || operation.inputs.len() != 4 { return Err(invalid()); }
        // The last quote has maximum rows and authenticates all source/layout
        // fields. Cache identity includes the complete initial operation.
        let build = |rows: usize| -> Result<WorkspaceTraceReport, Error> {
            let mut source = source;
            source.chunks.rows = rows;
            let mut inputs = context.metadata_vec(4)?;
            for (index, input) in operation.inputs.iter().enumerate() {
                let shape = [i32::try_from(rows).map_err(|_| invalid())?, if index == 0 { source.kernel.dimensions().0 } else { 1 }];
                inputs.push(WorkspaceTensor::existing(context.layout(&shape, if index == 1 { WorkspaceDtype::Int32 } else { input.dtype() })?
                    .with_representation(input.representation()), &context)?);
            }
            let routes = eredu_nn::GroupSelection::new(inputs[1].clone(), inputs[2].clone(), inputs[3].clone());
            let mut observe = |_: eredu_nn::workspace::WorkspaceAddressableObservationView<'_>| {
                let value = region.observation().ok_or_else(invalid)?;
                Ok(eredu_nn::workspace::WorkspaceAddressableObservationSource {
                    before: value.before, after: value.after, unit_dtype: value.unit_dtype,
                })
            };
            context.begin_span();
            let output = eredu_nn::workspace::record_addressable_region_with_observation(source, &inputs[0], &routes, &context,
                if region.observation().is_some() { Some(&mut observe) } else { None })?;
            let (value, bias) = output.into_parts();
            let mut outputs = context.metadata_vec(1 + usize::from(bias.is_some()))?;
            outputs.push(value); outputs.extend(bias);
            context.finish_report(&outputs)
        };
        // The source-aware mechanism is needed for observed physical layouts.
        // Its ordinary facts do not mint another accepted source.
        let maximum_report = build(maximum)?;
        let maximum_operation = maximum_report.operations.last().ok_or_else(invalid)?.as_view();
        if maximum_report.operations.len() != 1 { return Err(invalid()); }
        if let Some(quote) = self.inner().quotes.try_borrow().map_err(|_| invalid())?.iter()
            .find(|quote| quote.is_local() && quote.matches(maximum_operation)) { return Ok(quote.clone()); }
        let maximum_quote = self.quote(maximum_operation)?;
        let mut population = AddressableNumericalPopulation::from_recipe(maximum_quote.equation, false).ok_or_else(invalid)?;
        let residual_host = |quote: &AddressableQuote| -> Result<u64, Error> {
            let native = crate::backend::submission_recovery::addressable::control_bytes(quote)
                .map_err(|cause| context.metadata_source(cause))?;
            quote.host_bytes.checked_sub(u64::try_from(native).map_err(|_| invalid())?).ok_or_else(invalid)
        };
        let mut host_bytes = residual_host(&maximum_quote)?;
        let mut capture_publications = maximum_quote.capture_publications;
        let candidates = super::row_candidates::local_row_candidates(source.kernel, maximum, source.chunks.chunk_rows, &context)?;
        for count in candidates {
            let report = build(count)?;
            if report.operations.len() != 1 { return Err(invalid()); }
            let quote = self.quote(report.operations[0].as_view())?;
            population = population.union(AddressableNumericalPopulation::from_recipe(quote.equation, false)
                .ok_or_else(invalid)?).ok_or_else(invalid)?;
            host_bytes = host_bytes.max(residual_host(&quote)?);
            capture_publications = capture_publications.max(quote.capture_publications);
        }
        // Restore the equation's one role-exit completion, then join the
        // separately scoped copies from the authentic maximum-row plan. Its
        // immutable window is shared by all narrowed plans; only their number
        // of chunk occurrences decreases. Reimporting a copy-expanded graph
        // would mistake source constructors for numerical Eval entries.
        let equation = population.finish(maximum_quote.outputs.len(), 1, &context)?;
        let numerical = equation.with_indexed_source(&maximum_quote.residency, &context)?;
        let backing = safemlx::OriginalBufferBudget::population_layout(self.runtime(),
            usize::try_from(numerical.storage.mutable_bytes()).map_err(|_| invalid())?, numerical.storage.maximum_births())
            .map_err(|cause| context.metadata_source(cause))?;
        let capacity = BoundaryStageCapacity { graph: numerical.graph_capacity, records: numerical.record_capacity, backing: backing.capacity() };
        let native = crate::backend::submission_recovery::addressable::envelope_control_bytes(&maximum_quote, numerical, capacity)
            .map_err(|cause| context.metadata_source(cause))?;
        host_bytes = host_bytes.checked_add(u64::try_from(native).map_err(|_| invalid())?).ok_or_else(invalid)?;
        let envelope = LocalEnvelope { equation, numerical, capacity, host_bytes, capture_publications };
        context.charge_metadata(shared_bytes::<LocalEnvelope>().ok_or_else(invalid)?)?;
        let quote = AddressableQuoteRef(maximum_quote.0.clone(), Some(Rc::new(envelope)));
        let mut quotes = self.inner().quotes.try_borrow_mut().map_err(|_| invalid())?;
        context.reserve_metadata_vec(&mut quotes, 1)?;
        quotes.push(quote.clone());
        Ok(quote)
    }
    pub(crate) fn invocation(
        &self,
        operations: &[WorkspaceOperation],
    ) -> Result<AddressableInvocation, Error> {
        let context =
            WorkspaceContext::new_with_metadata_funding(self.mechanism(), self.funding().clone())?;
        context.charge_metadata(size_of::<(
            AddressableInvocation,
            Result<AddressableInvocation, Error>,
        )>())?;
        let count = operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::AddressableRegion(_)) || matches!(&op.kind, WorkspaceOperationKind::ExpertRegion(region) if region.as_view().addressable.is_some()))
            .count();
        let mut occurrences = context.metadata_vec(count)?;
        for (ordinal, op) in operations.iter().enumerate() {
            if matches!(op.kind, WorkspaceOperationKind::AddressableRegion(_)) {
                occurrences.push((ordinal, self.quote(op.as_view())?));
            } else if matches!(&op.kind, WorkspaceOperationKind::ExpertRegion(region) if region.as_view().addressable.is_some()) {
                occurrences.push((ordinal, self.local_quote(op.as_view())?));
            }
        }
        Ok(AddressableInvocation {
            occurrences,
            source: self.clone(),
        })
    }
}
#[derive(Debug)]
pub(crate) struct AddressableInvocation {
    occurrences: Vec<(usize, AddressableQuoteRef)>,
    source: AddressableSources,
}
impl AddressableInvocation {
    pub(crate) fn try_clone_for_retention(&self)->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(self.source.mechanism(),self.source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,Result<Self,Error>)>())?;
        let mut occurrences=context.metadata_vec(self.occurrences.len())?;
        occurrences.extend(self.occurrences.iter().map(|(ordinal,quote)|(*ordinal,quote.clone())));
        Ok(Self{occurrences,source:self.source.clone()})
    }

    pub(crate) fn occurrences(&self) -> &[(usize, AddressableQuoteRef)] {
        &self.occurrences
    }
    pub(crate) fn source(&self) -> &AddressableSources {
        &self.source
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
