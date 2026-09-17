//! Explicit model-bound immutable bank sources and their paid quote directory.
use super::*;
use crate::backend::runtime::residency::parameter_bank::IndexedBankSource;
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::{alloc::Layout, cell::RefCell, ops::Deref, rc::Rc};

pub(crate) struct AddressableQuoteRef(Option<Rc<AddressableQuote>>);
impl AddressableQuoteRef {
    pub(crate) fn same_quote(&self,other:&Self)->bool {
        match (&self.0,&other.0){(Some(a),Some(b))=>Rc::ptr_eq(a,b),_=>false}
    }
}
impl Clone for AddressableQuoteRef {
    fn clone(&self) -> Self {
        Self(self.0.clone())
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
    funding: WorkspaceMetadataFunding,
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
impl AddressableSources {
    pub(crate) fn new<'a, I>(
        banks: I,
        mechanism: ResidentExecutionMechanisms,
        runtime: &safemlx::PreparedInputRuntime,
        pool: Option<&WorkingMemoryPool>,
        funding: &WorkspaceMetadataFunding,
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
    pub(crate) fn funding(&self) -> &WorkspaceMetadataFunding {
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
            .find(|quote| quote.matches(operation))
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
        let quote = AddressableQuoteRef(Some(Rc::new(quote)));
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
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::AddressableRegion(_)))
            .count();
        let mut occurrences = context.metadata_vec(count)?;
        for (ordinal, op) in operations.iter().enumerate() {
            if matches!(op.kind, WorkspaceOperationKind::AddressableRegion(_)) {
                occurrences.push((ordinal, self.quote(op.as_view())?));
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
