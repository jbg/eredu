//! Addressable source activation around the existing embedded/external handler.
//! The selected neural bank retains authority; this payload owns only its exact
//! accepted source partition and a lexical access loan from that live bank.
use super::*;
use crate::backend::{
    nn::workspace::ResidentSpanRecipe,
    runtime::execution::generic::{OriginalSelectedResidencyAccess, SpeculativeNeuralOwner},
    submission_recovery::addressable::{
        AddressableExecutionRow, SpeculativeAddressableSources, SpeculativeAddressableSpan,
    },
};
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, OriginalHostSourceBank, WorkingMemoryError,
};
fn identity() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}
fn total(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(size_of_val(parts), usize::checked_add)
}
pub(crate) struct AddressableModelSource {
    pending: RefCell<Option<SpeculativeAddressableSpan>>,
    access: RefCell<Option<OriginalSelectedResidencyAccess>>,
    funding: HostMetadataFunding,
}
struct Call<'a, 'r, W, Q, R: ModelRole, T, F> {
    run: Option<F>,
    work: &'a mut W,
    payload: &'a Q,
    active: &'a ActiveEmbeddedNativeInvocation<'r, R>,
    output: Option<Result<T, Error>>,
}
impl<W, Q, R: ModelRole, T, F> Call<'_, '_, W, Q, R, T, F>
where
    F: FnOnce(&mut W, &Q, &ActiveEmbeddedNativeInvocation<'_, R>) -> Result<T, Error>,
{
    fn invoke(&mut self) -> Result<bool, Error> {
        let run = self.run.take().ok_or_else(identity)?;
        let result = run(self.work, self.payload, self.active);
        let success = result.is_ok();
        self.output = Some(result);
        Ok(success)
    }
}
impl AddressableModelSource {
    pub(crate) fn prepare(
        records: &[ResidentSpanRecipe],
        target: Option<HostSourceConstructionFacts>,
        funding: &HostMetadataFunding,
    ) -> Result<(Option<Self>, Option<HostSourceConstructionFacts>, u64), Error> {
        if records.len() != 1 {
            return Err(identity());
        }
        let Some(mut source) = SpeculativeAddressableSources::prepare(records, target, funding)?
        else {
            return Ok((None, target, 0));
        };
        funding
            .reserve_metadata(
                total(&[
                    size_of::<Self>(),
                    size_of::<SpeculativeAddressableSources>(),
                    size_of::<
                        Result<(Option<Self>, Option<HostSourceConstructionFacts>, u64), Error>,
                    >(),
                    size_of::<eredu_runtime::working_memory::SpeculativeHostSourceSpans>(),
                    size_of::<Option<SpeculativeAddressableSpan>>(),
                    size_of::<(
                        &[ResidentSpanRecipe],
                        Option<HostSourceConstructionFacts>,
                        &HostMetadataFunding,
                    )>(),
                    size_of::<u64>(),
                ])
                .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let facts = source.take_facts()?;
        if facts.as_slice().len() != 1 {
            return Err(identity());
        }
        let facts = facts.as_slice()[0].ok_or_else(identity)?;
        let controls = source
            .controls()
            .checked_add(
                u64::try_from(Self::binding_controls().ok_or_else(overflow)?)
                    .map_err(|_| overflow())?,
            )
            .ok_or_else(overflow)?;
        Ok((
            Some(Self {
                pending: RefCell::new(Some(source.take(0)?.ok_or_else(identity)?)),
                access: RefCell::new(None),
                funding: funding.clone(),
            }),
            Some(facts),
            controls,
        ))
    }
    fn binding_controls() -> Option<usize> {
        total(&[
            size_of::<&Self>(),
            size_of::<Option<OriginalHostSourceBank>>(),
            size_of::<Result<Option<OriginalHostSourceBank>, Error>>(),
            size_of::<std::cell::RefMut<'_, Option<SpeculativeAddressableSpan>>>(),
            size_of::<std::cell::RefMut<'_, Option<OriginalSelectedResidencyAccess>>>(),
            size_of::<OriginalSelectedResidencyAccess>(),
            size_of::<&SpeculativeNeuralOwner>(),
            size_of::<Result<(), Error>>(),
            size_of::<&Self>(),
            size_of::<
                Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
            >(),
        ])
    }
    pub(crate) fn accept(
        &self,
        root: Option<OriginalHostSourceBank>,
    ) -> Result<Option<OriginalHostSourceBank>, Error> {
        self.funding
            .reserve_metadata(Self::binding_controls().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        self.pending
            .try_borrow_mut()
            .map_err(|_| identity())?
            .as_mut()
            .ok_or_else(identity)?
            .accept(root)
    }
    pub(crate) fn bind(&self, bank: &SpeculativeNeuralOwner) -> Result<(), Error> {
        let mut slot = self.access.try_borrow_mut().map_err(|_| identity())?;
        if slot.is_some() {
            return Err(identity());
        }
        *slot = Some(bank.selected_residency_access()?);
        Ok(())
    }
    pub(crate) fn callback_controls<W, Q, R: ModelRole, T, F>(_: &F) -> Option<u64>
    where
        F: FnOnce(&mut W, &Q, &ActiveEmbeddedNativeInvocation<'_, R>) -> Result<T, Error>,
    {
        Self::call_controls::<W, Q, R, T, F>()
    }
    fn call_controls<W, Q, R: ModelRole, T, F>() -> Option<u64> {
        u64::try_from(total(&[
            size_of::<Call<'_, '_, W, Q, R, T, F>>(),
            size_of::<&mut Call<'_, '_, W, Q, R, T, F>>(),
            size_of::<(&Self, &mut W, &Q, &ActiveEmbeddedNativeInvocation<'_, R>, F)>(),
            size_of::<SpeculativeAddressableSpan>(),
            size_of::<Option<SpeculativeAddressableSpan>>(),
            size_of::<OriginalSelectedResidencyAccess>(),
            size_of::<Option<OriginalSelectedResidencyAccess>>(),
            size_of::<std::cell::RefMut<'_, Option<SpeculativeAddressableSpan>>>(),
            size_of::<std::cell::RefMut<'_, Option<OriginalSelectedResidencyAccess>>>(),
            size_of::<OriginalBufferBudget>(),
            size_of::<AddressableExecutionRow>(),
            size_of::<Result<AddressableExecutionRow, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<Option<Result<T, Error>>>(),
            size_of::<Result<T, Error>>(),
            size_of::<bool>(),
        ])?)
        .ok()
    }
    pub(crate) fn run<W, Q, R: ModelRole, T, F>(
        &self,
        work: &mut W,
        payload: &Q,
        active: &ActiveEmbeddedNativeInvocation<'_, R>,
        run: F,
    ) -> Result<T, Error>
    where
        F: FnOnce(&mut W, &Q, &ActiveEmbeddedNativeInvocation<'_, R>) -> Result<T, Error>,
    {
        self.funding
            .reserve_metadata(
                usize::try_from(Self::call_controls::<W, Q, R, T, F>().ok_or_else(overflow)?)
                    .map_err(|_| overflow())?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let source = self
            .pending
            .try_borrow_mut()
            .map_err(|_| identity())?
            .take()
            .ok_or_else(identity)?;
        active.bind_initialized(source.take_initialized()?)?;
        if !source.has_addressable() {
            return run(work, payload, active);
        }
        let access = self
            .access
            .try_borrow_mut()
            .map_err(|_| identity())?
            .take()
            .ok_or_else(identity)?;
        let row = source.activate(
            access,
            active.budget().clone(),
            active.observer(),
            None,
            &self.funding,
        )?;
        let mut call = Call {
            run: Some(run),
            work,
            payload,
            active,
            output: None,
        };
        let status = row.during(&mut || call.invoke());
        match call.output.take() {
            Some(Err(cause)) => Err(cause),
            Some(Ok(value)) => status.map(|()| value),
            None => Err(status.err().unwrap_or_else(identity)),
        }
    }
}
