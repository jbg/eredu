//! Borrowed active scalar/readout and the actual shared-driver completion point.
use super::super::workspace::quote::PreparedPredictionIo;
use super::*;
use crate::composition::mlx::speculative::embedded_native::ActiveEmbeddedNativeInvocation;
use crate::composition::mlx::speculative::{
    EmbeddedNumericalInvocation, OriginalSpeculativeNumericalSources, PendingModelLogits,
};
use eredu_runtime::working_memory::{
    OriginalEmbeddedSpeculativeRole, WorkingMemoryError,
};
use safemlx::{
    Array, OriginalScopeObserver,
};
use std::{
    cell::RefCell,
    mem::size_of,
};

use crate::backend::submission_recovery::prefill::nested::NestedRootCompletion;

/// Prediction points preserve their semantic identity over the same native worker.
pub(super) struct StateCompletion {
    point: PredictionCompletionPoint,
    roots: NestedRootCompletion,
}
impl StateCompletion {
    pub(super) fn control_bytes(roots:usize, validations:usize)->Option<usize> {
        NestedRootCompletion::control_bytes(roots,validations)?
            .checked_add(size_of::<Self>())?.checked_add(size_of::<PredictionCompletionPoint>())
    }
    pub(super) fn prepare(roots:usize,validations:usize,point:PredictionCompletionPoint,
        role:&OriginalEmbeddedSpeculativeRole,sources:&OriginalSpeculativeNumericalSources)->Result<Self,Error> {
        sources.metadata_funding().reserve_metadata(size_of::<Self>()).map_err(Error::WorkspacePlanning)?;
        NestedRootCompletion::prepare(roots,validations,role.budget_custody(),sources.metadata_funding())
            .map(|roots|Self{point,roots}).map_err(|cause|sources.retain_error(cause))
    }
    fn complete<'values>(&mut self,point:PredictionCompletionPoint,
        values:&mut dyn Iterator<Item=&'values MlxTensor>,observer:&OriginalScopeObserver,stream:&Stream)->Result<(),Error> {
        if point!=self.point { return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)); }
        self.roots.complete(|visitor|{ for value in values { visitor(value); } Ok(()) },observer,stream)
    }
    pub(super) fn validate_complete(&self)->Result<(),Error>{self.roots.validate_complete()}
}

pub(super) struct Binding<'a> {
    pub(super) active: &'a ActiveEmbeddedNativeInvocation<'a>,
    pub(super) sources: &'a OriginalSpeculativeNumericalSources,
    pub(super) io: Option<&'a RefCell<PreparedPredictionIo>>,
    pub(super) logits: Option<&'a PendingModelLogits>,
    pub(super) completion: Option<&'a RefCell<StateCompletion>>,
}
impl std::fmt::Debug for Binding<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EmbeddedPredictionInvocation")
    }
}
impl EmbeddedNumericalInvocation for Binding<'_> {
    fn sources(&self) -> &OriginalSpeculativeNumericalSources {
        self.sources
    }
    fn validate_scope(&self, stream: &Stream) -> Result<(), Error> {
        self.active.validate_equation_scope(stream)
    }
    fn token(&self, id: u32, stream: &Stream) -> Result<MlxTensor, Error> {
        self.validate_scope(stream)?;
        let io = self
            .io
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let mut io = io
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        io.token(id, self.active.role(), self.active.observer())
            .map(MlxTensor::from_array)
    }
    fn logits_row(
        &self,
        value: &Array,
        row: usize,
        stream: &Stream,
    ) -> Result<IndependentLogits, Error> {
        self.validate_scope(stream)?;
        let logits = self
            .logits
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let array = crate::composition::mlx::prepared_speculative::embedded_logits::row(
            value, row, stream,
        )?;
        logits.fill(
            array,
            self.active.role(),
            self.active.budget(),
            self.active.observer(),
        )
    }
    fn complete_state<'values>(
        &self,
        point: PredictionCompletionPoint,
        values: &mut dyn Iterator<Item = &'values MlxTensor>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.validate_scope(stream)?;
        self.completion
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .complete(point, values, self.active.observer(), stream)
    }
}
