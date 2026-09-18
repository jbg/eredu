//! One actual resident state source and its authenticated constructor tables.
use super::*;
use crate::backend::runtime::cache::state::{MlxHybridState, MlxKeyValueState};
use crate::composition::mlx::replicated_text::ResidentResetProfile;
use eredu_runtime::working_memory::{
    OriginalResidentResetSource, OriginalTextControlGuard, ResidentTableResetState,
    ResidentResetSource, WorkingMemoryFundingScope,
};

enum Source<'a> {
    KeyValue(ResidentResetSource<'a, MlxKeyValueState>),
    Hybrid(ResidentResetSource<'a, MlxHybridState>),
}
impl<'a> Source<'a> {
    fn current(session: &'a MlxModelSession) -> Result<Option<Self>, WorkingMemoryError> {
        let model = session.payload.model.erased();
        match model.resident_reset_profile() {
            Some(ResidentResetProfile::KeyValue) => {
                model.resident_reset_source().map(Self::KeyValue).map(Some)
            }
            Some(ResidentResetProfile::Hybrid) => model
                .resident_hybrid_reset_source()
                .map(Self::Hybrid)
                .map(Some),
            None => Ok(None),
        }
    }
    fn metadata(&self) -> &eredu_runtime::HostSlotMetadata {
        match self {
            Self::KeyValue(source) => source.state().resident_reset_layers().metadata(),
            Self::Hybrid(source) => source.state().resident_reset_layers().metadata(),
        }
    }
    fn same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::KeyValue(left), Self::KeyValue(right)) => left.same_source(right),
            (Self::Hybrid(left), Self::Hybrid(right)) => left.same_source(right),
            _ => false,
        }
    }
}
pub(super) struct Plan<'a> {
    source: Source<'a>,
    original: OriginalResidentResetSource,
}
impl<'a> Plan<'a> {
    pub(super) fn prepare(session: &'a MlxModelSession) -> Result<Option<Self>, Error> {
        let source = match Source::current(session) {
            Ok(Some(source)) => source,
            Ok(None) => return Ok(None),
            // Other ordinary state representations keep their existing path.
            // The complete idle inventory separately rejects any original table.
            Err(WorkingMemoryError::UnknownBound) => return Ok(None),
            Err(error) => return Err(memory(error)),
        };
        let table = source.metadata();
        let kind = session
            .payload
            .memory_pool
            .classify_host_slot_source(table)
            .map_err(memory)?;
        let Some(original) = kind.into_original() else {
            return Ok(None);
        };
        Ok(Some(Self { source, original }))
    }
    pub(super) fn source(&self) -> &OriginalResidentResetSource {
        &self.original
    }
    pub(super) fn finish(
        self,
        session: &MlxModelSession,
        request: &InferenceRequest,
        controls: OriginalTextControlGuard,
    ) -> Result<Owned, Error> {
        let actual = Source::current(session)
            .map_err(memory)?
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
        if !self.source.same_source(&actual) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        controls
            .validate_reservation(request.memory_reservation().ok_or_else(unknown)?)
            .map_err(memory)?;
        session
            .payload
            .memory_pool
            .pin_original_reset_slots(self.original.metadata())
            .map_err(memory)?;
        Ok(Owned {
            source: self.original,
            controls,
        })
    }
}

#[derive(Debug)]
pub(super) struct Owned {
    source: OriginalResidentResetSource,
    // Last: historical quote Q survives the source even after other banks move.
    controls: OriginalTextControlGuard,
}
impl Owned {
    #[cfg(test)]
    pub(super) fn source(&self) -> &OriginalResidentResetSource {
        &self.source
    }
    pub(super) fn model_source(
        &self,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<OriginalResidentResetSource, Error> {
        self.controls.validate_native_scope(scope).map_err(memory)?;
        scope
            .pool()
            .pin_original_reset_slots(self.source.metadata())
            .map_err(memory)
    }
}

pub(super) fn admission_controls() -> Option<u64> {
    // Owned is stored in the actual quoted payload and counted there already.
    [
        size_of::<Source<'static>>(),
        size_of::<Result<Option<Source<'static>>, WorkingMemoryError>>(),
        size_of::<Result<ResidentResetSource<'static, MlxKeyValueState>, WorkingMemoryError>>(),
        size_of::<Result<ResidentResetSource<'static, MlxHybridState>, WorkingMemoryError>>(),
        size_of::<Plan<'static>>(),
        size_of::<Option<Plan<'static>>>(),
        size_of::<Result<Option<Plan<'static>>, Error>>(),
        size_of::<Result<Owned, Error>>(),
        size_of::<Result<Option<Owned>, Error>>(),
        size_of::<eredu_runtime::working_memory::HostSlotSource>(),
        size_of::<Result<eredu_runtime::working_memory::HostSlotSource, WorkingMemoryError>>(),
        size_of::<Result<OriginalResidentResetSource, WorkingMemoryError>>(),
        size_of::<Option<OriginalResidentResetSource>>(),
        size_of::<Option<OriginalTextControlGuard>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
}
pub(super) fn work_controls() -> Option<u64> {
    // The actual Work field/Rc layout is counted by the existing owner fact.
    [
        size_of::<Result<OriginalResidentResetSource, Error>>(),
        size_of::<Result<Option<OriginalResidentResetSource>, Error>>(),
        size_of::<Result<OriginalResidentResetSource, WorkingMemoryError>>(),
        size_of::<WorkingMemoryPool>(),
        size_of::<Result<WorkingMemoryPool, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Option<super::super::text_funding::FundedWorkOwner>>(),
        size_of::<Option<super::super::SessionPayloadOwner>>(),
        size_of::<eredu_runtime::working_memory::HostSlotSource>(),
        size_of::<Result<eredu_runtime::working_memory::HostSlotSource, WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
}
