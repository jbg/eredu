//! Semantic state joins the same independently admitted cursor copy transaction.
use super::*;
use crate::working_memory::PreparedSemanticState;
use crate::execution_control::PreparedTextHostJournal;
use eredu_core::{HostPreparationAuthority, SemanticStateOwner, SpeculativeOutputError};

struct SemanticSequenceCopy<'a, J> {
    sequence: RetainedSequenceHostCopy<'a>,
    semantic: &'a PreparedSemanticState,
    journal: J,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SemanticCopyFailure {
    #[source]
    cause: SpeculativeOutputError,
    host: HostPreparationAuthority,
}
type Copied<J> = (RetainedGenerationSequence, SemanticStateOwner, <J as PreparedTextHostJournal>::Copied);

impl WorkingMemoryPool {
    #[cfg(test)]
    pub(in crate::working_memory) fn prepare_semantic_resume_provider_for_test<
        'a,
        C: 'static,
        E,
        J: PreparedTextHostJournal + 'a,
    >(
        &self,
        sequence: &'a RetainedGenerationSequence,
        semantic: &'a SemanticStateOwner,
        capacity: u64,
        journal: J,
    ) -> Result<impl PreparedTextHostCopy<Copied = Copied<J>> + 'a, WorkingMemoryError> {
        let controls = crate::execution_control::PendingSnapshotResumeRetention::control_bytes()
            .ok_or(WorkingMemoryError::Overflow)?;
        self.prepare_generation_host_copy_plan::<C, E>(sequence, capacity)?
            .with_controls(controls, 0)?
            .with_semantic(semantic, journal)
    }
    /// Borrows the actual cursor and semantic parser for one snapshot copy.
    /// Both destinations use the sequence provider's independently admitted
    /// account; the existing snapshot transaction pays cumulative copy usage.
    pub fn prepare_semantic_generation_snapshot_host_copy<'a, C: 'static, E, B, D, J>(
        &self,
        sequence: &'a RetainedGenerationSequence,
        semantic: &'a SemanticStateOwner,
        capacity: u64,
        native_preparation_bytes: u64,
        journal: J,
    ) -> Result<impl PreparedTextHostCopy<Copied = Copied<J>> + 'a, WorkingMemoryError>
    where
        B: crate::execution_control::TextSnapshotBackend,
        D: crate::execution_control::SnapshotTokenController,
        J: PreparedTextHostJournal + 'a,
    {
        let controls = crate::execution_control::TextContinuationSnapshot::<B, D>
            ::original_capture_control_bytes::<(C, SemanticStateOwner, J::Copied)>()
            .ok_or(WorkingMemoryError::Overflow)?;
        self.prepare_generation_host_copy_plan::<C, E>(sequence, capacity)?
            .with_controls(controls, native_preparation_bytes)?
            .with_semantic(semantic, journal)
    }

    /// The same semantic/cursor copy for a freshly admitted restore or branch.
    /// Every escaping alias retains the shared nonrefundable resume lease.
    pub fn prepare_semantic_generation_resume_host_copy<'a, C: 'static, E, B, D, J>(
        &self, sequence: &'a RetainedGenerationSequence, semantic: &'a SemanticStateOwner,
        capacity: u64, native_preparation_bytes: u64, journal: J,
    ) -> Result<impl PreparedTextHostCopy<Copied = Copied<J>> + 'a, WorkingMemoryError>
    where B: crate::execution_control::TextSnapshotBackend + eredu_core::TextResumeBackend<
            ResumeSource = <B as crate::execution_control::TextSnapshotBackend>::SavedTextComponents>,
          D: crate::execution_control::SnapshotTokenController,
          J: PreparedTextHostJournal + 'a,
    {
        let controls = crate::execution_control::TextContinuationSnapshot::<B, D>
            ::original_resume_control_bytes::<(C, SemanticStateOwner, J::Copied)>()
            .ok_or(WorkingMemoryError::Overflow)?;
        self.prepare_generation_host_copy_plan::<C, E>(sequence, capacity)?
            .with_controls(controls, native_preparation_bytes)?
            .with_semantic(semantic, journal)
    }
}
impl<'a> RetainedSequenceHostCopy<'a> {
    fn with_semantic<J: PreparedTextHostJournal>(
        mut self,
        semantic: &'a SemanticStateOwner,
        journal: J,
    ) -> Result<SemanticSequenceCopy<'a, J>, WorkingMemoryError> {
        let semantic = semantic
            .prepared_source()
            .and_then(|source| source.downcast_ref::<PreparedSemanticState>())
            .ok_or(WorkingMemoryError::UnknownBound)?;
        semantic.preparation().validate(
            self.source.authority.pool(),
            self.source.authority.source().execution(),
        )?;
        let parts = [
            semantic
                .copy_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<SemanticSequenceCopy<'_, J>>(),
            size_of::<Copied<J>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<J>(),
            size_of::<J::Copied>(),
            size_of::<Result<J::Copied, TextHostCopyError>>(),
            size_of::<SemanticCopyFailure>(),
            size_of::<Result<Copied<J>, TextHostCopyError>>(),
            size_of::<Result<(Copied<J>, HostPreparationAuthority), TextHostCopyError>>(),
            size_of::<Result<SemanticStateOwner, SpeculativeOutputError>>(),
            BackendFailure::source_retention_peak_bytes::<SemanticCopyFailure>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let journal_preparation = journal.preparation_bytes()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let journal_logical = journal.storage_bytes().ok_or(WorkingMemoryError::UnknownBound)?;
        self.bytes = self.bytes.checked_add(bytes)
            .and_then(|bytes| bytes.checked_add(journal_preparation))
            .ok_or(WorkingMemoryError::Overflow)?;
        self.logical_bytes = self.logical_bytes.checked_add(bytes)
            .and_then(|bytes| bytes.checked_add(journal_logical))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(SemanticSequenceCopy { sequence: self, semantic, journal })
    }
}
impl<J: PreparedTextHostJournal> SemanticSequenceCopy<'_, J> {
    fn join(
        semantic: &PreparedSemanticState,
        sequence: RetainedGenerationSequence,
        host: HostPreparationAuthority,
        journal: J,
        retained: u64,
    ) -> Result<(Copied<J>, HostPreparationAuthority), TextHostCopyError> {
        let copied = semantic.copy(host.clone(), true).map_err(|cause| {
            TextHostCopyError::Source(BackendFailure::from_error(SemanticCopyFailure {
                cause,
                host: host.clone(),
            }))
        })?;
        let journal = journal.copy(retained, &host)?;
        Ok(((sequence, copied, journal), host))
    }
}
impl<J: PreparedTextHostJournal> PreparedTextHostCopy for SemanticSequenceCopy<'_, J> {
    type Copied = Copied<J>;
    fn storage_bytes(&self) -> Option<u64> {
        self.sequence.storage_bytes()
    }
    fn original_control_bytes(&self) -> Option<usize> {
        self.sequence.original_control_bytes()
    }
    fn original_preparation_bytes(&self) -> Option<u64> {
        self.sequence.original_preparation_bytes()
    }
    fn copy(self, retained: u64) -> Result<Copied<J>, TextHostCopyError> {
        self.copy_original(retained).map(|(copied, _)| copied)
    }
    fn copy_original(
        self,
        retained: u64,
    ) -> Result<(Copied<J>, HostPreparationAuthority), TextHostCopyError> {
        let (sequence, host) = self.sequence.copy_original(retained)?;
        Self::join(self.semantic, sequence, host, self.journal, retained)
    }
    fn copy_original_resume(
        self,
        retained: u64,
        reservation: crate::execution_control::PendingSnapshotResumeRetention,
    ) -> Result<(Copied<J>, HostPreparationAuthority), TextHostCopyError> {
        let (sequence, host) = self.sequence.copy_original_resume(retained, reservation)?;
        Self::join(self.semantic, sequence, host, self.journal, retained)
    }
}
