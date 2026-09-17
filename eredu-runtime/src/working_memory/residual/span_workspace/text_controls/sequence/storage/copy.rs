//! One independently admitted copy of the actual retained generation provider.
use super::*;
use crate::execution_control::{PreparedTextHostCopy, TextHostCopyError};
use eredu_core::{RetainedGenerationSequenceCopy, RetainedSequenceCopyMismatch};

pub(in crate::working_memory) struct RetainedSequenceHostCopy<'a> {
    source: &'a Provider,
    sequence: RetainedGenerationSequenceCopy<'a>,
    bytes: u64,
    capacity: u64,
    snapshot_controls: Option<usize>,
    native_preparation_bytes: Option<u64>,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
    #[error(transparent)]
    Decoder(#[from] decoder::DecoderStateCopyError),
    #[error(transparent)]
    Mismatch(#[from] RetainedSequenceCopyMismatch),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct CopyFailure {
    #[source]
    cause: Cause,
    partial: Option<PayloadOwner>,
    // This holds only the new independently accepted destination account.
    custody: GenerationCopyCustody,
}
impl WorkingMemoryPool {
    /// Borrow an actual originally retained generation provider. Source identity,
    /// policy and state remain frozen through the shared host-copy transaction.
    /// This returns no grant; `PreparedTextHostCopy::copy` opens the independent
    /// destination account after the caller's SnapshotBudget reservation.
    pub fn prepare_generation_host_copy<'a, C: 'static, E>(
        &self,
        sequence: &'a RetainedGenerationSequence,
        capacity: u64,
    ) -> Result<
        impl PreparedTextHostCopy<Copied = RetainedGenerationSequence> + 'a,
        WorkingMemoryError,
    > {
        self.prepare_generation_host_copy_plan::<C, E>(sequence, capacity)
    }

    /// The same provider plan plus the actual shared snapshot transaction's
    /// fixed controls. Source rows and decoder state are borrowed and validated;
    /// no destination account is accepted until `copy_original` is consumed.
    pub fn prepare_generation_snapshot_host_copy<'a, C: 'static, E, B, D>(
        &self,
        sequence: &'a RetainedGenerationSequence,
        capacity: u64,
        native_preparation_bytes: u64,
    ) -> Result<
        impl PreparedTextHostCopy<Copied = RetainedGenerationSequence> + 'a,
        WorkingMemoryError,
    >
    where
        B: crate::execution_control::TextSnapshotBackend,
        D: crate::execution_control::SnapshotTokenController,
    {
        let mut plan = self.prepare_generation_host_copy_plan::<C, E>(sequence, capacity)?;
        let controls = crate::execution_control::TextContinuationSnapshot::<B, D>::original_capture_control_bytes::<C>()
            .ok_or(WorkingMemoryError::Overflow)?;
        plan.bytes = plan
            .bytes
            .checked_add(u64::try_from(controls).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        plan.bytes = plan
            .bytes
            .checked_add(native_preparation_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        plan.snapshot_controls = Some(controls);
        plan.native_preparation_bytes = Some(native_preparation_bytes);
        Ok(plan)
    }

    /// Same immutable provider and copy worker with fresh-resume constructor
    /// controls. The physical destination is accepted only at copy_original;
    /// neither saved source custody nor this layout supplies new run authority.
    pub fn prepare_generation_resume_host_copy<'a, C: 'static, E, B, D>(
        &self,
        sequence: &'a RetainedGenerationSequence,
        capacity: u64,
        native_preparation_bytes: u64,
    ) -> Result<impl PreparedTextHostCopy<Copied = RetainedGenerationSequence> + 'a, WorkingMemoryError>
    where
        B: crate::execution_control::TextSnapshotBackend + eredu_core::TextResumeBackend<ResumeSource = <B as crate::execution_control::TextSnapshotBackend>::SavedTextComponents>,
        D: crate::execution_control::SnapshotTokenController,
    {
        let mut plan = self.prepare_generation_host_copy_plan::<C, E>(sequence, capacity)?;
        let controls = crate::execution_control::TextContinuationSnapshot::<B,D>::original_resume_control_bytes::<C>().ok_or(WorkingMemoryError::Overflow)?;
        plan.bytes = plan
            .bytes
            .checked_add(u64::try_from(controls).map_err(|_| WorkingMemoryError::Overflow)?)
            .and_then(|n| n.checked_add(native_preparation_bytes))
            .ok_or(WorkingMemoryError::Overflow)?;
        plan.snapshot_controls = Some(controls);
        plan.native_preparation_bytes = Some(native_preparation_bytes);
        Ok(plan)
    }

    // Tests only the concrete provider/lease producer, without constructing a
    // core machine or asserting a native resume bound. All produced controls are
    // queried by the same pending-lease layout used by the complete transaction.
    #[cfg(test)]
    pub(in crate::working_memory) fn prepare_resume_provider_for_test<'a, C: 'static, E>(
        &self,
        sequence: &'a RetainedGenerationSequence,
        capacity: u64,
    ) -> Result<
        impl PreparedTextHostCopy<Copied = RetainedGenerationSequence> + 'a,
        WorkingMemoryError,
    > {
        let mut plan = self.prepare_generation_host_copy_plan::<C, E>(sequence, capacity)?;
        let controls = crate::execution_control::PendingSnapshotResumeRetention::control_bytes()
            .ok_or(WorkingMemoryError::Overflow)?;
        plan.bytes = plan
            .bytes
            .checked_add(u64::try_from(controls).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        plan.snapshot_controls = Some(controls);
        plan.native_preparation_bytes = Some(0);
        Ok(plan)
    }

    fn prepare_generation_host_copy_plan<'a, C: 'static, E>(
        &self,
        sequence: &'a RetainedGenerationSequence,
        capacity: u64,
    ) -> Result<RetainedSequenceHostCopy<'a>, WorkingMemoryError> {
        let sequence = sequence.prepare_host_copy();
        let source = sequence
            .provider()
            .copy_source()
            .and_then(|s| s.downcast_ref::<Provider>())
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if !self.same_domain(source.authority.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        source.authority.validate()?;
        if let Some(decoder) = &source.decoder {
            decoder.validate_pool(self)?;
        }
        if !source.consumer.as_ref().is_some_and(|c| c.is_cursor::<C>()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let consumer = size_of::<C>()
            .checked_add(size_of::<E>())
            .and_then(|n| n.checked_add(size_of::<Result<C, E>>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let bytes = source
            .copy_required_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?
            .checked_add(consumer)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(RetainedSequenceHostCopy {
            source,
            sequence,
            bytes,
            capacity,
            snapshot_controls: None,
            native_preparation_bytes: None,
        })
    }
}
impl Provider {
    fn copy_required_bytes(&self) -> Option<u64> {
        let p = self.payload.get();
        let token_bytes = p
            .tokens
            .capacity()
            .max(self.maximum)
            .checked_mul(size_of::<u32>())?;
        let eos_bytes = p.eos.capacity().checked_mul(size_of::<u32>())?;
        let terminal = p
            .terminal
            .as_ref()
            .map_or(0, |t| t.bound.max(t.bytes.capacity()));
        let decoder = match &self.decoder {
            Some(d) => d.copy_required_bytes()?,
            None => 0,
        };
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            token_bytes,
            eos_bytes,
            terminal,
            decoder,
            arc,
            size_of::<Provider>(),
            size_of::<Provider>(),
            size_of::<Payload>(),
            size_of::<Option<Payload>>(),
            size_of::<PayloadOwner>(),
            size_of::<ProviderAuthority>(),
            size_of::<PayloadCustody>(),
            size_of::<RetainedSequenceHostCopy<'_>>(),
            size_of::<RetainedGenerationSequence>(),
            size_of::<RetainedGenerationStorageOwner>(),
            size_of::<RetainedGenerationSequenceCopy<'_>>(),
            size_of::<Result<RetainedGenerationSequence, TextHostCopyError>>(),
            size_of::<Result<RetainedGenerationSequence, RetainedSequenceCopyMismatch>>(),
            self.consumer
                .as_ref()
                .map_or(0, |c| c.retention_peak_bytes()),
            GenerationCopyCustody::control_bytes()?,
            error_retention_peak_bytes()?,
            size_of::<Cause>(),
            size_of::<Result<Option<OriginalGenerationDecoderSource>, Cause>>(),
            size_of::<(&Payload, &mut Payload)>(),
            size_of::<(&mut TerminalText, &TerminalText)>(),
            BackendFailure::source_retention_peak_bytes::<CopyFailure>()?,
            BackendFailure::source_retention_peak_bytes::<
                crate::working_memory::funding::GenerationCopyAdmission,
            >()?,
        ];
        parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .and_then(|b| u64::try_from(b).ok())
    }
}
impl PreparedTextHostCopy for RetainedSequenceHostCopy<'_> {
    type Copied = RetainedGenerationSequence;
    fn storage_bytes(&self) -> Option<u64> {
        Some(self.bytes)
    }
    fn copy(self, _retained_bytes: u64) -> Result<Self::Copied, TextHostCopyError> {
        self.copy_with_custody(None).map(|(sequence, _)| sequence)
    }
    fn original_control_bytes(&self) -> Option<usize> {
        self.snapshot_controls
    }
    fn original_preparation_bytes(&self) -> Option<u64> {
        self.native_preparation_bytes
    }

    fn copy_original(
        self,
        _retained_bytes: u64,
    ) -> Result<(Self::Copied, eredu_core::HostPreparationAuthority), TextHostCopyError> {
        if self.snapshot_controls.is_none() {
            return Err(TextHostCopyError::Admission(
                WorkingMemoryError::UnknownBound,
            ));
        }
        let (sequence, custody) = self.copy_with_custody(None)?;
        Ok((
            sequence,
            eredu_core::HostPreparationAuthority::retain(custody),
        ))
    }
    fn copy_original_resume(
        self,
        _retained_bytes: u64,
        reservation: crate::execution_control::PendingSnapshotResumeRetention,
    ) -> Result<(Self::Copied, eredu_core::HostPreparationAuthority), TextHostCopyError> {
        if self.snapshot_controls.is_none() || self.native_preparation_bytes.is_none() {
            return Err(TextHostCopyError::Admission(
                WorkingMemoryError::UnknownBound,
            ));
        }
        let (sequence, custody) = self.copy_with_custody(Some(reservation))?;
        Ok((
            sequence,
            eredu_core::HostPreparationAuthority::retain(custody),
        ))
    }
}
impl RetainedSequenceHostCopy<'_> {
    fn copy_with_custody(
        self,
        reservation: Option<crate::execution_control::PendingSnapshotResumeRetention>,
    ) -> Result<(RetainedGenerationSequence, GenerationCopyCustody), TextHostCopyError> {
        let mut custody = match GenerationCopyCustody::admit(
            self.source.authority.source(),
            self.bytes,
            self.capacity,
        ) {
            Ok(c) => c,
            Err(error) if error.partial.is_none() => {
                return Err(TextHostCopyError::Admission(error.cause));
            }
            Err(error) => {
                return Err(TextHostCopyError::Source(BackendFailure::new(
                    BackendFailureKind::InvalidSession,
                    error,
                )));
            }
        };
        if let Some(reservation) = reservation {
            custody.attach_snapshot_resume(reservation);
        }
        let old = self.source.payload.get();
        let mut payload = PayloadOwner(Some(Arc::new(Payload {
            tokens: Vec::new(),
            eos: Vec::new(),
            committed: old.committed,
            terminal: old.terminal.as_ref().map(|t| TerminalText {
                bytes: Vec::new(),
                bound: t.bound,
            }),
            _custody: PayloadCustody::Copy(custody.clone()),
        })));
        let result = (|| {
            let out = payload.get_mut();
            out.tokens.try_reserve_exact(old.tokens.capacity())?;
            out.tokens.extend_from_slice(&old.tokens);
            out.eos.try_reserve_exact(old.eos.capacity())?;
            out.eos.extend_from_slice(&old.eos);
            if let (Some(to), Some(from)) = (&mut out.terminal, &old.terminal) {
                to.bytes.try_reserve_exact(from.bytes.capacity())?;
                to.bytes.extend_from_slice(&from.bytes);
            }
            self.source
                .decoder
                .as_ref()
                .map(OriginalGenerationDecoderSource::copy_state)
                .transpose()
                .map_err(Cause::from)
        })();
        let decoder = match result {
            Ok(d) => d,
            Err(cause) => {
                return Err(TextHostCopyError::Source(BackendFailure::new(
                    BackendFailureKind::ResourceExhausted,
                    CopyFailure {
                        cause,
                        partial: Some(payload),
                        custody,
                    },
                )));
            }
        };
        let owner = RetainedGenerationStorageOwner::new(Box::new(Provider {
            #[cfg(test)]
            fail_terminal_reserve: false,
            decoder,
            maximum: self.source.maximum,
            consumer: self.source.consumer,
            authority: ProviderAuthority::Copy(custody.clone()),
            payload,
        }));
        let sequence = self.sequence.complete(owner).map_err(|cause| {
            TextHostCopyError::Source(BackendFailure::new(
                BackendFailureKind::InvalidSession,
                CopyFailure {
                    cause: cause.into(),
                    partial: None,
                    custody: custody.clone(),
                },
            ))
        })?;
        Ok((sequence, custody))
    }
}
