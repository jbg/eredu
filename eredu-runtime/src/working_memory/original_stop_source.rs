//! Original stop compiler charge; immutable strings are borrowed only during construction.
use super::loaded_decode_source::Allowance;
use super::{MemoryLedger, WorkingMemoryError};
use eredu_core::BackendFailure;
use eredu_text::stop_storage::{PreparedStopSource, StopCompileFailure, StopCompilePlan};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
struct Payload {
    source: PreparedStopSource,
    allowance: Allowance,
}

/// Closed shared source retaining its complete original compiler allowance.
/// No Arc, Weak, source extraction/refill or independent-byte constructor escapes.
/// This does not retroactively fund the borrowed caller strings.
#[derive(Debug)]
pub struct OriginalStopSource(Option<Arc<Payload>>);
impl OriginalStopSource {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live original stop source")
    }
    /// Borrows the immutable program without transferring or replacing it.
    pub fn source(&self) -> &PreparedStopSource {
        &self.payload().source
    }
    /// Full original allowance held through final payload/control retirement.
    pub fn original_bytes(&self) -> u64 {
        self.payload().allowance.bytes()
    }
    /// Exact source identity, with no allocation or accounting change.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    pub(crate) fn validate_decoder_pool(
        &self,
        decoder: &super::LoadedDecodeSource,
    ) -> Result<(), WorkingMemoryError> {
        decoder.validate_pool(self.payload().allowance.pool())
    }
    /// Checks the original domain without minting a request or another hold.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.payload().allowance.pool().same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl Clone for OriginalStopSource {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live source"))))
    }
}
impl Drop for OriginalStopSource {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Every strong owner participates; no external Weak exists. Payload
            // returns after the Arc allocation is gone, with allowance last.
            drop(Arc::into_inner(owner));
        }
    }
}

#[derive(Debug)]
enum Cause {
    Admission(WorkingMemoryError),
    Compilation(StopCompileFailure),
}
/// Closed terminal failure retaining the compiler prefix and original allowance.
/// Admission rejection has no allowance. No partial/guard extraction or retry exists.
#[derive(Debug)]
pub struct OriginalStopSourceError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    _completed: Option<PreparedStopSource>,
    allowance: Option<Allowance>,
}
impl OriginalStopSourceError {
    fn rejected(cause: WorkingMemoryError) -> Self {
        Self {
            cause: Cause::Admission(cause),
            settlement: None,
            _completed: None,
            allowance: None,
        }
    }
    /// Retained original bytes; zero means rejection preceded compilation.
    pub fn retained_bytes(&self) -> u64 {
        self.allowance.as_ref().map_or(0, |a| a.bytes())
    }
    /// Actual compiler failure from an admitted reserve attempt.
    pub fn compiler_failure(&self) -> Option<&StopCompileFailure> {
        match &self.cause {
            Cause::Compilation(error) => Some(error),
            _ => None,
        }
    }
    /// Original admission or terminal accounting failure, if present.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Admission(error) => Some(error),
            _ => None,
        })
    }
}
impl fmt::Display for OriginalStopSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Admission(error) => fmt::Display::fmt(error, f),
            Cause::Compilation(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for OriginalStopSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Admission(error) => Some(error),
            Cause::Compilation(error) => Some(error),
        }
    }
}

impl MemoryLedger {
    /// Actual source-derived compiler plus closed owner/error control requirements.
    /// This query grants no budget and takes no ownership of the borrowed plan.
    pub fn stop_source_required_bytes(
        plan: &StopCompilePlan<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .0
            .pad_to_align()
            .size();
        let controls = [
            super::OriginalTextSourceError::stop_controls().ok_or(WorkingMemoryError::Overflow)?,
            arc,
            size_of::<Payload>(),
            size_of::<Option<Payload>>(),
            size_of::<Arc<Payload>>(),
            size_of::<OriginalStopSource>(),
            size_of::<Option<OriginalStopSource>>(),
            size_of::<Result<(), BackendFailure>>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<OriginalStopSourceError>(),
            size_of::<Result<OriginalStopSource, OriginalStopSourceError>>(),
            size_of::<Result<OriginalStopSource, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalStopSourceError>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        plan.requirements()
            .required_bytes()
            .checked_add(controls)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Admits before the first compiler reserve and consumes the actual plan once.
    /// Terminal compilation ends only its active count; every byte remains held
    /// through source/error retirement. No second quote, registration or HF claim.
    pub fn compile_stop_source(
        &self,
        plan: StopCompilePlan<'_>,
    ) -> Result<OriginalStopSource, OriginalStopSourceError> {
        self.compile_stop_source_with(plan, || {})
    }
    fn compile_stop_source_with(
        &self,
        plan: StopCompilePlan<'_>,
        after_admission: impl FnOnce(),
    ) -> Result<OriginalStopSource, OriginalStopSourceError> {
        let bytes =
            Self::stop_source_required_bytes(&plan).map_err(OriginalStopSourceError::rejected)?;
        let mut allowance = self
            .admit_source_compiler(bytes)
            .map_err(OriginalStopSourceError::rejected)?;
        // Private test hook runs with the actual original guard installed; the
        // public entry supplies only a zero-sized no-op.
        after_admission();
        // Inner compiler locals/prefix retire before allowance on unwind.
        let compiled = plan.compile();
        match compiled {
            Err(error) => {
                let settlement = allowance.end_compilation().err();
                Err(OriginalStopSourceError {
                    cause: Cause::Compilation(error),
                    settlement,
                    _completed: None,
                    allowance: Some(allowance),
                })
            }
            Ok(source) => {
                // Allocate the final source control while compilation is still
                // active. Only then may another host preparation begin.
                let mut owner = Arc::new(Payload { source, allowance });
                let settlement = Arc::get_mut(&mut owner)
                    .expect("unexposed source")
                    .allowance
                    .end_compilation();
                match settlement {
                    Ok(()) => Ok(OriginalStopSource(Some(owner))),
                    Err(error) => {
                        let Payload { source, allowance } =
                            Arc::into_inner(owner).expect("unexposed source");
                        Err(OriginalStopSourceError {
                            cause: Cause::Admission(error),
                            settlement: None,
                            _completed: Some(source),
                            allowance: Some(allowance),
                        })
                    }
                }
            }
        }
    }
}
#[cfg(test)]
mod tests;

/// Concrete source-only cold composition over a backend's actual original pool.
/// Tokenizer/compiler selection remains in text/facade; core has no stop-selection policy.
/// This capability does not activate managed public decoding or fund prior caller input.
pub trait OriginalStopSourceBackend: eredu_core::TextGenerationBackend {
    /// Consumes the exact borrowed compiler plan under the runtime's original
    /// source allowance. Owning failures use core's closed source directly.
    fn compile_original_stop_source(
        runtime: &eredu_core::ModelRuntime<Self>,
        plan: StopCompilePlan<'_>,
    ) -> Result<OriginalStopSource, BackendFailure>;
}
