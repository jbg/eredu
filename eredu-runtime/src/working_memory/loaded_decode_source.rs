//! One original cold compiler charge, separate from every generation request.
use super::{WorkingMemoryError, WorkingMemoryPool};
use eredu_core::BackendFailure;
use eredu_text::decoder_storage::{DecodeCompileFailure, DecodeCompilePlan, PreparedDecodeSource};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Uncharged,
    Compiling,
    Idle,
}
#[derive(Debug)]
pub(super) struct Allowance {
    pool: WorkingMemoryPool,
    bytes: u64,
    phase: Phase,
}
impl Allowance {
    pub(super) fn into_gguf_source_account(mut self) -> super::gguf_source::SourceAccount {
        debug_assert_eq!(self.phase, Phase::Compiling);
        let account = super::gguf_source::SourceAccount::new_unarmed(&self.pool, self.bytes);
        self.phase = Phase::Uncharged;
        account.activate();
        account
    }

    pub(super) fn into_prepared_native_account(
        mut self,
    ) -> super::original_prepared_native_input::Account {
        debug_assert_eq!(self.phase, Phase::Compiling);
        let account = super::original_prepared_native_input::Account::new_unarmed(
            self.pool.clone(),
            self.bytes,
        );
        // The new shell is unarmed during allocation, so even a constructor
        // unwind cannot refund twice. Disarm then activate with no fallible work
        // or user callback between these exclusive scalar state changes.
        self.phase = Phase::Uncharged;
        account.activate();
        account
    }
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(super) fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }

    /// Admit a reached input producer while retaining the complete prior prefix.
    /// Only the active original compiler may extend its own charge.
    pub(super) fn reserve_more(&mut self, bytes: u64) -> Result<(), WorkingMemoryError> {
        if self.phase != Phase::Compiling {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let total = self
            .bytes
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.pool.charge_source_compiler(bytes, false)?;
        // No fallible operation follows the ledger commit before ownership update.
        self.bytes = total;
        Ok(())
    }

    pub(super) fn end_compilation(&mut self) -> Result<(), WorkingMemoryError> {
        debug_assert_eq!(self.phase, Phase::Compiling);
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        // Compiler calls/locals are finished; retain every original byte.
        usage.reservations -= 1;
        self.phase = Phase::Idle;
        Ok(())
    }
}
impl Drop for Allowance {
    fn drop(&mut self) {
        if self.phase == Phase::Uncharged {
            return;
        }
        // Payload and control allocations retire first. Poison cannot certify a refund.
        if let Ok(mut usage) = self.pool.0.usage.lock() {
            if self.phase == Phase::Compiling {
                usage.reservations -= 1;
            }
            usage.reserved -= self.bytes;
        }
    }
}
#[derive(Debug)]
struct Payload {
    source: PreparedDecodeSource,
    allowance: Allowance,
}

/// Closed shared source retaining its complete original compiler allowance.
/// No Arc, Weak, source extraction/refill or independent-byte constructor escapes.
/// This does not retroactively fund the borrowed HF graph.
#[derive(Debug)]
pub struct LoadedDecodeSource(Option<Arc<Payload>>);
impl LoadedDecodeSource {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live loaded decoder source")
    }
    /// Borrows the immutable program without transferring or replacing it.
    pub fn source(&self) -> &PreparedDecodeSource {
        &self.payload().source
    }
    /// Full original allowance held through final payload/control retirement.
    pub fn original_bytes(&self) -> u64 {
        self.payload().allowance.bytes
    }
    /// Exact source identity, with no allocation or accounting change.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    /// Checks the original domain without minting a request or another hold.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.payload().allowance.pool.same_domain(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl Clone for LoadedDecodeSource {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live source"))))
    }
}
impl Drop for LoadedDecodeSource {
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
    Compilation(DecodeCompileFailure),
}
/// Closed terminal failure retaining the compiler prefix and original allowance.
/// Admission rejection has no allowance. No partial/guard extraction or retry exists.
#[derive(Debug)]
pub struct LoadedDecodeSourceError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    _completed: Option<PreparedDecodeSource>,
    allowance: Option<Allowance>,
}
impl LoadedDecodeSourceError {
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
        self.allowance.as_ref().map_or(0, |a| a.bytes)
    }
    /// Actual compiler failure from an admitted reserve attempt.
    pub fn compiler_failure(&self) -> Option<&DecodeCompileFailure> {
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
impl fmt::Display for LoadedDecodeSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Admission(error) => fmt::Display::fmt(error, f),
            Cause::Compilation(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for LoadedDecodeSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Admission(error) => Some(error),
            Cause::Compilation(error) => Some(error),
        }
    }
}

impl WorkingMemoryPool {
    /// Actual source-derived compiler plus closed owner/error control requirements.
    /// This query grants no budget and takes no ownership of the borrowed plan.
    pub fn decode_source_required_bytes(
        plan: &DecodeCompilePlan<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .0
            .pad_to_align()
            .size();
        let controls = [
            arc,
            size_of::<Payload>(),
            size_of::<Option<Payload>>(),
            size_of::<Arc<Payload>>(),
            size_of::<LoadedDecodeSource>(),
            size_of::<Option<LoadedDecodeSource>>(),
            size_of::<Result<(), BackendFailure>>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<LoadedDecodeSourceError>(),
            size_of::<Result<LoadedDecodeSource, LoadedDecodeSourceError>>(),
            size_of::<Result<LoadedDecodeSource, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<LoadedDecodeSourceError>()
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
    pub(super) fn admit_source_compiler(
        &self,
        bytes: u64,
    ) -> Result<Allowance, WorkingMemoryError> {
        let mut allowance = Allowance {
            pool: self.clone(),
            bytes,
            phase: Phase::Uncharged,
        };
        self.charge_source_compiler(bytes, true)?;
        // No fallible operation between commit and installing this guard.
        allowance.phase = Phase::Compiling;
        Ok(allowance)
    }
    fn charge_source_compiler(&self, bytes: u64, starting: bool) -> Result<(), WorkingMemoryError> {
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if usage.unquoted_owners != 0 {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let available = self.0.available(&usage, None)?;
        if bytes > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let reservations = usage
            .reservations
            .checked_add(usize::from(starting))
            .ok_or(WorkingMemoryError::Overflow)?;
        let reserved = usage
            .reserved
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let used = self
            .0
            .existing
            .checked_add(usage.registered)
            .and_then(|n| n.checked_add(reserved))
            .ok_or(WorkingMemoryError::Overflow)?;
        usage.reservations = reservations;
        usage.reserved = reserved;
        usage.peak = usage.peak.max(used);
        Ok(())
    }
    /// Admits before the first compiler reserve and consumes the actual plan once.
    /// Terminal compilation ends only its active count; every byte remains held
    /// through source/error retirement. No second quote, registration or HF claim.
    pub fn compile_decode_source(
        &self,
        plan: DecodeCompilePlan<'_>,
    ) -> Result<LoadedDecodeSource, LoadedDecodeSourceError> {
        self.compile_decode_source_with(plan, || {})
    }
    fn compile_decode_source_with(
        &self,
        plan: DecodeCompilePlan<'_>,
        after_admission: impl FnOnce(),
    ) -> Result<LoadedDecodeSource, LoadedDecodeSourceError> {
        let bytes =
            Self::decode_source_required_bytes(&plan).map_err(LoadedDecodeSourceError::rejected)?;
        let mut allowance = self
            .admit_source_compiler(bytes)
            .map_err(LoadedDecodeSourceError::rejected)?;
        // Private test hook runs with the actual original guard installed; the
        // public entry supplies only a zero-sized no-op.
        after_admission();
        // Inner compiler locals/prefix retire before allowance on unwind.
        let compiled = plan.compile();
        match compiled {
            Err(error) => {
                let settlement = allowance.end_compilation().err();
                Err(LoadedDecodeSourceError {
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
                    Ok(()) => Ok(LoadedDecodeSource(Some(owner))),
                    Err(error) => {
                        let Payload { source, allowance } =
                            Arc::into_inner(owner).expect("unexposed source");
                        Err(LoadedDecodeSourceError {
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
/// Tokenizer/compiler selection remains in text/facade; core has no HF policy.
/// This capability does not activate managed public decoding or fund prior HF.
pub trait LoadedDecodeSourceBackend: eredu_core::TextGenerationBackend {
    /// Consumes the exact borrowed compiler plan under the runtime's original
    /// source allowance. Owning failures use core's closed source directly.
    fn compile_loaded_decode_source(
        runtime: &eredu_core::ModelRuntime<Self>,
        plan: DecodeCompilePlan<'_>,
    ) -> Result<LoadedDecodeSource, BackendFailure>;
}
