//! Original packed BPE model residence; no full tokenizer or encoding operation.
use super::loaded_decode_source::Allowance;
use super::{WorkingMemoryError, WorkingMemoryPool};
use eredu_core::BackendFailure;
use eredu_text::bpe_storage::{BpeModelCompileFailure, BpeModelPlan, PreparedBpeModel};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{atomic::AtomicUsize, Arc},
};

#[derive(Debug)]
struct Payload {
    model: PreparedBpeModel,
    allowance: Allowance,
}

/// Closed shared source retaining its complete original compiler allowance.
/// No Arc, Weak, source extraction/refill or independent-byte constructor escapes.
/// This does not fund the borrowed JSON or any enclosing tokenizer components.
///
/// ```compile_fail
/// # use eredu_runtime::working_memory::OriginalBpeModel;
/// fn escape(model:OriginalBpeModel) -> &'static str { model.spelling(90).unwrap() }
/// ```
/// ```compile_fail
/// # use eredu_runtime::working_memory::OriginalBpeModel;
/// fn raw(model:OriginalBpeModel) { let _ = model.source(); }
/// ```
/// ```compile_fail
/// # use eredu_runtime::working_memory::{OriginalBpeModel,WorkingMemoryPool};
/// fn adopt(pool:&WorkingMemoryPool,model:OriginalBpeModel) { let _ = pool.compile_bpe_model(model); }
/// ```
pub struct OriginalBpeModel(Option<Arc<Payload>>);
impl OriginalBpeModel {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live original BPE model")
    }
    /// Canonical forward vocabulary length; no tokenizer or cache access.
    pub fn token_count(&self) -> usize {
        self.payload().model.token_count()
    }
    /// Looks up a spelling without allocating or mutating the model.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.payload().model.token_id(token)
    }
    /// Borrows one canonical spelling for no longer than this owner borrow.
    pub fn spelling(&self, id: u32) -> Option<&str> {
        self.payload().model.spelling(id)
    }
    /// Borrows canonical forward IDs without traversing their sparse extent.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.payload().model.ids()
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
    /// Checks the original domain without minting a request or another hold.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.payload().allowance.pool().same_domain(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl fmt::Debug for OriginalBpeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalBpeModel")
            .field("token_count", &self.token_count())
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}
impl Clone for OriginalBpeModel {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live source"))))
    }
}
impl Drop for OriginalBpeModel {
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
    Compilation(BpeModelCompileFailure),
}
/// Closed terminal failure retaining the compiler prefix and original allowance.
/// Admission rejection has no allowance. No partial/guard extraction or retry exists.
pub struct OriginalBpeModelError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    _completed: Option<PreparedBpeModel>,
    allowance: Option<Allowance>,
}
impl OriginalBpeModelError {
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
    pub fn compiler_failure(&self) -> Option<&BpeModelCompileFailure> {
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
impl fmt::Debug for OriginalBpeModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalBpeModelError")
            .field("cause", &self.cause)
            .field("settlement", &self.settlement)
            .field("completed", &self._completed.is_some())
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}
impl fmt::Display for OriginalBpeModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Admission(error) => fmt::Display::fmt(error, f),
            Cause::Compilation(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for OriginalBpeModelError {
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
    pub fn bpe_model_required_bytes(plan: &BpeModelPlan<'_>) -> Result<u64, WorkingMemoryError> {
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
            size_of::<Option<Arc<Payload>>>(),
            size_of::<OriginalBpeModel>(),
            size_of::<Option<OriginalBpeModel>>(),
            size_of::<Result<(), BackendFailure>>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<Cause>(),
            size_of::<Option<WorkingMemoryError>>(),
            size_of::<Option<PreparedBpeModel>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<OriginalBpeModelError>(),
            size_of::<Result<OriginalBpeModel, OriginalBpeModelError>>(),
            size_of::<Result<OriginalBpeModel, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalBpeModelError>()
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
    pub fn compile_bpe_model(
        &self,
        plan: BpeModelPlan<'_>,
    ) -> Result<OriginalBpeModel, OriginalBpeModelError> {
        self.compile_bpe_model_with(plan, || {})
    }
    fn compile_bpe_model_with(
        &self,
        plan: BpeModelPlan<'_>,
        after_admission: impl FnOnce(),
    ) -> Result<OriginalBpeModel, OriginalBpeModelError> {
        let bytes =
            Self::bpe_model_required_bytes(&plan).map_err(OriginalBpeModelError::rejected)?;
        let mut allowance = self
            .admit_source_compiler(bytes)
            .map_err(OriginalBpeModelError::rejected)?;
        // Private test hook runs with the actual original guard installed; the
        // public entry supplies only a zero-sized no-op.
        after_admission();
        // Inner compiler locals/prefix retire before allowance on unwind.
        let compiled = plan.compile();
        match compiled {
            Err(error) => {
                let settlement = allowance.end_compilation().err();
                Err(OriginalBpeModelError {
                    cause: Cause::Compilation(error),
                    settlement,
                    _completed: None,
                    allowance: Some(allowance),
                })
            }
            Ok(source) => {
                // Allocate the final source control while compilation is still
                // active. Only then may another host preparation begin.
                let mut owner = Arc::new(Payload {
                    model: source,
                    allowance,
                });
                let settlement = Arc::get_mut(&mut owner)
                    .expect("unexposed source")
                    .allowance
                    .end_compilation();
                match settlement {
                    Ok(()) => Ok(OriginalBpeModel(Some(owner))),
                    Err(error) => {
                        let Payload {
                            model: source,
                            allowance,
                        } = Arc::into_inner(owner).expect("unexposed source");
                        Err(OriginalBpeModelError {
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
