//! Original host-input residence, separate from numerical execution and requests.
use super::{loaded_decode_source::Allowance, WorkingMemoryError, WorkingMemoryPool};
use crate::input::host::{CompileFailure, PreparedHostInputPlan, PreparedHostPart, Storage};
use eredu_core::BackendFailure;
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{atomic::AtomicUsize, Arc},
};
#[derive(Debug)]
struct Payload {
    storage: Storage,
    digest: [u8; 32],
    allowance: Allowance,
}
/// Closed immutable prepared host source. This funds its newly constructed
/// buffers only: preceding caller/processor storage, native uploads, plans and
/// encoder work remain separate. No raw owning handle, Weak or grant escapes.
///
/// ```compile_fail
/// use eredu_runtime::{working_memory::OriginalPreparedHostInput,input::host::{HostInputPartView,HostTensorView}};
/// fn escape(source:OriginalPreparedHostInput) {
///     let part=source.parts().next().unwrap();
///     drop(source);
///     std::hint::black_box(part.payload());
/// }
/// ```
/// ```compile_fail
/// use eredu_runtime::working_memory::OriginalPreparedHostInput;
/// fn raw(source:OriginalPreparedHostInput) { let _=source.into_inner(); }
/// ```
pub struct OriginalPreparedHostInput(Option<Arc<Payload>>);
impl OriginalPreparedHostInput {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live original host input")
    }
    /// Borrows all parts and their immutable slots without allocating.
    pub fn parts(&self) -> impl ExactSizeIterator<Item = PreparedHostPart<'_>> + Clone {
        self.payload().storage.parts()
    }
    /// Borrows one actual source part without a descriptor, index table or copy.
    pub fn part(&self, index: usize) -> Option<PreparedHostPart<'_>> {
        self.payload().storage.part(index)
    }
    /// Number of payload and metadata slots.
    pub fn slot_count(&self) -> usize {
        self.payload().storage.slot_count()
    }
    /// Borrows a canonical slot: each payload followed by that part's ordered
    /// metadata. The source remains necessary for the returned shape and values.
    pub fn slot(&self, index: usize) -> Option<crate::input::host::HostTensorView<'_>> {
        self.payload().storage.slot(index)
    }
    /// Content-derived cache input. Equality never confers source authority.
    pub fn content_digest(&self) -> &[u8; 32] {
        &self.payload().digest
    }
    /// Complete original source construction allowance, retained until retirement.
    pub fn original_bytes(&self) -> u64 {
        self.payload().allowance.bytes()
    }
    /// Whether both wrappers retain this same actual original source.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    /// Confirms genuine ordinary diagnostic custody belongs to this source pool.
    /// This neither registers storage nor grants original execution/copy work.
    pub fn validate_unquoted(
        &self,
        lease: &super::WorkingMemoryUnquotedLease,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_pool(&lease.0.pool)
    }
    /// Rejects a foreign original domain without allocation or native work.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.payload().allowance.pool().same_domain(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl Clone for OriginalPreparedHostInput {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live source"))))
    }
}
impl Drop for OriginalPreparedHostInput {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl fmt::Debug for OriginalPreparedHostInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalPreparedHostInput")
            .field("parts", &self.parts().len())
            .field("slots", &self.slot_count())
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
enum Cause {
    Admission(WorkingMemoryError),
    Compilation(CompileFailure),
}
/// Closed construction failure retaining every allocated prefix and original
/// allowance. No partial payload, retry authority or accounting owner escapes.
pub struct OriginalPreparedHostInputError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    _completed: Option<Storage>,
    allowance: Option<Allowance>,
}
impl OriginalPreparedHostInputError {
    fn rejected(cause: WorkingMemoryError) -> Self {
        Self {
            cause: Cause::Admission(cause),
            settlement: None,
            _completed: None,
            allowance: None,
        }
    }
    /// Original allowance still held by this failure; zero for pre-admission rejection.
    pub fn retained_bytes(&self) -> u64 {
        self.allowance.as_ref().map_or(0, Allowance::bytes)
    }
    /// Actual failed reservation's zero-based destination index.
    pub fn failed_buffer(&self) -> Option<usize> {
        match &self.cause {
            Cause::Compilation(e) => Some(e.buffer),
            _ => None,
        }
    }
    /// Capacity retained by a failed reserve prefix, without exposing buffers.
    pub fn failed_prefix_bytes(&self) -> usize {
        match &self.cause {
            Cause::Compilation(e) => e.retained_heap_bytes(),
            _ => 0,
        }
    }
    /// Original admission or terminal accounting error, preserving exact cause.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Admission(e) => Some(e),
            _ => None,
        })
    }
}
impl fmt::Debug for OriginalPreparedHostInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalPreparedHostInputError")
            .field("cause", &self.cause)
            .field("settlement", &self.settlement)
            .field("completed", &self._completed.is_some())
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}
impl fmt::Display for OriginalPreparedHostInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Admission(e) => fmt::Display::fmt(e, f),
            Cause::Compilation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for OriginalPreparedHostInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Admission(e) => Some(e),
            Cause::Compilation(e) => Some(e),
        }
    }
}
impl WorkingMemoryPool {
    /// Concrete source-derived requested buffers and closed owner/error controls.
    /// This scalar query allocates nothing and grants no construction authority.
    pub fn prepared_host_input_required_bytes(
        plan: &PreparedHostInputPlan<'_>,
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
            size_of::<Option<Arc<Payload>>>(),
            size_of::<OriginalPreparedHostInput>(),
            size_of::<Option<OriginalPreparedHostInput>>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<Cause>(),
            size_of::<Option<WorkingMemoryError>>(),
            size_of::<Option<Storage>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<OriginalPreparedHostInputError>(),
            size_of::<Result<OriginalPreparedHostInput, OriginalPreparedHostInputError>>(),
            size_of::<Result<OriginalPreparedHostInput, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalPreparedHostInputError>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        plan.required_bytes()
            .checked_add(controls)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Compares once before all seven reserves and the final Arc. Success ends
    /// only active construction; the complete allowance remains as source residence.
    /// It cannot adopt an existing native input or evade real unquoted owners.
    pub fn compile_prepared_host_input(
        &self,
        plan: PreparedHostInputPlan<'_>,
    ) -> Result<OriginalPreparedHostInput, OriginalPreparedHostInputError> {
        self.compile_prepared_host_input_with(plan, || {}, Storage::compile)
    }
    fn compile_prepared_host_input_with<'a>(
        &self,
        plan: PreparedHostInputPlan<'a>,
        after_admission: impl FnOnce(),
        compile: impl FnOnce(PreparedHostInputPlan<'a>) -> Result<Storage, CompileFailure>,
    ) -> Result<OriginalPreparedHostInput, OriginalPreparedHostInputError> {
        let bytes = Self::prepared_host_input_required_bytes(&plan)
            .map_err(OriginalPreparedHostInputError::rejected)?;
        let mut allowance = self
            .admit_source_compiler(bytes)
            .map_err(OriginalPreparedHostInputError::rejected)?;
        after_admission();
        let digest = *plan.content_digest();
        match compile(plan) {
            Err(error) => {
                let settlement = allowance.end_compilation().err();
                Err(OriginalPreparedHostInputError {
                    cause: Cause::Compilation(error),
                    settlement,
                    _completed: None,
                    allowance: Some(allowance),
                })
            }
            Ok(storage) => {
                let mut owner = Arc::new(Payload {
                    storage,
                    digest,
                    allowance,
                });
                let settlement = Arc::get_mut(&mut owner)
                    .expect("unexposed source")
                    .allowance
                    .end_compilation();
                match settlement {
                    Ok(()) => Ok(OriginalPreparedHostInput(Some(owner))),
                    Err(error) => {
                        let Payload {
                            storage, allowance, ..
                        } = Arc::into_inner(owner).expect("unexposed source");
                        Err(OriginalPreparedHostInputError {
                            cause: Cause::Admission(error),
                            settlement: None,
                            _completed: Some(storage),
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
