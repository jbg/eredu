//! Native source leaves; selected plans and prompt controls remain separate.
use super::*;
use eredu_runtime::{
    input::host::{HostTensorValues, HostTensorView},
    working_memory::{
        OriginalPreparedInputCustody, OriginalPreparedInputMaterialization,
        OriginalPreparedInputMaterializationError, PreparedNativeInputCompiler,
    },
};
use safemlx::{
    PreparedInputArena, PreparedInputCause, PreparedInputLeaf, PreparedInputPlan,
    PreparedInputRuntime, PreparedSubmissionGraphQuota, SubmissionGraphQuotaCause,
};
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

mod model_input;
pub(in crate::composition::mlx::session::model_session) use model_input::{RetiredMlxPreparedModelInputError, RetiredMlxPreparedModelInputBindError};
pub(crate) use model_input::CompletedOriginalModelInput;
pub(in crate::composition::mlx::session::model_session) use model_input::CompletedOriginalTextInput;
pub use model_input::{
    MlxOriginalPreparedModelInput, MlxPreparedModelInputBindError, MlxPreparedModelInputError,
    MlxPreparedModelInputPlan,
};

/// Explicit ordinary initialization of the actual native allocator and thread
/// entry machinery. No source/selection is adopted or charged by this witness.
#[derive(Debug)]
pub struct MlxPreparedInputMaterializer(PreparedInputRuntime, InputAllocatorCoverage);
pub use crate::backend::managed_memory::input_allocator::{
    InputAllocatorCoverage, MlxInputAllocatorInitializationError,
};
impl MlxPreparedInputMaterializer {
    /// Initializes/reuses Device, process Scheduler and input allocator, each
    /// under its actual shared cold account in the same managed domain. No
    /// ordinary predecessor is promoted. Worker/stream/context ownership and
    /// complete execution fit remain separate prerequisites.
    pub fn prepare_admitted(
        pool: &WorkingMemoryPool,
    ) -> Result<Self, MlxInputAllocatorInitializationError> {
        crate::backend::managed_memory::input_allocator::prepare_admitted(pool)
            .map(|(runtime, coverage)| Self(runtime, coverage))
    }
    /// Actual constructor coverage, independent of all later source/request fit.
    pub fn allocator_coverage(&self) -> InputAllocatorCoverage {
        self.1
    }

    /// May allocate or execute ordinary housekeeping. Call before B1 admission,
    /// outside any managed operation, with genuine ordinary runtime ownership.
    pub fn prepare() -> Result<Self, safemlx::error::Exception> {
        crate::backend::managed_memory::input_allocator::prepare_ordinary()
            .map(|(runtime, coverage)| Self(runtime, coverage))
    }
    /// Pure plan over the exact original source. No descriptors or slot index
    /// collection is allocated before its original comparison.
    pub fn plan<'a>(
        &'a self,
        source: &'a OriginalPreparedHostInput,
    ) -> Result<MlxPreparedNativeInputPlan<'a>, WorkingMemoryError> {
        let (arena, native_required) = leaf_recipe(&self.0, source, 0)?;
        let required = [
            native_required,
            size_of::<MlxOriginalPreparedNativeInput>(),
            size_of::<MlxHostInputUploadError>(),
            size_of::<Result<MlxModelInput, MlxHostInputUploadError>>(),
            size_of::<MlxPreparedNativeInputError>(),
            size_of::<Result<MlxOriginalPreparedNativeInput, MlxPreparedNativeInputError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        Ok(MlxPreparedNativeInputPlan {
            runtime: &self.0,
            source,
            arena,
            required,
        })
    }
}
fn leaf_recipe(
    runtime: &PreparedInputRuntime,
    source: &OriginalPreparedHostInput,
    handles: usize,
) -> Result<(usize, usize), WorkingMemoryError> {
    let mut arena = handles;
    let mut backing = 0usize;
    let mut controls = 0usize;
    for index in 0..source.slot_count() {
        let plan = slot_plan(runtime, source.slot(index).expect("source slot"))
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        arena = arena
            .checked_add(plan.metadata_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        backing = backing
            .checked_add(plan.backing_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        controls = controls.max(plan.control_bytes());
    }
    let resource = PreparedInputArena::layout::<OriginalPreparedInputCustody>(arena)
        .map_err(|_| WorkingMemoryError::UnknownBound)?
        .total_bytes()
        .ok_or(WorkingMemoryError::Overflow)?;
    let slots = Layout::array::<PreparedInputLeaf>(source.slot_count())
        .map_err(|_| WorkingMemoryError::Overflow)?
        .size();
    let required = [
        resource,
        slots,
        backing,
        controls,
        size_of::<NativeCells>(),
        size_of::<MlxNativeInputCause>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(WorkingMemoryError::Overflow)?;
    Ok((arena, required))
}
/// Borrowed source recipe. It is consumed once and cannot change its capacity,
/// native runtime or source after the requirements query.
#[derive(Debug)]
pub struct MlxPreparedNativeInputPlan<'a> {
    runtime: &'a PreparedInputRuntime,
    source: &'a OriginalPreparedHostInput,
    arena: usize,
    required: usize,
}
impl MlxPreparedNativeInputPlan<'_> {
    /// Full original B1 charge including actual generic accounting controls.
    pub fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        WorkingMemoryPool::prepared_native_input_required_bytes(&Compiler {
            plan: self.borrowed(),
        })
    }
    fn borrowed(&self) -> MlxPreparedNativeInputPlan<'_> {
        MlxPreparedNativeInputPlan {
            runtime: self.runtime,
            source: self.source,
            arena: self.arena,
            required: self.required,
        }
    }
    /// One genuine comparison and complete source materialization; no late
    /// registration, evaluation, native submission or ordinary owner acquisition.
    pub fn materialize(
        self,
        pool: &WorkingMemoryPool,
    ) -> Result<MlxOriginalPreparedNativeInput, MlxPreparedNativeInputError> {
        pool.compile_prepared_native_input(Compiler { plan: self })
            .map(MlxOriginalPreparedNativeInput)
            .map_err(MlxPreparedNativeInputError)
    }
}
fn slot_plan<'a>(
    runtime: &'a PreparedInputRuntime,
    slot: HostTensorView<'a>,
) -> Result<PreparedInputPlan<'a>, PreparedInputCause> {
    match slot.values {
        HostTensorValues::U32(v) => runtime.u32(v, slot.shape),
        HostTensorValues::I32(v) => runtime.i32(v, slot.shape),
        HostTensorValues::F32(v) => runtime.f32(v, slot.shape),
        HostTensorValues::Bool(v) => runtime.boolean(v, slot.shape),
    }
}
#[derive(Debug, thiserror::Error)]
enum MlxNativeInputCause {
    #[error("native source arena: {0}")]
    Arena(#[source] SubmissionGraphQuotaCause),
    #[error("native source slot storage: {0}")]
    Slots(#[source] TryReserveError),
    #[error("native source leaf: {0}")]
    Leaf(#[source] PreparedInputCause),
}
struct NativeCells {
    // Exact cells first, then independently retained native resource/queue.
    leaves: Vec<PreparedInputLeaf>,
    arena: Option<PreparedInputArena>,
    prepared: Option<PreparedSubmissionGraphQuota<OriginalPreparedInputCustody>>,
    rejected: Option<OriginalPreparedInputCustody>,
}
impl NativeCells {
    fn empty() -> Self {
        Self {
            leaves: Vec::new(),
            arena: None,
            prepared: None,
            rejected: None,
        }
    }
}
struct Compiler<'a> {
    plan: MlxPreparedNativeInputPlan<'a>,
}
impl PreparedNativeInputCompiler for Compiler<'_> {
    type Output = NativeCells;
    type Error = MlxNativeInputCause;
    fn source(&self) -> &OriginalPreparedHostInput {
        self.plan.source
    }
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        Ok(self.plan.required)
    }
    fn compile(
        self,
        custody: OriginalPreparedInputCustody,
    ) -> Result<NativeCells, (NativeCells, MlxNativeInputCause)> {
        #[cfg(all(test, unix))]
        COMPILES.set(COMPILES.get() + 1);
        let mut cells = NativeCells::empty();
        let prepared = match PreparedSubmissionGraphQuota::try_new(self.plan.arena, custody) {
            Ok(v) => v,
            Err(e) => {
                let (cause, owner) = e.into_parts();
                cells.rejected = Some(owner);
                return Err((cells, MlxNativeInputCause::Arena(cause)));
            }
        };
        match PreparedInputArena::try_allocate(prepared) {
            Ok(v) => cells.arena = Some(v),
            Err(e) => {
                let (cause, owner) = e.into_parts();
                cells.prepared = Some(owner);
                return Err((cells, MlxNativeInputCause::Arena(cause)));
            }
        }
        let requested_slots = self.plan.source.slot_count();
        #[cfg(all(test, unix))]
        let requested_slots = if FAIL_VECTOR_RESERVE.replace(false) {
            usize::MAX
        } else {
            requested_slots
        };
        if let Err(e) = cells.leaves.try_reserve_exact(requested_slots) {
            return Err((cells, MlxNativeInputCause::Slots(e)));
        }
        for index in 0..self.plan.source.slot_count() {
            let slot = self.plan.source.slot(index).expect("source slot");
            let leaf = slot_plan(self.plan.runtime, slot)
                .and_then(|p| p.construct(cells.arena.as_ref().expect("prepared arena")));
            match leaf {
                Ok(v) => cells.leaves.push(v),
                Err(e) => return Err((cells, MlxNativeInputCause::Leaf(e))),
            }
        }
        Ok(cells)
    }
}

/// Closed original completed leaves. Source authority is actual owner identity;
/// equal data/digest cannot substitute. No raw native owning handles escape.
///
/// ```compile_fail
/// use eredu_backend_mlx::native::MlxOriginalPreparedNativeInput;
/// fn escape(source:MlxOriginalPreparedNativeInput) { source.into_arrays(); }
/// ```
#[derive(Debug)]
pub struct MlxOriginalPreparedNativeInput(OriginalPreparedInputMaterialization<NativeCells>);
impl MlxOriginalPreparedNativeInput {
    /// Exact immutable source used by every completed leaf.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.0.source()
    }
    /// Full original B1 charge, retained independently of source I.
    pub fn original_bytes(&self) -> u64 {
        self.0.original_bytes()
    }
    /// Number of actual completed canonical leaves.
    pub fn slot_count(&self) -> usize {
        self.0.storage().leaves.len()
    }
    pub(super) fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        self.0.validate_pool(pool)
    }
    /// Only the ordinary lowerer calls this after authenticating each actual
    /// source callback. The returned C wrapper is ordinary control storage.
    pub(super) fn clone_slot(
        &self,
        index: usize,
        values: HostTensorValues<'_>,
        shape: &[usize],
    ) -> Result<Array, Error> {
        let expected = self
            .source()
            .slot(index)
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)))?;
        let same = match (expected.values, values) {
            (HostTensorValues::U32(a), HostTensorValues::U32(b)) => {
                a.len() == b.len() && std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            (HostTensorValues::I32(a), HostTensorValues::I32(b)) => {
                a.len() == b.len() && std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            (HostTensorValues::F32(a), HostTensorValues::F32(b)) => {
                a.len() == b.len() && std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            _ => false,
        };
        if !same || expected.shape != shape {
            return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
        }
        let array = self.0.storage().leaves[index]
            .try_clone_array()
            .map_err(Error::from)?;
        #[cfg(all(test, unix))]
        CLONES.set(CLONES.get() + 1);
        Ok(array)
    }
}
/// Owning fixed native materialization failure, including every real prefix.
#[derive(Debug)]
pub struct MlxPreparedNativeInputError(
    OriginalPreparedInputMaterializationError<NativeCells, MlxNativeInputCause>,
);
impl MlxPreparedNativeInputError {
    /// Comparison/settlement failure, without extracting custody.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.0.accounting_failure()
    }
    /// Original retained B1 bytes; zero for a pre-work rejection.
    pub fn retained_bytes(&self) -> u64 {
        self.0.retained_bytes()
    }
}
impl std::fmt::Display for MlxPreparedNativeInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for MlxPreparedNativeInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[cfg(all(test, unix))]
thread_local! {
    static COMPILES: std::cell::Cell<usize> = const {std::cell::Cell::new(0)};
    static CLONES: std::cell::Cell<usize> = const {std::cell::Cell::new(0)};
    static FAIL_VECTOR_RESERVE: std::cell::Cell<bool> = const {std::cell::Cell::new(false)};
}
#[cfg(all(test, unix, target_vendor = "apple"))]
mod tests;
