//! Fixed semantic destinations. These records confer no architecture admission.
//! Equation and source/selection authority remain with the architecture compiler.
use super::{
    InferenceExecutionIdentity, InferenceStateRevision, OriginalPreparedHostInput,
    WorkingMemoryError, WorkingMemoryPool, loaded_decode_source::Allowance,
};
use eredu_core::{BackendFailure, InputModality};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};

/// Concrete coordinate destinations; not a native tensor or workspace quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeSemanticCoordinates {
    Ordinary,
    ThreeAxesAndPrefix,
}
/// Generic disposition; only the architecture-owned compiler admits its meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeSemanticRole {
    Tokens,
    Projected,
    Encoded,
}
/// Fixed record referring back to one actual immutable source part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeSemanticPartRecord {
    pub source_part: usize,
    pub grid_slot: Option<usize>,
    pub grid_rows: usize,
    pub start: u64,
    pub end: u64,
    pub role: CompositeSemanticRole,
    pub modality: InputModality,
    pub placeholder: u32,
    pub workspace_scalars: u64,
}
impl Default for CompositeSemanticPartRecord {
    fn default() -> Self {
        Self {
            source_part: 0,
            grid_slot: None,
            grid_rows: 0,
            start: 0,
            end: 0,
            role: CompositeSemanticRole::Tokens,
            modality: InputModality::Text,
            placeholder: 0,
            workspace_scalars: 0,
        }
    }
}
/// Scalar layout for the actual source's part population and concrete axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedCompositeSemanticLayout {
    parts: usize,
    positions: usize,
    coordinates: CompositeSemanticCoordinates,
}
impl PreparedCompositeSemanticLayout {
    pub fn for_source(
        source: &OriginalPreparedHostInput,
        positions: usize,
        coordinates: CompositeSemanticCoordinates,
    ) -> Result<Self, WorkingMemoryError> {
        if positions == 0 || positions > i32::MAX as usize {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let result = Self {
            parts: source.parts().len(),
            positions,
            coordinates,
        };
        result.heap_bytes()?;
        Ok(result)
    }
    pub const fn positions(self) -> usize {
        self.positions
    }
    pub const fn coordinate_mode(self) -> CompositeSemanticCoordinates {
        self.coordinates
    }
    fn coordinate_count(self) -> Result<usize, WorkingMemoryError> {
        match self.coordinates {
            CompositeSemanticCoordinates::Ordinary => Ok(0),
            CompositeSemanticCoordinates::ThreeAxesAndPrefix => self
                .positions
                .checked_mul(4)
                .ok_or(WorkingMemoryError::Overflow),
        }
    }
    fn heap_bytes(self) -> Result<usize, WorkingMemoryError> {
        Layout::array::<CompositeSemanticPartRecord>(self.parts)
            .map_err(|_| WorkingMemoryError::Overflow)?
            .size()
            .checked_add(
                Layout::array::<i32>(self.coordinate_count()?)
                    .map_err(|_| WorkingMemoryError::Overflow)?
                    .size(),
            )
            .ok_or(WorkingMemoryError::Overflow)
    }
}
/// Current actual session identities, minted only by the shared session boundary.
/// This carries no grant, byte bound, source choice, or freely selected frontier.
#[derive(Debug)]
pub struct MediaSessionBinding {
    pub(crate) execution: InferenceExecutionIdentity,
    pub(crate) revision: InferenceStateRevision,
    pub(crate) control: Arc<()>,
    pub(crate) frontier: u64,
}
impl MediaSessionBinding {
    pub const fn frontier(&self) -> u64 {
        self.frontier
    }
    pub fn same_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.execution.0, &other.execution.0)
            && Arc::ptr_eq(&self.control, &other.control)
    }
    pub fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.execution.0, &other.execution.0)
            && Arc::ptr_eq(&self.control, &other.control)
            && self.revision == other.revision
            && self.frontier == other.frontier
    }
    /// Compares an immutable paired snapshot's exact execution, revision and
    /// frontier. The enclosing copied-state owner separately validates its
    /// control origin; this comparison creates neither a binding nor a grant.
    pub fn matches_snapshot(
        &self,
        execution: &InferenceExecutionIdentity,
        revision: &InferenceStateRevision,
        frontier: u64,
    ) -> bool {
        Arc::ptr_eq(&self.execution.0, &execution.0)
            && &self.revision == revision
            && self.frontier == frontier
    }
    fn duplicate(&self) -> Self {
        Self {
            execution: self.execution.clone(),
            revision: self.revision.clone(),
            control: Arc::clone(&self.control),
            frontier: self.frontier,
        }
    }
}
/// Receipt from the shared validated state-slot exchange. It binds the exact
/// source semantics to the fresh installed revision, not merely its frontier.
/// There is no public constructor and it grants no copy or execution authority.
#[derive(Debug)]
pub struct CopiedMediaStateBinding {
    source: MediaSessionBinding,
    installed: MediaSessionBinding,
}
impl CopiedMediaStateBinding {
    pub(crate) fn new(source: &MediaSessionBinding, installed: MediaSessionBinding) -> Self {
        Self {
            source: source.duplicate(),
            installed,
        }
    }
    pub fn matches_installed(&self, current: &MediaSessionBinding) -> bool {
        self.installed.matches(current)
    }
}

#[derive(Debug)]
struct Buffers {
    records: Vec<CompositeSemanticPartRecord>,
    coordinates: Vec<i32>,
}
#[derive(Debug)]
struct Payload {
    buffers: Buffers,
    layout: PreparedCompositeSemanticLayout,
    source: OriginalPreparedHostInput,
    allowance: Allowance,
}
impl Payload {
    fn heap_bytes(&self) -> usize {
        self.buffers.records.capacity() * size_of::<CompositeSemanticPartRecord>()
            + self.buffers.coordinates.capacity() * size_of::<i32>()
    }
}
struct Shared(Option<Arc<Payload>>);
impl Shared {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live semantic owner")
    }
}
impl Clone for Shared {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live semantic owner"),
        )))
    }
}
impl Drop for Shared {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
/// Concrete borrowing recipe. It retains references only, never an owned generic
/// payload, callback, arbitrary byte request or semantic admission.
pub struct PreparedCompositeSemanticRecipe<'p, 'h, P: Sized> {
    provenance: &'p P,
    source: &'h OriginalPreparedHostInput,
    layout: PreparedCompositeSemanticLayout,
}
impl<'p, 'h, P: Sized> PreparedCompositeSemanticRecipe<'p, 'h, P> {
    pub fn new(
        provenance: &'p P,
        source: &'h OriginalPreparedHostInput,
        positions: usize,
        coordinates: CompositeSemanticCoordinates,
    ) -> Result<Self, WorkingMemoryError> {
        Ok(Self {
            provenance,
            source,
            layout: PreparedCompositeSemanticLayout::for_source(source, positions, coordinates)?,
        })
    }
    pub fn source(&self) -> &'h OriginalPreparedHostInput {
        self.source
    }
    pub fn provenance(&self) -> &'p P {
        self.provenance
    }
    pub const fn layout(&self) -> PreparedCompositeSemanticLayout {
        self.layout
    }
    pub fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        WorkingMemoryPool::composite_semantic_required_bytes::<P>(self.layout)
    }
    pub fn allocate(
        self,
        pool: &WorkingMemoryPool,
    ) -> Result<PreparedCompositeSemanticBuilder<'p, P>, OriginalCompositeSemanticStorageError>
    {
        pool.prepare_composite_semantic_storage(self.source, self.provenance, self.layout)
    }
}

/// Unique writable fixed destinations. No spare capacity, source replacement or
/// accounting handle can escape. Neutral storage alone is not semantic authority.
pub struct PreparedCompositeSemanticBuilder<'p, P: Sized> {
    payload: Payload,
    provenance: &'p P,
}
/// Immutable original semantic buffers retaining their actual cold provenance loan.
pub struct OriginalCompositeSemanticStorage<'p, P: Sized> {
    owner: Shared,
    provenance: &'p P,
}
/// Immutable storage after the architecture validates the actual source/destination.
/// Public construction of neutral storage does not construct a family admission.
pub struct BoundCompositeSemanticStorage {
    binding: MediaSessionBinding,
    owner: Shared,
}
impl Clone for BoundCompositeSemanticStorage {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            binding: self.binding.duplicate(),
        }
    }
}
impl<'p, P: Sized> PreparedCompositeSemanticBuilder<'p, P> {
    pub fn fail_semantic(
        mut self,
        cause: CompositeSemanticDiagnostic,
    ) -> OriginalCompositeSemanticStorageError {
        let settlement = self.payload.allowance.end_compilation().err();
        OriginalCompositeSemanticStorageError {
            cause: Cause::Semantic(cause),
            settlement,
            payload: Some(self.payload),
            completed: None,
        }
    }
    pub fn records_mut(&mut self) -> &mut [CompositeSemanticPartRecord] {
        &mut self.payload.buffers.records
    }
    pub fn coordinates_mut(&mut self) -> &mut [i32] {
        &mut self.payload.buffers.coordinates
    }
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.payload.source
    }
    pub fn fail(mut self, cause: WorkingMemoryError) -> OriginalCompositeSemanticStorageError {
        let settlement = self.payload.allowance.end_compilation().err();
        OriginalCompositeSemanticStorageError {
            cause: Cause::Account(cause),
            settlement,
            payload: Some(self.payload),
            completed: None,
        }
    }
    pub fn finish(
        self,
    ) -> Result<OriginalCompositeSemanticStorage<'p, P>, OriginalCompositeSemanticStorageError>
    {
        let mut owner = Arc::new(self.payload);
        match Arc::get_mut(&mut owner)
            .expect("unpublished semantic source")
            .allowance
            .end_compilation()
        {
            Ok(()) => Ok(OriginalCompositeSemanticStorage {
                owner: Shared(Some(owner)),
                provenance: self.provenance,
            }),
            Err(error) => Err(OriginalCompositeSemanticStorageError {
                cause: Cause::Account(error),
                settlement: None,
                payload: Some(Arc::into_inner(owner).expect("unpublished semantic source")),
                completed: None,
            }),
        }
    }
}
impl<'p, P: Sized> OriginalCompositeSemanticStorage<'p, P> {
    /// Retains the complete original body in a boundary rejection.
    pub fn reject_boundary(
        mut self,
        cause: WorkingMemoryError,
    ) -> OriginalCompositeSemanticStorageError {
        let owner = self.owner.0.take().expect("live original semantic storage");
        let payload = Arc::into_inner(owner).expect("unaliased unbound semantic storage");
        OriginalCompositeSemanticStorageError {
            cause: Cause::Account(cause),
            settlement: None,
            payload: Some(payload),
            completed: None,
        }
    }

    pub fn reject(
        mut self,
        cause: CompositeSemanticDiagnostic,
    ) -> OriginalCompositeSemanticStorageError {
        let owner = self.owner.0.take().expect("live original semantic storage");
        let payload = Arc::into_inner(owner).expect("unaliased unbound semantic storage");
        OriginalCompositeSemanticStorageError {
            cause: Cause::Semantic(cause),
            settlement: None,
            payload: Some(payload),
            completed: None,
        }
    }
    pub fn provenance(&self) -> &'p P {
        self.provenance
    }
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.owner.payload().source
    }
    pub fn records(&self) -> &[CompositeSemanticPartRecord] {
        &self.owner.payload().buffers.records
    }
    pub fn coordinates(&self) -> &[i32] {
        &self.owner.payload().buffers.coordinates
    }
    pub fn layout(&self) -> PreparedCompositeSemanticLayout {
        self.owner.payload().layout
    }
    pub fn original_bytes(&self) -> u64 {
        self.owner.payload().allowance.bytes()
    }
    pub fn bind(self, binding: MediaSessionBinding) -> BoundCompositeSemanticStorage {
        BoundCompositeSemanticStorage {
            owner: self.owner,
            binding,
        }
    }
}
impl BoundCompositeSemanticStorage {
    /// Retains the original admitted semantics with the exact completed-copy
    /// transition produced by the shared control exchange. No source is rebound
    /// from a freely supplied revision/frontier or matching contents.
    pub fn for_copied_state(
        &self,
        transition: &CopiedMediaStateBinding,
    ) -> Result<Self, WorkingMemoryError> {
        if !self.binding.matches(&transition.source) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self {
            binding: transition.installed.duplicate(),
            owner: self.owner.clone(),
        })
    }

    pub fn reject(
        self,
        cause: CompositeSemanticDiagnostic,
    ) -> OriginalCompositeSemanticStorageError {
        OriginalCompositeSemanticStorageError {
            cause: Cause::Semantic(cause),
            settlement: None,
            payload: None,
            completed: Some(self),
        }
    }
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.owner.payload().source
    }
    pub fn records(&self) -> &[CompositeSemanticPartRecord] {
        &self.owner.payload().buffers.records
    }
    pub fn coordinates(&self) -> &[i32] {
        &self.owner.payload().buffers.coordinates
    }
    pub fn layout(&self) -> PreparedCompositeSemanticLayout {
        self.owner.payload().layout
    }
    pub fn binding(&self) -> &MediaSessionBinding {
        &self.binding
    }
    pub fn original_bytes(&self) -> u64 {
        self.owner.payload().allowance.bytes()
    }
}
/// Fixed error detail supplied by an architecture equation, not an owned error
/// object or user callback. Diagnostic values confer no source authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeSemanticDiagnostic {
    pub operation: &'static str,
    pub part: Option<usize>,
    pub key: Option<eredu_core::InputMetadataKey>,
    pub expected: Option<u64>,
    pub actual: Option<i128>,
}
impl fmt::Display for CompositeSemanticDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (part {:?}, expected {:?}, actual {:?})",
            self.operation, self.part, self.expected, self.actual
        )
    }
}
impl std::error::Error for CompositeSemanticDiagnostic {}
#[derive(Debug)]
enum Cause {
    Semantic(CompositeSemanticDiagnostic),
    Account(WorkingMemoryError),
    Reserve {
        buffer: usize,
        cause: TryReserveError,
    },
}
/// Owns every real reserved prefix plus the original source and allowance.
/// No partial buffer, raw Arc, Weak, or retry grant escapes.
pub struct OriginalCompositeSemanticStorageError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    payload: Option<Payload>,
    completed: Option<BoundCompositeSemanticStorage>,
}
impl OriginalCompositeSemanticStorageError {
    pub fn semantic_failure(&self) -> Option<&CompositeSemanticDiagnostic> {
        match &self.cause {
            Cause::Semantic(value) => Some(value),
            _ => None,
        }
    }
    fn rejected(error: WorkingMemoryError) -> Self {
        Self {
            cause: Cause::Account(error),
            settlement: None,
            payload: None,
            completed: None,
        }
    }
    pub fn retained_bytes(&self) -> u64 {
        self.payload.as_ref().map_or_else(
            || self.completed.as_ref().map_or(0, |p| p.original_bytes()),
            |p| p.allowance.bytes(),
        )
    }
    pub fn retained_heap_bytes(&self) -> usize {
        self.payload.as_ref().map_or_else(
            || {
                self.completed
                    .as_ref()
                    .map_or(0, |value| value.owner.payload().heap_bytes())
            },
            Payload::heap_bytes,
        )
    }
    pub fn failed_buffer(&self) -> Option<usize> {
        match self.cause {
            Cause::Reserve { buffer, .. } => Some(buffer),
            _ => None,
        }
    }
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Account(e) => Some(e),
            _ => None,
        })
    }
}
impl fmt::Debug for OriginalCompositeSemanticStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalCompositeSemanticStorageError")
            .field("cause", &self.cause)
            .field("settlement", &self.settlement)
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}
impl fmt::Display for OriginalCompositeSemanticStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "original media semantic construction: {:?}", self.cause)
    }
}
impl std::error::Error for OriginalCompositeSemanticStorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Semantic(e) => Some(e),
            Cause::Account(e) => Some(e),
            Cause::Reserve { cause, .. } => Some(cause),
        }
    }
}
impl WorkingMemoryPool {
    /// Exact fixed destinations and named concrete control representations.
    /// P is borrowed only: its payload/layout/destructor is never charged here.
    pub fn composite_semantic_required_bytes<P: Sized>(
        layout: PreparedCompositeSemanticLayout,
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
            size_of::<Buffers>(),
            size_of::<PreparedCompositeSemanticLayout>(),
            size_of::<PreparedCompositeSemanticRecipe<'_, '_, P>>(),
            size_of::<PreparedCompositeSemanticBuilder<'_, P>>(),
            size_of::<OriginalCompositeSemanticStorage<'_, P>>(),
            size_of::<BoundCompositeSemanticStorage>(),
            size_of::<Shared>(),
            size_of::<Option<Arc<Payload>>>(),
            size_of::<OriginalPreparedHostInput>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<InferenceExecutionIdentity>(),
            size_of::<InferenceStateRevision>(),
            size_of::<crate::replicated_session::ReplicatedTextControlOrigin>(),
            size_of::<MediaSessionBinding>(),
            size_of::<OriginalCompositeSemanticStorageError>(),
            size_of::<
                Result<
                    PreparedCompositeSemanticBuilder<'_, P>,
                    OriginalCompositeSemanticStorageError,
                >,
            >(),
            size_of::<
                Result<
                    OriginalCompositeSemanticStorage<'_, P>,
                    OriginalCompositeSemanticStorageError,
                >,
            >(),
            size_of::<Result<BoundCompositeSemanticStorage, OriginalCompositeSemanticStorageError>>(
            ),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Result<BoundCompositeSemanticStorage, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalCompositeSemanticStorageError>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ];
        let bytes = controls
            .into_iter()
            .try_fold(layout.heap_bytes()?, usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }
    pub fn prepare_composite_semantic_storage<'p, P: Sized>(
        &self,
        source: &OriginalPreparedHostInput,
        provenance: &'p P,
        layout: PreparedCompositeSemanticLayout,
    ) -> Result<PreparedCompositeSemanticBuilder<'p, P>, OriginalCompositeSemanticStorageError>
    {
        self.prepare_composite_semantic_storage_inner(source, provenance, layout, None)
    }
    fn prepare_composite_semantic_storage_inner<'p, P: Sized>(
        &self,
        source: &OriginalPreparedHostInput,
        provenance: &'p P,
        layout: PreparedCompositeSemanticLayout,
        fail: Option<usize>,
    ) -> Result<PreparedCompositeSemanticBuilder<'p, P>, OriginalCompositeSemanticStorageError>
    {
        source
            .validate_pool(self)
            .map_err(OriginalCompositeSemanticStorageError::rejected)?;
        if layout.parts != source.parts().len() {
            return Err(OriginalCompositeSemanticStorageError::rejected(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let bytes = Self::composite_semantic_required_bytes::<P>(layout)
            .map_err(OriginalCompositeSemanticStorageError::rejected)?;
        let allowance = self
            .admit_source_compiler(bytes)
            .map_err(OriginalCompositeSemanticStorageError::rejected)?;
        let mut payload = Payload {
            buffers: Buffers {
                records: Vec::new(),
                coordinates: Vec::new(),
            },
            layout,
            source: source.clone(),
            allowance,
        };
        macro_rules! reserve {
            ($field:ident,$count:expr,$index:expr) => {{
                let requested = if fail == Some($index) {
                    usize::MAX
                } else {
                    $count
                };
                if let Err(cause) = payload.buffers.$field.try_reserve_exact(requested) {
                    let settlement = payload.allowance.end_compilation().err();
                    return Err(OriginalCompositeSemanticStorageError {
                        cause: Cause::Reserve {
                            buffer: $index,
                            cause,
                        },
                        settlement,
                        payload: Some(payload),
                        completed: None,
                    });
                }
            }};
        }
        reserve!(records, layout.parts, 0);
        reserve!(
            coordinates,
            layout.coordinate_count().expect("checked layout"),
            1
        );
        payload
            .buffers
            .records
            .resize(layout.parts, CompositeSemanticPartRecord::default());
        payload
            .buffers
            .coordinates
            .resize(layout.coordinate_count().expect("checked layout"), 0);
        Ok(PreparedCompositeSemanticBuilder {
            payload,
            provenance,
        })
    }
}

#[cfg(test)]
mod tests;
