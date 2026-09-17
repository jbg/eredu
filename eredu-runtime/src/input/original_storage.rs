//! Source-derived host controls constructed inside the one original B compiler.
use super::host::{HostInputPartView, HostTensorValues, HostTensorView};
use super::*;
use crate::working_memory::{
    OriginalPreparedHostInput, OriginalPreparedInputCustody, PreparedInputHostCustody,
    WorkingMemoryError,
};
use eredu_core::{checkpoint::TensorDtype, input::InputWordWriteError};
use eredu_nn::sequence_layout::{
    PatchEncoderTableError, PatchEncoderTableLayout, PatchEncoderTables,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    convert::Infallible,
    fmt,
    marker::PhantomData,
    mem::size_of,
    ops::Deref,
    sync::{Arc, atomic::AtomicUsize},
};

/// Common immutable owner consumed by the existing media driver. Ordinary
/// inputs preserve their owned representation; original clones share controls.
///
/// ```compile_fail
/// use eredu_runtime::input::PreparedModelInputOwner;
/// fn no_unboxing<T>(source:PreparedModelInputOwner<T>) { source.into_parts(); }
/// ```
pub struct PreparedModelInputOwner<T>(PreparedOwner<T>);
enum PreparedOwner<T> {
    Ordinary(PreparedModelInput<T>),
    Original(SharedPrepared<T>),
}
struct PreparedPayload<T> {
    value: PreparedModelInput<T>,
    encoder: Option<EncoderTables>,
    source: OriginalPreparedHostInput,
    custody: PreparedInputHostCustody,
}
struct EncoderTables {
    layout: PatchEncoderTableLayout,
    integers: Vec<i32>,
    floats: Vec<f32>,
}
impl EncoderTables {
    fn view(&self) -> PatchEncoderTables<'_> {
        PatchEncoderTables::new(self.layout, &self.integers, &self.floats)
            .expect("completed fixed encoder tables")
    }
}
// All typed and erased strong-owner exits consume through this concrete path.
// No Weak or owning raw Arc is exported.
fn retire_prepared<T>(value: Arc<PreparedPayload<T>>) {
    drop(Arc::into_inner(value));
}
trait EncoderStorage: Send + Sync {
    fn source(&self) -> &OriginalPreparedHostInput;
    fn identity(&self) -> &PreparedInputIdentity;
    fn tables(&self) -> Option<PatchEncoderTables<'_>>;
    fn retire(self: Arc<Self>);
}
impl<T: Send + Sync + 'static> EncoderStorage for PreparedPayload<T> {
    fn source(&self) -> &OriginalPreparedHostInput {
        &self.source
    }
    fn identity(&self) -> &PreparedInputIdentity {
        self.value.identity()
    }
    fn tables(&self) -> Option<PatchEncoderTables<'_>> {
        self.encoder.as_ref().map(EncoderTables::view)
    }
    fn retire(self: Arc<Self>) {
        retire_prepared(self);
    }
}
/// Closed table projection of the actual original prepared payload. Its Arc is
/// the existing B allocation, including native slots, source and original custody.
/// Cloning creates no table/handle allocation, grant or bind attempt.
///
/// ```compile_fail
/// use eredu_runtime::input::OriginalEncoderTableProjection;
/// fn no_weak(value: OriginalEncoderTableProjection) { value.downgrade(); }
/// ```
pub struct OriginalEncoderTableProjection(Option<Arc<dyn EncoderStorage>>);
impl OriginalEncoderTableProjection {
    /// Actual I identity, not a digest-derived source token.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.0.as_deref().expect("live projection").source()
    }
    /// Exact ordered descriptor identity of the retained prepared payload.
    pub fn identity(&self) -> &PreparedInputIdentity {
        self.0.as_deref().expect("live projection").identity()
    }
    /// Borrows the original fixed tables; no numerical copy occurs.
    pub fn tables(&self) -> PatchEncoderTables<'_> {
        self.0.as_deref().expect("live projection").tables().expect("completed encoder projection")
    }
}
impl Clone for OriginalEncoderTableProjection {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live projection"))))
    }
}
impl Drop for OriginalEncoderTableProjection {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            value.retire();
        }
    }
}
impl fmt::Debug for OriginalEncoderTableProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalEncoderTableProjection")
            .field("layout", &self.tables().layout())
            .finish_non_exhaustive()
    }
}
/// Closed projection of the actual original B payload and its host source.
/// Encoder tables are optional architecture data, not the proof of B custody.
/// Clones share the original allocation and cannot create a materialization or
/// native bind attempt. No Weak/raw owning handle or replacement slots escape.
pub struct OriginalPreparedInputProjection(Option<Arc<dyn EncoderStorage>>);
impl OriginalPreparedInputProjection {
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.0.as_deref().expect("live prepared projection").source()
    }
    pub fn identity(&self) -> &PreparedInputIdentity {
        self.0.as_deref().expect("live prepared projection").identity()
    }
    /// Whether this same B contains the completed selected encoder tables.
    pub fn has_encoder_tables(&self) -> bool {
        self.0.as_deref().expect("live prepared projection").tables().is_some()
    }
    /// Moves the same owner into its more restrictive table loan. Missing
    /// tables return the complete source without allocating or losing custody.
    pub fn into_encoder_tables(mut self) -> Result<OriginalEncoderTableProjection, Self> {
        if !self.has_encoder_tables() { return Err(self); }
        Ok(OriginalEncoderTableProjection(self.0.take()))
    }
}
impl Clone for OriginalPreparedInputProjection {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live prepared projection"))))
    }
}
impl Drop for OriginalPreparedInputProjection {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() { value.retire(); }
    }
}
impl fmt::Debug for OriginalPreparedInputProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalPreparedInputProjection")
            .field("encoder_tables", &self.has_encoder_tables()).finish_non_exhaustive()
    }
}
struct SharedPrepared<T>(Option<Arc<PreparedPayload<T>>>);
impl<T> Drop for SharedPrepared<T> {
    fn drop(&mut self) {
        if let Some(v) = self.0.take() {
            retire_prepared(v);
        }
    }
}
impl<T> Clone for SharedPrepared<T> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live prepared owner"),
        )))
    }
}
impl<T> From<PreparedModelInput<T>> for PreparedModelInputOwner<T> {
    fn from(value: PreparedModelInput<T>) -> Self {
        Self(PreparedOwner::Ordinary(value))
    }
}
impl<T> AsRef<PreparedModelInput<T>> for PreparedModelInputOwner<T> {
    fn as_ref(&self) -> &PreparedModelInput<T> {
        match &self.0 {
            PreparedOwner::Ordinary(v) => v,
            PreparedOwner::Original(v) => &v.0.as_deref().expect("live prepared owner").value,
        }
    }
}
impl<T> Deref for PreparedModelInputOwner<T> {
    type Target = PreparedModelInput<T>;
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}
impl<T: Clone> Clone for PreparedModelInputOwner<T> {
    fn clone(&self) -> Self {
        Self(match &self.0 {
            PreparedOwner::Ordinary(v) => PreparedOwner::Ordinary(v.clone()),
            PreparedOwner::Original(v) => PreparedOwner::Original(v.clone()),
        })
    }
}
impl<T> PreparedModelInputOwner<T> {
    // Only the shared original projection worker can obtain this account-only
    // alias. No native tensor or source payload is retained by the returned pin.
    pub(super) fn workspace_custody(&self) -> Option<PreparedInputHostCustody> {
        self.workspace_custody_ref().map(PreparedInputHostCustody::share)
    }
    pub(super) fn workspace_custody_ref(&self) -> Option<&PreparedInputHostCustody> {
        match &self.0 {
            PreparedOwner::Ordinary(_) => None,
            PreparedOwner::Original(value) => Some(
                &value.0.as_deref().expect("live prepared owner").custody,
            ),
        }
    }
    /// Fixed source-derived encoder tables installed by the same original B
    /// constructor. Ordinary values and copied fallback inputs have no table grant.
    pub fn original_encoder_tables(&self) -> Option<PatchEncoderTables<'_>> {
        match &self.0 {
            PreparedOwner::Ordinary(_) => None,
            PreparedOwner::Original(v) => {
                v.0.as_deref()
                    .expect("live prepared owner")
                    .encoder
                    .as_ref()
                    .map(EncoderTables::view)
            }
        }
    }
    /// Shares the actual native/source payload for ordinary metadata tracing.
    /// A non-Send tensor implementation can still use ordinary and typed original
    /// execution; only cross-type diagnostic projection needs this bound.
    pub fn original_encoder_projection(&self) -> Option<OriginalEncoderTableProjection>
    where
        T: Send + Sync + 'static,
    {
        let PreparedOwner::Original(value) = &self.0 else {
            return None;
        };
        let payload = value.0.as_ref().expect("live prepared owner");
        payload.encoder.as_ref()?;
        let shared = Arc::clone(payload);
        let erased: Arc<dyn EncoderStorage> = shared;
        Some(OriginalEncoderTableProjection(Some(erased)))
    }
    /// Retains the exact completed B allocation, including any native slots,
    /// original host source and its paid account. Ordinary inputs have no loan.
    pub fn original_projection(&self) -> Option<OriginalPreparedInputProjection>
    where T: Send + Sync + 'static,
    {
        let PreparedOwner::Original(value) = &self.0 else { return None; };
        let payload = value.0.as_ref().expect("live prepared owner");
        let shared = Arc::clone(payload);
        let erased: Arc<dyn EncoderStorage> = shared;
        Some(OriginalPreparedInputProjection(Some(erased)))
    }
    /// Genuine original source, when this owner was constructed by the B worker.
    /// Ordinary conversion never installs this identity.
    pub fn original_source(&self) -> Option<&OriginalPreparedHostInput> {
        match &self.0 {
            PreparedOwner::Ordinary(_) => None,
            PreparedOwner::Original(v) => {
                Some(&v.0.as_deref().expect("live prepared owner").source)
            }
        }
    }
}
impl<T: fmt::Debug> fmt::Debug for PreparedModelInputOwner<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

struct PartsPayload<T> {
    parts: Vec<PreparedInputPart<T>>,
    source: OriginalPreparedHostInput,
    custody: PreparedInputHostCustody,
}
/// Immutable second native-handle view of the same original source. It cannot
/// export its vector or acquire arbitrary metadata attachments.
pub struct SharedPreparedInputParts<T>(Option<Arc<PartsPayload<T>>>);
impl<T> SharedPreparedInputParts<T> {
    /// Exact original source used for every slot.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.0.as_deref().expect("live parts").source
    }
}
impl<T> AsRef<[PreparedInputPart<T>]> for SharedPreparedInputParts<T> {
    fn as_ref(&self) -> &[PreparedInputPart<T>] {
        &self.0.as_deref().expect("live parts").parts
    }
}
impl<T> Clone for SharedPreparedInputParts<T> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live parts"))))
    }
}
impl<T> Drop for SharedPreparedInputParts<T> {
    fn drop(&mut self) {
        if let Some(v) = self.0.take() {
            drop(Arc::into_inner(v));
        }
    }
}
impl<T> fmt::Debug for SharedPreparedInputParts<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPreparedInputParts")
            .field("len", &self.as_ref().len())
            .finish()
    }
}

struct Body<T, U, N> {
    prepared: Option<PreparedModelInputOwner<T>>,
    parts: Option<SharedPreparedInputParts<U>>,
    cache: Option<SharedPreparedInputCacheIdentity>,
    // Complete and partial prefixes remain owned on every returned failure.
    prepared_prefix: Vec<PreparedInputPart<T>>,
    parts_prefix: Vec<PreparedInputPart<U>>,
    identity_prefix: Vec<InputPartDescriptor>,
    cache_prefix: Vec<InputPartDescriptor>,
    // Kept here during every fallible reserve/fill; moved into the immutable
    // prepared owner only after both actual buffers are complete.
    encoder_prefix: Option<EncoderTables>,
    native: Option<N>,
    source: OriginalPreparedHostInput,
    custody: PreparedInputHostCustody,
}
/// Closed completed or failed B storage. Clones retain the same immutable body;
/// no clone creates a bind attempt, native handle, descriptor or fingerprint.
pub struct PreparedModelInputSource<T, U, N>(Option<Arc<Body<T, U, N>>>);
impl<T, U, N> PreparedModelInputSource<T, U, N> {
    fn body(&self) -> &Body<T, U, N> {
        self.0.as_deref().expect("live source body")
    }
    /// Exact I source, also retained after a partial construction failure.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.body().source
    }
    /// Borrow a completed prepared view. Partial construction returns None.
    pub fn prepared(&self) -> Option<&PreparedModelInputOwner<T>> {
        self.body().prepared.as_ref()
    }
    /// Borrow the completed native-handle view.
    pub fn parts(&self) -> Option<&SharedPreparedInputParts<U>> {
        self.body().parts.as_ref()
    }
    /// Borrow the completed cache description; every alias retains original B.
    pub fn cache(&self) -> Option<&SharedPreparedInputCacheIdentity> {
        self.body().cache.as_ref()
    }
    /// Borrow the exact native initializer output, including a failed prefix.
    pub fn native(&self) -> Option<&N> {
        self.body().native.as_ref()
    }
    /// Full B residence shared by this body and its escaped concrete owners.
    pub fn original_bytes(&self) -> u64 {
        self.body().custody.bytes()
    }
}
impl<T, U, N> Clone for PreparedModelInputSource<T, U, N> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live source body"))))
    }
}
impl<T, U, N> Drop for PreparedModelInputSource<T, U, N> {
    fn drop(&mut self) {
        if let Some(v) = self.0.take() {
            drop(Arc::into_inner(v));
        }
    }
}
impl<T, U, N> fmt::Debug for PreparedModelInputSource<T, U, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedModelInputSource")
            .field("complete", &self.body().cache.is_some())
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}

/// Fixed host failure or the actual native initializer/slot failure. Source and
/// partial storage are returned separately in the same owning compiler result.
#[derive(Debug, thiserror::Error)]
pub enum PreparedModelInputSourceError<E: std::error::Error> {
    /// A checked vector reservation failed.
    #[error("prepared source storage: {0}")]
    Allocation(#[source] TryReserveError),
    /// A source descriptor violated its prevalidated representation.
    #[error("prepared source descriptor: {0}")]
    Descriptor(#[source] PreparedInputError),
    /// Identity/custody construction failed without formatting.
    #[error("prepared source custody: {0}")]
    Custody(#[source] WorkingMemoryError),
    /// The concrete native initializer or source-slot handle constructor failed.
    #[error("prepared source native storage: {0}")]
    Native(#[source] E),
    /// Fixed semantic table construction failure; source and prefixes stay owned.
    #[error("prepared encoder tables: {0}")]
    Encoder(#[source] PatchEncoderTableError),
}

/// Allocation-free recipe over the genuine immutable source. T and U are the
/// two actual handle representations; N is the initializer's complete/partial
/// native output. This is not a second account or a late adoption operation.
pub struct PreparedModelInputSourcePlan<'a, T, U, N, E> {
    source: &'a OriginalPreparedHostInput,
    bytes: usize,
    words: usize,
    encoder: Option<PatchEncoderTableLayout>,
    marker: PhantomData<fn() -> (T, U, N, E)>,
}
fn add(total: &mut usize, bytes: usize) -> Result<(), WorkingMemoryError> {
    *total = total
        .checked_add(bytes)
        .ok_or(WorkingMemoryError::Overflow)?;
    Ok(())
}
fn array<T>(n: usize) -> Result<usize, WorkingMemoryError> {
    Layout::array::<T>(n)
        .map(|v| v.size())
        .map_err(|_| WorkingMemoryError::Overflow)
}
fn arc<T>() -> Result<usize, WorkingMemoryError> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map(|v| v.0.pad_to_align().size())
        .map_err(|_| WorkingMemoryError::Overflow)
}
fn extent_key(extent: InputExtent) -> Result<u32, WorkingMemoryError> {
    match extent {
        InputExtent::PatchGrid {
            time,
            height,
            width,
        } => {
            for v in [time, height, width] {
                u32::try_from(v).map_err(|_| WorkingMemoryError::Overflow)?;
            }
            Ok(0)
        }
        InputExtent::AudioValidFrames(v) => {
            u32::try_from(v).map_err(|_| WorkingMemoryError::Overflow)?;
            Ok(1)
        }
        _ => Err(WorkingMemoryError::UnknownBound),
    }
}
impl<'a, T, U, N, E: std::error::Error> PreparedModelInputSourcePlan<'a, T, U, N, E> {
    /// Measures both source-derived views, both descriptors, cache/identity/body
    /// controls and the concrete constructor/error population before B acceptance.
    /// Native resource/handle bytes are supplied by the enclosing native recipe.
    pub fn new(source: &'a OriginalPreparedHostInput) -> Result<Self, WorkingMemoryError> {
        let p = source.parts().len();
        u32::try_from(p).map_err(|_| WorkingMemoryError::Overflow)?;
        if p == 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut bytes = 0;
        let mut words = 1usize;
        for value in [
            array::<PreparedInputPart<T>>(p)?,
            array::<PreparedInputPart<U>>(p)?,
            array::<InputPartDescriptor>(p)?,
            array::<InputPartDescriptor>(p)?,
        ] {
            add(&mut bytes, value)?;
        }
        for part in source.parts() {
            // Same canonical word populations streamed by visit_encoded_words:
            // four part tags/counts, payload, keyed metadata and typed extents.
            add(&mut words, 4)?;
            add(&mut words, 2)?;
            add(&mut words, part.payload().shape.len())?;
            for (_, slot) in part.metadata() {
                add(&mut words, 3)?;
                add(&mut words, slot.shape.len())?;
            }
            for extent in part.extents() {
                add(
                    &mut words,
                    match extent {
                        InputExtent::PatchGrid { .. } => 4,
                        InputExtent::AudioValidFrames(_) => 2,
                        _ => return Err(WorkingMemoryError::UnknownBound),
                    },
                )?;
            }
            if !part.kind().accepts(part.modality()) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            let m = part.metadata().len();
            let e = part.extents().len();
            if m > 3 || e > 2 {
                return Err(WorkingMemoryError::UnknownBound);
            }
            for value in [
                array::<(InputMetadataKey, T)>(m)?,
                array::<(InputMetadataKey, U)>(m)?,
                array::<InputExtent>(e)?,
                array::<InputExtent>(e)?,
                array::<(InputMetadataKey, InputTensorIdentity)>(m)?,
                array::<(InputMetadataKey, InputTensorIdentity)>(m)?,
                array::<(u32, InputExtent)>(e)?,
                array::<(u32, InputExtent)>(e)?,
            ] {
                add(&mut bytes, value)?;
            }
            for extent in part.extents() {
                extent_key(*extent)?;
            }
        }
        for i in 0..source.slot_count() {
            let slot = source.slot(i).ok_or(WorkingMemoryError::IdentityMismatch)?;
            if slot.shape.is_empty() || slot.shape.contains(&0) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            if !matches!(
                slot.values,
                HostTensorValues::U32(_) | HostTensorValues::I32(_) | HostTensorValues::F32(_) | HostTensorValues::Bool(_)
            ) {
                return Err(WorkingMemoryError::UnknownBound);
            }
            u32::try_from(slot.shape.len()).map_err(|_| WorkingMemoryError::Overflow)?;
            for d in slot.shape {
                u32::try_from(*d).map_err(|_| WorkingMemoryError::Overflow)?;
            }
            add(&mut bytes, array::<usize>(slot.shape.len())?)?;
            add(&mut bytes, array::<usize>(slot.shape.len())?)?;
        }
        for value in [
            arc::<Body<T, U, N>>()?,
            size_of::<Body<T, U, N>>(),
            size_of::<Option<Body<T, U, N>>>(),
            arc::<PreparedPayload<T>>()?,
            size_of::<PreparedPayload<T>>(),
            size_of::<Option<PreparedPayload<T>>>(),
            size_of::<OriginalPreparedInputProjection>(),
            size_of::<Option<OriginalPreparedInputProjection>>(),
            size_of::<Result<OriginalEncoderTableProjection, OriginalPreparedInputProjection>>(),
            size_of::<Option<Arc<dyn EncoderStorage>>>(),
            size_of::<Arc<dyn EncoderStorage>>(),
            size_of::<Arc<PreparedPayload<T>>>(),
            arc::<PartsPayload<U>>()?,
            size_of::<PartsPayload<U>>(),
            size_of::<Option<PartsPayload<U>>>(),
            SharedPreparedInputCacheIdentity::original_control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            64,
            64, // fixed allocated fingerprint byte arrays, moved into String
            size_of::<Sha256>(),
            size_of::<Sha256>(),
            size_of::<[u8; 32]>(),
            size_of::<[u8; 64]>(),
            size_of::<u64>(),
            size_of::<u32>(),
            size_of::<Self>(),
            size_of::<(OriginalPreparedInputCustody, PreparedInputHostCustody)>(),
            size_of::<PreparedModelInputSourceError<E>>(),
            size_of::<Result<(), PreparedModelInputSourceError<E>>>(),
            size_of::<
                Result<
                    PreparedModelInputSource<T, U, N>,
                    (
                        PreparedModelInputSource<T, U, N>,
                        PreparedModelInputSourceError<E>,
                    ),
                >,
            >(),
            size_of::<Result<N, (N, E)>>(),
            size_of::<Result<(T, U), E>>(),
            size_of::<[Option<(InputMetadataKey, T)>; 3]>(),
            size_of::<[Option<(InputMetadataKey, U)>; 3]>(),
            size_of::<[Option<(InputMetadataKey, InputTensorIdentity)>; 3]>(),
            size_of::<[Option<(u32, InputExtent)>; 2]>(),
            size_of::<PreparedInputPart<T>>(),
            size_of::<PreparedInputPart<U>>(),
            size_of::<InputPartDescriptor>(),
            size_of::<InputTensorIdentity>(),
            size_of::<Result<InputTensorIdentity, PreparedInputError>>(),
            size_of::<Result<InputPartDescriptor, PreparedInputError>>(),
            size_of::<Result<PreparedInputIdentity, PreparedInputError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Vec<usize>>(),
            size_of::<PreparedInputCacheIdentity>(),
            size_of::<String>(),
            size_of::<String>(),
        ] {
            add(&mut bytes, value)?;
        }
        Ok(Self {
            source,
            bytes,
            words,
            encoder: None,
            marker: PhantomData,
        })
    }
    /// Extends this complete source recipe before its enclosing original
    /// comparison. A completed B1/B2 owner cannot acquire these buffers later.
    pub fn with_encoder_tables(
        mut self,
        layout: PatchEncoderTableLayout,
    ) -> Result<Self, WorkingMemoryError> {
        if self.encoder.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        for bytes in [
            array::<i32>(layout.integer_count())?,
            array::<f32>(layout.float_count())?,
            size_of::<EncoderTables>(),
            size_of::<PatchEncoderTableLayout>(),
            size_of::<PatchEncoderTableError>(),
            size_of::<Result<(), PatchEncoderTableError>>(),
            size_of::<PatchEncoderTables<'_>>(),
            size_of::<OriginalEncoderTableProjection>(),
            size_of::<Option<OriginalEncoderTableProjection>>(),
            size_of::<Option<Arc<dyn EncoderStorage>>>(),
            size_of::<Arc<dyn EncoderStorage>>(),
            size_of::<Arc<PreparedPayload<T>>>(),
            size_of::<&mut [i32]>(),
            size_of::<&mut [f32]>(),
        ] {
            add(&mut self.bytes, bytes)?;
        }
        self.encoder = Some(layout);
        Ok(self)
    }
    /// Concrete host allocation/control contribution; no native numerical bytes.
    pub fn required_storage_bytes(&self) -> usize {
        self.bytes
    }
    /// The exact immutable source used for both measurement and construction.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.source
    }
    /// Constructs once inside the enclosing original compiler. The initializer
    /// receives the single native custody; lowering only borrows its result and
    /// exact canonical slot. No public custody clone or replacement grant exists.
    pub fn construct(
        self,
        custody: OriginalPreparedInputCustody,
        initialize: impl FnOnce(OriginalPreparedInputCustody) -> Result<N, (N, E)>,
        lower: impl FnMut(&N, usize, HostTensorView<'_>) -> Result<(T, U), E>,
    ) -> Result<
        PreparedModelInputSource<T, U, N>,
        (
            PreparedModelInputSource<T, U, N>,
            PreparedModelInputSourceError<E>,
        ),
    > {
        self.construct_with_encoder(custody, initialize, lower, |_, _| {
            Err(PatchEncoderTableError::SourceGeometry)
        })
    }
    /// Shared constructor with an architecture-owned fixed table fill. The
    /// callback receives only the prepaid actual destinations, never an account.
    /// It is used only when this recipe includes encoder tables.
    pub fn construct_with_encoder(
        self,
        custody: OriginalPreparedInputCustody,
        initialize: impl FnOnce(OriginalPreparedInputCustody) -> Result<N, (N, E)>,
        mut lower: impl FnMut(&N, usize, HostTensorView<'_>) -> Result<(T, U), E>,
        fill: impl FnOnce(&mut [i32], &mut [f32]) -> Result<(), PatchEncoderTableError>,
    ) -> Result<
        PreparedModelInputSource<T, U, N>,
        (
            PreparedModelInputSource<T, U, N>,
            PreparedModelInputSourceError<E>,
        ),
    > {
        let (native, host) = custody.into_model_input_parts();
        let mut output = PreparedModelInputSource(Some(Arc::new(Body {
            prepared: None,
            parts: None,
            cache: None,
            prepared_prefix: Vec::new(),
            parts_prefix: Vec::new(),
            identity_prefix: Vec::new(),
            cache_prefix: Vec::new(),
            encoder_prefix: self.encoder.map(|layout| EncoderTables {
                layout,
                integers: Vec::new(),
                floats: Vec::new(),
            }),
            native: None,
            source: self.source.clone(),
            custody: host,
        })));
        let body = Arc::get_mut(output.0.as_mut().expect("new body")).expect("unique constructor");
        match initialize(native) {
            Ok(value) => body.native = Some(value),
            Err((value, error)) => {
                body.native = Some(value);
                return Err((output, PreparedModelInputSourceError::Native(error)));
            }
        }
        let result = (|| {
            if let Some(tables) = &mut body.encoder_prefix {
                reserve(&mut tables.integers, tables.layout.integer_count())
                    .map_err(PreparedModelInputSourceError::Allocation)?;
                reserve(&mut tables.floats, tables.layout.float_count())
                    .map_err(PreparedModelInputSourceError::Allocation)?;
                tables.integers.resize(tables.layout.integer_count(), 0);
                tables.floats.resize(tables.layout.float_count(), 0.);
                fill(&mut tables.integers, &mut tables.floats)
                    .map_err(PreparedModelInputSourceError::Encoder)?;
            }
            let p = self.source.parts().len();
            reserve(&mut body.prepared_prefix, p)
                .map_err(PreparedModelInputSourceError::Allocation)?;
            reserve(&mut body.parts_prefix, p)
                .map_err(PreparedModelInputSourceError::Allocation)?;
            reserve(&mut body.identity_prefix, p)
                .map_err(PreparedModelInputSourceError::Allocation)?;
            reserve(&mut body.cache_prefix, p)
                .map_err(PreparedModelInputSourceError::Allocation)?;
            let mut slot_index = 0;
            for part in self.source.parts() {
                let (t, u) = lower(
                    body.native.as_ref().expect("initialized native source"),
                    slot_index,
                    part.payload(),
                )
                .map_err(PreparedModelInputSourceError::Native)?;
                slot_index += 1;
                let mut ts = [const { None }; 3];
                let mut us = [const { None }; 3];
                for (i, (key, slot)) in part.metadata().enumerate() {
                    let (t, u) = lower(
                        body.native.as_ref().expect("initialized native source"),
                        slot_index,
                        slot,
                    )
                    .map_err(PreparedModelInputSourceError::Native)?;
                    ts[i] = Some((key, t));
                    us[i] = Some((key, u));
                    slot_index += 1;
                }
                let m = part.metadata().len();
                let tp = PreparedInputPart {
                    modality: part.modality(),
                    payload: payload(part.kind(), t),
                    metadata: fixed3(ts, m),
                    extents: copy_extents(part.extents())
                        .map_err(PreparedModelInputSourceError::Allocation)?,
                };
                body.prepared_prefix.push(tp);
                let up = PreparedInputPart {
                    modality: part.modality(),
                    payload: payload(part.kind(), u),
                    metadata: fixed3(us, m),
                    extents: copy_extents(part.extents())
                        .map_err(PreparedModelInputSourceError::Allocation)?,
                };
                body.parts_prefix.push(up);
                body.identity_prefix.push(descriptor(&part)?);
                body.cache_prefix.push(descriptor(&part)?);
            }
            let identity = PreparedInputIdentity::new(std::mem::take(&mut body.identity_prefix))
                .map_err(PreparedModelInputSourceError::Descriptor)?;
            let prepared = PreparedModelInput {
                parts: std::mem::take(&mut body.prepared_prefix),
                identity,
            };
            body.prepared = Some(PreparedModelInputOwner(PreparedOwner::Original(
                SharedPrepared(Some(Arc::new(PreparedPayload {
                    value: prepared,
                    encoder: body.encoder_prefix.take(),
                    source: body.source.clone(),
                    custody: body.custody.share(),
                }))),
            )));
            body.parts = Some(SharedPreparedInputParts(Some(Arc::new(PartsPayload {
                parts: std::mem::take(&mut body.parts_prefix),
                source: body.source.clone(),
                custody: body.custody.share(),
            }))));
            let identity = PreparedInputIdentity::new(std::mem::take(&mut body.cache_prefix))
                .map_err(PreparedModelInputSourceError::Descriptor)?;
            let semantic = super::text_identity::fixed_hex(*body.source.content_digest());
            let mut prefix = Sha256::new();
            prefix.update(b"eredu-prepared-input-cache-v1\0");
            let count = identity
                .encoded_word_count()
                .map_err(PreparedModelInputSourceError::Descriptor)?;
            debug_assert_eq!(
                count, self.words,
                "source plan and canonical encoder word populations"
            );
            prefix.update((count as u64).to_le_bytes());
            identity
                .visit_encoded_words(|word| {
                    prefix.update(word.to_le_bytes());
                    Ok::<_, Infallible>(())
                })
                .map_err(|e| match e {
                    InputWordWriteError::Encoding(e) => {
                        PreparedModelInputSourceError::Descriptor(e)
                    }
                    InputWordWriteError::Sink(never) => match never {},
                })?;
            prefix.update(64u64.to_le_bytes());
            prefix.update(semantic.as_bytes());
            let prefix = super::text_identity::fixed_hex(prefix.finalize().into());
            let cache =
                PreparedInputCacheIdentity::from_validated_text_parts(identity, semantic, prefix);
            body.cache = Some(
                SharedPreparedInputCacheIdentity::new_original(
                    cache,
                    body.custody.share(),
                    body.source.clone(),
                )
                .map_err(PreparedModelInputSourceError::Custody)?,
            );
            Ok(())
        })();
        match result {
            Ok(()) => Ok(output),
            Err(error) => Err((output, error)),
        }
    }
}
fn payload<T>(kind: InputPayloadKind, value: T) -> PreparedInputPayload<T> {
    match kind {
        InputPayloadKind::TokenIds => PreparedInputPayload::TokenIds(value),
        InputPayloadKind::Tensor => PreparedInputPayload::Tensor(value),
        InputPayloadKind::Embeddings => PreparedInputPayload::Embeddings(value),
        _ => unreachable!("validated source kind"),
    }
}
// Tests request a genuine checked capacity failure at one selected reservation;
// production uses exactly the measured source count, with no retry/growth path.
#[cfg(test)]
thread_local! { static FAIL_HOST_RESERVATION: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }
fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), TryReserveError> {
    #[cfg(test)]
    let additional = FAIL_HOST_RESERVATION.with(|counter| match counter.get() {
        Some(0) => {
            counter.set(None);
            usize::MAX
        }
        Some(n) => {
            counter.set(Some(n - 1));
            additional
        }
        None => additional,
    });
    values.try_reserve_exact(additional)
}
fn copy_extents(input: &[InputExtent]) -> Result<Vec<InputExtent>, TryReserveError> {
    let mut values = Vec::new();
    reserve(&mut values, input.len())?;
    values.extend_from_slice(input);
    Ok(values)
}
fn fixed3<K: Ord, V>(mut a: [Option<(K, V)>; 3], n: usize) -> InputIdentityMap<K, V> {
    let entries: Box<[(K, V)]> = match n {
        0 => Box::new([]),
        1 => Box::new([a[0].take().expect("entry")]),
        2 => Box::new([a[0].take().expect("entry"), a[1].take().expect("entry")]),
        3 => Box::new([
            a[0].take().expect("entry"),
            a[1].take().expect("entry"),
            a[2].take().expect("entry"),
        ]),
        _ => unreachable!("finite metadata schema"),
    };
    InputIdentityMap::from_sorted_entries(entries)
        .unwrap_or_else(|_| unreachable!("canonical source metadata"))
}
fn descriptor<E: std::error::Error>(
    part: &impl HostInputPartView,
) -> Result<InputPartDescriptor, PreparedModelInputSourceError<E>> {
    let payload = identity(part.payload())?;
    let mut metadata = [const { None }; 3];
    for (i, (key, slot)) in part.metadata().enumerate() {
        metadata[i] = Some((key, identity(slot)?));
    }
    let mut extents = [const { None }; 2];
    for (i, e) in part.extents().iter().copied().enumerate() {
        extents[i] = Some((
            extent_key(e).map_err(PreparedModelInputSourceError::Custody)?,
            e,
        ));
    }
    // Extent execution order is preserved in parts. Identity keys are canonical.
    if extents[0]
        .as_ref()
        .zip(extents[1].as_ref())
        .is_some_and(|(a, b)| a.0 > b.0)
    {
        extents.swap(0, 1);
    }
    let entries: Box<[(u32, InputExtent)]> = match part.extents().len() {
        0 => Box::new([]),
        1 => Box::new([extents[0].take().expect("extent")]),
        2 => Box::new([
            extents[0].take().expect("extent"),
            extents[1].take().expect("extent"),
        ]),
        _ => unreachable!("finite extent schema"),
    };
    let extents = InputIdentityMap::from_sorted_entries(entries)
        .unwrap_or_else(|_| unreachable!("unique source extents"));
    InputPartDescriptor::from_fixed_entries(
        part.modality(),
        part.kind(),
        payload,
        fixed3(metadata, part.metadata().len()),
        extents,
    )
    .map_err(PreparedModelInputSourceError::Descriptor)
}
fn identity<E: std::error::Error>(
    slot: HostTensorView<'_>,
) -> Result<InputTensorIdentity, PreparedModelInputSourceError<E>> {
    let dtype = match slot.values {
        HostTensorValues::U32(_) => TensorDtype::U32,
        HostTensorValues::I32(_) => TensorDtype::I32,
        HostTensorValues::F32(_) => TensorDtype::F32,
        HostTensorValues::Bool(_) => TensorDtype::Bool,
    };
    let mut shape = Vec::new();
    reserve(&mut shape, slot.shape.len()).map_err(PreparedModelInputSourceError::Allocation)?;
    shape.extend_from_slice(slot.shape);
    InputTensorIdentity::new(dtype, shape).map_err(PreparedModelInputSourceError::Descriptor)
}

#[cfg(test)]
mod tests;
