//! Closed source erasure: strong and opaque weak aliases retain one custody.
use super::{
    CheckpointSource, CompositeCheckpointSource, PreparedAcquisitionOwner, SafetensorsWeightStore, SharedCheckpointSource,
    storage::{CheckpointSourceHandle, SourceControl, SourceHandle, SourceStorageIdentity},
};
use crate::gguf_store::GgufWeightStore;
use std::{
    alloc::Layout,
    any::Any,
    fmt,
    mem::{size_of, size_of_val},
    sync::{Arc, Weak},
};

/// A retained checkpoint root whose source allocation is never exported as Arc.
/// Ordinary providers keep their existing Arc and callback behavior. The typed
/// constructors move the actual built-in source into one private shared allocation;
/// clones and identities retain its same constructor custody.
///
/// This is source lifetime, not physical payload inventory or a fit certificate.
/// Composite children still have their independent original ownership contracts.
///
/// ```compile_fail
/// use eredu_checkpoint::store::RetainedCheckpointSource;
/// fn raw_weak(source: &RetainedCheckpointSource) {
///     let _ = std::sync::Arc::downgrade(source);
/// }
/// ```
#[derive(Clone)]
pub struct RetainedCheckpointSource(Owner);
#[derive(Clone)]
enum Owner {
    Safetensors(SourceHandle<SafetensorsWeightStore>),
    Ordinary(SharedCheckpointSource),
    Gguf(SourceHandle<GgufWeightStore>),
    Composite(SourceHandle<CompositeCheckpointSource>),
    Closed(CheckpointSourceHandle),
}
impl From<SharedCheckpointSource> for RetainedCheckpointSource {
    fn from(source: SharedCheckpointSource) -> Self {
        Self(Owner::Ordinary(source))
    }
}
impl<T: CheckpointSource + 'static> From<Arc<T>> for RetainedCheckpointSource {
    fn from(source: Arc<T>) -> Self {
        Self(Owner::Ordinary(source))
    }
}
impl std::ops::Deref for RetainedCheckpointSource {
    type Target = dyn CheckpointSource;
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}
impl AsRef<dyn CheckpointSource> for RetainedCheckpointSource {
    fn as_ref(&self) -> &(dyn CheckpointSource + 'static) {
        match &self.0 {
            Owner::Safetensors(source) => &**source,
            Owner::Ordinary(source) => source.as_ref(),
            Owner::Gguf(source) => &**source,
            Owner::Composite(source) => &**source,
            Owner::Closed(source) => source.source(),
        }
    }
}
impl RetainedCheckpointSource {
    /// Move the concrete SafeTensors source into closed lifetime ownership.
    /// The typed source supplies the existing file acquisition route; arbitrary
    /// caller custody does not certify runtime admission of source or outer data.
    pub fn from_safetensors_with_custody<C: Any + fmt::Debug + Send + Sync>(
        source: SafetensorsWeightStore,
        custody: C,
    ) -> Self {
        Self(Owner::Safetensors(SourceHandle::new(
            source, Some(SourceControl::new(custody)),
        )))
    }
    /// Owning requests for the concrete SafeTensors source allocation and custody.
    pub fn safetensors_storage_request<C>() -> Option<SourceErasureStorageRequest> {
        request::<SafetensorsWeightStore, C>()
    }
    /// Ordinary typed GGUF construction using the same private allocation.
    /// No constructor custody or original accounting claim is installed.
    pub fn from_gguf(source: GgufWeightStore) -> Self {
        Self(Owner::Gguf(SourceHandle::new(source, None)))
    }
    /// Ordinary typed composite construction; children retain their own status.
    pub fn from_composite(source: CompositeCheckpointSource) -> Self {
        Self(Owner::Composite(SourceHandle::new(source, None)))
    }
    // Closed typed constructor input. An ordinary erased Arc cannot certify
    // its dynamic type by forwarding a source callback.
    pub(super) fn gguf(&self) -> Option<&GgufWeightStore> {
        match &self.0 {
            Owner::Gguf(source) => Some(source),
            _ => None,
        }
    }
    /// Move the exact GGUF value after its caller has admitted these requests.
    /// Arbitrary C cannot impersonate the runtime's private original custody.
    pub fn from_gguf_with_custody<C: Any + fmt::Debug + Send + Sync>(
        source: GgufWeightStore,
        custody: C,
    ) -> Self {
        Self(Owner::Gguf(SourceHandle::new(
            source,
            Some(SourceControl::new(custody)),
        )))
    }
    /// Move the exact composite value, without cloning its catalogs or children.
    /// Child legacy Arcs remain separate, potentially unqualified contributions.
    pub fn from_composite_with_custody<C: Any + fmt::Debug + Send + Sync>(
        source: CompositeCheckpointSource,
        custody: C,
    ) -> Self {
        Self(Owner::Composite(SourceHandle::new(
            source,
            Some(SourceControl::new(custody)),
        )))
    }
    /// Moves a caller's concrete source into closed lifetime ownership after
    /// admission. This erases source access, not acquisition qualification: a
    /// custom callback cannot become a typed GGUF, SafeTensors or composite acquisition.
    /// The caller separately accounts the source's owned payload and C's origin.
    pub fn from_source_with_custody<
        T: CheckpointSource + 'static,
        C: Any + fmt::Debug + Send + Sync,
    >(
        source: T,
        custody: C,
    ) -> Self {
        Self(Owner::Closed(
            SourceHandle::new(source, Some(SourceControl::new(custody))).into_checkpoint_source(),
        ))
    }
    /// Requests for the one concrete source allocation, erased custody and
    /// fixed lifetime transports. Shared headers still need caller qualification.
    /// No payload bound or memory authority is inferred from these layouts.
    pub fn source_storage_request<T: CheckpointSource + 'static, C>()
    -> Option<SourceErasureStorageRequest> {
        let mut request = request::<T, C>()?;
        request.controls = request
            .controls
            .checked_add(CheckpointSourceHandle::erasure_control_bytes::<T>()?)?;
        Some(request)
    }
    /// Actual owning requests for the concrete GGUF erasure constructor.
    pub fn gguf_storage_request<C>() -> Option<SourceErasureStorageRequest> {
        request::<GgufWeightStore, C>()
    }
    /// Actual owning requests for the concrete composite erasure constructor.
    pub fn composite_storage_request<C>() -> Option<SourceErasureStorageRequest> {
        request::<CompositeCheckpointSource, C>()
    }
    /// Read only this root's actual constructor origin, without extracting C.
    pub fn constructor_control_owner<C: Any>(&self) -> Option<&C> {
        match &self.0 {
            Owner::Ordinary(_) => None,
            Owner::Safetensors(source) => source.origin(),
            Owner::Gguf(source) => source.origin(),
            Owner::Composite(source) => source.origin(),
            Owner::Closed(source) => source.origin(),
        }
    }
    /// Exact root identity; comparing handles creates no allocation or grant.
    pub fn same_source(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Owner::Ordinary(a), Owner::Ordinary(b)) => Arc::ptr_eq(a, b),
            (Owner::Safetensors(a), Owner::Safetensors(b)) => a.same(b),
            (Owner::Gguf(a), Owner::Gguf(b)) => a.same(b),
            (Owner::Composite(a), Owner::Composite(b)) => a.same(b),
            (Owner::Closed(a), Owner::Closed(b)) => a.same(b),
            _ => false,
        }
    }
    /// Ordinary compatibility check never blesses an unrelated admitted root.
    pub fn matches_ordinary(&self, source: &SharedCheckpointSource) -> bool {
        matches!(&self.0, Owner::Ordinary(actual) if Arc::ptr_eq(actual, source))
    }
    /// Opaque identity keeps the exact allocation and its custody until last
    /// identity retirement. It cannot be upgraded or used as payload inventory.
    pub fn identity(&self) -> CheckpointSourceIdentity {
        CheckpointSourceIdentity(match &self.0 {
            Owner::Ordinary(source) => Identity::Ordinary(Arc::downgrade(source)),
            Owner::Safetensors(source) => Identity::Closed(source.identity()),
            Owner::Gguf(source) => Identity::Closed(source.identity()),
            Owner::Composite(source) => Identity::Closed(source.identity()),
            Owner::Closed(source) => Identity::Closed(source.identity()),
        })
    }
    pub(super) fn acquisition_owner(&self) -> Option<PreparedAcquisitionOwner> {
        match &self.0 {
            Owner::Ordinary(source) => {
                let owner = Arc::clone(source).prepared_acquisition_owner()?;
                std::ptr::addr_eq(owner.source(), source.as_ref()).then_some(owner)
            }
            Owner::Safetensors(source) => Some(PreparedAcquisitionOwner::retained_safetensors(source.clone())),
            Owner::Gguf(source) => Some(PreparedAcquisitionOwner::retained_gguf(source.clone())),
            Owner::Composite(source) => {
                Some(PreparedAcquisitionOwner::retained_composite(source.clone()))
            }
            // Generic closed lifetime ownership supplies no typed acquisition.
            Owner::Closed(_) => None,
        }
    }
}
impl fmt::Debug for RetainedCheckpointSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedCheckpointSource")
            .finish_non_exhaustive()
    }
}

/// Source identity only. No raw Weak, upgrade, storage amount, payload identity
/// conversion, or custody extraction is available.
#[derive(Clone)]
pub struct CheckpointSourceIdentity(Identity);
#[derive(Clone)]
enum Identity {
    Ordinary(Weak<dyn CheckpointSource>),
    // SourceStorageIdentity is an implementation detail, never projected.
    // Its Weak retires before the same independent constructor control.
    Closed(SourceStorageIdentity),
}
impl PartialEq for CheckpointSourceIdentity {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Identity::Ordinary(a), Identity::Ordinary(b)) => Weak::ptr_eq(a, b),
            (Identity::Closed(a), Identity::Closed(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for CheckpointSourceIdentity {}
impl fmt::Debug for CheckpointSourceIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckpointSourceIdentity")
            .finish_non_exhaustive()
    }
}

/// Owning layouts for one root, not a byte grant or arbitrary source estimate.
#[derive(Clone, Copy, Debug)]
pub struct SourceErasureStorageRequest {
    source: Layout,
    custody: Layout,
    controls: usize,
}
impl SourceErasureStorageRequest {
    /// Actual inline source value; shared headers need owning qualification.
    pub fn source_body(&self) -> Layout {
        self.source
    }
    /// Actual erased control value, excluding its qualified shared headers.
    pub fn custody_body(&self) -> Layout {
        self.custody
    }
    /// Constructor, clone, identity and borrowed-source control transports.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
}
fn request<T: Any + Send + Sync, C>() -> Option<SourceErasureStorageRequest> {
    let controls = [
        size_of::<T>(),
        size_of::<C>(),
        size_of::<RetainedCheckpointSource>(),
        size_of::<CheckpointSourceIdentity>(),
        size_of::<&RetainedCheckpointSource>(),
        size_of::<&dyn CheckpointSource>(),
        size_of::<Option<&C>>(),
        size_of::<bool>(),
        SourceHandle::<T>::owner_control_bytes()?,
        SourceControl::owner_control_bytes::<C>()?,
    ];
    Some(SourceErasureStorageRequest {
        source: Layout::new::<T>(),
        custody: SourceControl::body_layout::<C>(),
        controls: controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)?,
    })
}

#[cfg(test)]
mod tests;
