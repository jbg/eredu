//! The neutral supplied-storage family realized by intact original host owners.
use super::*;
use eredu_gguf::{InitializedStorage, StorageFamily};

#[derive(Debug)]
pub(crate) struct AdmittedFamily;
#[derive(Debug)]
pub(crate) struct AdmittedStorage<T>(pub(crate) OriginalHostVec<T>);
impl<T> AsRef<[T]> for AdmittedStorage<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}
impl<T> AsMut<[T]> for AdmittedStorage<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}
impl<T: std::fmt::Debug + 'static> InitializedStorage<T> for AdmittedStorage<T> {
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum StorageCause {
    #[error("GGUF supplied storage source identity differs")]
    Source,
    #[error("{0}")]
    Host(#[source] HostDestinationCause),
}
impl StorageFamily for AdmittedFamily {
    type Buffer<T: Copy + std::fmt::Debug + 'static> = AdmittedStorage<T>;
    type Error = StorageCause;
}
pub(crate) fn prepare<T: Copy + std::fmt::Debug + 'static>(
    bank: &mut OriginalHostDestinationBank,
    count: usize,
    initializer: T,
) -> Result<AdmittedStorage<T>, (StorageCause, Option<AdmittedStorage<T>>)> {
    let mut values = match bank.try_vec(count) {
        Ok(v) => v,
        Err(error) => {
            let (cause, prefix) = error.into_parts();
            return Err((StorageCause::Host(cause), prefix.map(AdmittedStorage)));
        }
    };
    if let Err(cause) = values.try_fill(count, std::iter::repeat(initializer).take(count)) {
        return Err((StorageCause::Host(cause), Some(AdmittedStorage(values))));
    }
    Ok(AdmittedStorage(values))
}
impl<T: Copy + std::fmt::Debug + 'static> From<eredu_gguf::StoredBuffer<AdmittedFamily, T>>
    for TypedValues<T>
{
    fn from(buffer: eredu_gguf::StoredBuffer<AdmittedFamily, T>) -> Self {
        let (storage, emitted, limit) = buffer.into_storage();
        let values = storage
            .expect("a reached completed payload owns its supplied buffer")
            .0;
        Self::Supplied {
            values,
            emitted,
            limit,
        }
    }
}
