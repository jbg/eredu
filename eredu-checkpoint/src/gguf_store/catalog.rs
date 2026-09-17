//! Immutable logical catalog and its actual shared retirement owner.
use super::*;
use crate::prepared_index::{PreparedIndex, PreparedIndexNode};
use std::{alloc::Layout, any::Any, fmt};

#[derive(Debug, Default)]
pub(super) struct CatalogRows(PreparedIndex<String, CatalogEntry, ()>);
impl CatalogRows {
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub(super) fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
    pub(super) fn get(&self, key: &str) -> Option<&CatalogEntry> {
        self.0.get_by(|stored| key.cmp(stored.as_str()))
    }
    pub(super) fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(k, _)| k)
    }
    pub(super) fn insert(&mut self, key: String, entry: CatalogEntry) {
        let mut node = Some(PreparedIndexNode::new(key, entry, ()));
        assert!(self.0.insert(&mut node), "prechecked unique catalog key");
    }
    pub(super) const fn node_layout() -> Layout {
        PreparedIndexNode::<String, CatalogEntry, ()>::storage_layout()
    }
}
#[derive(Debug, Default)]
pub(super) struct UnclaimedRows(PreparedIndex<String, (), ()>);
impl UnclaimedRows {
    pub(super) fn iter(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(k, _)| k)
    }
    pub(super) fn insert(&mut self, key: String) {
        if self.0.get_by(|stored| key.cmp(stored)).is_some() {
            return;
        }
        let mut node = Some(PreparedIndexNode::new(key, (), ()));
        assert!(self.0.insert(&mut node), "unclaimed key");
    }
    pub(super) const fn node_layout() -> Layout {
        PreparedIndexNode::<String, (), ()>::storage_layout()
    }
}
#[derive(Debug)]
pub(super) struct CatalogData {
    pub(super) rows: CatalogRows,
    pub(super) unclaimed: UnclaimedRows,
}
#[derive(Debug)]
struct CatalogBody<C> {
    data: CatalogData,
    // Actual nodes/nested buffers and the shared block retire before this.
    custody: C,
}
// Private erased access, implemented only by this module's concrete body. No
// outside callback can supply rows, a retirement policy or a claimed origin.
trait CatalogStorage: fmt::Debug + Send + Sync {
    fn data(&self) -> &CatalogData;
    fn origin(&self) -> &dyn Any;
    fn retire(self: Arc<Self>);
}
impl<C: Any + fmt::Debug + Send + Sync> CatalogStorage for CatalogBody<C> {
    fn data(&self) -> &CatalogData {
        &self.data
    }
    fn origin(&self) -> &dyn Any {
        &self.custody
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
/// The only strong owner. There is no Weak or bare Arc escape; every final
/// retirement dispatches to its concrete Arc::into_inner implementation.
#[derive(Debug)]
pub(super) struct CatalogHandle(Option<Arc<dyn CatalogStorage>>);
impl Clone for CatalogHandle {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("catalog"))))
    }
}
impl Drop for CatalogHandle {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
impl CatalogHandle {
    pub(super) fn new<C: Any + fmt::Debug + Send + Sync>(data: CatalogData, custody: C) -> Self {
        Self(Some(Arc::new(CatalogBody { data, custody })))
    }
    pub(super) fn data(&self) -> &CatalogData {
        self.0.as_ref().expect("catalog").data()
    }
    pub(super) fn get(&self, key: &str) -> Option<&CatalogEntry> {
        self.data().rows.get(key)
    }
    pub(super) fn keys(&self) -> impl Iterator<Item = &String> {
        self.data().rows.keys()
    }
    pub(super) fn origin<C: Any>(&self) -> Option<&C> {
        self.0.as_ref().expect("catalog").origin().downcast_ref()
    }
    pub(super) fn owner_control_bytes<C>() -> Option<usize> {
        let controls = [
            std::mem::size_of::<CatalogBody<C>>(),
            std::mem::size_of::<Option<CatalogBody<C>>>(),
            std::mem::size_of::<Arc<CatalogBody<C>>>(),
            std::mem::size_of::<Arc<dyn CatalogStorage>>(),
            std::mem::size_of::<Option<Arc<dyn CatalogStorage>>>(),
            std::mem::size_of::<&CatalogHandle>(),
            std::mem::size_of::<&CatalogData>(),
            std::mem::size_of::<&dyn Any>(),
            std::mem::size_of::<Option<&C>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
    pub(super) const fn body_layout<C>() -> Layout {
        Layout::new::<CatalogBody<C>>()
    }
}

// Preserve existing internal fixture observations; production uses fallible get.
#[cfg(test)]
impl<Q: AsRef<str> + ?Sized> std::ops::Index<&Q> for CatalogHandle {
    type Output = CatalogEntry;
    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key.as_ref())
            .expect("existing catalog fixture key")
    }
}
