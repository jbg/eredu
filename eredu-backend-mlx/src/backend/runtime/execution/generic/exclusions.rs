//! Immutable logical selection shared by native and cold unit binding.
use crate::backend::runtime::residency::manager::ManagerCustody;
use eredu_core::{SharedStorageOwner, SharedStorageRetirement};
use eredu_runtime::working_memory::{OriginalHostMetadataCustody, WorkingMemoryError};
use std::{alloc::Layout, collections::BTreeSet, mem::{size_of, size_of_val}, sync::Arc};

/// Exact independently owned parameter identities. Empty selections allocate
/// nothing; nonempty prepared selections retain their original source account.
#[derive(Clone, Debug)]
pub struct MlxParameterExclusions(Option<SharedStorageOwner<Payload>>);
#[derive(Debug)]
struct Payload {
    names: Vec<String>,
    // Names and their shared allocation retire before the final source hold.
    custody: ManagerCustody,
}
impl SharedStorageRetirement for Payload {
    fn retire(self: Arc<Self>) { drop(Arc::into_inner(self)); }
}
impl PartialEq for MlxParameterExclusions {
    fn eq(&self, other: &Self) -> bool { self.names() == other.names() }
}
impl Eq for MlxParameterExclusions {}
impl MlxParameterExclusions {
    pub(crate) fn ordinary(names: BTreeSet<String>) -> Self {
        let names = names.into_iter().collect::<Vec<_>>();
        Self::construct(&names, ManagerCustody::default())
    }
    /// Called only after the source initializer admitted this exact producer.
    pub(crate) fn construct(names: &[String], custody: ManagerCustody) -> Self {
        if names.is_empty() { return Self(None); }
        let mut copied = Vec::with_capacity(names.len());
        for name in names { copied.push(name.clone()); }
        Self(Some(SharedStorageOwner::new(Payload { names: copied, custody })))
    }
    /// Actual shared shell, exact copied names/vector and constructor controls.
    pub(crate) fn construction_bytes(names: &[String]) -> Result<usize, WorkingMemoryError> {
        let controls = [
            size_of::<(&[String], ManagerCustody, Self)>(),
            size_of::<(Vec<String>, String, &String, std::slice::Iter<'_, String>)>(),
            size_of::<Payload>(), size_of::<SharedStorageOwner<Payload>>(),
            size_of::<Option<SharedStorageOwner<Payload>>>(),
        ];
        let mut bytes = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        if !names.is_empty() {
            bytes = bytes.checked_add(usize::try_from(OriginalHostMetadataCustody::shared_storage_bytes(
                Layout::new::<Payload>())?).map_err(|_| WorkingMemoryError::Overflow)?)
                .and_then(|n| n.checked_add(Layout::array::<String>(names.len()).ok()?.size()))
                .ok_or(WorkingMemoryError::Overflow)?;
            for name in names { bytes = bytes.checked_add(name.len()).ok_or(WorkingMemoryError::Overflow)?; }
        }
        Ok(bytes)
    }
    pub(crate) fn names(&self) -> &[String] {
        self.0.as_ref().map_or(&[], |owner| owner.names.as_slice())
    }
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.names().binary_search_by(|value| value.as_str().cmp(name)).is_ok()
    }
    pub(crate) fn matches(&self, names: &BTreeSet<String>) -> bool {
        self.names().len() == names.len() && self.names().iter().eq(names.iter())
    }
    pub(crate) fn for_selection(&self, selected: &BTreeSet<String>)
        -> Result<Self, WorkingMemoryError> {
        if !self.is_source_funded() || !self.matches(selected) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(self.clone())
    }
    pub(crate) fn is_source_funded(&self) -> bool {
        self.0.as_ref().is_none_or(|owner| owner.custody.is_source_funded())
    }
    /// Ordinary cold inspection accounts the actual retained payload. A source
    /// funded owner already carries its original hold and receives no new debit.
    pub(crate) fn unfunded_retained_bytes(&self) -> Option<usize> {
        let Some(owner) = &self.0 else { return Some(0); };
        if owner.custody.is_source_funded() { return Some(0); }
        let mut bytes = usize::try_from(OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<Payload>()).ok()?).ok()?
            .checked_add(Layout::array::<String>(owner.names.capacity()).ok()?.size())?;
        for name in &owner.names { bytes = bytes.checked_add(name.capacity())?; }
        Some(bytes)
    }
}
