//! Closed compiler account shared by actual immutable observation/edit sources.
use super::{WorkingMemoryError, WorkingMemoryPool, loaded_decode_source::Allowance};
use eredu_core::HostPreparationAuthority;
use std::{
    alloc::Layout,
    mem::size_of,
    sync::{Arc, Mutex, atomic::AtomicUsize},
};
#[derive(Debug)]
struct AccountNode {
    allowance: Mutex<Allowance>,
}
// Erasure stops downstream auto-trait expansion at this actual compiler-account
// boundary. The trait has no constructor or generic custody/adoption operation.
trait SourceAccount: std::fmt::Debug + Send + Sync {
    fn finish(&self) -> Result<(), WorkingMemoryError>;
    fn validate(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError>;
    // Dynamic dispatch recovers the concrete Arc type before final retirement.
    // Unsizing Arc changes no allocation; its shell still dies before Allowance.
    fn retire(self: Arc<Self>);
}
#[derive(Debug)]
pub(super) struct Account(Option<Arc<dyn SourceAccount>>);
impl Clone for Account {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live source account").clone()))
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        if let Some(node) = self.0.take() {
            node.retire();
        }
    }
}
impl Account {
    pub(super) fn admit(pool: &WorkingMemoryPool, bytes: u64) -> Result<Self, WorkingMemoryError> {
        let allowance = pool.admit_source_compiler(bytes)?;
        Ok(Self(Some(Arc::new(AccountNode {
            allowance: Mutex::new(allowance),
        }))))
    }
    pub(super) fn control_bytes() -> Option<usize> {
        let shell = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<AccountNode>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let frames = [
            shell,
            size_of::<AccountNode>(),
            size_of::<Option<AccountNode>>(),
            size_of::<Self>(),
            size_of::<&dyn SourceAccount>(),
            size_of::<Arc<dyn SourceAccount>>(),
            size_of::<Option<Arc<dyn SourceAccount>>>(),
            size_of::<Arc<AccountNode>>(),
            size_of::<(&dyn SourceAccount, &WorkingMemoryPool)>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<std::sync::MutexGuard<'_, Allowance>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            HostPreparationAuthority::retention_bytes::<Self>()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }

    pub(super) fn finish(&self) -> Result<(), WorkingMemoryError> {
        self.0.as_ref().expect("live source account").finish()
    }
    pub(super) fn validate(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        self.0.as_ref().expect("live source account").validate(pool)
    }
}
impl SourceAccount for AccountNode {
    fn finish(&self) -> Result<(), WorkingMemoryError> {
        self.allowance
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?
            .end_compilation()
    }
    fn validate(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        let allowance = self
            .allowance
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if !allowance.pool().same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
