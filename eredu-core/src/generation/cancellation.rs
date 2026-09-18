//! Shared control cells retain the original account of their actual construction.
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use crate::HostPreparationAuthority;

#[derive(Debug)]
struct FlagOwner {
    flag: AtomicBool,
    // The cell and shared allocation retire before their payer.
    host: HostPreparationAuthority,
}
#[derive(Debug)]
pub(crate) struct ControlFlag(Option<Arc<FlagOwner>>);
impl ControlFlag {
    fn owner(&self) -> &Arc<FlagOwner> { self.0.as_ref().expect("live control flag") }
    pub(crate) fn new(host: HostPreparationAuthority) -> Self {
        Self(Some(Arc::new(FlagOwner { flag: AtomicBool::new(false), host })))
    }
    pub(crate) fn construction_bytes() -> Option<usize> {
        let shell = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<FlagOwner>()).ok()?.0.pad_to_align().size();
        let parts = [shell, std::mem::size_of::<Self>(), std::mem::size_of::<FlagOwner>(),
            std::mem::size_of::<Option<FlagOwner>>(), std::mem::size_of::<HostPreparationAuthority>()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn set(&self, value: bool) { self.owner().flag.store(value, Ordering::Release); }
    pub(crate) fn get(&self) -> bool { self.owner().flag.load(Ordering::Acquire) }
}
impl Clone for ControlFlag {
    fn clone(&self) -> Self { Self(Some(self.owner().clone())) }
}
impl Drop for ControlFlag {
    fn drop(&mut self) { if let Some(owner) = self.0.take() { drop(Arc::into_inner(owner)); } }
}

/// Cheap thread-safe cooperative cancellation observed between submissions.
#[derive(Debug, Clone)]
pub struct GenerationCancellationToken { cancelled: ControlFlag }
impl Default for GenerationCancellationToken {
    fn default() -> Self { Self::new_retained(HostPreparationAuthority::unmanaged()) }
}
impl GenerationCancellationToken {
    /// Exact shared cell, closed owner and fixed constructor controls. This is
    /// a layout fact only; the enclosing producer must pay before construction.
    pub fn construction_bytes() -> Option<usize> {
        ControlFlag::construction_bytes()?.checked_add(std::mem::size_of::<Self>())
    }
    /// Creates an active token under the ordinary caller's storage policy.
    pub fn new() -> Self { Self::default() }
    /// Constructs a fresh cell in an already admitted original host producer.
    /// The caller pays `construction_bytes` first. All escaping clones retain the
    /// supplied destination custody; existing ordinary cells cannot be adopted.
    pub fn new_retained(host: HostPreparationAuthority) -> Self {
        Self { cancelled: ControlFlag::new(host) }
    }
    /// Permanently requests cancellation for every clone.
    pub fn cancel(&self) { self.cancelled.set(true); }
    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool { self.cancelled.get() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug)]
    struct Retirement(Arc<AtomicBool>);
    impl Drop for Retirement {
        fn drop(&mut self) { self.0.store(true, Ordering::Release); }
    }
    #[test]
    fn escaped_pause_and_cancellation_aliases_retain_the_original_host() {
        let retired = Arc::new(AtomicBool::new(false));
        let host = HostPreparationAuthority::retain(Retirement(retired.clone()));
        assert!(crate::execution_control::GenerationControlHandle::construction_bytes().unwrap()
            > GenerationCancellationToken::construction_bytes().unwrap());
        let control = crate::execution_control::GenerationControlHandle::new_retained(host.clone());
        let pause = control.clone();
        let cancellation = control.cancellation().clone();
        drop((host, control));
        assert!(!retired.load(Ordering::Acquire));
        std::thread::spawn(move || {
            pause.request_pause();
            assert!(pause.pause_requested());
            pause.acknowledge_resume();
            assert!(!pause.pause_requested());
            pause.cancel();
        }).join().unwrap();
        assert!(cancellation.is_cancelled());
        assert!(!retired.load(Ordering::Acquire), "independent cancellation alias owns custody");
        drop(cancellation);
        assert!(retired.load(Ordering::Acquire));
    }
}
