//! Shared lifetime custody on the existing capture-run identity.
use eredu_core::{capture::CaptureError, HostPreparationAuthority};
use std::sync::Mutex;

/// Payload-free: no plan, checkpoint, session or callback is retained here.
/// Every containing owner must place its Arc after the payload it protects.
#[derive(Default, Debug)]
pub(super) struct CaptureHostOwner {
    authority: Mutex<HostPreparationAuthority>,
}

impl CaptureHostOwner {
    #[cfg(test)]
    pub(super) fn poison_for_test(&self) {
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lock = self.authority.lock().unwrap();
            panic!("injected ordinary custody poisoning");
        }));
        assert!(failed.is_err());
    }
    pub(super) fn retain(&self, incoming: &HostPreparationAuthority) -> Result<(), CaptureError> {
        let mut authority = self.authority.lock().map_err(|_| poisoned())?;
        let combined = HostPreparationAuthority::retain((authority.clone(), incoming.clone()));
        let previous = std::mem::replace(&mut *authority, combined);
        // An erased token's final destructor must never execute under this lock.
        drop(authority);
        drop(previous);
        Ok(())
    }

    pub(super) fn retained(&self) -> Result<HostPreparationAuthority, CaptureError> {
        self.authority
            .lock()
            .map(|authority| authority.clone())
            .map_err(|_| poisoned())
    }
}

fn poisoned() -> CaptureError {
    CaptureError::Invalid("capture host preparation custody is poisoned".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Drops(Arc<AtomicUsize>);
    impl Drop for Drops {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn poisoned_custody_rejects_attachment_without_consuming_incoming_owner() {
        let owner = Arc::new(CaptureHostOwner::default());
        let poisoned_owner = owner.clone();
        let result = std::panic::catch_unwind(move || {
            let _lock = poisoned_owner.authority.lock().unwrap();
            panic!("poison capture custody");
        });
        assert!(result.is_err());
        let drops = Arc::new(AtomicUsize::new(0));
        let incoming = HostPreparationAuthority::retain(Drops(drops.clone()));
        assert!(
            matches!(owner.retain(&incoming), Err(CaptureError::Invalid(reason)) if reason.contains("custody is poisoned"))
        );
        assert!(
            matches!(owner.retained(), Err(CaptureError::Invalid(reason)) if reason.contains("custody is poisoned"))
        );
        drop(incoming);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
