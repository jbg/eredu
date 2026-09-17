//! Immutable verification inputs retain their actual host-construction account.
use crate::backend::error::Error;
use eredu_core::{GenerationTokenIdStorage, GenerationTokenIds};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
struct Tokens {
    values: Vec<u32>,
    // Last: backing and all token owners retire before the cumulative account.
    _funding: WorkspaceMetadataFunding,
}
impl GenerationTokenIdStorage for Tokens {
    fn token_ids(&self) -> &[u32] {
        &self.values
    }
    fn retire(self: Arc<Self>) {
        // GenerationTokenIds closes every strong exit; no Weak or raw Arc escapes.
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: eredu_nn::Error,
    _funding: WorkspaceMetadataFunding,
}
fn controls() -> Option<usize> {
    let shared = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<Tokens>()).ok()?.0.pad_to_align().size();
    let parts = [
        shared,
        size_of::<Tokens>(),
        size_of::<Arc<Tokens>>(),
        size_of::<Arc<dyn GenerationTokenIdStorage>>(),
        size_of::<GenerationTokenIds>(),
        size_of::<Result<GenerationTokenIds, Error>>(),
        size_of::<Vec<u32>>(),
        size_of::<Failure>(),
        size_of::<(&[u32], &WorkspaceMetadataFunding)>(),
        eredu_nn::Error::retained_source_control_bytes::<Failure>()?,
    ];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn copy(
    tokens: &[u32],
    funding: &WorkspaceMetadataFunding,
) -> Result<GenerationTokenIds, Error> {
    funding.reserve_metadata(controls().ok_or(Error::WorkspacePlanning(
        WorkspaceMetadataFundingError::Overflow,
    ))?).map_err(Error::WorkspacePlanning)?;
    let mut values = funding.metadata_vec(tokens.len()).map_err(|cause| {
        // This erasure was paid above; it preserves the exact inner refusal.
        Error::Neural(eredu_nn::Error::backend_retained_source(Failure {
            cause,
            _funding: funding.clone(),
        }))
    })?;
    values.extend_from_slice(tokens);
    Ok(GenerationTokenIds::from_owner(Arc::new(Tokens {
        values,
        _funding: funding.clone(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::workspace::WorkspaceMetadataAccount;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct Account {
        available: Arc<AtomicUsize>,
        retired: Arc<AtomicBool>,
    }
    impl WorkspaceMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
            self.available.fetch_update(Ordering::SeqCst, Ordering::SeqCst,
                |available| available.checked_sub(bytes))
                .map(|_| ())
                .map_err(|available| WorkspaceMetadataFundingError::Capacity {
                    required: u64::try_from(bytes).unwrap(), available: u64::try_from(available).unwrap(),
                })
        }
    }
    impl Drop for Account {
        fn drop(&mut self) { self.retired.store(true, Ordering::SeqCst); }
    }
    #[test]
    fn refusal_precedes_copy_and_cloned_tokens_retain_the_account() {
        let available = Arc::new(AtomicUsize::new(1 << 20));
        let retired = Arc::new(AtomicBool::new(false));
        let funding = WorkspaceMetadataFunding::new(Account {
            available: available.clone(), retired: retired.clone(),
        }).unwrap();
        available.store(0, Ordering::SeqCst);
        let refusal = copy(&[7, 31, 5], &funding).unwrap_err();
        assert!(matches!(refusal, Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Capacity { available: 0, .. }
        )));
        assert_eq!(available.load(Ordering::SeqCst), 0);
        available.store(1 << 20, Ordering::SeqCst);
        let tokens = copy(&[7, 31, 5], &funding).unwrap();
        let remaining = available.load(Ordering::SeqCst);
        assert!(remaining < 1 << 20);
        let retained = tokens.clone();
        drop(funding);
        drop(tokens);
        assert!(!retired.load(Ordering::SeqCst));
        assert_eq!(retained.as_slice(), &[7, 31, 5]);
        assert_eq!(available.load(Ordering::SeqCst), remaining);
        drop(retained);
        assert!(retired.load(Ordering::SeqCst));
    }
}
