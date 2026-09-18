//! Private publication shell for the just-constructed source payload.
use eredu_core::{
    capture::{CaptureError, CaptureSourceConstruction},
    HostMetadataFunding,
};
use std::{
    alloc::Layout,
    cell::RefCell,
    mem::size_of,
    sync::{atomic::AtomicUsize, Arc},
};

struct Inner<T> {
    value: T,
    funding: Option<HostMetadataFunding>,
}
pub(crate) struct Publication<T>(Option<Arc<Inner<T>>>);
impl<T> Clone for Publication<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> Publication<T> {
    /// The ordinary and original entries share this success-only cache worker.
    /// A prior ordinary publication cannot satisfy original construction.
    pub(super) fn cached<E, F>(
        cache: &RefCell<Option<Self>>,
        construction: CaptureSourceConstruction<'_>,
        build: F,
    ) -> Result<Self, E>
    where
        E: From<CaptureError>,
        F: FnOnce() -> Result<Self, E>,
    {
        construction.controls(size_of::<(
            &RefCell<Option<Self>>,
            CaptureSourceConstruction<'_>,
            F,
            Option<Self>,
            Result<Self, E>,
            std::cell::Ref<'_, Option<Self>>,
            std::cell::RefMut<'_, Option<Self>>,
        )>())?;
        if let Some(source) = cache.borrow().as_ref() {
            if construction.funding().is_none() || source.is_funded() {
                return Ok(source.clone());
            }
        }
        let source = build()?;
        *cache.borrow_mut() = Some(source.clone());
        Ok(source)
    }
    /// Only the original enclosing source worker supplies this payload. It has
    /// already priced its fields with this same account before constructing them.
    pub(super) fn new(
        value: T,
        construction: CaptureSourceConstruction<'_>,
    ) -> Result<Self, CaptureError> {
        let shell = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Inner<T>>())
            .map_err(|_| CaptureError::Overflow)?
            .0
            .pad_to_align()
            .size();
        construction.controls(
            shell
                .checked_add(size_of::<(
                    T,
                    Self,
                    Result<Self, CaptureError>,
                    Arc<Inner<T>>,
                )>())
                .ok_or(CaptureError::Overflow)?,
        )?;
        Ok(Self(Some(Arc::new(Inner {
            value,
            funding: construction.funding().cloned(),
        }))))
    }
    pub(super) fn is_funded(&self) -> bool {
        self.0.as_ref().expect("live source").funding.is_some()
    }
}
impl<T> std::ops::Deref for Publication<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.as_ref().expect("live source").value
    }
}
impl<T> Drop for Publication<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            if let Some(inner) = Arc::into_inner(owner) {
                drop(inner);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{HostMetadataAccount, HostMetadataFundingError};
    use std::sync::atomic::{AtomicBool, Ordering};
    #[derive(Debug)]
    struct Account {
        calls: Arc<AtomicUsize>,
        retired: Arc<AtomicBool>,
        payload: Arc<AtomicBool>,
        refuse: usize,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(n <= self.refuse);
            if n == self.refuse {
                Err(HostMetadataFundingError::Unavailable)
            } else {
                Ok(())
            }
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            assert!(self.payload.load(Ordering::SeqCst));
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    struct Payload {
        retired: Arc<AtomicBool>,
        account: Arc<AtomicBool>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            assert!(!self.account.load(Ordering::SeqCst));
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    #[test]
    fn original_publication_pays_shell_once_and_retires_payload_before_account() {
        for refuse in [1, usize::MAX] {
            let calls = Arc::new(AtomicUsize::new(0));
            let retired = Arc::new(AtomicBool::new(false));
            let payload = Arc::new(AtomicBool::new(false));
            let funding = HostMetadataFunding::new(Account {
                calls: calls.clone(),
                retired: retired.clone(),
                payload: payload.clone(),
                refuse,
            })
            .unwrap();
            let result = Publication::new(
                Payload {
                    retired: payload.clone(),
                    account: retired.clone(),
                },
                CaptureSourceConstruction::new(Some(&funding)),
            );
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            match result {
                Ok(source) => {
                    let alias = source.clone();
                    drop(source);
                    drop(funding);
                    assert!(!retired.load(Ordering::SeqCst));
                    assert!(!payload.load(Ordering::SeqCst));
                    assert_eq!(calls.load(Ordering::SeqCst), 2, "aliases create no shell");
                    drop(alias);
                }
                Err(error) => {
                    assert!(matches!(
                        error,
                        CaptureError::AdmissionStorage(
                            eredu_core::capture::CaptureAdmissionStorageError::Funding(
                                HostMetadataFundingError::Unavailable
                            )
                        )
                    ));
                    assert!(payload.load(Ordering::SeqCst));
                    assert!(!retired.load(Ordering::SeqCst));
                    drop(funding);
                }
            }
            assert!(retired.load(Ordering::SeqCst));
        }
    }

    struct CachedPayload {
        value: String,
        custody: Payload,
    }
    fn cache_attempt(ordinary: bool, refuse: usize) -> usize {
        let old_payload = Arc::new(AtomicBool::new(false));
        let cache = RefCell::new(None);
        let old_alias = ordinary.then(|| {
            Publication::cached(&cache, CaptureSourceConstruction::new(None), || {
                Publication::new(
                    CachedPayload {
                        value: "ordinary".into(),
                        custody: Payload {
                            retired: old_payload.clone(),
                            account: Arc::new(AtomicBool::new(false)),
                        },
                    },
                    CaptureSourceConstruction::new(None),
                )
            })
            .unwrap()
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicBool::new(false));
        let payload = Arc::new(AtomicBool::new(false));
        let funding = HostMetadataFunding::new(Account {
            calls: calls.clone(),
            retired: retired.clone(),
            payload: payload.clone(),
            refuse,
        })
        .unwrap();
        let construction = CaptureSourceConstruction::new(Some(&funding));
        let built = AtomicUsize::new(0);
        let result = Publication::cached(&cache, construction, || {
            built.fetch_add(1, Ordering::SeqCst);
            // The producer's own payload and backing precede publication.
            let custody = Payload {
                retired: payload.clone(),
                account: retired.clone(),
            };
            let value = construction.text("original")?;
            Publication::new(CachedPayload { value, custody }, construction)
        });
        let requests = calls.load(Ordering::SeqCst);
        match result {
            Ok(source) => {
                assert_eq!(refuse, usize::MAX);
                assert_eq!(built.load(Ordering::SeqCst), 1);
                assert_eq!(source.value, "original");
                assert!(source.is_funded());
                let again = Publication::cached::<CaptureError, _>(&cache, construction, || {
                    panic!("a precompiled source must reuse the exact funded publication")
                })
                .unwrap();
                assert!(Arc::ptr_eq(
                    source.0.as_ref().unwrap(),
                    again.0.as_ref().unwrap()
                ));
                // Ordinary discovery likewise reads the already published source.
                let unaccounted = Publication::cached::<CaptureError, _>(
                    &cache,
                    CaptureSourceConstruction::new(None),
                    || panic!("ordinary discovery must share the funded cache"),
                )
                .unwrap();
                assert!(Arc::ptr_eq(
                    source.0.as_ref().unwrap(),
                    unaccounted.0.as_ref().unwrap()
                ));
                drop(funding);
                drop(cache);
                drop(source);
                drop(again);
                assert!(!payload.load(Ordering::SeqCst));
                assert!(!retired.load(Ordering::SeqCst));
                drop(unaccounted);
            }
            Err(error) => {
                assert!(matches!(
                    error,
                    CaptureError::AdmissionStorage(
                        eredu_core::capture::CaptureAdmissionStorageError::Funding(
                            HostMetadataFundingError::Unavailable
                        )
                    )
                ));
                assert_eq!(requests, refuse + 1);
                if let Some(old) = &old_alias {
                    let current = cache.borrow();
                    assert!(Arc::ptr_eq(
                        old.0.as_ref().unwrap(),
                        current.as_ref().unwrap().0.as_ref().unwrap()
                    ));
                    assert!(!current.as_ref().unwrap().is_funded());
                } else {
                    assert!(cache.borrow().is_none());
                }
                if built.load(Ordering::SeqCst) == 0 {
                    // No producer was entered, so there is no payload to retire.
                    payload.store(true, Ordering::SeqCst);
                }
                assert!(payload.load(Ordering::SeqCst));
                assert!(!retired.load(Ordering::SeqCst));
                drop(funding);
                drop(cache);
            }
        }
        assert!(payload.load(Ordering::SeqCst));
        assert!(retired.load(Ordering::SeqCst));
        if let Some(old) = old_alias {
            assert_eq!(old.value, "ordinary");
            assert!(!old.custody.retired.load(Ordering::SeqCst));
            drop(old);
            assert!(old_payload.load(Ordering::SeqCst));
        }
        requests
    }
    #[test]
    fn original_cold_cache_publishes_only_after_every_destination_succeeds() {
        let requests = cache_attempt(false, usize::MAX);
        for refuse in 1..requests {
            cache_attempt(false, refuse);
        }
    }
    #[test]
    fn original_cache_replaces_ordinary_publication_without_adopting_aliases() {
        let requests = cache_attempt(true, usize::MAX);
        for refuse in 1..requests {
            cache_attempt(true, refuse);
        }
    }
}
