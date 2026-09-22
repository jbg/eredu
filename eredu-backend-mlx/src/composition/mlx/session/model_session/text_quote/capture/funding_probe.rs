//! Test-only, lexical observation of successful original captured admission.
//! No pool-derived bound, retained funding owner, allocation grant or production
//! callback is introduced. The test installs one exact source slot beforehand.
use super::*;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Clone, Copy, Debug)]
pub(in crate::composition::mlx::session::model_session) struct CapturedFundingQuote {
    pub reservation: u64,
    pub protected: u64,
    pub capture: u64,
    pub source: u64,
}
impl CapturedFundingQuote {
    pub fn source_tail(self) -> u64 {
        self.protected.checked_add(self.source).unwrap()
    }
}
struct Slot {
    source: eredu_core::SharedStorageIdentity,
    value: Cell<Option<CapturedFundingQuote>>,
}
thread_local! {
    static CURRENT: RefCell<Option<Rc<Slot>>> = const { RefCell::new(None) };
}
pub(in crate::composition::mlx::session::model_session) struct CaptureFundingProbe(Rc<Slot>);
impl CaptureFundingProbe {
    pub fn new(source: &SharedCapturePlan) -> Self {
        let slot = Rc::new(Slot {
            source: source.storage_identity().clone(),
            value: Cell::new(None),
        });
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            assert!(
                current.is_none(),
                "capture funding fixture scopes cannot nest"
            );
            *current = Some(slot.clone());
        });
        Self(slot)
    }
    pub fn take(&self) -> CapturedFundingQuote {
        self.0
            .value
            .take()
            .expect("actual successful original capture admission")
    }
    pub fn is_empty(&self) -> bool {
        self.0.value.get().is_none()
    }
}
impl Drop for CaptureFundingProbe {
    fn drop(&mut self) {
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            assert!(
                current
                    .as_ref()
                    .is_some_and(|slot| Rc::ptr_eq(slot, &self.0))
            );
            current.take();
        });
    }
}
pub(super) fn record(
    source: &SharedCapturePlan,
    span: &OwnedTextSpanWorkspace,
    bank: &PreparedCaptureRun,
) {
    CURRENT.with(|current| {
        let current = current.borrow();
        if let Some(slot) = current.as_ref() {
            assert_eq!(
                &slot.source,
                source.storage_identity(),
                "wrong original source in fixture scope"
            );
            assert!(
                slot.value.get().is_none(),
                "fixture must consume its original admission record"
            );
            slot.value.set(Some(CapturedFundingQuote {
                reservation: span
                    .reservation()
                    .requirements()
                    .get(crate::memory_topology().unwrap().host_domain())
                    .unwrap()
                    .total()
                    .unwrap(),
                protected: span.protected_host_bytes(),
                capture: bank.protected_bytes(),
                source: source.capacity_bytes().expect("admitted source extent"),
            }));
        }
    });
}
