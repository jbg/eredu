use super::*;
use crate::backend::submission_recovery::Status;
use std::{
    cell::Cell,
    rc::{Rc, Weak},
};

struct Terminal;
impl Probe for Terminal {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        Status {
            settled: true,
            failed: false,
            blocked: false,
        }
    }
}
type Slot = RefCell<Option<Recovery<Reenter, Terminal>>>;
struct Reenter {
    slot: Weak<Slot>,
    drops: Rc<Cell<usize>>,
}
impl Retention for Reenter {
    fn observe(&self, _: Status) {}
}
impl Drop for Reenter {
    fn drop(&mut self) {
        let slot = self.slot.upgrade().unwrap();
        let loan = slot
            .try_borrow_mut()
            .expect("completion slot remained borrowed during retirement");
        assert!(loan.is_none());
        assert!(!safemlx::can_reclaim_submission_resources());
        self.drops.set(self.drops.get() + 1);
    }
}
#[test]
fn actual_completion_extractor_releases_its_loan_before_finish_or_drop_callbacks() {
    for finish in [true, false] {
        let slot = Rc::new(RefCell::new(None));
        let drops = Rc::new(Cell::new(0));
        *slot.borrow_mut() = Some(Recovery::with_probe(
            Reenter {
                slot: Rc::downgrade(&slot),
                drops: drops.clone(),
            },
            Terminal,
        ));
        // This is the same private extractor used by the completion's actual
        // branches; the callback reenters its precise original slot.
        let owner = take_recovery(&slot);
        if finish {
            let status = owner.map(Recovery::finish).unwrap();
            assert!(status.settled && !status.failed && !status.blocked);
        } else {
            drop(owner);
        }
        assert_eq!(drops.get(), 1);
        assert!(slot.borrow().is_none());
    }
}
