//! Completed private readout roots never become escaping model/state storage.
use super::*;
use std::mem::{size_of, size_of_val};

// Move the existing allocation out of its slot while native descriptor
// validation/drop can retire event handles. Restore the same allocation on every
// return/unwind; no RefCell loan spans a native call or an Array destructor.
struct Roots<'a> {
    values: Vec<Array>,
    slot: &'a RefCell<Vec<Array>>,
}
impl<'a> Roots<'a> {
    fn take(slot: &'a RefCell<Vec<Array>>) -> Result<Self, Error> {
        let values = std::mem::take(
            &mut *slot
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?,
        );
        Ok(Self { values, slot })
    }
}
impl Drop for Roots<'_> {
    fn drop(&mut self) {
        // Activity excludes another capture/publication or certification. No
        // callback can activate this already-live Work's collectors again.
        let mut slot = self.slot.borrow_mut();
        assert!(slot.is_empty() && slot.capacity() == 0);
        *slot = std::mem::take(&mut self.values);
    }
}

impl FundedWork {
    pub(super) fn retire_completed_readout_roots(
        &self,
        first: usize,
        completion: CaptureCompletion<'_>,
    ) -> Result<(), Error> {
        let CaptureCompletion::Original(observer) = completion else {
            return Ok(());
        };
        // Every scalar frontier is already consumed. Retiring proven completed
        // roots authenticates the owner, but admits no additional submission.
        self.validate_capture_identity(completion)?;
        let _activity = publication_scope::Activity::begin(&self.publishing)?;
        let mut roots = Roots::take(&self.roots)?;
        let suffix = roots
            .values
            .get(first..)
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        // Scalar completion proves its ancestors finished, but their retained
        // descriptors can still carry completed events. Use the existing exact
        // observer check to detach only matching completed events. Unscheduled,
        // pending, foreign, busy and failed roots refuse without any Eval/wait.
        // Validate the whole suffix before retiring even one retained root.
        for root in suffix {
            observer.validate_completed_array(root)?;
        }
        while roots.values.len() > first {
            drop(roots.values.pop());
        }
        // The same capacity and preceding model/token roots return to Work.
        // Native Record/Graph and original accounting remain with the live role;
        // this grants no reuse credit, certification, refund or record retirement.
        Ok(())
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Roots<'static>>(),
        size_of::<Result<Roots<'static>, Error>>(),
        size_of::<Vec<Array>>(),
        size_of::<std::cell::RefMut<'static, Vec<Array>>>(),
        size_of::<Option<Array>>(),
        size_of::<std::slice::Iter<'static, Array>>(),
        size_of::<(&FundedWork, usize, CaptureCompletion<'static>)>(),
        size_of::<Result<(), safemlx::error::Exception>>(),
        size_of::<Result<(), Error>>(),
        safemlx::OriginalScopeObserver::control_bytes()?,
        publication_scope::control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
