use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

static RUNTIME_LOCK: ReentrantMutex<()> = ReentrantMutex::new(());
type HousekeepingHooks = Option<Rc<Vec<fn()>>>;

thread_local! {
    static HOUSEKEEPING_HOOKS: RefCell<HousekeepingHooks> = const { RefCell::new(None) };
    static RUNNING_HOUSEKEEPING: Cell<bool> = const { Cell::new(false) };
    static RETIRING_SUBMISSIONS: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct RuntimeLockGuard {
    _guard: ReentrantMutexGuard<'static, ()>,
}

pub(crate) fn enter() -> RuntimeLockGuard {
    let guard = RuntimeLockGuard {
        _guard: RUNTIME_LOCK.lock(),
    };
    run_housekeeping_hooks();
    guard
}

pub(crate) fn try_enter() -> Option<RuntimeLockGuard> {
    let guard = RUNTIME_LOCK
        .try_lock()
        .map(|guard| RuntimeLockGuard { _guard: guard })?;
    run_housekeeping_hooks();
    Some(guard)
}

/// Recovery must not clone or invoke housekeeping hooks while retiring an
/// error/unwind owner. Those hooks can allocate and reenter application code.
pub(crate) fn try_enter_for_recovery() -> Option<RuntimeLockGuard> {
    RUNTIME_LOCK
        .try_lock()
        .map(|guard| RuntimeLockGuard { _guard: guard })
}

pub(crate) fn try_retire<R>(retire: impl FnOnce() -> R) -> Option<R> {
    let _guard = try_enter_for_recovery()?;
    let previous = RETIRING_SUBMISSIONS
        .try_with(|state| state.replace(true))
        .ok();
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            if let Some(previous) = self.0 {
                let _ = RETIRING_SUBMISSIONS.try_with(|state| state.set(previous));
            }
        }
    }
    let _reset = Reset(previous);
    Some(retire())
}

pub(crate) fn can_reclaim_submission_resources() -> bool {
    !std::thread::panicking() && !RUNTIME_LOCK.is_owned_by_current_thread()
}

pub(crate) fn register_housekeeping_hook(hook: fn()) {
    let _ = HOUSEKEEPING_HOOKS.try_with(|hooks| {
        let mut hooks = hooks.borrow_mut();
        let hooks = hooks.get_or_insert_with(|| Rc::new(Vec::new()));
        if !hooks
            .iter()
            .any(|candidate| std::ptr::fn_addr_eq(*candidate, hook))
        {
            Rc::make_mut(hooks).push(hook);
        }
    });
}

pub(crate) fn unregister_housekeeping_hook(hook: fn()) {
    let _ = HOUSEKEEPING_HOOKS.try_with(|hooks| {
        if let Some(hooks) = hooks.borrow_mut().as_mut() {
            Rc::make_mut(hooks).retain(|candidate| !std::ptr::fn_addr_eq(*candidate, hook));
        }
    });
}

fn run_housekeeping_hooks() {
    if RETIRING_SUBMISSIONS.try_with(Cell::get).unwrap_or(true) {
        return;
    }
    let _ = RUNNING_HOUSEKEEPING.try_with(|running| {
        if running.replace(true) {
            return;
        }
        struct Reset<'a>(&'a Cell<bool>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(false);
            }
        }
        let _reset = Reset(running);
        let hooks = HOUSEKEEPING_HOOKS
            .try_with(|hooks| hooks.borrow().clone())
            .unwrap_or_default();
        if let Some(hooks) = hooks {
            for hook in hooks.iter() {
                hook();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    thread_local! { static CALLS: Cell<usize> = const { Cell::new(0) }; }
    fn second() {
        CALLS.with(|calls| calls.set(calls.get() + 10));
    }
    fn first() {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        unregister_housekeeping_hook(first);
        register_housekeeping_hook(second);
        let _reentrant = enter();
    }

    #[test]
    fn housekeeping_uses_stable_snapshots_and_recovery_never_invokes_hooks() {
        CALLS.with(|calls| calls.set(0));
        register_housekeeping_hook(first);
        drop(enter());
        assert_eq!(CALLS.with(Cell::get), 1);
        drop(try_enter_for_recovery().unwrap());
        assert_eq!(CALLS.with(Cell::get), 1);
        try_retire(|| {
            assert!(!can_reclaim_submission_resources());
            drop(enter());
            try_retire(|| drop(enter())).unwrap();
        })
        .unwrap();
        assert_eq!(CALLS.with(Cell::get), 1);
        assert!(can_reclaim_submission_resources());
        drop(enter());
        assert_eq!(CALLS.with(Cell::get), 11);
        unregister_housekeeping_hook(second);
    }
}
