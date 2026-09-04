use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use std::cell::{Cell, RefCell};

static RUNTIME_LOCK: ReentrantMutex<()> = ReentrantMutex::new(());

thread_local! {
    static HOUSEKEEPING_HOOKS: RefCell<Vec<fn()>> = const { RefCell::new(Vec::new()) };
    static RUNNING_HOUSEKEEPING: Cell<bool> = const { Cell::new(false) };
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

pub(crate) fn register_housekeeping_hook(hook: fn()) {
    let _ = HOUSEKEEPING_HOOKS.try_with(|hooks| {
        let mut hooks = hooks.borrow_mut();
        if !hooks
            .iter()
            .any(|candidate| std::ptr::fn_addr_eq(*candidate, hook))
        {
            hooks.push(hook);
        }
    });
}

pub(crate) fn unregister_housekeeping_hook(hook: fn()) {
    let _ = HOUSEKEEPING_HOOKS.try_with(|hooks| {
        hooks
            .borrow_mut()
            .retain(|candidate| !std::ptr::fn_addr_eq(*candidate, hook));
    });
}

fn run_housekeeping_hooks() {
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
        for hook in hooks {
            hook();
        }
    });
}
