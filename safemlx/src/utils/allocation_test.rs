//! Unit-test-only allocation observation through the existing sole allocator.
use std::cell::Cell;
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}
pub(crate) fn allocated() {
    let _ = ENABLED.try_with(|enabled| {
        if enabled.get() {
            let _ = COUNT.try_with(|count| count.set(count.get() + 1));
        }
    });
}
pub(crate) fn measure<T>(work: impl FnOnce() -> T) -> (T, usize) {
    assert!(!ENABLED.with(Cell::get));
    COUNT.with(|count| count.set(0));
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            let _ = ENABLED.try_with(|enabled| enabled.set(false));
        }
    }
    ENABLED.with(|enabled| enabled.set(true));
    let reset = Reset;
    let value = work();
    drop(reset);
    (value, COUNT.with(Cell::get))
}
