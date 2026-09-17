//! Canonical tickets from the real shared SessionPrefill fixture.
use super::*;
type State = DeviceState<FakeBackend, FakeLayerState>;
#[derive(Debug, PartialEq, Eq)]
enum Event {
    Opening(u64),
    End(u64),
    Retire(u64),
}
thread_local! {
    static FAIL_END: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TRACE: RefCell<Option<Vec<Event>>> = const { RefCell::new(None) };
}
pub(crate) fn enabled() -> bool {
    TRACE.with(|t| t.borrow().is_some())
}
pub(crate) fn opening(
    _: &State,
    context: &PrefillChunkRetentionContext<'_>,
    execution: &PrefillOpeningExecution<'_, FakeTensor>,
) -> Result<(), &'static str> {
    INFERENCE_SCOPE_TRACE.with(|t| assert_eq!(t.borrow().active, 1));
    assert!(
        execution.prepared_paths().is_none(),
        "legacy no-opt-in view remains unprepared"
    );
    TRACE.with(|t| {
        t.borrow_mut()
            .as_mut()
            .unwrap()
            .push(Event::Opening(context.chunk().input.start))
    });
    Ok(())
}
pub(crate) fn completed(
    state: &State,
    ticket: &SettledPrefillChunkRetention,
    execution: &PrefillOpeningExecution<'_, FakeTensor>,
) -> Result<(), &'static str> {
    if !enabled() {
        return Ok(());
    }
    INFERENCE_SCOPE_TRACE.with(|t| {
        let t = t.borrow();
        assert_eq!(t.active, 1);
        assert_eq!(t.opened, t.finished + 1);
    });
    assert!(execution.prepared_paths().is_none());
    assert_eq!(
        state.as_ref()[0].0,
        1,
        "actual updated decoder state is lent after work"
    );
    TRACE.with(|t| {
        t.borrow_mut()
            .as_mut()
            .unwrap()
            .push(Event::End(ticket.chunk().input.end))
    });
    if FAIL_END.replace(false) {
        return Err("original completed source error");
    }
    Ok(())
}
pub(crate) fn retiring(ticket: &SettledPrefillChunkRetention) {
    TRACE.with(|t| {
        if let Some(events) = t.borrow_mut().as_mut() {
            assert_eq!(events.last(), Some(&Event::End(ticket.chunk().input.end)));
            events.push(Event::Retire(ticket.chunk().input.end));
        }
    });
}
struct Trace;
impl Trace {
    fn new() -> Self {
        TRACE.with(|t| {
            assert!(t.borrow().is_none());
            *t.borrow_mut() = Some(vec![]);
        });
        Self
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        FAIL_END.set(false);
        TRACE.with(|t| *t.borrow_mut() = None);
    }
}
#[test]
fn completed_current_sources_precede_observer_retirement_under_existing_guard() {
    let _trace = Trace::new();
    exercise(Failure::None, false);
    TRACE.with(|t| {
        assert_eq!(
            t.borrow().as_ref().unwrap(),
            &[Event::Opening(0), Event::End(1), Event::Retire(1)]
        )
    });
}
#[test]
fn initial_cancellation_never_lends_opening_or_completed_sources() {
    let _trace = Trace::new();
    exercise(Failure::None, true);
    TRACE.with(|t| assert!(t.borrow().as_ref().unwrap().is_empty()));
}
#[test]
fn completed_sources_still_precede_final_cancellation_and_aborted_frame() {
    let _trace = Trace::new();
    exercise(Failure::FinalCancellation, false);
    TRACE.with(|t| {
        assert_eq!(
            t.borrow().as_ref().unwrap(),
            &[Event::Opening(0), Event::End(1), Event::Retire(1)]
        )
    });
}

#[test]
fn original_completed_source_error_abandons_guard_and_prevents_observer_retirement() {
    let _trace = Trace::new();
    FAIL_END.set(true);
    exercise(Failure::CompletedSources, false);
    TRACE.with(|t| {
        assert_eq!(
            t.borrow().as_ref().unwrap(),
            &[Event::Opening(0), Event::End(1)]
        )
    });
}
