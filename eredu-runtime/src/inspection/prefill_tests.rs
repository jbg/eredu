use super::*;
use crate::prefill::PrefillChunk;
use std::{fmt, rc::Rc};

#[derive(Debug)]
struct Original(Rc<()>);
impl fmt::Display for Original {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("the error bridge must retain the original without formatting it")
    }
}
impl std::error::Error for Original {}

#[derive(Default)]
struct Observer {
    chunks: Vec<usize>,
    terminals: Vec<bool>,
    failure: Option<Rc<()>>,
}
impl ActivationObserver<i32, Original> for Observer {
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Original> {
        self.chunks.push(std::ptr::from_ref(chunk) as usize);
        match &self.failure {
            Some(identity) => Err(Original(identity.clone())),
            None => Ok(()),
        }
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.terminals.push(committed);
    }
    fn observe(&mut self, _: &str, _: &i32) -> Result<(), Original> {
        Ok(())
    }
}
fn chunk(start: u64) -> PrefillChunk {
    PrefillChunk {
        input: start..start + 1,
        position: 7 + start,
        output: if start == 2 {
            eredu_core::OutputDemand::LastPosition
        } else {
            eredu_core::OutputDemand::StateOnly
        },
    }
}

#[test]
fn prefill_guard_borrows_exact_chunks_and_delivers_one_successful_terminal() {
    let chunks = [chunk(0), chunk(1), chunk(2)];
    let mut observer = Observer::default();
    {
        let mut borrowed = BorrowedActivationObserver(&mut observer);
        // Terminal forwarding is independent of the pre-existing transactional
        // flag; the new callbacks have defaults for ordinary observers.
        assert!(!borrowed.transactional());
        let guard = PrefillObservationGuard::<i32, Original, _>::new(&mut borrowed);
        for chunk in &chunks {
            guard.observer.begin_prefill_chunk(chunk).unwrap();
        }
        guard.finish(true);
    }
    assert_eq!(
        observer.chunks,
        chunks
            .iter()
            .map(|c| std::ptr::from_ref(c) as usize)
            .collect::<Vec<_>>()
    );
    assert_eq!(observer.terminals, [true]);
}

#[test]
fn prefill_guard_aborts_once_for_explicit_failure_drop_and_unwind() {
    for mode in 0..3 {
        let mut observer = Observer::default();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = PrefillObservationGuard::<i32, Original, _>::new(&mut observer);
            guard.observer.begin_prefill_chunk(&chunk(0)).unwrap();
            match mode {
                0 => guard.finish(false),
                1 => drop(guard),
                _ => panic!("later prefill failure"),
            }
        }));
        assert_eq!(outcome.is_err(), mode == 2);
        assert_eq!(observer.chunks.len(), 1);
        assert_eq!(observer.terminals, [false]);
    }
}

#[test]
fn prefill_error_bridge_keeps_original_cause_and_forwards_abort_without_copying_input() {
    let identity = Rc::new(());
    let mut observer = Observer {
        failure: Some(identity.clone()),
        ..Observer::default()
    };
    let chunk = chunk(0);
    let mut bridge = ObserverErrorBridge::new(
        &mut observer,
        |_: u8| Original(Rc::new(())),
        |_: &Original| 17u8,
    );
    {
        let guard = PrefillObservationGuard::<i32, u8, _>::new(&mut bridge);
        assert_eq!(guard.observer.begin_prefill_chunk(&chunk), Err(17));
        // The same outer failure guard must abort even if an enclosing layer
        // swallowed the propagation signal before the original is resolved.
    }
    let original = bridge.resolve(Ok(())).unwrap_err();
    assert!(Rc::ptr_eq(&original.0, &identity));
    assert_eq!(observer.chunks, [std::ptr::from_ref(&chunk) as usize]);
    assert_eq!(observer.terminals, [false]);
}

#[test]
fn legacy_noop_accepts_prefill_annotations_without_new_requirements() {
    let mut observer = NoopObserver;
    let guard = PrefillObservationGuard::<i32, (), _>::new(&mut observer);
    <NoopObserver as ActivationObserver<i32, ()>>::begin_prefill_chunk(guard.observer, &chunk(0))
        .unwrap();
    <NoopObserver as ActivationObserver<i32, ()>>::observe(guard.observer, "unchanged", &9)
        .unwrap();
    guard.finish(true);
}
