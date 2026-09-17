//! Scoped observation of actual source-root readiness, without evaluation.
use crate::MlxTensor;
use std::{cell::RefCell, rc::Rc};
#[derive(Clone, Debug)]
pub(crate) struct Sample {
    pub after: bool,
    pub shapes: Vec<Vec<i32>>,
    pub ready: Vec<bool>,
    pub backing: Vec<Option<safemlx::AllocationIdentity>>,
}
struct Active {
    samples: Vec<Sample>,
    cancel: Option<(eredu_core::GenerationCancellationToken, usize)>,
    completed: usize,
}
thread_local! {
    static ACTIVE: RefCell<Option<Rc<RefCell<Active>>>> = const { RefCell::new(None) };
}
pub(crate) fn record(
    roots: &eredu_runtime::media_prefill::RetainedMediaRoots<'_, MlxTensor>,
    after: bool,
) {
    let active = ACTIVE.with(|slot| slot.borrow().clone());
    if let Some(active) = active {
        let mut sample = Sample {
            after,
            shapes: Vec::new(),
            ready: Vec::new(),
            backing: Vec::new(),
        };
        roots.visit(&mut |root| {
            sample.shapes.push(root.as_array().shape().to_vec());
            let allocation = root.as_array().try_allocation_info().unwrap();
            sample.ready.push(allocation.is_some());
            sample
                .backing
                .push(allocation.map(|allocation| allocation.identity()));
        });
        let mut active = active.borrow_mut();
        active.samples.push(sample);
        if after {
            active.completed += 1;
            if let Some((cancel, boundary)) = &active.cancel {
                if active.completed == *boundary {
                    cancel.cancel();
                }
            }
        }
    }
}
pub(crate) fn observe<T>(
    cancel: Option<(eredu_core::GenerationCancellationToken, usize)>,
    operation: impl FnOnce() -> T,
) -> (T, Vec<Sample>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
    let samples = Rc::new(RefCell::new(Active {
        samples: Vec::new(),
        cancel,
        completed: 0,
    }));
    ACTIVE.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut() = Some(samples.clone());
    });
    let reset = Reset;
    let result = operation();
    drop(reset);
    let samples = Rc::try_unwrap(samples)
        .ok()
        .expect("probe owner retired")
        .into_inner()
        .samples;
    (result, samples)
}
