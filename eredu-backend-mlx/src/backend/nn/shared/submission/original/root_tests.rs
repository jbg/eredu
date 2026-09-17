use super::*;

impl PreparedNeuralSubmission {
    pub(crate) fn exercise_root_iterator_unwind(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        input: &Array,
        stream: &Stream,
    ) -> Rc<Cell<bool>> {
        let prepared =
            Self::try_new(NeuralSubmissionShape::new(2, 0).unwrap(), controls.clone()).unwrap();
        let dropped = Rc::new(Cell::new(false));
        prepared
            .shared
            .payload
            .borrow_mut()
            .as_mut()
            .unwrap()
            .witness = Some(tests::PayloadWitness(dropped.clone()));
        let mut first = true;
        let roots = std::iter::from_fn(|| {
            if std::mem::replace(&mut first, false) {
                Some(input)
            } else {
                std::panic::panic_any(37_u8)
            }
        });
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = prepared.submit_arrays(roots, observer.clone(), stream);
        }));
        assert_eq!(*unwind.unwrap_err().downcast::<u8>().unwrap(), 37);
        assert!(
            !dropped.get(),
            "actual payload must await unlocked retirement"
        );
        dropped
    }

    pub(crate) fn exercise_exact_root_iterator(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        input: &Array,
        stream: &Stream,
    ) {
        struct Misleading<'a> {
            root: &'a Array,
            left: usize,
        }
        impl<'a> Iterator for Misleading<'a> {
            type Item = &'a Array;
            fn next(&mut self) -> Option<Self::Item> {
                if self.left == 0 {
                    None
                } else {
                    self.left -= 1;
                    Some(self.root)
                }
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (2, Some(2))
            }
        }
        for (actual, prefix, extra) in [(1, 1, false), (3, 2, true)] {
            let prepared =
                Self::try_new(NeuralSubmissionShape::new(2, 0).unwrap(), controls.clone()).unwrap();
            let (address, capacity, slots_address) = {
                let loan = prepared.shared.payload.borrow();
                let payload = loan.as_ref().unwrap();
                let arrays = &payload.arrays;
                assert_eq!(payload.clone_slots.len(), 2);
                (
                    arrays.as_ptr(),
                    arrays.capacity(),
                    payload.clone_slots.as_ptr(),
                )
            };
            let failed = match prepared.submit_arrays(
                Misleading {
                    root: input,
                    left: actual,
                },
                observer.clone(),
                stream,
            ) {
                Ok(_) => panic!("incorrect root extent accepted"),
                Err(failed) => failed,
            };
            assert!(
                matches!(&failed.cause, OriginalArraySubmissionCause::Extent { expected: 2, prefix: n, extra: e } if *n == prefix && *e == extra)
            );
            let OriginalSubmissionOwner::Active(active) = &failed.owner else {
                panic!("cloned prefix not armed")
            };
            {
                let loan = active.retained.retention().shared.payload.borrow();
                let payload = loan.as_ref().unwrap();
                assert_eq!(payload.arrays.len(), prefix);
                assert_eq!(payload.arrays.as_ptr(), address);
                assert_eq!(payload.arrays.capacity(), capacity);
                assert_eq!(payload.clone_slots.len(), 2);
                assert_eq!(payload.clone_slots.as_ptr(), slots_address);
                assert!(payload.event.is_none(), "mismatch dispatched native work");
            }
            drop(failed);
        }
    }
}
