//! The existing ordinary callback order, shared with the native owned boundary.
use eredu_runtime::{ActivationObserver, prefill::PrefillChunk, observe_and_intervene};

/// Executes the ordinary outer observation contract. Unchanged values retain
/// the previous Clone behavior; a structural empty slot has Noop semantics.
pub fn observe_embedded_tensor_ordinary<T: Clone, E>(
    value: T,
    path: &str,
    chunk: Option<&PrefillChunk>,
    observer: Option<&mut dyn ActivationObserver<T, E>>,
) -> Result<T, E> {
    match observer {
        Some(observer) => {
            if let Some(chunk) = chunk {
                observer.begin_prefill_chunk(chunk)?;
            }
            observe_and_intervene(observer, path, &value)
        }
        None => Ok(value.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };
    struct Value(u32, Rc<Cell<usize>>);
    impl Clone for Value {
        fn clone(&self) -> Self {
            self.1.set(self.1.get() + 1);
            Self(self.0, self.1.clone())
        }
    }
    struct Observer {
        events: Rc<RefCell<Vec<&'static str>>>,
        replacement: Option<u32>,
        fail: bool,
    }
    impl ActivationObserver<Value, &'static str> for Observer {
        fn begin_prefill_chunk(&mut self, _: &PrefillChunk) -> Result<(), &'static str> {
            self.events.borrow_mut().push("begin");
            Ok(())
        }
        fn observe(&mut self, _: &str, _: &Value) -> Result<(), &'static str> {
            self.events.borrow_mut().push("observe");
            if self.fail { Err("observer") } else { Ok(()) }
        }
        fn intervene(&mut self, _: &str, value: &Value) -> Result<Option<Value>, &'static str> {
            self.events.borrow_mut().push("intervene");
            Ok(self.replacement.map(|v| Value(v, value.1.clone())))
        }
    }
    #[test]
    fn owned_outer_boundary_preserves_ordinary_clone_replacement_and_failure_order() {
        let clones = Rc::new(Cell::new(0));
        let events = Rc::new(RefCell::new(Vec::new()));
        let empty = observe_embedded_tensor_ordinary::<_, &'static str>(
            Value(17, clones.clone()),
            "p",
            None,
            None,
        )
        .unwrap();
        assert_eq!(empty.0, 17);
        assert_eq!(clones.get(), 1);
        let chunk = PrefillChunk {
            input: 0..2,
            position: 0,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let mut observer = Observer {
            events: events.clone(),
            replacement: Some(19),
            fail: false,
        };
        let replaced = observe_embedded_tensor_ordinary(
            Value(23, clones.clone()),
            "p",
            Some(&chunk),
            Some(&mut observer),
        )
        .unwrap();
        assert_eq!(replaced.0, 19);
        assert_eq!(clones.get(), 1);
        assert_eq!(&*events.borrow(), &["begin", "observe", "intervene"]);
        events.borrow_mut().clear();
        observer.replacement = None;
        let unchanged = observe_embedded_tensor_ordinary(
            Value(31, clones.clone()),
            "p",
            None,
            Some(&mut observer),
        )
        .unwrap();
        assert_eq!(unchanged.0, 31);
        assert_eq!(clones.get(), 2);
        assert_eq!(&*events.borrow(), &["observe", "intervene"]);
        events.borrow_mut().clear();
        observer.fail = true;
        let failed = observe_embedded_tensor_ordinary(
            Value(47, clones.clone()),
            "p",
            Some(&chunk),
            Some(&mut observer),
        );
        assert!(matches!(failed, Err("observer")));
        assert_eq!(clones.get(), 2);
        assert_eq!(&*events.borrow(), &["begin", "observe"]);
    }
}
