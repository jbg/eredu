use super::*;
use std::{cell::Cell, rc::Rc};

struct Value { scalar: u32, clones: Rc<Cell<usize>> }
impl Clone for Value {
    fn clone(&self) -> Self {
        self.clones.set(self.clones.get() + 1);
        Self { scalar: self.scalar, clones: self.clones.clone() }
    }
}

#[test]
fn absent_activation_interest_keeps_component_values_without_hooks_or_clones() {
    struct CutOnly;
    impl eredu_runtime::ActivationObserver<Value, Error> for CutOnly {
        fn observes_activations(&self) -> bool { false }
        fn observe(&mut self, _: &str, _: &Value) -> Result<(), Error> {
            panic!("activation hook must be absent")
        }
        fn intervene(&mut self, _: &str, _: &Value) -> Result<Option<Value>, Error> {
            panic!("intervention hook must be absent")
        }
    }
    let clones = Rc::new(Cell::new(0));
    let mut observer = CutOnly;
    let mut instrument = ComponentInstrumentation::new("model.layers.2", &mut observer);
    assert!(!instrument.enabled());
    let value = instrument.with_scope("mlp", |instrument| {
        instrument.apply("output", Value { scalar: 7, clones: clones.clone() })
    }).unwrap();
    instrument.observe("retained", &value).unwrap();
    assert_eq!(value.scalar, 7);
    assert_eq!(clones.get(), 0);
}

#[test]
fn default_activation_interest_preserves_original_effective_and_nested_paths() {
    struct Observer { events: Vec<(String, u32)> }
    impl eredu_runtime::ActivationObserver<Value, Error> for Observer {
        fn observe(&mut self, path: &str, value: &Value) -> Result<(), Error> {
            self.events.push((path.into(), value.scalar)); Ok(())
        }
        fn intervene(&mut self, _: &str, value: &Value) -> Result<Option<Value>, Error> {
            Ok(Some(Value { scalar: value.scalar + 3, clones: value.clones.clone() }))
        }
    }
    let mut observer = Observer { events: Vec::new() };
    let mut instrument = ComponentInstrumentation::new("model.layers.2", &mut observer);
    let value = instrument.with_scope("mlp", |instrument| {
        instrument.apply("output", Value { scalar: 7, clones: Rc::new(Cell::new(0)) })
    }).unwrap();
    assert_eq!(value.scalar, 10);
    assert_eq!(observer.events, [
        ("model.layers.2.mlp.output".into(), 7),
        ("model.layers.2.mlp.output.effective".into(), 10),
    ]);
}
