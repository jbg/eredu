use super::*;
use std::cell::Cell;

// A provider whose inventory fails after a real borrowed value. Consumers must
// retain the error even though earlier values have reached their collector.
struct PartialSource {
    values: [Box<[f32]>; 2],
    visits: Cell<usize>,
}
impl OpeningValues<Box<[f32]>> for PartialSource {
    fn visit(&self, visitor: &mut dyn FnMut(&Box<[f32]>)) -> Result<(), StateError> {
        self.visits.set(self.visits.get() + 1);
        visitor(&self.values[0]);
        Err(StateError::PreparedLayerCount {
            expected: 2,
            actual: 1,
        })
    }
}

#[test]
fn partial_inventory_keeps_typed_failure_and_borrow_identity_through_adapter() {
    let source = PartialSource {
        values: [
            vec![1.25, -3.5].into_boxed_slice(),
            vec![7.75].into_boxed_slice(),
        ],
        visits: Cell::new(0),
    };
    let opening = PrefillOpeningState { source: &source };
    let mut visited = 0;
    let error = opening
        .visit(&mut |value| {
            assert!(std::ptr::eq(value, &source.values[0]));
            assert_eq!(&**value, &[1.25, -3.5]);
            visited += 1;
        })
        .unwrap_err();
    assert!(matches!(
        error,
        StateError::PreparedLayerCount {
            expected: 2,
            actual: 1
        }
    ));
    assert_eq!(visited, 1);
    visited = 0;
    let error = opening
        .with_tensor_adapter(
            |value| &value[0],
            |mapped| {
                mapped.visit(&mut |value| {
                    assert!(std::ptr::eq(value, &source.values[0][0]));
                    assert_eq!(*value, 1.25);
                    visited += 1;
                })
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        StateError::PreparedLayerCount {
            expected: 2,
            actual: 1
        }
    ));
    assert_eq!(visited, 1);
    assert_eq!(source.visits.get(), 2);
    let error = opening
        .with_tensor_adapter(
            |value| &value[0],
            |mapped| {
                mapped.with_tensor_adapter(
                    |value| value,
                    |second| {
                        second.visit(&mut |value| {
                            assert!(std::ptr::eq(value, &source.values[0][0]));
                            assert_eq!(*value, 1.25);
                        })
                    },
                )
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        StateError::PreparedLayerCount {
            expected: 2,
            actual: 1
        }
    ));
    assert_eq!(source.visits.get(), 3);
}
