use super::*;
use crate::store::{checked_elements, invalid_selection};

fn old_validate_selection(
    key: &str,
    shape: &[usize],
    selection: &TensorSelection,
) -> Result<Vec<usize>, StoreError> {
    checked_elements(key, shape)?;
    let mut output = shape.to_vec();
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| invalid_selection(key, "axis outside rank"))?;
            if start >= end || *end > *dimension {
                return Err(invalid_selection(key, "range outside dimension"));
            }
            output[*axis] = end - start;
        }
        TensorSelection::Indices { axis, indices } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| invalid_selection(key, "axis outside rank"))?;
            if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                return Err(invalid_selection(
                    key,
                    "indices are empty or outside dimension",
                ));
            }
            output[*axis] = indices.len();
        }
        TensorSelection::Contiguous {
            offset_elements,
            shape: selected,
        } => {
            if selected.is_empty() || selected.contains(&0) {
                return Err(invalid_selection(key, "contiguous output shape is empty"));
            }
            let end = offset_elements
                .checked_add(checked_elements(key, selected)?)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("contiguous selection end for {key:?}"),
                })?;
            if end > checked_elements(key, shape)? {
                return Err(invalid_selection(key, "contiguous span outside tensor"));
            }
            output = selected.clone();
        }
    }
    checked_elements(key, &output)?;
    Ok(output)
}

#[test]
fn destination_selection_matches_untouched_ordinary_reference_and_error_order() {
    let shapes = [
        vec![],
        vec![0],
        vec![2, 3],
        vec![usize::MAX, 2, 0],
        vec![0, usize::MAX, 2],
        vec![usize::MAX, 1],
    ];
    let selections = [
        TensorSelection::Full,
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        },
        TensorSelection::Range {
            axis: 9,
            start: 0,
            end: 0,
        },
        TensorSelection::Indices {
            axis: 1,
            indices: vec![0, 0],
        },
        TensorSelection::Indices {
            axis: 0,
            indices: vec![],
        },
        TensorSelection::Contiguous {
            offset_elements: 1,
            shape: vec![1, 2],
        },
        TensorSelection::Contiguous {
            offset_elements: usize::MAX,
            shape: vec![1],
        },
        TensorSelection::Contiguous {
            offset_elements: 0,
            shape: vec![usize::MAX, 2],
        },
        TensorSelection::Contiguous {
            offset_elements: 0,
            shape: vec![0],
        },
    ];
    for shape in &shapes {
        for selection in &selections {
            let expected =
                old_validate_selection("real\"key", shape, selection).map_err(|e| e.to_string());
            let ordinary = crate::store::validate_selection("real\"key", shape, selection)
                .map_err(|e| e.to_string());
            assert_eq!(ordinary, expected);
            let plan = SelectionValidationPlan::new("real\"key", shape, selection).unwrap();
            let mut initial = vec![77; shape.len()];
            let mut replacement =
                vec![88; plan.replacement_layout().size() / std::mem::size_of::<usize>()];
            let actual = plan
                .validate_into(&mut initial, &mut replacement)
                .map(|value| value.to_vec())
                .map_err(|e| {
                    assert_eq!(e.to_string(), StoreError::from(e).to_string());
                    assert!(std::error::Error::source(&e).is_none());
                    e.to_string()
                });
            assert_eq!(actual, expected, "{shape:?} / {selection:?}");
        }
    }
}

#[test]
fn destination_geometry_refuses_short_and_long_before_writes_and_returns_actual_slice() {
    let shape = [2, 3];
    let selection = TensorSelection::Contiguous {
        offset_elements: 1,
        shape: vec![1, 2],
    };
    let plan = SelectionValidationPlan::new("tensor", &shape, &selection).unwrap();
    for (n, m) in [(1, 2), (3, 2), (2, 1), (2, 3)] {
        let mut first = vec![17; n];
        let mut second = vec![29; m];
        assert!(plan.validate_into(&mut first, &mut second).is_err());
        assert_eq!(first, vec![17; n]);
        assert_eq!(second, vec![29; m]);
    }
    let mut first = [17; 2];
    let mut second = [29; 2];
    let address = second.as_ptr();
    let output = plan.validate_into(&mut first, &mut second).unwrap();
    assert_eq!(output, [1, 2]);
    assert_eq!(output.as_ptr(), address);
    assert_eq!(first, shape); // distinct initial copy persists in caller storage
    let full = TensorSelection::Full;
    let scalar = SelectionValidationPlan::new("scalar", &[], &full).unwrap();
    assert!(scalar.validate_into(&mut [], &mut []).unwrap().is_empty());
    let wide = vec![1; 257];
    let plan = SelectionValidationPlan::new("wide", &wide, &full).unwrap();
    let mut destination = vec![0; wide.len()];
    assert_eq!(plan.validate_into(&mut destination, &mut []).unwrap(), wide);
}

#[test]
fn destination_selection_preserves_initial_overflow_and_late_failure_writes() {
    let invalid = TensorSelection::Range {
        axis: 9,
        start: 0,
        end: 1,
    };
    let shape = [usize::MAX, 2, 0];
    let plan = SelectionValidationPlan::new("tensor", &shape, &invalid).unwrap();
    let mut initial = [71; 3];
    assert!(matches!(
        plan.validate_into(&mut initial, &mut []).unwrap_err().cause,
        Cause::Elements
    ));
    assert_eq!(initial, [71; 3]);
    let shape = [0, usize::MAX, 2];
    let plan = SelectionValidationPlan::new("tensor", &shape, &invalid).unwrap();
    assert!(matches!(
        plan.validate_into(&mut initial, &mut []).unwrap_err().cause,
        Cause::Invalid("axis outside rank")
    ));
    assert_eq!(initial, shape);
}

#[test]
fn shared_selection_driver_keeps_replacement_and_failure_retirement_order() {
    use std::{cell::RefCell, rc::Rc};
    type Log = Rc<RefCell<Vec<&'static str>>>;
    struct Shape {
        value: Vec<usize>,
        name: &'static str,
        log: Log,
    }
    impl AsRef<[usize]> for Shape {
        fn as_ref(&self) -> &[usize] {
            &self.value
        }
    }
    impl AsMut<[usize]> for Shape {
        fn as_mut(&mut self) -> &mut [usize] {
            &mut self.value
        }
    }
    impl Drop for Shape {
        fn drop(&mut self) {
            self.log.borrow_mut().push(self.name);
        }
    }
    struct Policy(Log);
    impl Drop for Policy {
        fn drop(&mut self) {
            self.0.borrow_mut().push("policy retired");
        }
    }
    impl<'a> Storage<'a> for Policy {
        type Shape = Shape;
        type Error = Cause;
        fn initial(&mut self, shape: &[usize]) -> Shape {
            self.0.borrow_mut().push("initial copied");
            Shape {
                value: shape.to_vec(),
                name: "initial retired",
                log: self.0.clone(),
            }
        }
        fn replace(&mut self, output: &mut Shape, shape: &Vec<usize>) {
            self.0.borrow_mut().push("replacement copied");
            *output = Shape {
                value: shape.to_vec(),
                name: "replacement retired",
                log: self.0.clone(),
            };
        }
        fn error(&self, _: &'a str, cause: Cause) -> Cause {
            self.0.borrow_mut().push("error created");
            cause
        }
    }
    let log = Log::default();
    let output = validate(
        "tensor",
        &[2, 3],
        &TensorSelection::Contiguous {
            offset_elements: 1,
            shape: vec![2],
        },
        Policy(log.clone()),
    )
    .unwrap();
    assert_eq!(
        &*log.borrow(),
        &[
            "initial copied",
            "replacement copied",
            "initial retired",
            "policy retired"
        ]
    );
    drop(output);
    assert_eq!(log.borrow().last(), Some(&"replacement retired"));
    log.borrow_mut().clear();
    let result = validate(
        "tensor",
        &[2, 3],
        &TensorSelection::Range {
            axis: 3,
            start: 0,
            end: 1,
        },
        Policy(log.clone()),
    );
    assert!(result.is_err());
    assert_eq!(
        &*log.borrow(),
        &[
            "initial copied",
            "error created",
            "initial retired",
            "policy retired"
        ]
    );
}
