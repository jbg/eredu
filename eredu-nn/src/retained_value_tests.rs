use super::*;

fn parameter(id: &str, value: i32) -> Parameter<i32> {
    Parameter::new(ParameterSpec::trainable(id).unwrap(), value)
}

fn retained(module: &impl Parameterized<i32>) -> (bool, Vec<i32>) {
    let mut values = Vec::new();
    let complete = module.visit_retained_values(&mut |value| values.push(*value));
    (complete, values)
}

struct Legacy {
    weight: Parameter<i32>,
    hidden: Vec<i32>,
}

impl Parameterized<i32> for Legacy {
    fn visit_parameters<'a, V: ParameterVisitor<'a, i32>>(&'a self, visitor: &mut V) {
        self.weight.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, i32>>(&'a mut self, visitor: &mut V) {
        self.weight.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.weight.set_trainable(trainable);
    }
}

#[derive(Parameterized)]
#[parameterized(tensor = "i32")]
struct ExplicitValues {
    weight: Parameter<i32>,
    #[parameter(skip, retained_value)]
    frequencies: i32,
    #[parameter(skip, retained_optional_value)]
    scale: Option<i32>,
    #[parameter(skip, metadata)]
    width: usize,
}

#[derive(Parameterized)]
#[parameterized(tensor = "i32")]
struct Nested {
    opaque: Legacy,
    known_after_opaque: ExplicitValues,
}

#[derive(Parameterized)]
#[parameterized(tensor = "i32")]
enum Choice {
    Named {
        #[parameter(skip, metadata)]
        width: usize,
        #[parameter(skip, retained_value)]
        buffer: i32,
        #[parameter(skip, retained_optional_value)]
        optional: Option<i32>,
        weight: Parameter<i32>,
    },
    Tuple(
        #[parameter(skip)] Vec<i32>,
        #[parameter(skip, retained_value)] i32,
        Parameter<i32>,
    ),
    Empty,
}

#[test]
fn legacy_default_exposes_parameters_but_cannot_certify_hidden_buffers() {
    let legacy = Legacy {
        weight: parameter("weight", 3),
        hidden: vec![5, 7],
    };
    assert_eq!(retained(&legacy), (false, vec![3]));
    assert_eq!(legacy.hidden, [5, 7]);
}

#[test]
fn retained_buffers_are_borrowed_without_changing_parameter_topology() {
    let mut module = ExplicitValues {
        weight: parameter("weight", 3),
        frequencies: 5,
        scale: Some(7),
        width: 8,
    };
    let before = validate_parameter_topology(&module).unwrap();
    let mut pointers = Vec::new();
    assert!(module.visit_retained_values(&mut |value| pointers.push(std::ptr::from_ref(value))));
    assert_eq!(
        pointers,
        [
            std::ptr::from_ref(module.weight.as_ref()),
            std::ptr::from_ref(&module.frequencies),
            std::ptr::from_ref(module.scale.as_ref().unwrap()),
        ]
    );
    assert_eq!(module.width, 8);
    assert_eq!(validate_parameter_topology(&module).unwrap(), before);
    struct Replace;
    impl<'a> ParameterVisitorMut<'a, i32> for Replace {
        fn visit_mut(&mut self, _: ParameterMetadata, value: &'a mut i32) {
            *value = 11;
        }
    }
    module.visit_parameters_mut(&mut Replace);
    module.set_trainable(false);
    assert_eq!(retained(&module), (true, vec![11, 5, 7]));
    let after = validate_parameter_topology(&module).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].id, before[0].id);
    assert!(!after[0].trainable);
    module.scale = None;
    assert_eq!(retained(&module), (true, vec![11, 5]));
}

#[test]
fn incomplete_children_never_suppress_later_known_values() {
    let module = Nested {
        opaque: Legacy {
            weight: parameter("legacy", 2),
            hidden: vec![99],
        },
        known_after_opaque: ExplicitValues {
            weight: parameter("later", 3),
            frequencies: 5,
            scale: Some(7),
            width: 8,
        },
    };
    assert_eq!(retained(&module), (false, vec![2, 3, 5, 7]));
    let modules = vec![
        Legacy {
            weight: parameter("first", 11),
            hidden: vec![13],
        },
        Legacy {
            weight: parameter("second", 17),
            hidden: vec![19],
        },
    ];
    assert_eq!(retained(&modules), (false, vec![11, 17]));
    assert_eq!(retained(&Some(module)), (false, vec![2, 3, 5, 7]));
    assert_eq!(retained(&None::<Legacy>), (true, vec![]));
    assert_eq!(retained(&Vec::<Legacy>::new()), (true, vec![]));
}

#[test]
fn enum_variants_distinguish_metadata_buffers_and_unknown_skips() {
    let named = Choice::Named {
        width: 8,
        buffer: 2,
        optional: Some(3),
        weight: parameter("w", 5),
    };
    assert_eq!(retained(&named), (true, vec![2, 3, 5]));
    if let Choice::Named { width, .. } = named {
        assert_eq!(width, 8);
    }
    let tuple = Choice::Tuple(vec![7], 11, parameter("tail", 13));
    assert_eq!(retained(&tuple), (false, vec![11, 13]));
    if let Choice::Tuple(hidden, ..) = tuple {
        assert_eq!(hidden, [7]);
    }
    assert_eq!(retained(&Choice::Empty), (true, vec![]));
}

#[test]
fn ordinary_affine_leaves_and_empty_normalization_are_complete() {
    let linear = Linear {
        weight: parameter("weight", 2),
        bias: Some(parameter("bias", 3)),
    };
    assert_eq!(retained(&linear), (true, vec![2, 3]));
    let norm = LayerNorm {
        epsilon: 1e-5,
        weight: Some(parameter("scale", 5)),
        bias: Some(parameter("shift", 7)),
    };
    assert_eq!(retained(&norm), (true, vec![5, 7]));
    assert_eq!(
        retained(&LayerNorm {
            epsilon: 1e-5,
            weight: None,
            bias: None
        }),
        (true, vec![])
    );
}

mod slot_bounds;
