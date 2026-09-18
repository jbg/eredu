use super::*;
use crate::{LayerNorm, Linear, Parameter, ParameterVisitor, ParameterVisitorMut, Parameterized};
use std::cell::Cell;

#[derive(Default)]
struct Rows<'a> {
    named: Vec<(ParameterMetadataView<'a>, &'a i32)>,
    auxiliary: Vec<&'a i32>,
}
impl<'a> ParameterSourceVisitor<'a, i32> for Rows<'a> {
    fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a i32) {
        self.named.push((metadata, value));
    }
    fn retained(&mut self, value: &'a i32) {
        self.auxiliary.push(value);
    }
}
fn parameter(id: &str, value: i32) -> Parameter<i32> {
    Parameter::new(ParameterSpec::trainable(id).unwrap(), value)
}

#[test]
fn metadata_borrows_all_spec_fields_and_current_trainability() {
    for role in [
        None,
        Some(LinearCompanionRole::Scale),
        Some(LinearCompanionRole::AffineBias),
    ] {
        let mut spec = ParameterSpec::trainable("模型.échelle").unwrap();
        spec.alias_of = Some(ParameterId::new("共享.weight").unwrap());
        spec.group = Some("réplique.0".into());
        spec.linear_companion = role;
        spec.linear_companion_of = Some(ParameterId::new("primary.weight").unwrap());
        spec.linear_row_layout = LinearRowLayout::equal_partitions(3).unwrap();
        let view = ParameterMetadataView::from_spec(&spec, false);
        assert!(std::ptr::eq(view.id(), &spec.id));
        assert!(std::ptr::eq(
            view.alias_of().unwrap(),
            spec.alias_of.as_ref().unwrap()
        ));
        assert!(std::ptr::eq(
            view.group().unwrap().as_ptr(),
            spec.group.as_ref().unwrap().as_ptr()
        ));
        assert_eq!(view.linear_companion(), role);
        assert!(std::ptr::eq(
            view.linear_companion_of().unwrap(),
            spec.linear_companion_of.as_ref().unwrap()
        ));
        assert_eq!(view.linear_row_layout().partitions(), 3);
        let expected = ParameterMetadata {
            id: spec.id.clone(),
            trainable: false,
            alias_of: spec.alias_of.clone(),
            group: spec.group.clone(),
            linear_companion: role,
            linear_companion_of: spec.linear_companion_of.clone(),
            linear_row_layout: spec.linear_row_layout,
        };
        assert_eq!(view.to_owned(), expected);
        let mut actual = Parameter::new(spec, 19);
        actual.set_trainable(false);
        let mut rows = Rows::default();
        actual.visit_parameter_sources(&mut rows).unwrap();
        assert!(!rows.named[0].0.trainable());
        assert_eq!(*rows.named[0].1, 19);
        drop(rows);
        actual.set_trainable(true);
        let mut rows = Rows::default();
        actual.visit_parameter_sources(&mut rows).unwrap();
        assert!(rows.named[0].0.trainable());
    }
}
struct Incomplete {
    source_calls: Cell<usize>,
}
impl Parameterized<i32> for Incomplete {
    fn visit_parameter_sources<'a, V: crate::ParameterSourceVisitor<'a, i32>>(&'a self, _: &mut V) -> Result<(), crate::ParameterSourceError> {


        self.source_calls.set(self.source_calls.get() + 1);

 Err(ParameterSourceError::UnclassifiedRetainedField)
}
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, i32>>(&'a mut self, _: &mut V) {}
    fn set_trainable(&mut self, _: bool) {}
}
#[derive(crate::Parameterized)]
#[parameterized(tensor = "i32")]
struct Known {
    named: Parameter<i32>,
    #[parameter(skip, retained_value)]
    auxiliary: i32,
    #[parameter(skip, retained_optional_value)]
    optional: Option<i32>,
    #[parameter(skip, metadata)]
    width: usize,
}
#[derive(crate::Parameterized)]
#[parameterized(tensor = "i32")]
struct Nested {
    first: Incomplete,
    later: Known,
}
#[derive(crate::Parameterized)]
#[parameterized(tensor = "i32")]
enum Choice {
    Named {
        weight: Parameter<i32>,
        #[parameter(skip, retained_optional_value)]
        auxiliary: Option<i32>,
    },
    Tuple(
        #[parameter(skip)] Vec<i32>,
        Parameter<i32>,
        #[parameter(skip, retained_value)] i32,
    ),
    Empty,
}
#[test]
fn derived_nested_failure_visits_all_known_children_without_partial_success() {
    let value = Some(vec![Nested {
        first: Incomplete {
            source_calls: Cell::new(0),
        },
        later: Known {
            named: parameter("later", 7),
            auxiliary: 11,
            optional: Some(13),
            width: 2,
        },
    }]);
    let mut rows = Rows::default();
    assert_eq!(
        value.visit_parameter_sources(&mut rows),
        Err(ParameterSourceError::UnclassifiedRetainedField)
    );
    assert_eq!(rows.named.len(), 1);
    assert_eq!(*rows.named[0].1, 7);
    assert_eq!(
        rows.auxiliary.iter().map(|v| **v).collect::<Vec<_>>(),
        [11, 13]
    );
    let nested = &value.as_ref().unwrap()[0];
    assert_eq!(nested.first.source_calls.get(), 1);
    assert_eq!(nested.later.width, 2);
    assert!(std::ptr::eq(rows.named[0].1, nested.later.named.as_ref()));
    assert!(std::ptr::eq(rows.auxiliary[0], &nested.later.auxiliary));
}
#[test]
fn derived_variants_distinguish_bare_skip_optional_auxiliary_and_empty() {
    let named = Choice::Named {
        weight: parameter("w", 17),
        auxiliary: Some(19),
    };
    let mut rows = Rows::default();
    named.visit_parameter_sources(&mut rows).unwrap();
    assert_eq!(*rows.named[0].1, 17);
    assert_eq!(*rows.auxiliary[0], 19);
    let tuple = Choice::Tuple(vec![23], parameter("tail", 29), 31);
    let mut rows = Rows::default();
    assert_eq!(
        tuple.visit_parameter_sources(&mut rows),
        Err(ParameterSourceError::UnclassifiedRetainedField)
    );
    assert_eq!(*rows.named[0].1, 29);
    assert_eq!(*rows.auxiliary[0], 31);
    if let Choice::Tuple(hidden, _, _) = &tuple {
        assert_eq!(hidden, &[23]);
    }
    let empty = Choice::Empty;
    empty.visit_parameter_sources(&mut Rows::default()).unwrap();
    Vec::<Incomplete>::new()
        .visit_parameter_sources(&mut Rows::default())
        .unwrap();
    None::<Incomplete>
        .visit_parameter_sources(&mut Rows::default())
        .unwrap();
}
#[test]
fn handwritten_affine_and_normalization_use_actual_slots() {
    let linear = Linear {
        weight: parameter("w", 2),
        bias: Some(parameter("b", 3)),
    };
    let norm = LayerNorm {
        epsilon: 1e-5,
        weight: Some(parameter("scale", 5)),
        bias: Some(parameter("shift", 7)),
    };
    let mut rows = Rows::default();
    linear.visit_parameter_sources(&mut rows).unwrap();
    norm.visit_parameter_sources(&mut rows).unwrap();
    assert_eq!(
        rows.named.iter().map(|r| *r.1).collect::<Vec<_>>(),
        [2, 3, 5, 7]
    );
    assert!(std::ptr::eq(rows.named[0].1, linear.weight.as_ref()));
    let empty: LayerNorm<i32> = LayerNorm {
        epsilon: 1e-5,
        weight: None,
        bias: None,
    };
    empty.visit_parameter_sources(&mut rows).unwrap();
    assert_eq!(rows.named.len(), 4);
}
