use super::*;

#[test]
fn explicit_optional_tensor_population_stays_inside_derived_bound() {
    let mut module = ExplicitValues {
        weight: parameter("weight", 3),
        frequencies: 5,
        scale: None,
        width: 8,
    };
    assert_eq!(module.retained_value_slot_bound(), Some(3));
    assert_eq!(retained(&module).1, [3, 5]);
    module.scale = Some(7);
    assert_eq!(module.retained_value_slot_bound(), Some(3));
    assert_eq!(retained(&module).1, [3, 5, 7]);
    let choice = Choice::Named {
        width: 8,
        buffer: 11,
        optional: None,
        weight: parameter("w", 13),
    };
    assert_eq!(choice.retained_value_slot_bound(), Some(3));
    assert_eq!(Choice::Empty.retained_value_slot_bound(), Some(0));
    assert_eq!(
        Choice::Tuple(vec![17], 19, parameter("x", 23)).retained_value_slot_bound(),
        None
    );
}

#[test]
fn unknown_children_remain_unknown_and_affine_future_fields_keep_their_bound() {
    let opaque = Incomplete {
        weight: parameter("w", 3),
        hidden: vec![5],
    };
    assert_eq!(opaque.retained_value_slot_bound(), None);
    let absent: Option<Parameter<i32>> = None;
    assert_eq!(absent.retained_value_slot_bound(), Some(0));
    assert_eq!(Some(parameter("w", 7)).retained_value_slot_bound(), Some(1));
    let mut affine = Linear {
        weight: parameter("weight", 11),
        bias: None,
    };
    assert_eq!(affine.retained_value_slot_bound(), Some(2));
    affine.bias = Some(parameter("bias", 13));
    assert_eq!(retained(&affine).1, [11, 13]);
    assert_eq!(affine.retained_value_slot_bound(), Some(2));
}

struct Ceiling(usize);
impl Parameterized<i32> for Ceiling {
    fn visit_parameter_sources<'a, V: crate::ParameterSourceVisitor<'a, i32>>(&'a self, _: &mut V) -> Result<(), crate::ParameterSourceError> {
 let mut __source_result = Ok(());

 __source_result
}
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, i32>>(&'a mut self, _: &mut V) {}
    fn set_trainable(&mut self, _: bool) {}
    fn retained_value_slot_bound(&self) -> Option<usize> {
        Some(self.0)
    }
}
#[derive(Parameterized)]
#[parameterized(tensor = "i32")]
struct Sum {
    first: Ceiling,
    second: Ceiling,
}
#[test]
fn checked_container_and_derived_sums_preserve_overflow_as_unknown() {
    assert_eq!(
        vec![Ceiling(2), Ceiling(3)].retained_value_slot_bound(),
        Some(5)
    );
    assert_eq!(
        vec![Ceiling(usize::MAX), Ceiling(1)].retained_value_slot_bound(),
        None
    );
    assert_eq!(
        Sum {
            first: Ceiling(usize::MAX),
            second: Ceiling(1)
        }
        .retained_value_slot_bound(),
        None
    );
}

#[test]
fn absent_nested_modules_count_empty_topology_and_present_unknown_stays_unknown() {
    let mut children: Vec<Option<Option<Parameter<i32>>>> = vec![None, Some(None)];
    assert_eq!(children.retained_value_slot_bound(), Some(0));
    assert!(retained(&children).1.is_empty());
    // Installing a child changes topology. The freshly queried bound includes it.
    children[0] = Some(Some(parameter("installed", 17)));
    assert_eq!(children.retained_value_slot_bound(), Some(1));
    assert_eq!(retained(&children).1, [17]);
    children.push(Some(Some(parameter("appended", 23))));
    assert_eq!(children.retained_value_slot_bound(), Some(2));
    assert_eq!(retained(&children).1, [17, 23]);
    let unknown = Some(Incomplete {
        weight: parameter("unknown", 29),
        hidden: vec![31],
    });
    assert_eq!(unknown.retained_value_slot_bound(), None);
    assert_eq!(
        Vec::<Option<Parameter<i32>>>::new().retained_value_slot_bound(),
        Some(0)
    );
    assert_eq!(Some(Choice::Empty).retained_value_slot_bound(), Some(0));
    assert_eq!(
        Some(vec![Ceiling(usize::MAX), Ceiling(1)]).retained_value_slot_bound(),
        None
    );
}
