use super::*;
use std::collections::BTreeMap;

#[test]
fn active_dense_forward_is_unknown_even_with_known_empty_override_topology() {
    let mut f = fixture(false, 1);
    assert_eq!(f.policy.retained_value_slot_bound(), Some(0));
    let initial = MlxTensor::from_array(Array::from_slice(&[3_i32], &[1, 1]));
    f.policy.begin(&initial, &f.stream).unwrap();
    assert_eq!(f.policy.retained_value_slot_bound(), None);
    f.policy.finish(&initial, &f.stream).unwrap();
    assert_eq!(f.policy.retained_value_slot_bound(), Some(0));
}

#[test]
fn idle_override_ceiling_follows_actual_replacement_owner_and_excludes_unloaded_modules() {
    let mut f = fixture(false, 1);
    let original = MlxTensor::from_array(Array::from_slice(&[7_i32, 11], &[2]));
    let replacements = BTreeMap::from([("weight".to_owned(), original.clone())]);
    assert!(MlxUnitPopulator::<Unit>::publish_parameter_replacements(
        &mut f.policy.populator,
        &replacements,
        true
    ));
    assert_eq!(f.policy.retained_value_slot_bound(), Some(1));
    let mut values = 0;
    assert!(f.policy.visit_retained_values(&mut |value| {
        assert_eq!(
            value.as_array().allocation_info().unwrap(),
            original.as_array().allocation_info().unwrap()
        );
        values += 1;
    }));
    assert_eq!(values, 1);
    assert!(MlxUnitPopulator::<Unit>::publish_parameter_replacements(
        &mut f.policy.populator,
        &replacements,
        false
    ));
    assert_eq!(f.policy.retained_value_slot_bound(), Some(0));
}
