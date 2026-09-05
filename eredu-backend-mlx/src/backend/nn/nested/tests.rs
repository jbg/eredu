use crate::array;

use super::*;

#[test]
fn test_flatten_nested_hash_map_of_owned_arrays() {
    let first_entry = NestedValue::Value(array!([1, 2, 3]));
    let second_entry = NestedValue::Map({
        let mut map = HashMap::new();
        map.insert("a", NestedValue::Value(array!([4, 5, 6])));
        map.insert("b", NestedValue::Value(array!([7, 8, 9])));
        map
    });

    let map = NestedHashMap {
        entries: {
            let mut map = HashMap::new();
            map.insert("first", first_entry);
            map.insert("second", second_entry);
            map
        },
    };

    let flattened = map.flatten();

    assert_eq!(flattened.len(), 3);
    assert!(crate::array::eval_equal_values(
        &flattened["first"],
        &array!([1, 2, 3])
    ));
    assert!(crate::array::eval_equal_values(
        &flattened["second.a"],
        &array!([4, 5, 6])
    ));
    assert!(crate::array::eval_equal_values(
        &flattened["second.b"],
        &array!([7, 8, 9])
    ));
}

#[test]
fn test_flatten_nested_hash_map_of_borrowed_arrays() {
    let first_entry_content = array!([1, 2, 3]);
    let first_entry = NestedValue::Value(&first_entry_content);

    let second_entry_content_a = array!([4, 5, 6]);
    let second_entry_content_b = array!([7, 8, 9]);
    let second_entry = NestedValue::Map({
        let mut map = HashMap::new();
        map.insert("a", NestedValue::Value(&second_entry_content_a));
        map.insert("b", NestedValue::Value(&second_entry_content_b));
        map
    });

    let map = NestedHashMap {
        entries: {
            let mut map = HashMap::new();
            map.insert("first", first_entry);
            map.insert("second", second_entry);
            map
        },
    };

    let flattened = map.flatten();

    assert_eq!(flattened.len(), 3);
    assert!(crate::array::eval_equal_values(
        flattened["first"],
        &first_entry_content
    ));
    assert!(crate::array::eval_equal_values(
        flattened["second.a"],
        &second_entry_content_a
    ));
    assert!(crate::array::eval_equal_values(
        flattened["second.b"],
        &second_entry_content_b
    ));
}

#[test]
fn test_flatten_nested_hash_map_of_mut_borrowed_arrays() {
    let mut first_entry_content = array!([1, 2, 3]);
    let first_entry = NestedValue::Value(&mut first_entry_content);

    let mut second_entry_content_a = array!([4, 5, 6]);
    let mut second_entry_content_b = array!([7, 8, 9]);
    let second_entry = NestedValue::Map({
        let mut map = HashMap::new();
        map.insert("a", NestedValue::Value(&mut second_entry_content_a));
        map.insert("b", NestedValue::Value(&mut second_entry_content_b));
        map
    });

    let map = NestedHashMap {
        entries: {
            let mut map = HashMap::new();
            map.insert("first", first_entry);
            map.insert("second", second_entry);
            map
        },
    };

    let flattened = map.flatten();

    assert_eq!(flattened.len(), 3);
    assert!(crate::array::eval_equal_values(
        flattened["first"],
        &array!([1, 2, 3])
    ));
    assert!(crate::array::eval_equal_values(
        flattened["second.a"],
        &array!([4, 5, 6])
    ));
    assert!(crate::array::eval_equal_values(
        flattened["second.b"],
        &array!([7, 8, 9])
    ));
}

#[test]
fn test_flatten_empty_nested_hash_map() {
    let map = NestedHashMap::<&str, i32>::new();
    let flattened = map.flatten();

    assert!(flattened.is_empty());

    // Insert another empty map
    let mut map = NestedHashMap::<&str, i32>::new();
    let empty_map = NestedValue::Map(HashMap::new());
    map.insert("empty", empty_map);

    let flattened = map.flatten();
    assert!(flattened.is_empty());
}
