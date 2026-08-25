//! Canonical prompt-cache identity components shared by architecture families.

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

pub(crate) fn string_set(values: Option<&HashSet<String>>) -> String {
    let mut values = values
        .map(|values| values.iter().map(String::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    values.sort_unstable();
    values.join(";")
}

pub(crate) fn debug_map<T: Debug>(values: Option<&HashMap<String, T>>) -> String {
    let mut values = values
        .map(|values| {
            values
                .iter()
                .map(|(name, value)| format!("{name}={value:?}"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    values.sort_unstable();
    values.join(";")
}
