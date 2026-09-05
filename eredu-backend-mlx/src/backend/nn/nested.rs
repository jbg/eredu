//! Nested maps used to traverse backend module parameters.

use std::{collections::HashMap, fmt::Display, rc::Rc};

const DELIMITER: char = '.';

/// A nested value that can be either a value or a map of nested values
#[derive(Debug, Clone)]
pub enum NestedValue<K, T> {
    /// A value
    Value(T),

    /// A map of nested values
    Map(HashMap<K, NestedValue<K, T>>),
}

impl<K, V> NestedValue<K, V> {
    /// Flattens the nested value into a hashmap
    pub fn flatten(self, prefix: &str) -> HashMap<Rc<str>, V>
    where
        K: Display,
    {
        match self {
            NestedValue::Value(array) => {
                let mut map = HashMap::new();
                map.insert(prefix.into(), array);
                map
            }
            NestedValue::Map(entries) => entries
                .into_iter()
                .flat_map(|(key, value)| value.flatten(&format!("{prefix}{DELIMITER}{key}")))
                .collect(),
        }
    }
}

/// A nested hashmap
#[derive(Debug, Clone)]
pub struct NestedHashMap<K, V> {
    /// The internal hashmap
    pub entries: HashMap<K, NestedValue<K, V>>,
}

impl<K, V> From<NestedHashMap<K, V>> for NestedValue<K, V> {
    fn from(map: NestedHashMap<K, V>) -> Self {
        NestedValue::Map(map.entries)
    }
}

impl<K, V> Default for NestedHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> NestedHashMap<K, V> {
    /// Creates a new nested hashmap
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Inserts a new entry into the nested hashmap
    pub fn insert(&mut self, key: K, value: NestedValue<K, V>)
    where
        K: Eq + std::hash::Hash,
    {
        self.entries.insert(key, value);
    }

    /// Flattens the nested hashmap into a hashmap
    pub fn flatten(self) -> HashMap<Rc<str>, V>
    where
        K: AsRef<str> + Display,
    {
        self.entries
            .into_iter()
            .flat_map(|(key, value)| value.flatten(key.as_ref()))
            .collect()
    }
}

#[cfg(test)]
mod tests;
