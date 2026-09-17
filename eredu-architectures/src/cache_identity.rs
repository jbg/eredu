
//! Canonical prompt-cache identity components shared by architecture families.
use crate::decoder::identity::Metadata;
use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Display},
};

pub(crate) fn string_set(values: Option<&HashSet<String>>) -> String {
    string_set_with_metadata(values, Metadata::new(None))
        .expect("ordinary fingerprint formatting is infallible")
}
pub(crate) fn string_set_with_metadata(
    values: Option<&HashSet<String>>,
    metadata: Metadata<'_>,
) -> Result<String, eredu_nn::Error> {
    let mut rows = metadata.vector(values.map_or(0, |values| values.len()))?;
    if let Some(values) = values {
        rows.extend(values.iter().map(String::as_str));
    }
    rows.sort_unstable();
    metadata.join(&rows, ";")
}
pub(crate) fn debug_map<T: Debug>(values: Option<&HashMap<String, T>>) -> String {
    debug_map_with_metadata(values, Metadata::new(None))
        .expect("ordinary fingerprint formatting is infallible")
}
pub(crate) fn debug_map_with_metadata<T: Debug>(
    values: Option<&HashMap<String, T>>,
    metadata: Metadata<'_>,
) -> Result<String, eredu_nn::Error> {
    let mut rows = metadata.vector(values.map_or(0, |values| values.len()))?;
    if let Some(values) = values {
        for (name, value) in values {
            rows.push(metadata.format(format_args!("{name}={value:?}"))?);
        }
    }
    // Preserve complete formatted-row order, including prefix/punctuation keys.
    rows.sort_unstable();
    metadata.join(&rows, ";")
}

pub(crate) struct Joined<F>(pub(crate) F, pub(crate) &'static str);
impl<F, I> Display for Joined<F>
where
    F: Fn() -> I,
    I: Iterator,
    I::Item: Display,
{
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, value) in (self.0)().enumerate() {
            if index != 0 {
                output.write_str(self.1)?;
            }
            Display::fmt(&value, output)?;
        }
        Ok(())
    }
}
pub(crate) struct DebugValue<T>(pub(crate) T);
impl<T: Debug> Display for DebugValue<T> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, output)
    }
}
