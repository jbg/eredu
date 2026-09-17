//! Shared ordered selection validation with ordinary or borrowed destinations.
use super::{StoreError, TensorSelection};
use std::alloc::Layout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cause {
    Elements,
    ContiguousEnd,
    Invalid(&'static str),
    Destination {
        replacement: bool,
        expected: usize,
        actual: usize,
    },
}

/// Fixed selection-validation failure retaining the actual borrowed key.
/// Conversion to the ordinary error preserves its existing owned diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionValidationError<'a> {
    key: &'a str,
    cause: Cause,
}
impl std::fmt::Display for SelectionValidationError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cause {
            Cause::Elements => write!(
                f,
                "checkpoint size overflow: element count for {:?}",
                self.key
            ),
            Cause::ContiguousEnd => write!(
                f,
                "checkpoint size overflow: contiguous selection end for {:?}",
                self.key
            ),
            Cause::Invalid(message) => {
                write!(f, "invalid selection for tensor {:?}: {message}", self.key)
            }
            Cause::Destination {
                replacement,
                expected,
                actual,
            } => write!(
                f,
                "selection {} destination has {actual} elements; expected {expected}",
                if replacement {
                    "replacement"
                } else {
                    "initial"
                }
            ),
        }
    }
}
impl std::error::Error for SelectionValidationError<'_> {}

pub(super) fn element_overflow(key: &str) -> SelectionValidationError<'_> {
    SelectionValidationError {
        key,
        cause: Cause::Elements,
    }
}

impl From<SelectionValidationError<'_>> for StoreError {
    fn from(error: SelectionValidationError<'_>) -> Self {
        match error.cause {
            Cause::Elements => Self::Overflow {
                context: format!("element count for {:?}", error.key),
            },
            Cause::ContiguousEnd => Self::Overflow {
                context: format!("contiguous selection end for {:?}", error.key),
            },
            Cause::Invalid(message) => super::invalid_selection(error.key, message),
            Cause::Destination { .. } => Self::Internal(error.to_string()),
        }
    }
}

/// Two checked destination layouts for the original validation transition.
/// The initial source-shape copy and late contiguous replacement are distinct.
/// This describes validation storage, not read or admission authority.
#[derive(Clone, Copy, Debug)]
pub struct SelectionValidationPlan<'a> {
    key: &'a str,
    shape: &'a [usize],
    selection: &'a TensorSelection,
    initial: Layout,
    replacement: Layout,
}
impl<'a> SelectionValidationPlan<'a> {
    /// Checks destination representability without validating tensor geometry.
    pub fn new(key: &'a str, shape: &'a [usize], selection: &'a TensorSelection) -> Option<Self> {
        let replacement = match selection {
            TensorSelection::Contiguous { shape, .. } => shape.len(),
            _ => 0,
        };
        Some(Self {
            key,
            shape,
            selection,
            initial: Layout::array::<usize>(shape.len()).ok()?,
            replacement: Layout::array::<usize>(replacement).ok()?,
        })
    }
    /// Initial source-shape destination.
    pub fn initial_layout(&self) -> Layout {
        self.initial
    }
    /// Late contiguous-shape destination (empty for all other variants).
    pub fn replacement_layout(&self) -> Layout {
        self.replacement
    }
    /// Runs the same ordered validator in exact caller-owned destinations.
    /// A wrong destination is rejected before either slice is written.
    pub fn validate_into<'d>(
        &self,
        initial: &'d mut [usize],
        replacement: &'d mut [usize],
    ) -> Result<&'d mut [usize], SelectionValidationError<'a>> {
        for (is_replacement, expected, actual) in [
            (false, self.shape.len(), initial.len()),
            (
                true,
                self.replacement.size() / std::mem::size_of::<usize>(),
                replacement.len(),
            ),
        ] {
            if expected != actual {
                return Err(SelectionValidationError {
                    key: self.key,
                    cause: Cause::Destination {
                        replacement: is_replacement,
                        expected,
                        actual,
                    },
                });
            }
        }
        validate(
            self.key,
            self.shape,
            self.selection,
            Borrowed {
                initial: Some(initial),
                replacement: Some(replacement),
            },
        )
    }
}

trait Storage<'a> {
    type Shape: AsRef<[usize]> + AsMut<[usize]>;
    type Error;
    fn initial(&mut self, shape: &[usize]) -> Self::Shape;
    fn replace(&mut self, output: &mut Self::Shape, shape: &Vec<usize>);
    fn error(&self, key: &'a str, cause: Cause) -> Self::Error;
}
struct Ordinary;
impl<'a> Storage<'a> for Ordinary {
    type Shape = Vec<usize>;
    type Error = StoreError;
    fn initial(&mut self, shape: &[usize]) -> Self::Shape {
        shape.to_vec()
    }
    fn replace(&mut self, output: &mut Self::Shape, shape: &Vec<usize>) {
        *output = shape.clone();
    }
    fn error(&self, key: &'a str, cause: Cause) -> Self::Error {
        SelectionValidationError { key, cause }.into()
    }
}
struct Borrowed<'d> {
    initial: Option<&'d mut [usize]>,
    replacement: Option<&'d mut [usize]>,
}
impl<'a, 'd> Storage<'a> for Borrowed<'d> {
    type Shape = &'d mut [usize];
    type Error = SelectionValidationError<'a>;
    fn initial(&mut self, shape: &[usize]) -> Self::Shape {
        let output = self.initial.take().expect("initial destination used once");
        output.copy_from_slice(shape);
        output
    }
    fn replace(&mut self, output: &mut Self::Shape, shape: &Vec<usize>) {
        let replacement = self
            .replacement
            .take()
            .expect("replacement destination used once");
        replacement.copy_from_slice(shape);
        *output = replacement;
    }
    fn error(&self, key: &'a str, cause: Cause) -> Self::Error {
        SelectionValidationError { key, cause }
    }
}
fn elements<'a, S: Storage<'a>>(
    key: &'a str,
    shape: &[usize],
    storage: &S,
) -> Result<usize, S::Error> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| storage.error(key, Cause::Elements))
    })
}
fn validate<'a, S: Storage<'a>>(
    key: &'a str,
    shape: &[usize],
    selection: &TensorSelection,
    mut storage: S,
) -> Result<S::Shape, S::Error> {
    elements(key, shape, &storage)?;
    let mut output = storage.initial(shape);
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| storage.error(key, Cause::Invalid("axis outside rank")))?;
            if start >= end || *end > *dimension {
                return Err(storage.error(key, Cause::Invalid("range outside dimension")));
            }
            output.as_mut()[*axis] = end - start;
        }
        TensorSelection::Indices { axis, indices } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| storage.error(key, Cause::Invalid("axis outside rank")))?;
            if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                return Err(storage.error(
                    key,
                    Cause::Invalid("indices are empty or outside dimension"),
                ));
            }
            output.as_mut()[*axis] = indices.len();
        }
        TensorSelection::Contiguous {
            offset_elements,
            shape: selected,
        } => {
            if selected.is_empty() || selected.contains(&0) {
                return Err(storage.error(key, Cause::Invalid("contiguous output shape is empty")));
            }
            let end = offset_elements
                .checked_add(elements(key, selected, &storage)?)
                .ok_or_else(|| storage.error(key, Cause::ContiguousEnd))?;
            if end > elements(key, shape, &storage)? {
                return Err(storage.error(key, Cause::Invalid("contiguous span outside tensor")));
            }
            storage.replace(&mut output, selected);
        }
    }
    elements(key, output.as_ref(), &storage)?;
    Ok(output)
}
pub(super) fn ordinary(
    key: &str,
    shape: &[usize],
    selection: &TensorSelection,
) -> Result<Vec<usize>, StoreError> {
    validate(key, shape, selection, Ordinary)
}

#[cfg(test)]
mod tests;
