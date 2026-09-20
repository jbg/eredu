//! Output views preserve the materialization's original completion authority.
use safemlx::{Array, EvaluatedArray, OriginalScopeObserver};

pub(super) fn completed_outputs<'a>(
    outputs: &'a [Array],
    observer: Option<&'a OriginalScopeObserver>,
) -> impl ExactSizeIterator<Item = safemlx::error::Result<EvaluatedArray<'a>>> + 'a {
    outputs.iter().map(move |output| match observer {
        Some(observer) => output.completed_in_original_scope(observer),
        None => output.evaluated(),
    })
}

#[cfg(test)]
#[path = "readback/tests.rs"]
mod tests;
