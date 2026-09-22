//! Same ordinary completion with exact admitted nested traversal in original work.
use safemlx::{
    error::Exception, Array, EvaluatedArray, OperationEvent, OriginalScopeObserver, Stream,
};

/// Complete the same quoted frontier and certify its descriptor before a host
/// reader or a later numerical role borrows the resulting backing.
pub(crate) fn complete_and_borrow<'a>(
    value: &'a Array,
    stream: &Stream,
) -> Result<EvaluatedArray<'a>, Exception> {
    if let Some(observer) = OriginalScopeObserver::try_current()? {
        complete_values([value], stream)?;
        value.completed_in_original_scope(&observer)
    } else {
        value.evaluated()
    }
}

pub(crate) fn completed_borrow_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        control_bytes(1)?,
        OperationEvent::nested_completion_control_bytes::<1>()?,
        OriginalScopeObserver::control_bytes()?,
        size_of::<(&Array, &Stream)>(),
        size_of::<EvaluatedArray<'_>>(),
        size_of::<Result<EvaluatedArray<'_>, Exception>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

pub(crate) fn complete_values<const N: usize>(
    values: [&Array; N],
    stream: &Stream,
) -> Result<(), Exception> {
    if let Some(observer) = OriginalScopeObserver::try_current()? {
        if !matches!(N, 1 | 2 | 4) {
            return Err(observer.capacity_error());
        }
        OperationEvent::validate_nested_completion(N)?;
        OperationEvent::complete_nested(values, stream)
    } else {
        safemlx::transforms::eval(values)
    }
}
pub(crate) fn control_bytes(roots: usize) -> Option<usize> {
    match roots {
        1 => fixed::<1>(),
        2 => fixed::<2>(),
        4 => fixed::<4>(),
        _ => None,
    }
}
fn fixed<const N: usize>() -> Option<usize> {
    use std::mem::size_of;
    let frames = [
        size_of::<[&Array; N]>(),
        size_of::<std::array::IntoIter<&Array, N>>(),
        size_of::<&Stream>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<Option<OriginalScopeObserver>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<usize>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
