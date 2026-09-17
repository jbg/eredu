//! Shared monotonic attempt spending; rollback never calls this in reverse.
#[derive(Debug)]
pub(super) enum AttemptError {
    Exhausted,
    Overflow,
}
pub(super) fn consume<const N: usize>(
    spent: &mut [usize; N],
    total: &mut usize,
    index: usize,
    limit: usize,
) -> Result<(usize, usize), AttemptError> {
    if spent[index] >= limit {
        return Err(AttemptError::Exhausted);
    }
    let next = total.checked_add(1).ok_or(AttemptError::Overflow)?;
    let ordinals = (*total, spent[index]);
    *total = next;
    spent[index] += 1;
    Ok(ordinals)
}
