//! Source-order capture spans shared by ordinary and paid definitive parsing.
use super::{Lexeme, RowInfo};
pub(super) fn visit<E>(
    rows: &[RowInfo],
    start: usize,
    current: usize,
    lexeme: &Lexeme,
    is_lexeme: bool,
    stop: bool,
    mut emit: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    if stop {
        return emit(lexeme.hidden_bytes());
    }
    if start < current {
        for row in &rows[start..current] {
            emit(row.lexeme.upper_visible_bytes(is_lexeme))?;
        }
    }
    if is_lexeme || start < current {
        emit(lexeme.upper_visible_bytes(is_lexeme))?;
    }
    Ok(())
}
