//! Allocation-free byte-chart rollback using actual shared history storage.
use super::{LexerState, RowInfo};
use std::{fmt, mem::{size_of, size_of_val}};
#[derive(Debug)]
pub(crate) enum Failure {
    Bytes { requested: usize, available: usize },
    Source,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes { requested, available } => write!(f, "rollback: too many bytes {requested} > {available}"),
            Self::Source => f.write_str("rollback history source differs"),
        }
    }
}
impl std::error::Error for Failure {}
pub(super) struct Fields<'a> {
    pub(super) byte_to_token_idx: &'a mut Vec<u32>,
    pub(super) bytes: &'a mut Vec<u8>,
    pub(super) lexer_stack: &'a mut Vec<LexerState>,
    pub(super) row_infos: &'a mut Vec<RowInfo>,
    pub(super) token_idx: &'a mut usize,
    pub(super) last_force_bytes_len: &'a mut usize,
    pub(super) lexer_stack_top_eos: &'a mut bool,
    pub(super) rows_valid_end: &'a mut usize,
}
pub(super) fn controls() -> Option<usize> {
    let parts = [size_of::<Fields<'_>>(), size_of::<Failure>(),
        size_of::<Result<(), Failure>>(), size_of::<Option<&LexerState>>(),
        size_of::<(usize, usize, usize)>(), size_of::<Option<usize>>()];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn run(fields: Fields<'_>, removed: usize) -> Result<(), Failure> {
    let old_len = fields.byte_to_token_idx.len();
    let new_len = old_len.checked_sub(removed).ok_or(Failure::Bytes {
        requested: removed, available: old_len,
    })?;
    let stack_len = new_len.checked_add(1).ok_or(Failure::Source)?;
    let rows = fields.lexer_stack.get(new_len).ok_or(Failure::Source)?.row_idx as usize + 1;
    if new_len > fields.bytes.len() || rows > fields.row_infos.len() {
        return Err(Failure::Source);
    }
    fields.byte_to_token_idx.truncate(new_len);
    fields.bytes.truncate(new_len);
    fields.lexer_stack.truncate(stack_len);
    fields.row_infos.truncate(rows);
    *fields.token_idx = *fields.byte_to_token_idx.last().unwrap_or(&0) as usize;
    *fields.last_force_bytes_len = usize::MAX;
    *fields.lexer_stack_top_eos = false;
    *fields.rows_valid_end = rows;
    Ok(())
}
