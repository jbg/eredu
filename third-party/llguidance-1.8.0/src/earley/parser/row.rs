//! Shared skip selection and publication into actual parser row destinations.
use super::{Lexeme, LexemeClass, Row, RowInfo, Scratch};

pub(super) fn add_skip(scratch: &mut Scratch, grammar: LexemeClass, allow: bool) {
    let mut selected = None;
    if allow {
        if scratch.push_allowed_grammar_ids.get(grammar.as_usize()) {
            selected = Some(grammar);
        } else {
            let mut pointer = scratch.push_grm_top;
            while pointer.as_usize() > 0 {
                pointer = scratch.grammar_stack[pointer.as_usize()].back_ptr;
                let parent = scratch.grammar_stack[pointer.as_usize()].grammar_id;
                if scratch.push_allowed_grammar_ids.get(parent.as_usize()) {
                    selected = Some(parent);
                    break;
                }
            }
        }
    }
    if let Some(class) = selected {
        scratch
            .push_allowed_lexemes
            .add(scratch.grammar.lexer_spec().skip_id(class));
    }
}

pub(super) fn store(rows: &mut Vec<Row>, index: usize, row: Row) {
    if rows.is_empty() || rows.len() == index {
        rows.push(row);
    } else {
        rows[index] = row;
    }
}

pub(super) fn store_info(rows: &mut Vec<RowInfo>, index: usize, token: usize, bytes: usize) {
    if rows.len() > index {
        rows.drain(index..);
    }
    let start_byte_idx = if bytes > 0 { bytes + 1 } else { bytes };
    rows.push(RowInfo {
        lexeme: Lexeme::bogus(),
        token_idx_start: token,
        token_idx_stop: token,
        start_byte_idx,
    });
}
