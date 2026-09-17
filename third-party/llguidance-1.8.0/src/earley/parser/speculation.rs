//! Shared committed-history boundary for trie walks and parser queries.
use super::{CGrammar, CSymIdx, LexerState, MatchingLexemesIdx, Row, Scratch};
#[derive(Clone, Copy)]
pub(super) struct Snapshot {
    pub(super) lexer: usize,
    pub(super) grammar: usize,
}
pub(super) fn begin(scratch: &mut Scratch, stack: &[LexerState]) -> Snapshot {
    assert!(scratch.definitive);
    let saved = Snapshot {
        lexer: stack.len(),
        grammar: scratch.grammar_stack.len(),
    };
    scratch.definitive = false;
    saved
}
pub(super) fn finish(scratch: &mut Scratch, stack: &mut Vec<LexerState>, saved: Snapshot) {
    assert!(!scratch.definitive);
    assert!(scratch.grammar_stack.len() >= saved.grammar);
    assert!(stack.len() >= saved.lexer);
    scratch.grammar_stack.truncate(saved.grammar);
    stack.truncate(saved.lexer);
    scratch.definitive = true;
    scratch.log_override = false;
}
pub(super) fn cached(
    definitive: bool,
    next: usize,
    valid: usize,
    rows: &[Row],
    index: MatchingLexemesIdx,
) -> bool {
    !definitive && next < valid && rows[next].lexeme_idx == index
}
pub(super) fn pending(stack: &[LexerState]) -> bool {
    let row = stack.last().expect("lexical history").row_idx;
    stack
        .iter()
        .rev()
        .take_while(|state| state.row_idx == row)
        .any(|state| state.byte.is_some())
}
pub(super) fn accepting(grammar: &CGrammar, scratch: &Scratch, row: &Row) -> bool {
    row.item_indices().any(|index| {
        let position = scratch.items[index].rhs_ptr();
        grammar.sym_idx_dot(position) == CSymIdx::NULL
            && grammar.sym_idx_lhs(position) == grammar.start()
    })
}

pub(super) fn can_advance(grammar: &CGrammar, scratch: &Scratch, row: &Row) -> bool {
    row.item_indices().any(|index| {
        let data = grammar.sym_data_dot(scratch.items[index].rhs_ptr());
        data.idx != CSymIdx::NULL && (data.is_terminal || data.gen_grammar.is_some())
    })
}
