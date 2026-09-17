//! Shared scan selection/copy and grammar-stack return over the actual chart.
use super::{GrammarStackPtr, Item, LexemeClass, ParamValue, Scratch};
use crate::earley::regexvec::MatchingLexemes;
use std::ops::Range;

pub(super) fn items<E>(
    scratch: &mut Scratch,
    source: Range<usize>,
    matches: Option<&MatchingLexemes>,
    mut push: impl FnMut(&mut Scratch, Item, ParamValue) -> Result<(), E>,
) -> Result<(), E> {
    for index in source {
        let previous = scratch.items[index];
        let item = if let Some(matches) = matches {
            let symbol = scratch.grammar.sym_data_dot(previous.rhs_ptr());
            if !symbol.lexeme.is_some_and(|id| matches.contains(id)) {
                continue;
            }
            previous.advance_dot()
        } else {
            previous
        };
        let arg = if scratch.parametric {
            scratch.item_args[index]
        } else {
            ParamValue::default()
        };
        push(scratch, item, arg)?;
    }
    Ok(())
}

pub(super) fn pop(
    scratch: &mut Scratch,
    matches: &MatchingLexemes,
    mut top: GrammarStackPtr,
    token: usize,
) -> (LexemeClass, Option<GrammarStackPtr>) {
    let contains = |class| {
        matches
            .as_slice()
            .iter()
            .any(|id| scratch.grammar.lexer_spec().lexeme_spec(*id).class() == class)
    };
    let mut limit = None;
    while top.as_usize() > 0 {
        let node = &scratch.grammar_stack[top.as_usize()];
        if contains(node.grammar_id) {
            if node.token_horizon <= token as u32 {
                limit = Some(top);
            }
            break;
        }
        top = node.back_ptr;
    }
    if top.as_usize() == 0 {
        assert!(
            contains(LexemeClass::ROOT),
            "grammar stack empty for non-root grammar"
        );
    }
    scratch.push_grm_top = top;
    (scratch.grammar_stack[top.as_usize()].grammar_id, limit)
}

pub(super) fn forced(spec: &super::LexerSpec, allowed: &super::LexemeSet, bytes: &[u8]) -> bool {
    if allowed.is_empty() {
        return false;
    }
    let mut matched = false;
    for index in allowed.iter() {
        let lexeme = &spec.lexemes[index.as_usize()];
        if lexeme.is_skip && matches!(lexeme.rx, super::RegexAst::NoMatch) {
            continue;
        }
        if !spec.has_forced_bytes(lexeme, bytes) {
            return false;
        }
        matched = true;
    }
    matched
}

pub(super) fn row_bytes(stack: &[super::LexerState], row: u32) -> impl Iterator<Item = u8> + '_ {
    stack
        .iter()
        .rev()
        .take_while(move |state| state.row_idx == row)
        .filter_map(|state| state.byte)
}

pub(super) fn special(
    spec: &super::LexerSpec,
    possible: &super::LexemeSet,
    bytes: &[u8],
) -> Option<super::LexemeIdx> {
    let token = std::str::from_utf8(bytes.get(2..)?)
        .ok()?
        .parse::<u32>()
        .ok()?;
    possible.iter().find(|id| {
        spec.lexeme_spec(*id)
            .token_ranges
            .iter()
            .any(|range| range.contains(&token))
    })
}
