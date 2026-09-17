//! Paid scan destinations feeding the same agenda and row publication workers.
use super::super::{row, scan, GrammarStackPtr, Item, Lexeme, ParamValue, Scratch};
use super::{agenda, reserve, Cause, PreparedEarleySeed};
use crate::{
    api::SkipRepetition,
    earley::{PreparedLexer, PreparedLexerOperationError},
};
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};
use toktrie::{TokTrie, TokenMaskConstructionPlan};

fn append<F: Fn(usize) -> Result<(), E>, E>(
    scratch: &mut Scratch,
    item: Item,
    arg: ParamValue,
    funding: &F,
) -> Result<(), Cause<E>> {
    let count = scratch.row_end.checked_add(1).ok_or(Cause::Overflow)?;
    u32::try_from(count).map_err(|_| Cause::Overflow)?;
    reserve(&mut scratch.items, count, funding)?;
    if scratch.parametric {
        reserve(&mut scratch.item_args, count, funding)?;
    }
    scratch.put_item(item, arg);
    scratch.row_end = count;
    Ok(())
}
impl PreparedEarleySeed {
    pub(super) fn scan_lexeme<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        lexeme: &Lexeme,
        token: usize,
        bytes: usize,
        funding: &F,
    ) -> Result<bool, Cause<E>> {
        let controls = [
            size_of::<(&mut Self, &mut PreparedLexer, &TokTrie, &Lexeme, &F)>(),
            size_of::<Arc<super::super::CGrammar>>(),
            size_of::<Cause<E>>(),
            size_of::<PreparedLexerOperationError<E>>(),
            size_of::<super::super::Row>(),
            size_of::<super::super::RowInfo>(),
            size_of::<super::super::GrammarStackNode>(),
            size_of::<(usize, usize, usize, bool, Item, ParamValue)>(),
            size_of::<Option<SkipRepetition>>(),
            size_of::<Option<GrammarStackPtr>>(),
            size_of::<TokenMaskConstructionPlan<'_>>(),
            size_of::<Result<bool, Cause<E>>>(),
            size_of::<Result<(), Cause<E>>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Option<derivre::StateID>>(),
        ];
        funding(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(Cause::Overflow)?,
        )
        .map_err(Cause::Funding)?;
        if !self.row_complete || self.initial_selection.is_some() {
            return Err(Cause::Source);
        }
        let index = self.lexer_stack.last().ok_or(Cause::Source)?.row_idx as usize;
        let previous = self.rows.get(index).ok_or(Cause::Source)?;
        let source_items = previous.item_indices();
        let previous_state = previous.lexer_start_state;
        let previous_top = previous.grammar_stack_ptr;
        let next = index.checked_add(1).ok_or(Cause::Overflow)?;
        let grammar = Arc::clone(&self.grammar);
        let matches = crate::earley::lexer::matching(grammar.lexer_spec(), lexeme.idx, |state| {
            lexer.vector().state_desc(state)
        })
        .ok_or(Cause::Source)?;
        let skip = matches.as_slice().iter().find_map(|id| {
            let spec = grammar.lexer_spec().lexeme_spec(*id);
            spec.is_skip.then_some(spec.skip_repetition)
        });
        if skip.is_some() && source_items.is_empty() {
            return Ok(false);
        }
        let scratch = self.scratch.as_mut().ok_or(Cause::Source)?;
        scratch.new_row(source_items.end);
        scratch.push_lexeme_idx = lexeme.idx;
        scan::items(
            scratch,
            source_items,
            skip.is_none().then_some(matches),
            |scratch, item, arg| append(scratch, item, arg, funding),
        )?;
        let (mut class, limit) = scan::pop(scratch, matches, previous_top, token);
        let mut reuse = skip.map(|_| previous_state);
        if skip.is_none() {
            agenda::run(self, next, lexeme, trie, token, funding)?;
        }
        if let Some(pointer) = limit {
            let scratch = self.scratch.as_mut().expect("scan scratch");
            let node = &scratch.grammar_stack[pointer.as_usize()];
            let item = node.start_item.advance_dot();
            let origin = node.start_item_idx;
            scratch.push_grm_top = node.back_ptr;
            scratch.row_end = scratch.row_start;
            let arg = if scratch.parametric {
                scratch.item_args[origin]
            } else {
                ParamValue::default()
            };
            append(scratch, item, arg, funding)?;
            agenda::run(self, next, lexeme, trie, token, funding)?;
            if skip.is_some() {
                reuse = None;
                let scratch = self.scratch.as_ref().expect("scan scratch");
                class = scratch.grammar_stack[scratch.push_grm_top.as_usize()].grammar_id;
            }
        } else if matches!(skip, Some(SkipRepetition::Once)) {
            let possible = &lexer
                .vector()
                .state_desc(previous_state)
                .ok_or(Cause::Source)?
                .possible;
            let plan = possible.copy_plan().map_err(Cause::MaskSource)?;
            funding(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
            self.initial_selection = Some(super::super::LexemeSet::from_owned_vob(
                plan.compile().map_err(Cause::Mask)?,
            ));
            let selected = self.initial_selection.as_mut().expect("skip selection");
            selected.remove(grammar.lexer_spec().skip_id(class));
            reuse = Some(lexer.start_state(selected, funding).map_err(Cause::Lexer)?);
            self.initial_selection = None;
        }
        let scratch = self.scratch.as_mut().expect("scan scratch");
        self.stats.rows = self.stats.rows.checked_add(1).ok_or(Cause::Overflow)?;
        if scratch.row_len() == 0 {
            return Ok(false);
        }
        self.stats.all_items = self
            .stats
            .all_items
            .checked_add(scratch.row_len())
            .ok_or(Cause::Overflow)?;
        let state = if let Some(state) = reuse {
            state
        } else {
            let allow = skip.is_none()
                || limit.is_some()
                || matches!(skip, Some(SkipRepetition::Unbounded));
            row::add_skip(scratch, class, allow);
            lexer
                .start_state(&scratch.push_allowed_lexemes, funding)
                .map_err(Cause::Lexer)?
        };
        let count = next.checked_add(1).ok_or(Cause::Overflow)?;
        u32::try_from(count).map_err(|_| Cause::Overflow)?;
        reserve(&mut self.rows, count, funding)?;
        row::store(&mut self.rows, next, scratch.work_row(state));
        self.rows_valid_end = count;
        if scratch.definitive {
            reserve(&mut self.row_infos, count, funding)?;
            row::store_info(&mut self.row_infos, next, token, bytes);
        }
        Ok(true)
    }
}
