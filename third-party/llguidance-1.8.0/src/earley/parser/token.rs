//! Shared numeric, EOS and token-history progression over the actual byte worker.
use super::{
    advance, GrammarStackPtr, LexemeIdx, LexemeSet, LexerResult, LexerState, MatchingLexemesIdx,
    PreLexeme, Scratch,
};
use crate::earley::lexerspec::LexerSpec;
use toktrie::{parse_numeric_token, TokTrie, TokenId};

pub(super) trait Context: advance::Context {
    fn rows(&self) -> usize;
    fn row_start(&self, row: usize) -> usize;
    fn apply_row(&mut self, row: usize, reset: bool);
    fn bytes(&self) -> &[u8];
    fn applied(&self) -> usize;
    fn append_applied(&mut self, count: usize) -> Result<(), Self::Error>;
    fn truncate_applied(&mut self, count: usize);
    fn append_numeric(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
    fn trie(&self) -> &TokTrie;
    fn numeric_possible(&mut self, token: TokenId) -> Result<bool, Self::Error>;
    fn numeric_index(&self, token: TokenId) -> Option<LexemeIdx>;
    fn pending(&self) -> bool;
    fn try_end(&mut self) -> Result<LexerResult, Self::Error>;
    fn stack_len(&self) -> usize;
    fn set_flush_position(&mut self, position: usize);
    fn flush_position(&self) -> usize;
    fn remove_state(&mut self, position: usize);
    fn allows_eos(&self) -> bool;
    fn set_top_eos(&mut self);
    fn limit_tokens(&mut self) -> Result<bool, Self::Error>;
    fn check_row_items(&self) -> Result<(), Self::Error>;
    fn rejected_byte(&self, token: &[u8], byte: u8, forced: Option<u8>) -> Self::Error;
    fn rejected_numeric(&self) -> Self::Error;
}

pub(super) fn flush<C: Context>(context: &mut C) -> Result<bool, C::Error> {
    if !context.pending() {
        return Ok(true);
    }
    let current = context.current();
    let result = context.try_end()?;
    let previous = context.stack_len();
    let accepted = advance::route(context, result, current)?;
    if context.stack_len() != previous {
        assert_eq!(context.stack_len(), previous + 1);
        assert!(previous > 0);
        context.set_flush_position(previous);
    }
    assert_eq!(context.backtrack_count(), 0);
    Ok(accepted)
}
pub(super) fn numeric<C: Context>(
    context: &mut C,
    token: TokenId,
) -> Result<Option<LexemeIdx>, C::Error> {
    Ok(if flush(context)? {
        context.numeric_index(token)
    } else {
        None
    })
}
pub(super) fn add_numeric<C: Context>(
    context: &mut C,
    index: LexemeIdx,
    bytes: &[u8],
) -> Result<(), C::Error> {
    if bytes.is_empty() {
        return Err(context.rejected_numeric());
    }
    let state = context.current();
    for &byte in &bytes[..bytes.len() - 1] {
        context.push_state(LexerState {
            byte: Some(byte),
            ..state
        })?;
    }
    if context.definitive() {
        context.append_numeric(bytes)?;
    }
    if !advance::run(
        context,
        PreLexeme::just_idx(MatchingLexemesIdx::Single(index)),
    )? {
        return Err(context.rejected_numeric());
    }
    if context.definitive() {
        context.apply_row(context.rows() - 1, false);
    }
    Ok(())
}
pub(super) fn eos<C: Context>(context: &mut C) -> Result<bool, C::Error> {
    assert!(context.definitive());
    let accepted = context.allows_eos();
    let before = context.stack_len();
    if !flush(context)? {
        assert_eq!(context.stack_len(), before);
        return Ok(false);
    }
    if accepted {
        return Ok(true);
    }
    if context.stack_len() != before {
        assert_eq!(context.stack_len(), before + 1);
        context.set_top_eos();
    }
    context.assert_finished();
    Ok(false)
}
pub(super) fn apply<C: Context>(
    context: &mut C,
    bytes: &[u8],
    token: TokenId,
) -> Result<usize, C::Error> {
    assert!(context.definitive());
    let mut check_max = false;
    let mut first_row = context.rows() - 1;
    let first_byte = context.applied();
    while first_row > 0 && context.row_start(first_row) > first_byte {
        first_row -= 1;
    }
    if context.trie().token(token) == bytes
        && context.applied() == context.bytes().len()
        && context.numeric_possible(token)?
    {
        context.apply_row(context.rows() - 1, false);
        context.set_flush_position(0);
        let index = numeric(context, token)?.expect("same numeric source after speculative query");
        add_numeric(context, index, bytes)?;
        let position = context.flush_position();
        if position > 0 {
            assert!(position + 1 < context.stack_len());
            context.remove_state(position);
        }
        context.assert_finished();
        return Ok(0);
    }
    for (index, &byte) in bytes.iter().enumerate() {
        check_max = false;
        let applied = context.applied();
        if applied >= context.bytes().len() {
            assert_eq!(applied, context.bytes().len());
            let row = context.rows() - 1;
            context.apply_row(row, false);
            let (accepted, backtrack) = advance::definitive(context, Some(byte))?;
            if !accepted {
                return Err(context.rejected_byte(bytes, byte, None));
            }
            if backtrack > 0 {
                context.truncate_applied(context.bytes().len());
                return Ok(backtrack + (bytes.len() - index - 1));
            }
            check_max = row == context.rows() - 1;
        } else {
            if index == 0 && context.bytes()[applied] == TokTrie::SPECIAL_TOKEN_MARKER {
                if let Some(id) = context.trie().token_id_at_bytes(bytes) {
                    if let Some((length, actual)) =
                        parse_numeric_token(&context.bytes()[applied + 1..])
                    {
                        if id == actual {
                            context.append_applied(length + 1)?;
                            break;
                        }
                    }
                }
            }
            let expected = context.bytes()[applied];
            if expected != byte {
                return Err(context.rejected_byte(bytes, byte, Some(expected)));
            }
        }
        context.append_applied(1)?;
    }
    for row in first_row..context.rows() {
        context.apply_row(row, context.row_start(row) >= first_byte);
    }
    if check_max && !context.limit_tokens()? {
        return Ok(0);
    }
    context.check_row_items()?;
    context.assert_finished();
    Ok(0)
}

// Exact max-token class ancestry and lexical filter, without a temporary class
// HashSet. Repeated source classes have the same membership semantics.
pub(super) fn limit(
    spec: &LexerSpec,
    scratch: &Scratch,
    top: GrammarStackPtr,
    possible: &LexemeSet,
    token: usize,
    first_token: usize,
    selected: &mut LexemeSet,
) -> usize {
    let tokens = std::cmp::max(0, token as isize + 1 - first_token as isize) as usize;
    let mut removed = 0;
    for index in possible.iter() {
        let lexeme = spec.lexeme_spec(index);
        let mut pointer = top;
        let mut class_ok = true;
        while pointer.as_usize() > 0 {
            let node = &scratch.grammar_stack[pointer.as_usize()];
            if node.token_horizon > token as u32 + 1 {
                break;
            }
            if node.grammar_id == lexeme.class() {
                class_ok = false;
                break;
            }
            pointer = node.back_ptr;
        }
        if tokens < lexeme.max_tokens() && class_ok {
            selected.add(index);
        } else {
            removed += 1;
        }
    }
    removed
}

impl Context for super::ParserState {
    fn rows(&self) -> usize {
        self.num_rows()
    }
    fn row_start(&self, row: usize) -> usize {
        self.row_infos[row].start_byte_idx
    }
    fn apply_row(&mut self, row: usize, reset: bool) {
        if reset {
            self.row_infos[row].set_token_idx(self.token_idx);
        } else {
            self.row_infos[row].apply_token_idx(self.token_idx);
        }
    }
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn applied(&self) -> usize {
        self.byte_to_token_idx.len()
    }
    fn append_applied(&mut self, count: usize) -> Result<(), Self::Error> {
        for _ in 0..count {
            self.byte_to_token_idx
                .push(self.token_idx.try_into().unwrap());
        }
        Ok(())
    }
    fn truncate_applied(&mut self, count: usize) {
        self.byte_to_token_idx.truncate(count);
    }
    fn append_numeric(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.bytes.extend_from_slice(bytes);
        self.append_applied(bytes.len())
    }
    fn trie(&self) -> &TokTrie {
        self.tok_env.tok_trie()
    }
    fn numeric_possible(&mut self, token: TokenId) -> Result<bool, Self::Error> {
        self.run_speculative("numeric_apply_token", |state| numeric(state, token))
            .map(|index| index.is_some())
    }
    fn numeric_index(&self, token: TokenId) -> Option<LexemeIdx> {
        self.token_range_lexemes()
            .find(|spec| spec.contains_token(token))
            .map(|spec| spec.idx)
    }
    fn pending(&self) -> bool {
        self.has_pending_lexeme_bytes()
    }
    fn try_end(&mut self) -> Result<LexerResult, Self::Error> {
        let state = self.lexer_state().lexer_state;
        Ok(self.lexer_mut().try_lexeme_end(state))
    }
    fn stack_len(&self) -> usize {
        self.lexer_stack.len()
    }
    fn set_flush_position(&mut self, position: usize) {
        self.lexer_stack_flush_position = position;
    }
    fn flush_position(&self) -> usize {
        self.lexer_stack_flush_position
    }
    fn remove_state(&mut self, position: usize) {
        self.lexer_stack.remove(position);
    }
    fn allows_eos(&self) -> bool {
        self.pending()
            && crate::earley::lexer::allows_eos(
                self.lexer_spec(),
                self.lexer().dfa.state_desc(self.lexer_state().lexer_state),
            )
    }
    fn set_top_eos(&mut self) {
        self.lexer_stack_top_eos = true;
    }
    fn limit_tokens(&mut self) -> Result<bool, Self::Error> {
        let row = self.num_rows() - 1;
        let state = self.lexer_state().lexer_state;
        let mut selected = self.lexer_spec().alloc_lexeme_set();
        let removed = limit(
            self.lexer_spec(),
            &self.scratch,
            self.rows[row].grammar_stack_ptr,
            self.lexer().possible_lexemes(state),
            self.token_idx,
            self.row_infos[row].token_idx_start,
            &mut selected,
        );
        if removed > 0 {
            let next = self.lexer_mut().limit_state_to(state, &selected);
            if next.is_dead() {
                let (accepted, backtrack) = advance::definitive(self, None)?;
                assert_eq!(backtrack, 0);
                if !accepted {
                    return Ok(false);
                }
            } else {
                self.lexer_stack.last_mut().unwrap().lexer_state = next;
            }
        }
        Ok(true)
    }
    fn check_row_items(&self) -> Result<(), Self::Error> {
        let count = self.curr_row().item_indices().count();
        anyhow::ensure!(count <= self.limits.max_items_in_row,
            "Current row has {} items; max is {}; consider making your grammar left-recursive if it's right-recursive", count, self.limits.max_items_in_row);
        Ok(())
    }
    fn rejected_byte(&self, token: &[u8], byte: u8, forced: Option<u8>) -> Self::Error {
        match forced {
            Some(expected) => anyhow::anyhow!(
                "token {:?} doesn't satisfy the grammar; forced bytes: got {:?}; applying {:?}",
                String::from_utf8_lossy(token),
                expected as char,
                byte as char
            ),
            None => anyhow::anyhow!(
                "token {:?} doesn't satisfy the grammar; byte {:?} fails parse",
                String::from_utf8_lossy(token),
                byte as char
            ),
        }
    }
    fn rejected_numeric(&self) -> Self::Error {
        anyhow::anyhow!("failed to advance parser after adding bytes ignoring lexer")
    }
}
