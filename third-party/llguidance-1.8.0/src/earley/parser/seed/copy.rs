//! Exact independent copies of the actual committed chart storage.
use super::{agenda::InitialCapture, reserve, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use super::super::{GrammarStackNode, Item, Lexeme, LexemeSet, LexerState, ParamValue, Row, RowInfo, Scratch};
use std::{mem::{size_of, size_of_val}, sync::Arc};
use toktrie::{SimpleVob, TokenMaskConstructionPlan};
// T: Copy closes this helper over fixed data; nested heap owners require their
// own paid constructor and failure prefix below.
fn fixed_controls<T: Copy, E>() -> Option<usize> {
    let parts = [size_of::<(&mut Vec<T>, &[T], &())>(), size_of::<T>(),
        size_of::<Result<(), Cause<E>>>(), size_of::<std::slice::Iter<'_, T>>()];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn fixed_required_bytes<T: Copy, E>(source: &[T]) -> Option<usize> {
    fixed_controls::<T, E>()?.checked_add(super::allocation_bytes::<T>(source.len())?)
}
pub(super) fn fixed<T: Copy, F: Fn(usize) -> Result<(), E>, E>(
    target: &mut Vec<T>, source: &[T], funding: &F,
) -> Result<(), Cause<E>> {
    funding(fixed_controls::<T, E>().ok_or(Cause::Overflow)?)
        .map_err(Cause::Funding)?;
    reserve(target, source.len(), funding)?;
    target.extend_from_slice(source);
    Ok(())
}
pub(super) fn mask<F: Fn(usize) -> Result<(), E>, E>(
    source: &SimpleVob, funding: &F,
) -> Result<SimpleVob, Cause<E>> {
    let plan = TokenMaskConstructionPlan::copy(source).map_err(Cause::MaskSource)?;
    funding(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
    plan.compile().map_err(Cause::Mask)
}
impl PreparedEarleySeed {
    fn copy_source(&self) -> Option<&Scratch> {
        let source = self.scratch.as_ref()?;
        if !self.row_complete || !source.definitive || self.backtrack_bytes != 0
            || self.lexemes.is_some() || self.grammars.is_some()
            || self.initial_selection.is_some() || self.capture_pending.is_some() {
            return None;
        }
        Some(source)
    }
    /// Exact prospective copy payment from initialized chart rows, capture
    /// bytes, token mask and scratch. No destination is allocated or mutated.
    pub fn copy_required_bytes<E>(&self) -> Option<usize> {
        let source = self.copy_source()?;
        let mut bytes = Self::copy_controls::<E>()?
            .checked_add(source.push_allowed_lexemes.copy_plan().ok()?.requirements().required_bytes())?
            .checked_add(TokenMaskConstructionPlan::copy(&source.push_allowed_grammar_ids).ok()?.requirements().required_bytes())?
            .checked_add(fixed_required_bytes::<_, E>(&source.items)?)?
            .checked_add(fixed_required_bytes::<_, E>(&source.item_args)?)?
            .checked_add(fixed_required_bytes::<_, E>(&source.grammar_stack)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.rows)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.lexer_stack)?)?
            .checked_add(super::allocation_bytes::<RowInfo>(self.row_infos.len())?)?;
        for row in &self.row_infos {
            u32::try_from(row.lexeme.num_hidden_bytes()).ok()?;
            bytes = bytes.checked_add(fixed_required_bytes::<_, E>(row.lexeme.all_bytes())?)?;
        }
        bytes = bytes.checked_add(fixed_required_bytes::<_, E>(&self.bytes)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.byte_to_token_idx)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.lexeme_bytes)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.capture_raw)?)?
            .checked_add(super::allocation_bytes::<InitialCapture>(self.captures.len())?)?;
        for capture in &self.captures {
            bytes = bytes.checked_add(fixed_required_bytes::<_, E>(&capture.bytes)?)?;
        }
        if let Some(mask) = &self.token_mask {
            bytes = bytes.checked_add(TokenMaskConstructionPlan::copy(mask).ok()?.requirements().required_bytes())?;
        }
        Some(bytes)
    }
    fn copy_controls<E>() -> Option<usize> {
            let parts = [
                Self::controls::<&(), E>()?,
                size_of::<Self>(), size_of::<&Self>(), size_of::<Scratch>(),
                size_of::<RowInfo>(), size_of::<InitialCapture>(),
                size_of::<(Item, ParamValue, GrammarStackNode, LexerState, Row)>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<Result<Self, PreparedEarleySeedError<E>>>(),
                size_of::<std::slice::Iter<'_, RowInfo>>(),
                size_of::<std::slice::Iter<'_, InitialCapture>>(),
                size_of::<Option<&Scratch>>(), size_of::<Option<&SimpleVob>>(),
                size_of::<(usize, u32, bool)>(),
            ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies a completed chart into independent paid storage, preserving every
    /// initialized history, capture, mask and current-row scalar. Immutable
    /// grammar ownership is shared; no lexer or source growth authority is copied.
    /// The enclosing owner retains the new funding through success or failure.
    pub fn try_copy<F: Fn(usize) -> Result<(), E>, E>(
        &self, funding: &F,
    ) -> Result<Self, PreparedEarleySeedError<E>> {
        let mut copied = Self::vacant(Arc::clone(&self.grammar));
        let result = (|| -> Result<(), Cause<E>> {
            funding(Self::copy_controls::<E>().ok_or(Cause::Overflow)?)
                .map_err(Cause::Funding)?;
            let source = self.copy_source().ok_or(Cause::Source)?;
            let lexemes = source.push_allowed_lexemes.copy_plan().map_err(Cause::MaskSource)?;
            funding(lexemes.requirements().required_bytes()).map_err(Cause::Funding)?;
            copied.lexemes = Some(LexemeSet::from_owned_vob(lexemes.compile().map_err(Cause::Mask)?));
            copied.grammars = Some(mask(&source.push_allowed_grammar_ids, funding)?);
            copied.scratch = Some(Scratch::from_masks(
                Arc::clone(&copied.grammar), copied.lexemes.take().expect("copied lexemes"),
                copied.grammars.take().expect("copied grammars"),
            ));
            let scratch = copied.scratch.as_mut().expect("copied scratch");
            fixed(&mut scratch.items, &source.items, funding)?;
            fixed(&mut scratch.item_args, &source.item_args, funding)?;
            fixed(&mut scratch.grammar_stack, &source.grammar_stack, funding)?;
            scratch.row_start = source.row_start;
            scratch.row_end = source.row_end;
            scratch.push_grm_top = source.push_grm_top;
            scratch.push_lexeme_idx = source.push_lexeme_idx;
            scratch.definitive = source.definitive;
            scratch.log_override = source.log_override;
            scratch.parametric = source.parametric;
            fixed(&mut copied.rows, &self.rows, funding)?;
            fixed(&mut copied.lexer_stack, &self.lexer_stack, funding)?;
            reserve(&mut copied.row_infos, self.row_infos.len(), funding)?;
            for row in &self.row_infos {
                // Keep the current nested byte destination in the copied owner
                // until the complete Lexeme is moved into its pre-paid row.
                fixed(&mut copied.lexeme_bytes, row.lexeme.all_bytes(), funding)?;
                let lexeme = Lexeme::new(row.lexeme.idx, std::mem::take(&mut copied.lexeme_bytes),
                    u32::try_from(row.lexeme.num_hidden_bytes()).map_err(|_| Cause::Overflow)?,
                    row.lexeme.is_suffix());
                copied.row_infos.push(RowInfo {
                    start_byte_idx: row.start_byte_idx, lexeme,
                    token_idx_start: row.token_idx_start, token_idx_stop: row.token_idx_stop,
                });
            }
            fixed(&mut copied.bytes, &self.bytes, funding)?;
            fixed(&mut copied.byte_to_token_idx, &self.byte_to_token_idx, funding)?;
            fixed(&mut copied.lexeme_bytes, &self.lexeme_bytes, funding)?;
            fixed(&mut copied.capture_raw, &self.capture_raw, funding)?;
            reserve(&mut copied.captures, self.captures.len(), funding)?;
            for capture in &self.captures {
                copied.capture_pending = Some(Vec::new());
                fixed(copied.capture_pending.as_mut().expect("capture destination"), &capture.bytes, funding)?;
                copied.captures.push(InitialCapture {
                    symbol: capture.symbol, stop: capture.stop,
                    bytes: copied.capture_pending.take().expect("copied capture"),
                });
            }
            if let Some(source) = &self.token_mask { copied.token_mask = Some(mask(source, funding)?); }
            copied.row_complete = self.row_complete;
            copied.rows_valid_end = self.rows_valid_end;
            copied.mask_key = self.mask_key;
            copied.lexer_stack_flush_position = self.lexer_stack_flush_position;
            copied.lexer_stack_top_eos = self.lexer_stack_top_eos;
            copied.max_items_in_row = self.max_items_in_row;
            copied.backtrack_bytes = self.backtrack_bytes;
            copied.token_idx = self.token_idx;
            copied.max_all_items = self.max_all_items;
            copied.last_force_bytes_len = self.last_force_bytes_len;
            copied.stats = self.stats.clone();
            copied.agenda_closed = self.agenda_closed;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(copied),
            Err(cause) => Err(PreparedEarleySeedError { cause, prefix: copied }),
        }
    }
}
