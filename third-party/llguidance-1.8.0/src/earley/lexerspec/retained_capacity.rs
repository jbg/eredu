//! Exact live declaration capacity, distinct from a compact destination quote.
use super::*;
use super::source_copy::{Cause, frame};
use derivre::{ParserAllocationFailure, prepared_funding::PreparedFunding};
use std::mem::size_of;

impl LexerSpec {
    pub(crate) fn retained_capacity_bytes<F: PreparedFunding<Error = ParserAllocationFailure>>(
        &self, funding: &F,
    ) -> Result<usize, LexerSpecCopyFailure> {
        let inspect = || -> Result<usize, Cause> {
            let _frame = funding.frame(size_of::<(
                &Self, &F, usize, Option<usize>, Result<usize, Cause>,
                std::slice::Iter<'_, LexemeSpec>, std::slice::Iter<'_, (String, usize)>,
            )>()).map_err(frame)?;
            let builder = {
                let _builder_frame = funding.frame(
                    RegexBuilder::source_inspection_control_bytes().ok_or(Cause::Overflow)?
                ).map_err(frame)?;
                self.regex_builder.retained_capacity_bytes().ok_or(Cause::Overflow)?
            };
            let mut bytes = (|| self.lexemes.capacity().checked_mul(size_of::<LexemeSpec>())?
                .checked_add(self.skip_by_class.capacity().checked_mul(size_of::<LexemeIdx>())?)?
                .checked_add(self.grammar_warnings.capacity().checked_mul(size_of::<(String, usize)>())?)?
                .checked_add(self.class_by_skip.allocation_size())?
                .checked_add(builder))().ok_or(Cause::Overflow)?;
            for (message, _) in &self.grammar_warnings {
                bytes = bytes.checked_add(message.capacity()).ok_or(Cause::Overflow)?;
            }
            for lexeme in &self.lexemes {
                let ast = lexeme.rx.retained_capacity_bytes(funding).map_err(Cause::Ast)?;
                bytes = (|| bytes.checked_add(lexeme.name.capacity())?
                    .checked_add(ast)?
                    .checked_add(lexeme.token_ranges.capacity().checked_mul(size_of::<RangeInclusive<TokenId>>())?))()
                    .ok_or(Cause::Overflow)?;
                if let MatchingLexemes::Many(values) = &lexeme.single_set {
                    bytes = values.capacity().checked_mul(size_of::<LexemeIdx>())
                        .and_then(|n| bytes.checked_add(n)).ok_or(Cause::Overflow)?;
                }
                if let Some(options) = &lexeme.json_options {
                    bytes = bytes.checked_add(options.allowed_escapes.capacity()).ok_or(Cause::Overflow)?;
                }
            }
            Ok(bytes)
        };
        inspect().map_err(LexerSpecCopyFailure::inspection)
    }
}
