//! Independent lexer state, including the complete mutable lexical DFA.
use super::PreparedLexer;
use crate::earley::regexvec::prepared::PreparedRegexVectorCopyFailure;
use derivre::raw::ParserAllocationFunding;
use std::{fmt, mem::{size_of, size_of_val}};
use toktrie::{SimpleVob, TokenMaskConstructionPlan, TokenMaskConstructionFailure, TokenMaskSourceError};
#[derive(Debug)]
enum Cause<E> {
    Source, Overflow, Funding(E), Vector(PreparedRegexVectorCopyFailure<E>),
    MaskSource(TokenMaskSourceError), Mask(TokenMaskConstructionFailure),
}
/// Failed lexer copies retain their actual mutable destination and new backing
/// account separately from the borrowed original source.
pub struct PreparedLexerCopyFailure<E> {
    cause: Cause<E>, prefix: Option<PreparedLexer>, backing: Option<ParserAllocationFunding>,
}
impl<E: fmt::Debug> fmt::Debug for PreparedLexerCopyFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedLexerCopyFailure").field("cause", &self.cause)
            .field("retains_prefix", &self.prefix.is_some()).finish()
    }
}
impl<E: fmt::Display> fmt::Display for PreparedLexerCopyFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source => f.write_str("lexer copy source is incomplete"),
            Cause::Overflow => f.write_str("lexer copy geometry overflow"),
            Cause::Funding(e) => fmt::Display::fmt(e,f), Cause::Vector(e) => fmt::Display::fmt(e,f),
            Cause::MaskSource(e) => fmt::Display::fmt(e,f), Cause::Mask(e) => fmt::Display::fmt(e,f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedLexerCopyFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause { Cause::Funding(e) => e, Cause::Vector(e) => e,
            Cause::MaskSource(e) => e, Cause::Mask(e) => e, _ => return None })
    }
}
impl PreparedLexer {
    fn copy_ready(&self) -> bool {
        !self.failed && self.selected.is_none() && self.allowed_first_byte.is_some()
    }
    /// Exact complete lexer/DFA copy payment from the retained source, without
    /// allocation, mutation or a funding callback.
    pub fn copy_required_bytes<E>(&self) -> Option<usize> {
        if !self.copy_ready() { return None; }
        Self::copy_controls::<E>()?
            .checked_add(self.vector.copy_required_bytes::<E>()?)?
            .checked_add(TokenMaskConstructionPlan::copy(self.allowed_first_byte.as_ref()?).ok()?
                .requirements().required_bytes())
    }
    fn copy_controls<E>() -> Option<usize> {
            let parts = [
                TokenMaskConstructionPlan::inspection_control_bytes()?,
                size_of::<Self>(), size_of::<&Self>(), size_of::<Option<Self>>(),
                size_of::<Cause<E>>(), size_of::<PreparedLexerCopyFailure<E>>(),
                size_of::<Option<ParserAllocationFunding>>(),
                size_of::<Result<Self, PreparedLexerCopyFailure<E>>>(),
                size_of::<Result<crate::earley::regexvec::prepared::PreparedRegexVector, PreparedRegexVectorCopyFailure<E>>>(),
                size_of::<Result<(), Cause<E>>>(), size_of::<Result<(), E>>(),
                size_of::<Option<SimpleVob>>(), size_of::<(&Self, &())>(),
            ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies this complete lexer, including its independently copied DFA and
    /// exact first-byte mask. Every destination is paid before construction;
    /// only the explicitly supplied new backing account permits future growth.
    pub fn try_copy<F: Fn(usize) -> Result<(), E>, E>(
        &self, backing: Option<ParserAllocationFunding>, funding: &F,
    ) -> Result<Self, PreparedLexerCopyFailure<E>> {
        let mut prefix = None;
        let result = (|| -> Result<(), Cause<E>> {
            funding(Self::copy_controls::<E>().ok_or(Cause::Overflow)?)
                .map_err(Cause::Funding)?;
            if !self.copy_ready() {
                return Err(Cause::Source);
            }
            let vector = self.vector.try_copy(backing.clone(), funding).map_err(Cause::Vector)?;
            prefix = Some(Self { vector, allowed_first_byte: None, selected: None, failed: true });
            let plan = TokenMaskConstructionPlan::copy(self.allowed_first_byte.as_ref().expect("checked lexer mask")).map_err(Cause::MaskSource)?;
            funding(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
            let target = prefix.as_mut().expect("lexer copy prefix");
            target.allowed_first_byte = Some(plan.compile().map_err(Cause::Mask)?);
            target.failed = self.failed;
            Ok(())
        })();
        match result { Ok(()) => Ok(prefix.expect("complete lexer copy")),
            Err(cause) => Err(PreparedLexerCopyFailure { cause, prefix, backing }) }
    }
}
