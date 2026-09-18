//! Independent token/session output around the actual paid chart copy.
use super::{Cause, PreparedEarleySeedError, PreparedTokenParser, PreparedTokenParserError, State};
use super::super::{copy, PreparedEarleySeed};
use std::{mem::{size_of, size_of_val}};
impl PreparedTokenParser {
    /// Exact complete committed-session copy payment from its chart and actual
    /// initialized output. The lexer copy remains a separate paired source.
    pub fn copy_required_bytes<E>(&self) -> Option<usize> {
        let mut bytes = Self::copy_controls::<E>()?
            .checked_add(self.chart.copy_required_bytes::<E>()?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.state.tokens)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.state.bytes)?)?;
        if let Some(mask) = &self.state.mask {
            bytes = bytes.checked_add(toktrie::TokenMaskConstructionPlan::copy(mask).ok()?.requirements().required_bytes())?;
        }
        Some(bytes)
    }
    fn copy_controls<E>() -> Option<usize> {
            let parts = [size_of::<Self>(), size_of::<&Self>(), size_of::<State>(),
                size_of::<PreparedTokenParserError<E>>(),
                size_of::<Result<Self, PreparedTokenParserError<E>>>(),
                size_of::<Result<(), Cause<E>>>(), size_of::<Result<(), E>>(),
                size_of::<(&Self, &())>(),
                toktrie::TokenMaskConstructionPlan::inspection_control_bytes()?,
            ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies the committed token/chart state into independent destinations.
    /// The enclosing owner must pair this with its independently copied lexer
    /// and retain new funding; no mutable lexer or growth authority is aliased.
    pub fn try_copy<F: crate::earley::PreparedFunding<Error = E>, E>(
        &self, funding: &F,
    ) -> Result<Self, PreparedTokenParserError<E>> {
        let mut state = State {
            tokens: Vec::new(), bytes: Vec::new(), mask: None,
            remaining: self.state.remaining, stop: self.state.stop,
            accepting: self.state.accepting, had_backtrack: self.state.had_backtrack,
            had_rollback: self.state.had_rollback,
        };
        let preparation = (|| -> Result<(), Cause<E>> {
            funding.reserve(Self::copy_controls::<E>().ok_or(Cause::Overflow)?)
                .map_err(Cause::Funding)
        })();

        if let Err(cause) = preparation {
            return Err(PreparedTokenParserError {
                cause: PreparedEarleySeedError {
                    cause, prefix: PreparedEarleySeed::vacant(self.chart.grammar.clone()),
                }, state,
            });
        }
        let chart = match self.chart.try_copy(funding) {
            Ok(chart) => chart,
            Err(cause) => return Err(PreparedTokenParserError { cause, state }),
        };
        let result = (|| -> Result<(), Cause<E>> {
            copy::fixed(&mut state.tokens, &self.state.tokens, funding)?;
            copy::fixed(&mut state.bytes, &self.state.bytes, funding)?;
            if let Some(source) = &self.state.mask { state.mask = Some(copy::mask(source, funding)?); }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Self { chart, state }),
            Err(cause) => Err(PreparedTokenParserError {
                cause: PreparedEarleySeedError { cause, prefix: chart }, state,
            }),
        }
    }
}
