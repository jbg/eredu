//! One ordinary/prepared grammar commitment and terminal-EOS policy.
use std::mem::{size_of, size_of_val};
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub(super) enum Failure {
    #[error("generation grammar already committed its terminal EOS token")]
    Terminal,
    #[error("token {0} is not allowed by the generation grammar")]
    Token(u32),
}
pub(super) trait Context {
    type Error;
    fn alias_committed(&self) -> bool;
    fn mark_alias(&mut self);
    fn try_token(&mut self, token: u32) -> Result<bool, Self::Error>;
    fn is_eos(&self, token: u32) -> Result<bool, Self::Error>;
    fn mask_allows(&mut self, token: u32) -> Result<bool, Self::Error>;
    fn accepting(&mut self) -> Result<bool, Self::Error>;
    fn stopped(&self) -> bool;
    fn refusal(&self, cause: Failure) -> Self::Error;
}
pub(super) fn commit<C: Context>(context: &mut C, token: u32) -> Result<(), C::Error> {
    if context.alias_committed() {
        return Err(context.refusal(Failure::Terminal));
    }
    if context.try_token(token)? {
        return Ok(());
    }
    // Secondary EOS aliases can be structural terminators. Accept one only
    // when this exact token is present in the actual current grammar mask.
    if context.is_eos(token)? && context.mask_allows(token)? {
        context.mark_alias();
        return Ok(());
    }
    Err(context.refusal(Failure::Token(token)))
}
pub(super) fn complete<C: Context>(context: &mut C) -> Result<bool, C::Error> {
    if context.alias_committed() {
        return Ok(true);
    }
    context.accepting()
}
pub(super) fn terminal<C: Context>(context: &mut C) -> Result<bool, C::Error> {
    Ok(complete(context)? && (context.alias_committed() || context.stopped()))
}
pub(super) fn control_bytes<C: Context>() -> Option<usize> {
    let parts = [
        size_of::<(&mut C, u32)>(),
        size_of::<&C>(),
        size_of::<Failure>(),
        size_of::<Result<(), C::Error>>(),
        size_of::<Result<bool, C::Error>>(),
        size_of::<(bool, bool)>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
