//! Borrowed generated programs and lossless retention/error forwarding.
use crate::BlockFp8InputReconstructionPlan;

/// Actual checked operator program, not allocation or execution authority.
#[derive(Debug, Clone, Copy)]
pub enum GeneratedTensorProgram<'a> {
    /// Reconstruct compact E4M3 activations with their actual F32 block scales.
    BlockFp8Input(BlockFp8InputReconstructionPlan<'a>),
}
/// Meaning of a borrowed input to the generated program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedTensorSourceRole {
    /// Compact U8 values, shaped `[rows, width]`.
    CompactValues,
    /// F32 scales, shaped `[rows, ceil(width / 128)]`.
    BlockScales,
}
/// A real operator factory whose partial graph can be retained by its caller.
///
/// Concrete producers borrow the checked plan and actual inputs. Neither this
/// contract nor its callbacks grant funding, source registration or completion.
/// The caller validates its original operation before visiting or generating.
/// Source visitation never constructs a tensor. Generation calls `retain` for
/// each created output before its next fallible operation, including the final
/// output. A failure stops construction; earlier retained roots remain with the
/// caller. Multiple invocations construct independent graphs, as on the legacy
/// callback route; callers own selection/quota and once-per-hook caching policy.
pub trait RetainedGeneratedTensorFactory<T, E> {
    /// Borrow the actual immutable checked plan, without manufacturing a bound.
    fn program(&self) -> GeneratedTensorProgram<'_>;
    /// Visit actual inputs in program order, without cloning or evaluating them.
    fn visit_sources(
        &mut self,
        retain: &mut dyn FnMut(GeneratedTensorSourceRole, &T) -> Result<(), E>,
    ) -> Result<(), E>;
    /// Construct and expose every output before continuing the fixed program.
    fn generate(&mut self, retain: &mut dyn FnMut(&T) -> Result<(), E>) -> Result<T, E>;
}

/// Fixed propagation signal used only while the original sink error is retained.
/// It carries no duplicated diagnostic payload and is never the final cause when
/// the lossless factory/observer adapter has an original error to return.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("generated tensor retention failed")]
pub struct GeneratedTensorRetentionSignal;

/// Borrowed type/error bridge for the same factory, never a new program.
///
/// A sink's original error is retained even if the inner factory returns or
/// swallows its execution-domain propagation signal. Payload references use the
/// supplied borrowed representation cast; returned handles move exactly once.
/// Mapping functions create no source/funding proof. The original caller owns
/// any error payload and the actual operation's completion/recovery.
pub struct MappedGeneratedTensorFactory<'a, S, T, X, E, Ref, Owned, ToOuter, ToInner> {
    inner: &'a mut dyn RetainedGeneratedTensorFactory<S, X>,
    reference: Ref,
    owned: Owned,
    to_outer: ToOuter,
    to_inner: ToInner,
    types: std::marker::PhantomData<fn() -> (T, E)>,
}
impl<'a, S, T, X, E, Ref, Owned, ToOuter, ToInner>
    MappedGeneratedTensorFactory<'a, S, T, X, E, Ref, Owned, ToOuter, ToInner>
{
    /// Borrow the original factory and lossless error conversion machinery.
    /// `to_inner` supplies a bounded propagation signal; it must not duplicate
    /// arbitrary original error text. The original error stays owned here.
    pub fn new(
        inner: &'a mut dyn RetainedGeneratedTensorFactory<S, X>,
        reference: Ref,
        owned: Owned,
        to_outer: ToOuter,
        to_inner: ToInner,
    ) -> Self {
        Self {
            inner,
            reference,
            owned,
            to_outer,
            to_inner,
            types: std::marker::PhantomData,
        }
    }
}
impl<S, T, X, E, Ref, Owned, ToOuter, ToInner> RetainedGeneratedTensorFactory<T, E>
    for MappedGeneratedTensorFactory<'_, S, T, X, E, Ref, Owned, ToOuter, ToInner>
where
    Ref: for<'b> Fn(&'b S) -> &'b T,
    Owned: FnMut(S) -> T,
    ToOuter: FnMut(X) -> E,
    ToInner: FnMut(&E) -> X,
{
    fn program(&self) -> GeneratedTensorProgram<'_> {
        self.inner.program()
    }
    fn visit_sources(
        &mut self,
        retain: &mut dyn FnMut(GeneratedTensorSourceRole, &T) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut failure = None;
        let reference = &self.reference;
        let to_inner = &mut self.to_inner;
        let result = self.inner.visit_sources(&mut |role, value| {
            retain(role, reference(value)).map_err(|error| {
                let signal = to_inner(&error);
                if failure.is_none() {
                    failure = Some(error);
                }
                signal
            })
        });
        match failure {
            Some(error) => Err(error),
            None => result.map_err(&mut self.to_outer),
        }
    }
    fn generate(&mut self, retain: &mut dyn FnMut(&T) -> Result<(), E>) -> Result<T, E> {
        let mut failure = None;
        let reference = &self.reference;
        let to_inner = &mut self.to_inner;
        let result = self.inner.generate(&mut |value| {
            retain(reference(value)).map_err(|error| {
                let signal = to_inner(&error);
                if failure.is_none() {
                    failure = Some(error);
                }
                signal
            })
        });
        match failure {
            Some(error) => Err(error),
            None => result.map(&mut self.owned).map_err(&mut self.to_outer),
        }
    }
}

#[cfg(test)]
mod tests;
