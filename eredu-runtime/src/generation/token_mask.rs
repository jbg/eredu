//! One borrowed Boolean mask expansion shared by ordinary and original sampling.
use eredu_core::{TokenFilter, TokenFilterError};
#[derive(Debug, thiserror::Error)]
pub enum TokenMaskError {
    /// Preserves the existing exact vocabulary/mask validation.
    #[error(transparent)]
    Filter(#[from] TokenFilterError),
    /// A mask requires nonempty finite distribution geometry.
    #[error("token mask source geometry is invalid")]
    Shape,
    /// A prospective forced token is not in the actual valid source domain.
    #[error("forced token conflicts with the actual tokenizer domain")]
    Forced,
    /// The caller did not supply the exact already-paid destination.
    #[error("token mask destination differs from its prepared extent")]
    Destination,
}
#[derive(Debug, Clone, Copy)]
enum Predicate<'a> {
    Fixed(Option<&'a [bool]>),
    Packed(eredu_core::PackedTokenFilter<'a>),
    Forbidden(eredu_core::speculative::ForbiddenControllerDecision<'a>),
}
impl Predicate<'_> {
    fn allows(self, token: usize) -> bool {
        match self {
            Self::Fixed(mask) => mask.is_none_or(|mask| mask.get(token).copied().unwrap_or(false)),
            Self::Packed(filter) => u32::try_from(token).is_ok_and(|token| filter.allows(token)),
            Self::Forbidden(decision) => {
                u32::try_from(token).is_ok_and(|token| decision.allows(token))
            }
        }
    }
    fn is_identity(self) -> bool {
        matches!(self, Self::Fixed(None))
    }
}
/// Finite borrowed source/predicate. It owns no buffer and grants no funding.
#[derive(Debug, Clone, Copy)]
pub struct TokenMaskPlan<'a> {
    predicate: Predicate<'a>,
    shape: &'a [i32],
    forced: Option<u32>,
    vocabulary: usize,
    elements: usize,
}
impl<'a> TokenMaskPlan<'a> {
    /// Same mask-width validation, followed by the optional fixed choice.
    pub fn new(
        filter: &'a TokenFilter,
        shape: &'a [i32],
        forced: Option<u32>,
    ) -> Result<Self, TokenMaskError> {
        let vocabulary = shape
            .last()
            .copied()
            .filter(|&d| d >= 0)
            .ok_or(TokenMaskError::Shape)? as usize;
        filter.validate_output_width(vocabulary)?;
        Self::with_predicate(
            Predicate::Fixed(filter.allowed_mask()),
            shape,
            forced,
            vocabulary,
        )
    }
    /// The actual forbidden predicate over source-derived vocabulary bytes.
    /// Validation preserves an empty semantic intersection before executable
    /// width refusal; missing output IDs remain false just like ordinary masks.
    pub fn forbidden(
        decision: eredu_core::speculative::ForbiddenControllerDecision<'a>,
        shape: &'a [i32],
        forced: Option<u32>,
    ) -> Result<Self, TokenMaskError> {
        let entries = decision.source().inputs().vocabulary_len();
        if entries == 0 {
            return Err(TokenFilterError::EmptyVocabulary.into());
        }
        if !(0..entries).any(|token| u32::try_from(token).is_ok_and(|token| decision.allows(token)))
        {
            return Err(TokenFilterError::NoAllowedToken.into());
        }
        let vocabulary = shape
            .last()
            .copied()
            .filter(|&d| d >= 0)
            .ok_or(TokenMaskError::Shape)? as usize;
        if vocabulary == 0 {
            return Err(TokenFilterError::EmptyVocabulary.into());
        }
        if !(0..entries.min(vocabulary))
            .any(|token| u32::try_from(token).is_ok_and(|token| decision.allows(token)))
        {
            return Err(TokenFilterError::NoExecutableToken {
                output_width: vocabulary,
            }
            .into());
        }
        Self::with_predicate(Predicate::Forbidden(decision), shape, forced, vocabulary)
    }
    /// Expands the actual packed grammar mask through the same invalid-Boolean
    /// destination worker. Borrowing words does not grant source authentication.
    pub fn packed(
        filter: eredu_core::PackedTokenFilter<'a>,
        shape: &'a [i32],
        forced: Option<u32>,
    ) -> Result<Self, TokenMaskError> {
        let vocabulary = shape.last().copied().filter(|&d| d >= 0)
            .ok_or(TokenMaskError::Shape)? as usize;
        filter.validate_output_width(vocabulary)?;
        Self::with_predicate(Predicate::Packed(filter), shape, forced, vocabulary)
    }
    fn with_predicate(
        predicate: Predicate<'a>,
        shape: &'a [i32],
        forced: Option<u32>,
        vocabulary: usize,
    ) -> Result<Self, TokenMaskError> {
        let elements = shape
            .iter()
            .try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or(TokenMaskError::Shape)?;
        if let Some(token) = forced {
            if token as usize >= vocabulary || !predicate.allows(token as usize) {
                return Err(TokenMaskError::Forced);
            }
        }
        Ok(Self {
            predicate,
            shape,
            forced,
            vocabulary,
            elements,
        })
    }
    /// Identity filters preserve their exact source alias.
    pub fn is_identity(self) -> bool {
        self.predicate.is_identity() && self.forced.is_none()
    }
    /// Exact invalid-mask destination; no forced intermediate mask is needed.
    pub fn elements(self) -> usize {
        if self.is_identity() { 0 } else { self.elements }
    }
    /// Revalidates the complete actual array shape without owned geometry.
    pub fn matches_shape(self, shape: &[i32]) -> bool {
        shape == self.shape
    }

    /// Expands the same row-major invalid predicate into an empty destination.
    /// Original callers provide capacity only after admission; this never grows.
    pub fn fill(self, destination: &mut Vec<bool>) -> Result<(), TokenMaskError> {
        if !destination.is_empty() || destination.capacity() < self.elements() {
            return Err(TokenMaskError::Destination);
        }
        if self.is_identity() {
            return Ok(());
        }
        for _ in 0..self.elements / self.vocabulary {
            for token in 0..self.vocabulary {
                destination.push(
                    !self.predicate.allows(token)
                        || self.forced.is_some_and(|forced| forced as usize != token),
                );
            }
        }
        Ok(())
    }
    /// Fixed expansion/validation transports; the operation's existing host
    /// workspace fact separately prices the actual Boolean buffer.
    pub fn control_bytes() -> usize {
        eredu_core::PackedTokenFilter::control_bytes()
            + std::mem::size_of::<Self>()
            + std::mem::size_of::<Predicate<'_>>()
            + std::mem::size_of::<(Predicate<'_>, &[i32], Option<u32>, usize)>()
            + std::mem::size_of::<std::ops::Range<usize>>()
            + std::mem::size_of::<Option<u32>>()
            + std::mem::size_of::<Vec<bool>>()
            + std::mem::size_of::<Result<Self, TokenMaskError>>()
            + std::mem::size_of::<Result<(), TokenMaskError>>()
            + std::mem::size_of::<Result<(), std::collections::TryReserveError>>()
    }
}
