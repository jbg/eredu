//! Native tensor views and completion for one selected refinement invocation.

use std::num::NonZeroU32;

use crate::{
    sampling::{SamplingBackend, TokenDomain},
    Submission, TextGenerationBackend,
};

type Token<B> = <<B as TextRefinementNative>::BlockSampling as SamplingBackend>::Token;
type Logits<B> = <<B as TextRefinementNative>::BlockSampling as SamplingBackend>::Logits;
type Invocation<B> = Submission<
    <B as TextRefinementNative>::ScoreTensor,
    <B as TextRefinementNative>::InvocationCompletion,
>;

/// Tensor views, token transfer, and resource retention for a single refinement invocation.
pub trait TextRefinementNative: TextGenerationBackend {
    /// Native numerical operations for scalar commitment.
    type BlockSampling: SamplingBackend;
    /// Raw score tensor produced by an executable.
    type ScoreTensor;
    /// Completion that retains an executable submission.
    type InvocationCompletion;
    /// Physical storage format used by a supplied conditioning tensor.
    type ConditioningFormat;

    /// Native submission scope retained through numerical work.
    type BlockScope;

    /// Shapes canvas and optional feedback for one model invocation.
    fn block_canvas(
        backend: &Self,
        canvas: &Token<Self>,
        canvas_shape: &[i32],
        feedback: Option<(&Logits<Self>, &[i32])>,
    ) -> Result<(Token<Self>, Option<Token<Self>>), Self::Error>;
    /// Interprets one output row as causal next-token scores.
    fn block_causal_scores(
        backend: &Self,
        scores: &Self::ScoreTensor,
    ) -> Result<Logits<Self>, Self::Error>;
    /// Shapes physical scores into the admitted canvas.
    fn block_refinement_scores(
        backend: &Self,
        scores: &Self::ScoreTensor,
        shape: &[i32],
    ) -> Result<Logits<Self>, Self::Error>;
    /// Reads the admitted selected scores for one completed refinement forward.
    fn read_refinement_scores(
        backend: &Self,
        scores: &Self::ScoreTensor,
        selections: Vec<super::RefinementScoreSelection>,
    ) -> Result<super::RefinementScoreRows, Self::Error>;
    /// Native floating format used for feedback.
    fn block_conditioning_format(scores: &Self::ScoreTensor) -> Self::ConditioningFormat;
    /// Reads completed native token IDs in position order.
    fn block_token_ids(backend: &Self, tokens: &Token<Self>) -> Result<Vec<u32>, Self::Error>;
    /// Opens a native scope retaining the model submission through numerical advancement.
    fn open_block_scope(submission: &Invocation<Self>) -> Result<Self::BlockScope, Self::Error>;
    /// Publishes a committed token while retaining the model submission.
    fn block_output_token(
        backend: &Self,
        token: Token<Self>,
        submission: &Invocation<Self>,
    ) -> Self::Token;
    /// Gets the native completion root for an intermediate canvas.
    fn block_canvas_root(canvas: &Token<Self>) -> Self::ScoreTensor;
    /// Gets the native completion root for a committed token.
    fn block_token_root(token: &Self::Token) -> Self::ScoreTensor;
    /// Retains native resources until the exact invocation is complete.
    fn finish_block_scope<P>(
        scope: Self::BlockScope,
        submission: Invocation<Self>,
        result: Result<(P, Self::ScoreTensor), Self::Error>,
    ) -> Result<Submission<P, Self::TextCompletion>, Self::Error>;
}

/// Numerical mechanisms used by a single-sequence refinement policy.
///
/// Operations return new immutable values. They never mutate supplied tensors,
/// retained model context, or the random stream except for explicit draw calls.
/// The owning session retains submission authority through settlement.
pub trait RefinementBackend {
    /// Immutable backend-native position scores.
    type Logits: Clone;
    /// Immutable backend-native token block.
    type Token: Clone;
    /// Independently copyable explicit random stream.
    type RandomState;
    /// Native execution context.
    type Context: ?Sized;
    /// Native mechanism failure.
    type Error;
    /// Native boolean position mask.
    type Mask: Clone;
    /// Architecture-selected embedding arithmetic used for retained feedback.
    type ConditioningFormat: Clone;

    /// Reads an exact one-dimensional token tensor into host storage.
    fn read_tokens(tokens: &Self::Token, context: &Self::Context) -> Result<Vec<u32>, Self::Error>;
    /// Reads an exact one-dimensional boolean tensor into host storage.
    fn read_mask(mask: &Self::Mask, context: &Self::Context) -> Result<Vec<bool>, Self::Error>;
    /// Reports the geometry of a nonempty score matrix without submitting work.
    fn masked_score_shape(scores: &Self::Logits) -> Option<[usize; 2]>;
    /// Reads a score matrix as flat row-major F32.
    fn read_scores(scores: &Self::Logits, context: &Self::Context)
        -> Result<Vec<f32>, Self::Error>;
    /// Supplies exact open-interval F64 draws using the native random stream.
    fn masked_uniforms(
        count: usize,
        random: &mut Self::RandomState,
        context: &Self::Context,
    ) -> Result<Vec<f64>, Self::Error>;
    /// Materializes a token vector on the selected context.
    fn materialize_tokens(
        tokens: &[u32],
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;
    /// Materializes a boolean vector on the selected context.
    fn materialize_mask(flags: &[bool], context: &Self::Context)
        -> Result<Self::Mask, Self::Error>;
    /// Materializes a flat row-major F32 score matrix.
    fn materialize_scores(
        scores: &[f32],
        shape: [usize; 2],
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Fixed positions have one-hot conditioning: zero at their seed and negative infinity elsewhere.
    fn pin_scores(
        scores: &Self::Logits,
        tokens: &Self::Token,
        pins: &Self::Mask,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Rejects NaN, positive infinity and rows without a finite candidate, then
    /// divides scores by a validated positive temperature.
    fn scale_temperature(
        logits: &Self::Logits,
        temperature: f32,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Checks `[canvas, vocabulary]` score geometry without execution or draws.
    fn validate_refinement_logits(
        logits: &Self::Logits,
        canvas: NonZeroU32,
        domain: TokenDomain,
    ) -> Result<(), Self::Error>;

    /// Uniform categorical token IDs for the complete canvas.
    fn uniform_tokens(
        canvas: NonZeroU32,
        domain: TokenDomain,
        random: &mut Self::RandomState,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;

    /// Per-position categorical draws from already temperature-scaled logits.
    fn categorical_tokens(
        logits: &Self::Logits,
        random: &mut Self::RandomState,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;

    /// Highest-scoring token at every canvas position.
    fn argmax_tokens(
        logits: &Self::Logits,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;

    /// Selects lowest-entropy positions with `cumsum(entropy) - entropy <= bound`.
    /// Ties are ordered by increasing original position.
    fn entropy_bound_mask(
        logits: &Self::Logits,
        bound: f32,
        context: &Self::Context,
    ) -> Result<Self::Mask, Self::Error>;

    /// Selects `accepted` at true mask positions and `other` elsewhere.
    fn select_tokens(
        mask: &Self::Mask,
        accepted: &Self::Token,
        other: &Self::Token,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;

    /// Converts processed logits to the decoder's self-conditioning dtype.
    fn retain_self_conditioning(
        logits: &Self::Logits,
        format: &Self::ConditioningFormat,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Observes one stop flag: mean entropy below the threshold and all supplied
    /// preceding argmax canvases identical to the current one.
    fn stable_and_confident(
        logits: &Self::Logits,
        argmax: &Self::Token,
        preceding: &[Self::Token],
        confidence_threshold: f32,
        pins: Option<&Self::Mask>,
        context: &Self::Context,
    ) -> Result<bool, Self::Error>;

    /// Establishes exact successful completion of all transition-local outputs.
    /// A failed poll is not completion; native resources stay with the session's
    /// recovery owner. No logical sampler state is published on error.
    fn settle_refinement(
        canvas: &Self::Token,
        argmax: &Self::Token,
        scores: &Self::Logits,
        conditioning: &Self::Logits,
        acceptance: &Self::Mask,
        random: &Self::RandomState,
        context: &Self::Context,
    ) -> Result<(), Self::Error>;

    /// Completes the initial noise and explicit key before publishing a block.
    fn settle_initial_canvas(
        canvas: &Self::Token,
        random: &Self::RandomState,
        context: &Self::Context,
    ) -> Result<(), Self::Error>;
}
