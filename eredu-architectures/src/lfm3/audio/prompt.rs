//! Speech embedding replacement through the shared LFM3 decoder traversal.

use super::AudioTokens;
use crate::{
    decoder::LayeredInput,
    lfm3::{ForwardContext, LayeredModel},
};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Error, GroupedNeuralBackend, Index, NeuralBackend,
    Tensor,
};
use eredu_runtime::{
    LayerRuntimeState, LayeredTraversalHook, LayerwisePolicy, LayerwiseRuntime,
    LayerwiseRuntimeError, ResidentRuntime, RuntimeStateComponents, SubmissionBackend,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Source {
    Text,
    SpeechInput,
    SpeechOutput,
}

#[derive(Debug)]
struct Span {
    source: Source,
    start: i32,
    end: i32,
}

/// A single prompt span with the speech positions already mapped to projected rows.
/// The same input representation serves initial prefill and subsequent speech feedback.
pub struct AudioPrompt<'a, T: Tensor> {
    tokens: T,
    input: Option<&'a T>,
    output: Option<&'a T>,
    spans: Vec<Span>,
    width: i32,
    length: i32,
}

impl<'a, T: Tensor> AudioPrompt<'a, T> {
    /// Validates exact placeholder counts before submitting any decoder layers.
    /// Replacement tensors are in prompt order with shape `[positions, hidden]`.
    pub fn new(
        tokens: &[i32],
        identities: AudioTokens,
        width: i32,
        input: Option<&'a T>,
        output: Option<&'a T>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        let length = i32::try_from(tokens.len()).map_err(Error::backend_source)?;
        if length <= 0 || width <= 0 || identities.input == identities.output {
            return Err(Error::backend("invalid LFM3 Audio prompt geometry"));
        }
        let source_of = |token| {
            if token == identities.input {
                Source::SpeechInput
            } else if token == identities.output {
                Source::SpeechOutput
            } else {
                Source::Text
            }
        };
        let mut spans: Vec<Span> = Vec::new();
        let mut input_count = 0;
        let mut output_count = 0;
        for (index, &token) in tokens.iter().enumerate() {
            let source = source_of(token);
            let start = match source {
                Source::Text => index as i32,
                Source::SpeechInput => {
                    input_count += 1;
                    input_count - 1
                }
                Source::SpeechOutput => {
                    output_count += 1;
                    output_count - 1
                }
            };
            if let Some(last) = spans.last_mut().filter(|span| span.source == source) {
                last.end += 1;
            } else {
                spans.push(Span {
                    source,
                    start,
                    end: start + 1,
                });
            }
        }
        for (tensor, count) in [(input, input_count), (output, output_count)] {
            if (count > 0 && tensor.is_none())
                || tensor.is_some_and(|tensor| tensor.shape() != [count, width])
            {
                return Err(Error::backend(
                    "LFM3 Audio placeholders differ from projected speech rows",
                ));
            }
        }
        Ok(Self {
            tokens: T::from_i32_slice(tokens, &[1, length], context)?,
            input,
            output,
            spans,
            width,
            length,
        })
    }

    /// Runs one resident mixed input pass and returns normalized language states.
    /// The caller owns cache continuation and establishes native completion.
    pub fn forward_resident<B, S>(
        &self,
        runtime: &mut ResidentRuntime<LayeredModel<B>, B, S>,
        state: &mut S,
        context: &T::Context,
    ) -> Result<T, Error>
    where
        B: GroupedNeuralBackend<Tensor = T> + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<T> + RuntimeStateComponents<B>,
    {
        runtime
            .forward_with_traversal_hook(
                LayeredInput {
                    tokens: &self.tokens,
                    mask: None,
                },
                state,
                context,
                &mut SpeechReplacement { prompt: self },
            )
            .map(|(output, _)| output)
    }

    /// Runs the same mixed input on an already configured bounded language runtime.
    /// The parameter policy owns language-layer acquisition and completion.
    pub fn forward_layerwise<B, S, P>(
        &self,
        runtime: &mut LayerwiseRuntime<LayeredModel<B>, B, S, P>,
        state: &mut S,
        context: &T::Context,
    ) -> Result<T, LayerwiseRuntimeError<Error, P::Error>>
    where
        B: GroupedNeuralBackend<Tensor = T>
            + DistributedNeuralBackend
            + SubmissionBackend<Executor = T::Context>,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<T> + RuntimeStateComponents<B>,
        P: LayerwisePolicy<B, crate::lfm3::Block<B>>,
        P::Error: std::fmt::Display,
    {
        runtime
            .forward_with_traversal_hook(
                LayeredInput {
                    tokens: &self.tokens,
                    mask: None,
                },
                state,
                context,
                &mut SpeechReplacement { prompt: self },
            )
            .map(|(output, _)| output)
    }

    fn replace(&self, text: &T, context: &T::Context) -> Result<T, Error> {
        if text.shape() != [1, self.length, self.width] {
            return Err(Error::backend(
                "LFM3 Audio prompt does not match language embedding geometry",
            ));
        }
        if self.input.is_none() && self.output.is_none() {
            return Ok(text.clone());
        }
        let dtype = text
            .element_type()
            .ok_or_else(|| Error::backend("unknown language embedding dtype"))?;
        let spans = self
            .spans
            .iter()
            .map(|span| {
                let range = Index::Range(span.start, span.end);
                match span.source {
                    Source::Text => text.index(&[Index::Full, range, Index::Full], context),
                    source => {
                        let rows = match source {
                            Source::SpeechInput => self.input,
                            Source::SpeechOutput => self.output,
                            Source::Text => unreachable!(),
                        }
                        .ok_or_else(|| Error::backend("LFM3 Audio replacement rows are missing"))?;
                        rows.index(&[range, Index::Full], context)?
                            .cast_float(dtype, context)?
                            .expand_dims(0, context)
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        T::concatenate(&spans, 1, context)
    }
}

struct SpeechReplacement<'a, 'b, T: Tensor> {
    prompt: &'a AudioPrompt<'b, T>,
}

impl<B: NeuralBackend> LayeredTraversalHook<B, ForwardContext<B::Tensor>, Error>
    for SpeechReplacement<'_, '_, B::Tensor>
{
    fn after_group_begin(
        &mut self,
        group: usize,
        hidden: &mut B::Tensor,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        if group != 0 {
            return Err(Error::backend(
                "LFM3 Audio expects one language decoder group",
            ));
        }
        *hidden = self.prompt.replace(hidden, context)?;
        forward.normalized_audio_output = true;
        Ok(())
    }
}
