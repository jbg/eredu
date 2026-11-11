//! Speech embedding replacement through the shared LFM3 decoder traversal.

use super::AudioTokens;
use crate::lfm3::{LayeredModel, Lfm3Input};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Error, GroupedNeuralBackend, Index, Tensor,
};
use eredu_runtime::{
    LayerRuntimeState, LayerwisePolicy, LayerwiseRuntime, LayerwiseRuntimeError, ResidentRuntime,
    RuntimeStateComponents, SubmissionBackend,
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
        runtime.forward(Lfm3Input::Audio(self), state, context)
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
        runtime.forward(Lfm3Input::Audio(self), state, context)
    }

    /// Substitutes speech rows after the vocabulary collective and returns normalized hidden states.
    pub fn forward_tensor_parallel<B, S, P>(
        &self,
        runtime: &mut LayerwiseRuntime<LayeredModel<B>, B, S, P>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &T::Context,
    ) -> Result<T, LayerwiseRuntimeError<Error, P::Error>>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend<Tensor = T>
            + DistributedNeuralBackend
            + SubmissionBackend<Executor = T::Context>,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<T> + RuntimeStateComponents<B>,
        P: LayerwisePolicy<B, crate::lfm3::Block<B>>,
        P::Error: std::fmt::Display,
    {
        runtime.forward_parallel(Lfm3Input::Audio(self), state, parallel, context)
    }

    /// Tensor roots retained through a distributed prefill or feedback invocation.
    pub fn retained_tensors(&self) -> impl Iterator<Item = &T> {
        [Some(&self.tokens), self.input, self.output]
            .into_iter()
            .flatten()
    }

    /// Describes the exact prepared input shared by the pipeline ranks.
    pub fn collective_descriptor(&self, context: &T::Context) -> Result<[u64; 8], Error> {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(self.length.to_le_bytes());
        digest.update(self.width.to_le_bytes());
        for span in &self.spans {
            digest.update([span.source as u8]);
            digest.update(span.start.to_le_bytes());
            digest.update(span.end.to_le_bytes());
        }
        for id in self.tokens.to_i32_vec(context)? {
            digest.update(id.to_le_bytes());
        }
        for tensor in [self.input, self.output].into_iter().flatten() {
            for value in tensor
                .cast_float(eredu_nn::TensorElementType::F32, context)?
                .to_f32_vec(context)?
            {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
        let digest = digest.finalize();
        let mut descriptor = [0x4c464d3341554449, 0, 0, 0, 0, 0, 0, 0];
        for (word, bytes) in descriptor[1..].iter_mut().zip(digest.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().expect("eight digest bytes"));
        }
        Ok(descriptor)
    }

    pub(in crate::lfm3) fn tokens(&self) -> &T {
        &self.tokens
    }

    pub(in crate::lfm3) fn replace(&self, text: &T, context: &T::Context) -> Result<T, Error> {
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
