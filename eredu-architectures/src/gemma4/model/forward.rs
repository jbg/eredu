//! Exact forward metadata is borrowed from the admitted input or cold context.
use super::*;
pub(super) use crate::decoder::identity::Metadata;
mod shared;
pub use shared::SharedAttentionPublications;

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Error,
    // The paid diagnostic dies before its actual host account.
    _funding: eredu_nn::workspace::HostMetadataFunding,
}
fn retain(metadata: Metadata<'_>, cause: Error) -> Error {
    match metadata.context().and_then(|context| context.metadata_funding()) {
        Some(funding) => funding.clone().metadata_source(Failure { cause, _funding: funding }),
        None => cause,
    }
}
#[inline(never)]
pub(in crate::gemma4) fn with_metadata<R, F>(metadata: Metadata<'_>, operation: F) -> Result<R, Error>
where F: FnOnce() -> Result<R, Error> {
    metadata.controls::<(F, Result<R,Error>, Option<eredu_nn::workspace::HostMetadataFunding>)>()?;
    operation().map_err(|cause| retain(metadata, cause))
}

impl<B> LayeredModel<B>
where B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend {
    pub(super) fn validate_partition_state_with_metadata<S: LayerRuntimeState<B>>(
        &self, state: &S, metadata: Metadata<'_>) -> Result<(), Error> {
        metadata.controls::<(&Self, &S, std::borrow::Cow<'_,StateLayout>,StateLayout,
            std::ops::Range<usize>)>()?;
        let complete = match metadata.context() {
            Some(_) => std::borrow::Cow::Borrowed(&self.checked_graph(metadata)?.state),
            None => std::borrow::Cow::Owned(self.state_layout_impl()?),
        };
        let end = self.partition_state_offset.checked_add(state.layout().len())
            .ok_or_else(||metadata.error(format_args!("Gemma 4 partition state interval overflow")))?;
        let expected = match metadata.context() {
            Some(context) => complete.slice_with_metadata(self.partition_state_offset..end,context),
            None => complete.slice(self.partition_state_offset..end),
        }.map_err(|cause|metadata.source(cause))?;
        if state.layout()!=&expected {
            return Err(metadata.error(format_args!("Gemma 4 rank-local state layout mismatch")));
        }
        Ok(())
    }

    pub(super) fn partition_position_offset_with_metadata<S: LayerRuntimeState<B>>(
        &self, state: &mut S, metadata: Metadata<'_>) -> Result<i32,Error>
    where S::LayerState: AttentionCache<B::Tensor> {
        metadata.controls::<(&Self,&mut S,i32,usize)>()?;
        let mut position_offset=0;
        for local in 0..state.layout().len() {
            position_offset=position_offset.max(AttentionCache::<B::Tensor>::offset(
                state.layer(local).map_err(|cause|metadata.source(cause))?));
        }
        Ok(position_offset)
    }

    pub(super) fn prepare_parts_with_metadata(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        with_metadata(metadata, || {
        metadata.controls::<(Vec<PreparedPart<B::Tensor>>, PreparedPart<B::Tensor>,
            &[DecoderInputPart<'_, B::Tensor>], Option<&B::ParallelContext>, [i32;3])>()?;
        if parts.is_empty() {
            return Err(metadata.error(format_args!("Gemma 4 input has no ordered parts")));
        }
        let mut prepared = metadata.vector(parts.len())?;
        for part in parts {
            let value: Result<PreparedPart<B::Tensor>,Error> = match part {
                DecoderInputPart::Text(tokens) if self.partition_state_offset > 0 => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: B::Tensor::full_f32(0.0, &[tokens.dim(0), tokens.dim(1), self.args.text.hidden_size], context)?,
                }),
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: match parallel {
                        Some(parallel) => B::vocabulary_parallel_lookup(
                            &mut self.static_modules.text.embeddings, tokens,
                            EmbeddingLookupPolicy::Strict, parallel, context)?,
                        None => self.static_modules.text.embeddings.forward(tokens, context)?,
                    }
                        .multiply_scalar((self.args.text.hidden_size as f32).sqrt(), context)?,
                }),
                DecoderInputPart::Image(tokens) | DecoderInputPart::Video(tokens) => {
                    Ok(PreparedPart::Vision {
                        tokens: (*tokens).clone(),
                    })
                }
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(metadata.error(format_args!(
                            "Gemma 4 projected input shape {:?} does not match tokens {:?} and hidden width {}",
                            embeddings.shape(),
                            tokens.shape(),
                            self.args.text.hidden_size
                        )));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
            };
            prepared.push(value?);
        }
        Ok(prepared)
        })
    }

    pub(super) fn assemble_with_metadata(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        audio: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        with_metadata(metadata, || {
        metadata.controls::<(Vec<B::Tensor>, Vec<OrderedInputPart<'_, B::Tensor>>,
            eredu_nn::multimodal::OrderedModelInput<B::Tensor>, i32, i32)>()?;
        let vision_tokens = token_count(parts, |part| matches!(part, PreparedPart::Vision { .. }), metadata)?;
        let audio_tokens = token_count(parts, |part| matches!(part, PreparedPart::Audio { .. }), metadata)?;
        validate_component_with_metadata("vision", vision, vision_tokens, self.args.text.hidden_size, metadata)?;
        validate_component_with_metadata("audio", audio, audio_tokens, self.args.text.hidden_size, metadata)?;
        let mut vision_offset = 0;
        let mut audio_offset = 0;
        let mut embeddings = metadata.vector(parts.len())?;
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
                PreparedPart::Vision { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(slice_component(
                        vision.expect("validated vision component"),
                        vision_offset,
                        length,
                        context,
                    )?);
                    vision_offset += length;
                }
                PreparedPart::Audio { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(slice_component(
                        audio.expect("validated audio component"),
                        audio_offset,
                        length,
                        context,
                    )?);
                    audio_offset += length;
                }
            }
        }
        let mut ordered = metadata.vector(parts.len())?;
        ordered.extend(parts
            .iter()
            .zip(&embeddings)
            .map(|(part, embeddings)| OrderedInputPart {
                token_ids: match part {
                    PreparedPart::Text { tokens, .. }
                    | PreparedPart::Vision { tokens }
                    | PreparedPart::Audio { tokens } => tokens,
                },
                embeddings,
            }));
        match metadata.context() {
            Some(metadata) => eredu_nn::multimodal::assemble_ordered_inputs_with_metadata(
                &ordered, self.args.text.hidden_size, context, metadata),
            None => assemble_ordered_inputs(&ordered, self.args.text.hidden_size, context),
        }
        })
    }

    pub(super) fn begin_forward_with_metadata<'a, S>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where S: LayerRuntimeState<B>, S::LayerState: AttentionCache<B::Tensor> {
        with_metadata(metadata, || {
        metadata.controls::<(LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            ModelInput<'a, B::Tensor>, Option<&B::ParallelContext>, &mut S, Option<eredu_nn::workspace::WorkspaceContext>,
            eredu_runtime::layered::LayeredMetadata<Error>)>()?;
        if parallel.is_some() && self.parallel_geometry.is_none() {
            return Err(metadata.error(format_args!("Gemma 4 model was not built with local geometry")));
        }
        self.validate_partition_state_with_metadata(state, metadata)?;
        if metadata.context().is_some() && (input.vision.is_some() || input.audio.is_some()) {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        let parts = self.prepare_parts_with_metadata(input.parts, parallel, context, metadata)?;
        let (vision_initial, vision_state) = match input.vision {
            Some(vision) => {
                let (hidden, state) = self.begin_partition_vision(vision, context, metadata)?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let (audio_initial, audio_valid) = match input.audio {
            Some(audio) => {
                let (hidden, valid) = self.begin_partition_audio(audio, context, metadata)?;
                (Some(hidden), Some(valid))
            }
            None => (None, None),
        };
        let assembled = self.assemble_with_metadata(&parts, None, None, context, metadata);
        let hidden = vision_initial
            .as_ref()
            .or(audio_initial.as_ref())
            .cloned()
            .or_else(|| {
                assembled
                    .as_ref()
                    .ok()
                    .map(|assembled| assembled.embeddings.clone())
            })
            .ok_or_else(|| {
                assembled
                    .err()
                    .unwrap_or_else(|| metadata.error(format_args!("empty Gemma input")))
            })?;
        let position_offset = self.partition_position_offset_with_metadata(state, metadata)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                position_offset,
                parts,
                per_layer_token_override: input.per_layer_tokens.cloned(),
                per_layer_inputs: None,
                shared: SharedAttentionPublications::prepare(&self.args.text, metadata)?,
                shared_pending: false,
                vision_state,
                vision_initial,
                vision_output: None,
                audio_valid,
                audio_initial,
                audio_output: None,
                pending_media: None,
                media_span: false,
                metadata: metadata.context().cloned(),
            },
        })
        })
    }

}

impl<T> ForwardContext<T> {
    pub(super) fn visit_values<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        self.mask.iter().for_each(&mut *visitor);
        self.per_layer_token_override.iter().for_each(&mut *visitor);
        self.per_layer_inputs.iter().for_each(&mut *visitor);
        for part in &self.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => [tokens, embeddings].into_iter().for_each(&mut *visitor),
                PreparedPart::Vision { tokens } | PreparedPart::Audio { tokens } => {
                    visitor(tokens)
                }
            }
        }
        if let Some(state) = &self.vision_state {
            state.retained_values().into_iter().for_each(&mut *visitor);
        }
        self.vision_initial.iter().for_each(&mut *visitor);
        self.vision_output.iter().for_each(&mut *visitor);
        self.audio_initial.iter().for_each(&mut *visitor);
        self.audio_output.iter().for_each(&mut *visitor);
        if let Some(pending) = &self.pending_media {
            media_prefill::visit_pending(pending, visitor);
        }
        for (key, value) in self.shared.values() {
            visitor(key); visitor(value);
        }
    }
}

#[cfg(test)]
mod tests;
pub(super) fn validate_component_with_metadata<T: Tensor>(
    name: &str,
    value: Option<&T>,
    tokens: i32,
    hidden: i32,
    metadata: Metadata<'_>,
) -> Result<(), Error> {
    match value {
        Some(value) if value.shape() == [1, tokens, hidden] => Ok(()),
        None if tokens == 0 => Ok(()),
        Some(value) => Err(metadata.error(format_args!(
            "Gemma 4 {name} output has shape {:?}, expected [1, {tokens}, {hidden}]",
            value.shape()
        ))),
        None => Err(metadata.error(format_args!(
            "Gemma 4 {name} placeholders require projected media"
        ))),
    }
}

fn token_count<T:Tensor>(parts:&[PreparedPart<T>], select:impl Fn(&PreparedPart<T>)->bool,
    metadata:Metadata<'_>)->Result<i32,Error> {
    parts.iter().filter(|part|select(part)).try_fold(0i32,|count,part| {
        let tokens=match part {PreparedPart::Text{tokens,..}|PreparedPart::Vision{tokens}|PreparedPart::Audio{tokens}=>tokens};
        count.checked_add(tokens.dim(1)).ok_or_else(||metadata.error(format_args!("Gemma ordered token extent overflow")))
    })
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
