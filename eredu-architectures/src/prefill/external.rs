//! External capture uses shared text spans or one unchanged prepared invocation.
use super::*;
use crate::media_plan::AdmittedCompositeInput;
use eredu_runtime::PreparedModelInput;

/// Ordinary source for the existing external capture transaction. Token inputs
/// use the ordinary text source. A default non-token request retains its genuine
/// prepared/admitted pair for one complete invocation; it does not slice media,
/// rerun admission, manufacture hidden values, or grant original-budget authority.
pub struct PreparedExternalPrefill<C, T, I, P> {
    body: Source<C, T, I, P>,
}

enum Source<C, T, I, P> {
    Text(PreparedCompositeTextPrefill<C, T, I>),
    Whole {
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        admitted: AdmittedCompositeInput<P>,
        geometry: InferenceGeometry,
        metadata: Option<WorkspaceContext>,
    },
}

impl<C, T: Tensor, I, P> PreparedExternalPrefill<C, T, I, P> {
    pub(crate) fn original_prepared_input(
        &self,
    ) -> Option<&eredu_runtime::input::PreparedModelInputOwner<T>> {
        match &self.body {
            Source::Text(text) => text.text.original_prepared_input(),
            Source::Whole { prepared, .. } => Some(prepared),
        }
    }
    pub(crate) fn completed_token_packet<'a>(
        &self,
        chunk: &'a ExternalPrefillChunk<T, P>,
    ) -> Option<&'a crate::speculative_execution::EmbeddedPredictionTensor<T>> {
        match &chunk.0 {
            Chunk::Text(chunk) => chunk.packet.as_ref(),
            Chunk::Whole => None,
        }
    }
    pub(crate) fn token_ids<'a>(&self, chunk: &'a ExternalPrefillChunk<T, P>) -> Option<&'a T> {
        match &chunk.0 {
            Chunk::Text(chunk) => match chunk.prepared.parts().first()?.payload() {
                eredu_runtime::PreparedInputPayload::TokenIds(tokens) => Some(tokens),
                _ => None,
            },
            Chunk::Whole => None,
        }
    }
    pub(crate) fn validate_completed_span(&self, chunk: &PrefillChunk) -> Result<(), Error> {
        match &self.body {
            Source::Text(text) => text.text.validate_completed_span(chunk),
            Source::Whole { .. } => {
                Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
            }
        }
    }
    pub(crate) fn prepare_completed_chunk<A, B, S>(
        &self,
        chunk: &PrefillChunk,
        supplied: (T, crate::speculative_execution::EmbeddedPredictionTensor<T>),
        context: &T::Context,
    ) -> Result<ExternalPrefillChunk<T, P>, Error>
    where
        B: NeuralBackend<Tensor = T>,
        S: RuntimeState<B>,
        A: CompositeArchitecture<B, S, Error = Error, AdmissionConfig = C, InputPartPlan = P>
            + 'static,
        P: 'static,
        I: eredu_runtime::PreparedInputInspector<T>,
    {
        match &self.body {
            Source::Text(text) => {
                prepare_composite_chunk::<A, B, S, I>(text, chunk, context, Some(supplied))
                    .map(|chunk| ExternalPrefillChunk(Chunk::Text(chunk)))
            }
            Source::Whole { .. } => {
                drop(supplied);
                Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
            }
        }
    }

    /// Adopts actual ordinary preparation. `allow_whole_input` is false for an
    /// explicit split request, whose media source remains the separate retained
    /// ingress integration. Returning None preserves that existing source gap.
    pub fn from_prepared(
        prepared: PreparedModelInput<T>,
        admitted: AdmittedCompositeInput<P>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
        allow_whole_input: bool,
    ) -> Result<Option<Self>, Error> {
        Self::from_source(
            prepared.into(),
            admitted,
            geometry,
            config,
            inspector,
            allow_whole_input,
            None,
        )
    }

    pub(crate) fn from_source_owner(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        admitted: AdmittedCompositeInput<P>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
        allow_whole_input: bool,
    ) -> Result<Option<Self>, Error> {
        Self::from_source(
            prepared,
            admitted,
            geometry,
            config,
            inspector,
            allow_whole_input,
            None,
        )
    }

    /// Retains the existing original prepared owner through the same selection,
    /// exact geometry and source-lifetime rules; no tensor handles are cloned.
    pub fn from_prepared_owner_with_metadata(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        admitted: AdmittedCompositeInput<P>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
        allow_whole_input: bool,
        context: &WorkspaceContext,
    ) -> Result<Option<Self>, Error> {
        Self::from_source(
            prepared,
            admitted,
            geometry,
            config,
            inspector,
            allow_whole_input,
            Some(context),
        )
    }
    fn from_source(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        admitted: AdmittedCompositeInput<P>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
        allow_whole_input: bool,
        context: Option<&WorkspaceContext>,
    ) -> Result<Option<Self>, Error> {
        let metadata = Metadata::new(context);
        if context.is_some()
            && (prepared.original_source().is_none()
                || context
                    .and_then(WorkspaceContext::metadata_funding)
                    .is_none())
        {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        metadata.controls::<(
            Self,
            Option<Self>,
            eredu_runtime::input::PreparedModelInputOwner<T>,
            AdmittedCompositeInput<P>,
            C,
            I,
        )>()?;
        match context {
            Some(context) => {
                PreparedCompositeInput::new_with_metadata(&prepared, &admitted, context)?;
            }
            None => {
                PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::backend)?;
            }
        }
        if is_prepared_token_input(&prepared) {
            let text = match context {
                Some(context) => PreparedCompositeTextPrefill::from_prepared_owner_with_metadata(
                    prepared, geometry, config, inspector, context,
                )?,
                None => PreparedCompositeTextPrefill::from_prepared_text(
                    &prepared, geometry, config, inspector,
                )?,
            };
            return Ok(text.map(|text| Self {
                body: Source::Text(text),
            }));
        }
        if !allow_whole_input {
            return Ok(None);
        }
        geometry
            .validate_fixed()
            .map_err(|cause| metadata.source(cause))?;
        if admitted.decoder_shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.prefill_chunk_positions < geometry.input_positions
        {
            return Err(metadata.error(format_args!(
                "whole external input requires its exact complete admitted decoder span"
            )));
        }
        Ok(Some(Self {
            body: Source::Whole {
                prepared,
                admitted,
                geometry,
                metadata: context.cloned(),
            },
        }))
    }
}

/// Per-span owned token view, or a marker borrowing the full retained source.
pub struct ExternalPrefillChunk<T, P>(Chunk<T, P>);
enum Chunk<T, P> {
    Text(CompositeTextPrefillChunk<T, P>),
    Whole,
}

impl<A, B, S, I> PreparedPrefillSource<PreparedCompositeArchitecture<A>, B, S>
    for PreparedExternalPrefill<A::AdmissionConfig, B::Tensor, I, A::InputPartPlan>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: eredu_runtime::PreparedInputInspector<B::Tensor>,
{
    type Chunk = ExternalPrefillChunk<B::Tensor, A::InputPartPlan>;

    fn geometry(&self) -> InferenceGeometry {
        match &self.body {
            Source::Text(text) => text.text.geometry,
            Source::Whole { geometry, .. } => *geometry,
        }
    }

    fn prepare_chunk(
        &self,
        chunk: &PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, Error> {
        match &self.body {
            Source::Text(text) => {
                <PreparedCompositeTextPrefill<_, _, _> as PreparedPrefillSource<
                    PreparedCompositeArchitecture<A>,
                    B,
                    S,
                >>::prepare_chunk(text, chunk, context)
                .map(|chunk| ExternalPrefillChunk(Chunk::Text(chunk)))
            }
            Source::Whole {
                geometry, metadata, ..
            } => {
                if chunk.input != (0..geometry.input_positions)
                    || chunk.position != geometry.cached_positions
                    || chunk.output != geometry.output
                {
                    return Err(Metadata::new(metadata.as_ref()).error(format_args!(
                        "whole external input span differs from its source",
                    )));
                }
                Ok(ExternalPrefillChunk(Chunk::Whole))
            }
        }
    }

    fn input<'a>(
        &'a self,
        chunk: &'a Self::Chunk,
    ) -> PreparedCompositeInput<'a, B::Tensor, A::InputPartPlan> {
        match (&self.body, &chunk.0) {
            (Source::Text(text), Chunk::Text(chunk)) => {
                <PreparedCompositeTextPrefill<_, _, _> as PreparedPrefillSource<
                    PreparedCompositeArchitecture<A>,
                    B,
                    S,
                >>::input(text, chunk)
            }
            (
                Source::Whole {
                    prepared,
                    admitted,
                    metadata,
                    ..
                },
                Chunk::Whole,
            ) => PreparedCompositeInput::new(prepared, admitted)
                .expect("private whole source retains its actual admission")
                .with_metadata_loan(metadata.as_ref()),
            _ => unreachable!("private chunk kind comes from its source"),
        }
    }
}
