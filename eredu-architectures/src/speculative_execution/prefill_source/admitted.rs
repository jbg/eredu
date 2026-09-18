//! Actual composite admission retained across captured-prefill source creation.
use super::*;
use crate::composite_execution::PreparedCompositeInput;
use crate::media_plan::AdmittedCompositeInput;
use crate::prefill::{ExternalPrefillChunk, PreparedExternalPrefill, PrefillAdmission};
use eredu_nn::Tensor;
use eredu_runtime::{PreparedInputInspector, PreparedInputPayload, PreparedModelInput};

/// Preserves the prepared owner, its actual admission and its speculative identity.
/// Token inputs use shared text spans; default media retains one whole invocation
/// until selected retained-media ingress is connected to captured seeding.
pub struct AdmittedPredictionPrefill<C, T, I, P> {
    input: eredu_runtime::input::PreparedModelInputOwner<T>,
    admitted: PrefillAdmission<P>,
    config: C,
    inspector: I,
    identity: Option<SharedPreparedInputCacheIdentity>,
    chunk: Option<NonZeroU64>,
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}
impl<C, T, I, P> AdmittedPredictionPrefill<C, T, I, P> {
    /// Adopts existing owners; it neither repeats admission nor grants a budget.
    pub fn new(
        input: PreparedModelInput<T>,
        admitted: AdmittedCompositeInput<P>,
        config: C,
        inspector: I,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: Option<NonZeroU64>,
    ) -> Self {
        Self {
            input: input.into(),
            admitted: PrefillAdmission::Ordinary(admitted),
            config,
            inspector,
            identity,
            chunk,
            metadata: None,
        }
    }
    /// Keeps actual original input/admission owners and the paid host context.
    /// This does not grant authority for any later native view or token program.
    pub fn from_prepared_owner_with_metadata(
        input: eredu_runtime::input::PreparedModelInputOwner<T>,
        admitted: AdmittedCompositeInput<P>,
        config: C,
        inspector: I,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: Option<NonZeroU64>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, Error> {
        crate::decoder::identity::Metadata::new(Some(context)).controls::<Self>()?;
        if input.original_source().is_none() || context.metadata_funding().is_none() {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        Ok(Self {
            input,
            admitted: PrefillAdmission::Ordinary(admitted),
            config,
            inspector,
            identity,
            chunk,
            metadata: Some(context.clone()),
        })
    }
    /// Retains the original compiled semantic admission instead of re-reading
    /// native metadata. The input account must be the exact source of that plan.
    pub fn from_compiled_source_with_metadata(
        input: eredu_runtime::input::PreparedModelInputOwner<T>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        config: C, inspector: I,
        identity: Option<SharedPreparedInputCacheIdentity>, chunk: Option<NonZeroU64>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, Error> {
        crate::decoder::identity::Metadata::new(Some(context)).controls::<(Self, Result<Self, Error>)>()?;
        if context.metadata_funding().is_none() {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        let admitted = PrefillAdmission::Original(original);
        admitted.input(&input, Some(context))?;
        Ok(Self { input, admitted, config, inspector, identity, chunk, metadata: Some(context.clone()) })
    }
}

pub struct AdmittedPredictionSource<C, T, I, P> {
    input: PreparedExternalPrefill<C, T, I, P>,
}
pub struct AdmittedPredictionChunk<T, P> {
    input: ExternalPrefillChunk<T, P>,
    tokens: Option<T>,
}

impl<A, B, S, I> PredictionPrefillPlan<PreparedCompositeArchitecture<A>, B, S>
    for AdmittedPredictionPrefill<A::AdmissionConfig, B::Tensor, I, A::InputPartPlan>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: PreparedInputInspector<B::Tensor>,
{
    type Source = AdmittedPredictionSource<A::AdmissionConfig, B::Tensor, I, A::InputPartPlan>;
    fn shape(&self) -> Result<[u64; 2], Error> {
        Ok(self.admitted.input(&self.input, self.metadata.as_ref())?.admitted().decoder_shape())
    }

    fn chunk_positions(&self) -> Option<NonZeroU64> {
        self.chunk
    }
    fn requires_whole_input(&self) -> bool {
        self.input.parts().iter().any(|part| {
            part.modality() != eredu_core::InputModality::Text
                || !matches!(part.payload(), PreparedInputPayload::TokenIds(_))
        })
    }
    fn identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.identity.as_ref().map(AsRef::as_ref)
    }
    fn visit_token_parts(&self,
        visitor: &mut dyn FnMut(crate::composite_execution::PredictionTokenPart<'_, B::Tensor>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let input = self.admitted.input(&self.input, self.metadata.as_ref())?;
        A::visit_prepared_prediction_tokens(input, visitor)
    }
    fn into_source(self, geometry: InferenceGeometry) -> Result<Option<Self::Source>, Error> {
        let input = PreparedExternalPrefill::from_source(
            self.input, self.admitted, geometry, self.config, self.inspector,
            self.chunk.is_none(), self.metadata.as_ref(),
        )?;
        Ok(input.map(|input| AdmittedPredictionSource { input }))
    }
}

impl<A, B, S, I> PreparedPrefillSource<PreparedCompositeArchitecture<A>, B, S>
    for AdmittedPredictionSource<A::AdmissionConfig, B::Tensor, I, A::InputPartPlan>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: PreparedInputInspector<B::Tensor>,
{
    type Chunk = AdmittedPredictionChunk<B::Tensor, A::InputPartPlan>;
    fn geometry(&self) -> InferenceGeometry {
        <_ as PreparedPrefillSource<PreparedCompositeArchitecture<A>, B, S>>::geometry(&self.input)
    }
    fn prepare_chunk(
        &self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, Error> {
        let input =
            <_ as PreparedPrefillSource<PreparedCompositeArchitecture<A>, B, S>>::prepare_chunk(
                &self.input,
                chunk,
                context,
            )?;
        let paired = <_ as PreparedPrefillSource<PreparedCompositeArchitecture<A>, B, S>>::input(
            &self.input,
            &input,
        );
        // This is the same architecture-owned extraction used by ordinary
        // embedded prefill, now inside the existing guarded source preparation.
        let tokens = A::prepared_prediction_token_ids(paired, context)?;
        Ok(AdmittedPredictionChunk {
            input,
            tokens: Some(tokens),
        })
    }
    fn input<'a>(
        &'a self,
        chunk: &'a Self::Chunk,
    ) -> PreparedCompositeInput<'a, B::Tensor, A::InputPartPlan> {
        <_ as PreparedPrefillSource<PreparedCompositeArchitecture<A>, B, S>>::input(
            &self.input,
            &chunk.input,
        )
    }
}

impl<A, B, S, I> PredictionPrefillSource<PreparedCompositeArchitecture<A>, B, S>
    for AdmittedPredictionSource<A::AdmissionConfig, B::Tensor, I, A::InputPartPlan>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: PreparedInputInspector<B::Tensor>,
{
    fn tokens<'a>(&self, chunk: &'a Self::Chunk) -> &'a B::Tensor {
        chunk.tokens.as_ref().unwrap_or_else(|| {
            self.input
                .token_ids(&chunk.input)
                .expect("completed text chunk retains one token part")
        })
    }
    fn original_prepared_input(
        &self,
    ) -> Option<&eredu_runtime::input::PreparedModelInputOwner<B::Tensor>> {
        self.input.original_prepared_input()
    }
    fn validate_completed_token_span(
        &self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), Error> {
        self.input.validate_completed_span(chunk)
    }
    fn prepare_chunk_from_completed_tokens(
        &self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        supplied: (B::Tensor, super::super::EmbeddedPredictionTensor<B::Tensor>),
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, Error> {
        self.input
            .prepare_completed_chunk::<A, B, S>(chunk, supplied, context)
            .map(|input| AdmittedPredictionChunk {
                input,
                tokens: None,
            })
    }
    fn completed_token_packet<'a>(
        &self,
        chunk: &'a Self::Chunk,
    ) -> Option<&'a super::super::EmbeddedPredictionTensor<B::Tensor>> {
        self.input.completed_token_packet(&chunk.input)
    }
}
