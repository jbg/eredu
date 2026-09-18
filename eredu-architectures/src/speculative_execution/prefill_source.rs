//! Owned source plans; native token buffers are retained, never re-copied.
use crate::composite_execution::{CompositeArchitecture, PreparedCompositeArchitecture};
use crate::prefill::{PreparedCompositeTextPrefill, PreparedTextPrefill};
use eredu_core::InferenceGeometry;
use eredu_nn::{Error, NeuralBackend};
use eredu_runtime::replicated_session::PreparedPrefillSource;
use eredu_runtime::{
    LayeredArchitecture, PreparedInputCacheIdentity, ReplicatedTextArchitecture, RuntimeState,
    SharedPreparedInputCacheIdentity,
};
use std::num::NonZeroU64;

mod admitted;
pub use admitted::AdmittedPredictionPrefill;

/// Semantic token access to the same chunk passed to the target. This lends a
/// handle, not a second input allocation or an independently reconstructed span.
pub trait PredictionPrefillSource<A, B, S>: PreparedPrefillSource<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    /// Borrows the exact tokens retained by this submitted chunk.
    fn tokens<'a>(&self, chunk: &'a Self::Chunk) -> &'a B::Tensor;
    /// Exact immutable original input retained by this source. This descriptive
    /// loan grants no execution scope or source registration.
    fn original_prepared_input(
        &self,
    ) -> Option<&eredu_runtime::input::PreparedModelInputOwner<B::Tensor>> {
        None
    }
    /// Checks the source's exact chunk contract before its numerical producer.
    fn validate_completed_token_span(
        &self,
        _chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), A::Error>
    where
        A::Error: From<Error>,
    {
        Err(Error::from(eredu_nn::workspace::WorkspaceMetadataError::Unqualified).into())
    }
    /// Retains a completed packet after its paid input alias in the same chunk.
    /// Native implementations authenticate the actual alias before this handoff.
    fn prepare_chunk_from_completed_tokens(
        &self,
        _chunk: &eredu_runtime::prefill::PrefillChunk,
        supplied: (B::Tensor, super::EmbeddedPredictionTensor<B::Tensor>),
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Chunk, A::Error>
    where
        A::Error: From<Error>,
    {
        drop(supplied);
        Err(Error::from(eredu_nn::workspace::WorkspaceMetadataError::Unqualified).into())
    }
    /// Account/source evidence for precisely this retained input chunk.
    fn completed_token_packet<'a>(
        &self,
        _chunk: &'a Self::Chunk,
    ) -> Option<&'a super::EmbeddedPredictionTensor<B::Tensor>> {
        None
    }
}

/// Prepared immutable input and exact identity retained until source creation
/// inside the existing source guard. The native owner remains with the lending
/// input adapter for the complete synchronous driver call.
pub trait PredictionPrefillPlan<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    /// Guard-owned concrete source used by the existing prefill executor.
    type Source: PredictionPrefillSource<A, B, S>;
    /// Returns checked batch and complete prompt extents.
    fn shape(&self) -> Result<[u64; 2], A::Error>;
    /// Caller-selected maximum span width, distinct from model context size.
    fn chunk_positions(&self) -> Option<NonZeroU64>;
    /// Whether the actual retained source needs one complete default invocation.
    /// Explicit chunk policies remain authoritative and are never widened.
    fn requires_whole_input(&self) -> bool {
        false
    }
    /// Exact prepared content binding for this speculative lane.
    fn identity(&self) -> Option<&PreparedInputCacheIdentity>;
    /// Visits the architecture's actual admitted semantic token sequence before
    /// source creation. No tensor is copied and no native authority is granted.
    fn visit_token_parts(&self,
        _visitor: &mut dyn FnMut(crate::composite_execution::PredictionTokenPart<'_, B::Tensor>) -> Result<(), Error>,
    ) -> Result<(), A::Error> where A::Error: From<Error> {
        Err(Error::from(eredu_nn::workspace::WorkspaceMetadataError::Unqualified).into())
    }
    /// Creates the source inside the existing source preparation guard.
    fn into_source(self, geometry: InferenceGeometry) -> Result<Option<Self::Source>, A::Error>;
}

/// Existing ordered native token segments and one exact prepared identity.
pub struct TextPredictionPrefill<T> {
    tokens: PredictionTokens<T>,
    identity: Option<SharedPreparedInputCacheIdentity>,
    chunk: Option<NonZeroU64>,
    metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
}
enum PredictionTokens<T> {
    Owned(Vec<T>),
    Prepared(eredu_runtime::input::PreparedModelInputOwner<T>),
}
impl<T> TextPredictionPrefill<T> {
    /// Retains already prepared owners and their exact requested span width.
    pub fn new(
        tokens: Vec<T>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: Option<NonZeroU64>,
    ) -> Self {
        Self {
            tokens: PredictionTokens::Owned(tokens),
            identity,
            chunk,
            metadata: None,
        }
    }
    /// Shares the actual original input owner; no handle/segment/identity clone.
    /// This is source construction, not authority to execute native span views.
    pub fn from_prepared_owner_with_metadata(
        input: eredu_runtime::input::PreparedModelInputOwner<T>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: Option<NonZeroU64>,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        let metadata = crate::prefill::Metadata::funded(Some(funding));
        metadata.controls::<(Self, eredu_runtime::input::PreparedModelInputOwner<T>)>()?;
        if input.original_source().is_none() || !crate::prefill::is_prepared_token_input(&input) {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        Ok(Self {
            tokens: PredictionTokens::Prepared(input),
            identity,
            chunk,
            metadata: Some(funding.clone()),
        })
    }
}
impl<A, B, S> PredictionPrefillPlan<A, B, S> for TextPredictionPrefill<B::Tensor>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S, Error = Error>,
{
    type Source = PreparedTextPrefill<B::Tensor>;
    /// Returns checked batch and complete prompt extents.
    fn shape(&self) -> Result<[u64; 2], Error> {
        let metadata = crate::prefill::Metadata::funded(self.metadata.as_ref());
        match &self.tokens {
            PredictionTokens::Owned(tokens) => {
                crate::prefill::segmented_token_shape_with_metadata(tokens.iter(), metadata)
            }
            PredictionTokens::Prepared(input) => {
                crate::prefill::segmented_token_shape_with_metadata(
                    input.parts().iter().map(|part| part.payload().value()),
                    metadata,
                )
            }
        }
    }
    /// Caller-selected maximum span width, distinct from model context size.
    fn chunk_positions(&self) -> Option<NonZeroU64> {
        self.chunk
    }
    /// Exact prepared content binding for this speculative lane.
    fn identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.identity.as_ref().map(AsRef::as_ref)
    }
    fn visit_token_parts(&self,
        visitor: &mut dyn FnMut(crate::composite_execution::PredictionTokenPart<'_, B::Tensor>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        use crate::composite_execution::PredictionTokenPart;
        match &self.tokens {
            PredictionTokens::Owned(tokens) => {
                for token in tokens { visitor(PredictionTokenPart::Tokens(token))?; }
            }
            PredictionTokens::Prepared(input) => {
                for part in input.parts() { visitor(PredictionTokenPart::Tokens(part.payload().value()))?; }
            }
        }
        Ok(())
    }
    /// Creates the source inside the existing source preparation guard.
    fn into_source(self, geometry: InferenceGeometry) -> Result<Option<Self::Source>, Error> {
        // Identity belongs to the speculative lane, not the borrowed canonical
        // model session. The lane binds it before the first source is consumed.
        match self.tokens {
            PredictionTokens::Owned(tokens) => PreparedTextPrefill::from_tensors(tokens, geometry),
            PredictionTokens::Prepared(input) => {
                PreparedTextPrefill::from_prepared_owner_with_metadata(
                    input,
                    geometry,
                    self.metadata.as_ref().expect("retained source funding"),
                )
            }
        }
        .map(Some)
    }
}

/// Composite preparation keeps architecture admission and ordered token segments.
pub struct CompositePredictionPrefill<C, T, I> {
    input: eredu_runtime::PreparedModelInput<T>,
    shape: [u64; 2],
    config: C,
    inspector: I,
    identity: Option<SharedPreparedInputCacheIdentity>,
    chunk: Option<NonZeroU64>,
}
impl<C, T, I> CompositePredictionPrefill<C, T, I> {
    /// Retains already prepared owners and their exact requested span width.
    pub fn new(
        input: eredu_runtime::PreparedModelInput<T>,
        shape: [u64; 2],
        config: C,
        inspector: I,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: Option<NonZeroU64>,
    ) -> Self {
        Self {
            input,
            shape,
            config,
            inspector,
            identity,
            chunk,
        }
    }
}
impl<A, B, S, I> PredictionPrefillPlan<PreparedCompositeArchitecture<A>, B, S>
    for CompositePredictionPrefill<A::AdmissionConfig, B::Tensor, I>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: eredu_runtime::PreparedInputInspector<B::Tensor>,
{
    type Source = PreparedCompositeTextPrefill<A::AdmissionConfig, B::Tensor, I>;
    /// Returns checked batch and complete prompt extents.
    fn shape(&self) -> Result<[u64; 2], Error> {
        Ok(self.shape)
    }
    /// Caller-selected maximum span width, distinct from model context size.
    fn chunk_positions(&self) -> Option<NonZeroU64> {
        self.chunk
    }
    /// Exact prepared content binding for this speculative lane.
    fn identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.identity.as_ref().map(AsRef::as_ref)
    }
    /// Creates the source inside the existing source preparation guard.
    fn into_source(self, geometry: InferenceGeometry) -> Result<Option<Self::Source>, Error> {
        PreparedCompositeTextPrefill::from_prepared_text(
            &self.input,
            geometry,
            self.config,
            self.inspector,
        )
    }
}
