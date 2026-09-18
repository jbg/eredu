//! Architecture-owned decoder spans and unchanged whole prepared invocations.

mod admission;
pub(crate) use admission::PrefillAdmission;
mod external;
pub use external::{ExternalPrefillChunk, PreparedExternalPrefill};

mod metadata;
use eredu_core::InferenceGeometry;
use eredu_nn::workspace::WorkspaceContext;
use eredu_nn::{Error, Index, NeuralBackend, Tensor};
use eredu_runtime::{
    PreparedInputCacheIdentity, ReplicatedTextArchitecture, RuntimeState,
    SharedPreparedInputCacheIdentity, prefill::PrefillChunk,
    replicated_session::PreparedPrefillSource,
};
pub(crate) use metadata::Metadata;
use std::sync::Arc;

use crate::composite_execution::{
    CompositeArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
};

/// Inspects architecture-declared batch/decoder axes without materialization.
pub(crate) fn token_shape<T: Tensor>(tokens: &T) -> Result<[u64; 2], Error> {
    token_shape_with_metadata(tokens, Metadata::new(None))
}
fn token_shape_with_metadata<T: Tensor>(
    tokens: &T,
    metadata: Metadata<'_>,
) -> Result<[u64; 2], Error> {
    let [batch, positions] = tokens.shape() else {
        return Err(metadata.error(format_args!(
            "decoder tokens require batch and sequence axes",
        )));
    };
    if *batch <= 0 || *positions <= 0 {
        return Err(metadata.error(format_args!("decoder input dimensions must be positive")));
    }
    Ok([*batch as u64, *positions as u64])
}

/// Sums ordered semantic token segments, preserving a common batch dimension.
pub(crate) fn segmented_token_shape<'a, T: Tensor + 'a>(
    tokens: impl IntoIterator<Item = &'a T>,
) -> Result<[u64; 2], Error> {
    segmented_token_shape_with_metadata(tokens, Metadata::new(None))
}
pub(crate) fn segmented_token_shape_with_metadata<
    'a,
    T: Tensor + 'a,
    I: IntoIterator<Item = &'a T>,
>(
    tokens: I,
    metadata: Metadata<'_>,
) -> Result<[u64; 2], Error> {
    metadata.controls::<(I, I::IntoIter, Option<[u64; 2]>, [u64; 2], Option<&T>)>()?;
    let mut shape: Option<[u64; 2]> = None;
    for tokens in tokens {
        let [batch, positions] = token_shape_with_metadata(tokens, metadata)?;
        match &mut shape {
            Some([expected_batch, total]) if *expected_batch == batch => {
                *total = total.checked_add(positions).ok_or_else(|| {
                    metadata.error(format_args!("decoder input position overflow"))
                })?;
            }
            None => shape = Some([batch, positions]),
            _ => {
                return Err(metadata.error(format_args!(
                    "decoder segments have different batch dimensions",
                )));
            }
        }
    }
    shape.ok_or_else(|| metadata.error(format_args!("decoder input has no token segments")))
}

#[derive(Debug, Clone)]
enum TextTokens<T> {
    Host(Arc<[i32]>),
    Native(T),
    Segments(Vec<T>),
    Prepared(eredu_runtime::input::PreparedModelInputOwner<T>),
}

/// Retained text ingress with explicit decoder coordinates. This does not accept
/// prepared media embeddings: their semantic spans belong to media ingress.
/// Host construction is cold; only the current span becomes a native tensor,
/// inside the session's reservation scope. Existing native tokens/masks must be
/// included in the admission's retained or existing allocation accounting.
#[derive(Debug, Clone)]
pub struct PreparedTextPrefill<T> {
    tokens: TextTokens<T>,
    mask: Option<T>,
    geometry: InferenceGeometry,
    identity: Option<SharedPreparedInputCacheIdentity>,
    // Source construction and native views are separate authorities. This
    // funding retains only the paid host destinations used by shared geometry.
    metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
}

/// Owned input views retained until a decoder chunk completes.
#[derive(Debug)]
pub struct TextPrefillChunk<T> {
    tokens: T,
    mask: Option<T>,
    // The actual native alias above retires before its completed packet/Q/H.
    packet: Option<crate::speculative_execution::EmbeddedPredictionTensor<T>>,
}
impl<T> TextPrefillChunk<T> {
    /// Batch by chunk-position token IDs.
    pub fn tokens(&self) -> &T {
        &self.tokens
    }
    /// Optional attention mask with the current query and visible key positions.
    pub fn mask(&self) -> Option<&T> {
        self.mask.as_ref()
    }
}

impl<T: Tensor> PreparedTextPrefill<T> {
    /// Retains ordered native text segments without concatenating the complete
    /// prompt. Only intersecting slices are concatenated while preparing a span.
    pub fn from_tensors(tokens: Vec<T>, geometry: InferenceGeometry) -> Result<Self, Error> {
        Self::from_tensors_with_metadata(tokens, geometry, Metadata::new(None))
    }

    fn from_tensors_with_metadata(
        mut tokens: Vec<T>,
        geometry: InferenceGeometry,
        metadata: Metadata<'_>,
    ) -> Result<Self, Error> {
        metadata.controls::<(Self, Vec<T>, u64)>()?;
        validate_geometry_with_metadata(geometry, metadata)?;
        validate_token_segments(tokens.iter(), geometry, metadata)?;
        // A single retained tensor needs neither the segment container nor a
        // temporary selected-view Vec for every later span.
        let tokens = if tokens.len() == 1 {
            let token = tokens.pop().expect("one validated text segment");
            drop(tokens);
            TextTokens::Native(token)
        } else {
            TextTokens::Segments(tokens)
        };
        Ok(Self {
            tokens,
            mask: None,
            geometry,
            identity: None,
            metadata: None,
        })
    }

    /// Retains the actual immutable original prepared input without cloning any
    /// tensor handle, segment container, identity descriptor or fingerprint.
    /// Native span operations still require their caller's admitted scope.
    pub fn from_prepared_owner_with_metadata(
        input: eredu_runtime::input::PreparedModelInputOwner<T>,
        geometry: InferenceGeometry,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        let metadata = Metadata::funded(Some(funding));
        metadata.controls::<(Self, eredu_runtime::input::PreparedModelInputOwner<T>)>()?;
        if input.original_source().is_none() {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        validate_geometry_with_metadata(geometry, metadata)?;
        if !is_prepared_token_input(&input) {
            return Err(metadata.error(format_args!("text prefill requires token-ID input")));
        }
        validate_token_segments(
            input.parts().iter().map(|part| part.payload().value()),
            geometry,
            metadata,
        )?;
        Ok(Self {
            tokens: TextTokens::Prepared(input),
            mask: None,
            geometry,
            identity: None,
            metadata: Some(funding.clone()),
        })
    }

    /// Retains row-major host token IDs without allocating or submitting native
    /// work. Each batch row has `geometry.input_positions` positions.
    pub fn from_token_ids(tokens: Arc<[i32]>, geometry: InferenceGeometry) -> Result<Self, Error> {
        validate_geometry(geometry)?;
        let expected = geometry
            .batch_size
            .checked_mul(geometry.input_positions)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::backend("text prefill token count overflow"))?;
        if tokens.len() != expected {
            return Err(Error::backend(
                "text prefill tokens do not match admitted geometry",
            ));
        }
        Ok(Self {
            tokens: TextTokens::Host(tokens),
            mask: None,
            geometry,
            identity: None,
            metadata: None,
        })
    }

    /// Retains existing native text input. A supplied mask uses the conventional
    /// last two axes `[query, key]`, with optional leading broadcast/batch axes.
    /// Query extent is one or the full new prompt; key extent is one or the
    /// cached prefix plus prompt. All masks are validated before creating views.
    pub fn from_tensor(
        tokens: T,
        mask: Option<T>,
        geometry: InferenceGeometry,
    ) -> Result<Self, Error> {
        validate_geometry(geometry)?;
        if tokens.shape() != [geometry.batch_size as i32, geometry.input_positions as i32] {
            return Err(Error::backend(
                "text prefill tokens do not match admitted geometry",
            ));
        }
        if let Some(mask) = &mask {
            let shape = mask.shape();
            if shape.len() < 2 || shape.iter().any(|&n| n <= 0) {
                return Err(Error::backend(
                    "text prefill mask requires explicit query/key axes",
                ));
            }
            let query = shape[shape.len() - 2] as u64;
            let key = shape[shape.len() - 1] as u64;
            if (query != 1 && query != geometry.input_positions)
                || (key != 1 && key != geometry.cached_positions + geometry.input_positions)
            {
                return Err(Error::backend(
                    "text prefill mask does not match admitted positions",
                ));
            }
        }
        Ok(Self {
            tokens: TextTokens::Native(tokens),
            mask,
            geometry,
            identity: None,
            metadata: None,
        })
    }

    /// Installs the identity of the complete prepared prompt after its final span
    /// commits. Intermediate chunks never claim that complete prompt identity.
    pub fn with_cache_identity(self, identity: PreparedInputCacheIdentity) -> Self {
        self.with_shared_cache_identity(SharedPreparedInputCacheIdentity::new(identity))
    }

    /// Retains the existing immutable prompt identity and its attached custody
    /// without copying descriptions or recomputing fingerprints.
    pub fn with_shared_cache_identity(
        mut self,
        identity: SharedPreparedInputCacheIdentity,
    ) -> Self {
        self.identity = Some(identity);
        self
    }

    /// Prepares one validated semantic span. Runtime calls this only while the
    /// corresponding request's native reservation scope is active.
    pub fn prepare_span(
        &self,
        chunk: &PrefillChunk,
        context: &T::Context,
    ) -> Result<TextPrefillChunk<T>, Error> {
        self.prepare_span_with_metadata(chunk, context, Metadata::funded(self.metadata.as_ref()))
    }

    pub(crate) fn original_prepared_input(
        &self,
    ) -> Option<&eredu_runtime::input::PreparedModelInputOwner<T>> {
        match &self.tokens {
            TextTokens::Prepared(input) => Some(input),
            _ => None,
        }
    }
    fn validate_span(&self, chunk: &PrefillChunk, metadata: Metadata<'_>) -> Result<(), Error> {
        let g = self.geometry;
        if chunk.input.start >= chunk.input.end
            || chunk.input.end > g.input_positions
            || chunk.input.end - chunk.input.start > g.prefill_chunk_positions
            || chunk.position != g.cached_positions + chunk.input.start
            || chunk.output != g.output.for_chunk(chunk.input.end == g.input_positions)
        {
            return Err(metadata.error(format_args!(
                "text prefill span does not match admitted coordinates",
            )));
        }
        Ok(())
    }
    fn validate_completed_span(&self, chunk: &PrefillChunk) -> Result<(), Error> {
        let metadata = Metadata::funded(self.metadata.as_ref());
        metadata.controls::<(&Self, &PrefillChunk)>()?;
        if self.original_prepared_input().is_none() || self.mask.is_some() {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        self.validate_span(chunk, metadata)
    }
    fn prepare_completed_span(
        &self,
        chunk: &PrefillChunk,
        supplied: (T, crate::speculative_execution::EmbeddedPredictionTensor<T>),
    ) -> Result<TextPrefillChunk<T>, Error> {
        let metadata = Metadata::funded(self.metadata.as_ref());
        metadata.controls::<(
            TextPrefillChunk<T>,
            (T, crate::speculative_execution::EmbeddedPredictionTensor<T>),
        )>()?;
        self.validate_span(chunk, metadata)?;
        if self.original_prepared_input().is_none()
            || self.mask.is_some()
            || supplied.1.evidence().is_none()
            || supplied.0.shape()
                != [
                    self.geometry.batch_size as i32,
                    (chunk.input.end - chunk.input.start) as i32,
                ]
            || supplied.0.shape() != supplied.1.shape()
        {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        let (tokens, packet) = supplied;
        Ok(TextPrefillChunk {
            tokens,
            mask: None,
            packet: Some(packet),
        })
    }

    fn prepare_span_with_metadata(
        &self,
        chunk: &PrefillChunk,
        context: &T::Context,
        metadata: Metadata<'_>,
    ) -> Result<TextPrefillChunk<T>, Error> {
        metadata.controls::<(TextPrefillChunk<T>, [Index; 2], u64, i32, i32)>()?;
        let g = self.geometry;
        self.validate_span(chunk, metadata)?;
        let start = chunk.input.start as i32;
        let end = chunk.input.end as i32;
        let tokens = match &self.tokens {
            TextTokens::Segments(segments) => {
                slice_segments(segments.iter(), start, end, context, metadata)?
            }
            TextTokens::Prepared(input) => slice_segments(
                input.parts().iter().map(|part| part.payload().value()),
                start,
                end,
                context,
                metadata,
            )?,
            TextTokens::Native(tokens) => {
                tokens.index(&[Index::Full, Index::Range(start, end)], context)?
            }
            TextTokens::Host(tokens) => {
                let shape = [g.batch_size as i32, end - start];
                if g.batch_size == 1 {
                    // One row is already contiguous in the retained host owner.
                    T::from_i32_slice(&tokens[start as usize..end as usize], &shape, context)?
                } else if start == 0 && end as u64 == g.input_positions {
                    // A full span preserves the existing row-major layout.
                    T::from_i32_slice(tokens, &shape, context)?
                } else {
                    // Narrow multi-row spans still need row-preserving packing.
                    let count = (g.batch_size as usize)
                        .checked_mul((end - start) as usize)
                        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    let mut values = metadata.vector(count)?;
                    for row in tokens.chunks_exact(g.input_positions as usize) {
                        values.extend_from_slice(&row[start as usize..end as usize]);
                    }
                    T::from_i32_slice(&values, &shape, context)?
                }
            }
        };
        let mask = self
            .mask
            .as_ref()
            .map(|mask| {
                let rank = mask.shape().len();
                let mut indices = metadata.vector(rank)?;
                indices.resize(rank, Index::Full);
                if mask.shape()[rank - 2] != 1 {
                    indices[rank - 2] = Index::Range(start, end);
                }
                if mask.shape()[rank - 1] != 1 {
                    indices[rank - 1] =
                        Index::Range(0, (g.cached_positions + chunk.input.end) as i32);
                }
                mask.index(&indices, context)
            })
            .transpose()?;
        Ok(TextPrefillChunk {
            tokens,
            mask,
            packet: None,
        })
    }
}

fn validate_token_segments<'a, T: Tensor + 'a, I: Iterator<Item = &'a T>>(
    tokens: I,
    geometry: InferenceGeometry,
    metadata: Metadata<'_>,
) -> Result<(), Error> {
    metadata.controls::<(I, InferenceGeometry, u64, Option<&T>, [i32; 2])>()?;
    let mut positions = 0u64;
    for token in tokens {
        let [batch, sequence] = token.shape() else {
            return Err(metadata.error(format_args!(
                "text segment requires batch and sequence axes",
            )));
        };
        if *batch as u64 != geometry.batch_size || *sequence <= 0 {
            return Err(metadata.error(format_args!(
                "text segment differs from admitted batch geometry",
            )));
        }
        positions = positions
            .checked_add(*sequence as u64)
            .ok_or_else(|| metadata.error(format_args!("text segment position overflow")))?;
    }
    if positions != geometry.input_positions {
        return Err(metadata.error(format_args!(
            "text segments differ from admitted prompt positions",
        )));
    }
    Ok(())
}

fn slice_segments<'a, T: Tensor + 'a, I: ExactSizeIterator<Item = &'a T>>(
    mut segments: I,
    start: i32,
    end: i32,
    context: &T::Context,
    metadata: Metadata<'_>,
) -> Result<T, Error> {
    metadata.controls::<(I, i32, i32, i32, i32, i32, Vec<T>, [Index; 2], Option<T>)>()?;
    if segments.len() == 1 {
        return segments
            .next()
            .expect("one retained text segment")
            .index(&[Index::Full, Index::Range(start, end)], context);
    }
    let mut offset = 0i32;
    let mut selected = metadata.vector(segments.len())?;
    for segment in segments {
        let segment_end = offset + segment.shape()[1];
        let left = start.max(offset);
        let right = end.min(segment_end);
        if left < right {
            selected.push(segment.index(
                &[Index::Full, Index::Range(left - offset, right - offset)],
                context,
            )?);
        }
        offset = segment_end;
        if offset >= end {
            break;
        }
    }
    if selected.len() == 1 {
        Ok(selected.pop().expect("one text slice"))
    } else {
        T::concatenate(&selected, 1, context)
    }
}

fn validate_geometry(geometry: InferenceGeometry) -> Result<(), Error> {
    validate_geometry_with_metadata(geometry, Metadata::new(None))
}

fn validate_geometry_with_metadata(
    geometry: InferenceGeometry,
    metadata: Metadata<'_>,
) -> Result<(), Error> {
    // Preserve the ordinary diagnostic adapter, and use the same fixed
    // validator without constructing an intermediate owned cause when counted.
    if metadata.checked() {
        geometry
            .validate_fixed()
            .map_err(|cause| metadata.source(cause))?;
    } else {
        geometry
            .validate()
            .map_err(|error| Error::backend(error.to_string()))?;
    }
    if geometry.batch_size > i32::MAX as u64
        || geometry.cached_positions + geometry.input_positions > i32::MAX as u64
    {
        return Err(metadata.error(format_args!(
            "text prefill geometry exceeds tensor index range",
        )));
    }
    Ok(())
}

impl<A, B, S> PreparedPrefillSource<A, B, S> for PreparedTextPrefill<B::Tensor>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S, Error = Error>,
{
    type Chunk = TextPrefillChunk<B::Tensor>;
    fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(
        &self,
        chunk: &PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, Error> {
        self.prepare_span(chunk, context)
    }
    fn input<'a>(&'a self, chunk: &'a Self::Chunk) -> A::Input<'a> {
        A::text_input(&chunk.tokens, chunk.mask.as_ref())
    }
    fn shared_cache_identity(&self) -> Option<SharedPreparedInputCacheIdentity> {
        self.identity.clone()
    }
}

/// Text-only decoder spans for a composite architecture. Each span uses that
/// architecture's ordinary prepared-input admission, so its decoder coordinates
/// and optional execution groups keep their existing semantics. The constructor
/// accepts token IDs only; media encoders need a separate retained ingress.
pub struct PreparedCompositeTextPrefill<C, T, I> {
    text: PreparedTextPrefill<T>,
    config: C,
    inspector: I,
    // Retire all paid source destinations before their continuation account.
    metadata: Option<WorkspaceContext>,
}

/// Tests the exact token-only payload contract used by the composite text
/// source, without cloning input or allocating a source. This is source
/// selection, not a model-family capability or a media geometry substitution.
pub fn is_prepared_token_input<T>(input: &eredu_runtime::PreparedModelInput<T>) -> bool {
    input.parts().iter().all(|part| {
        matches!(
            (part.modality(), part.payload()),
            (
                eredu_core::InputModality::Text,
                eredu_runtime::PreparedInputPayload::TokenIds(_)
            )
        )
    })
}

impl<C, T: Tensor, I> PreparedCompositeTextPrefill<C, T, I> {
    /// Adapts an existing prepared token-only request. Other semantic payloads
    /// require their own retained ingress and return `None` without slicing them.
    pub fn from_prepared_text(
        input: &eredu_runtime::PreparedModelInput<T>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
    ) -> Result<Option<Self>, Error> {
        if !is_prepared_token_input(input) {
            return Ok(None);
        }
        let tokens = input
            .parts()
            .iter()
            .map(|part| match part.payload() {
                eredu_runtime::PreparedInputPayload::TokenIds(value) => value.clone(),
                _ => unreachable!("token-only source predicate was checked"),
            })
            .collect();
        Self::new(
            PreparedTextPrefill::from_tensors(tokens, geometry)?,
            config,
            inspector,
        )
        .map(Some)
    }

    /// Retains original input segments without cloning native handles or maps.
    pub fn from_prepared_owner_with_metadata(
        input: eredu_runtime::input::PreparedModelInputOwner<T>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
        context: &WorkspaceContext,
    ) -> Result<Option<Self>, Error> {
        if !is_prepared_token_input(&input) {
            return Ok(None);
        }
        let funding = context
            .metadata_funding()
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
        let text =
            PreparedTextPrefill::from_prepared_owner_with_metadata(input, geometry, &funding)?;
        Self::new_with_metadata(text, config, inspector, Some(context)).map(Some)
    }

    /// Consumes the actual prepared token handles through the same text
    /// geometry worker. The caller retains independent funding with escaped
    /// errors/outputs; this source owns it through all chunk preparation.
    pub fn from_prepared_text_with_metadata(
        input: eredu_runtime::PreparedModelInput<T>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
        context: &WorkspaceContext,
    ) -> Result<Option<Self>, Error> {
        let metadata = Metadata::new(Some(context));
        metadata.controls::<(Self, Option<Self>, eredu_runtime::PreparedModelInput<T>)>()?;
        if !is_prepared_token_input(&input) {
            return Ok(None);
        }
        let mut tokens = metadata.vector(input.len())?;
        for part in input.into_parts() {
            match part.into_payload() {
                eredu_runtime::PreparedInputPayload::TokenIds(value) => tokens.push(value),
                _ => unreachable!("token-only source predicate was checked"),
            }
        }
        Self::new_with_metadata(
            PreparedTextPrefill::from_tensors_with_metadata(tokens, geometry, metadata)?,
            config,
            inspector,
            Some(context),
        )
        .map(Some)
    }

    /// Retains host tokens and cold admission configuration. Native allocation
    /// and architecture admission happen inside the per-chunk retention scope.
    pub fn from_token_ids(
        tokens: Arc<[i32]>,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
    ) -> Result<Self, Error> {
        Self::new(
            PreparedTextPrefill::from_token_ids(tokens, geometry)?,
            config,
            inspector,
        )
    }

    /// Retains an existing native token tensor, charged as retained or existing
    /// storage by the caller. Composite masks belong to prepared-input metadata.
    pub fn from_tensor(
        tokens: T,
        geometry: InferenceGeometry,
        config: C,
        inspector: I,
    ) -> Result<Self, Error> {
        Self::new(
            PreparedTextPrefill::from_tensor(tokens, None, geometry)?,
            config,
            inspector,
        )
    }

    fn new(text: PreparedTextPrefill<T>, config: C, inspector: I) -> Result<Self, Error> {
        Self::new_with_metadata(text, config, inspector, None)
    }

    fn new_with_metadata(
        text: PreparedTextPrefill<T>,
        config: C,
        inspector: I,
        context: Option<&WorkspaceContext>,
    ) -> Result<Self, Error> {
        let metadata = Metadata::new(context);
        metadata.controls::<Self>()?;
        // This is the existing composite admission constraint, checked while
        // still cold rather than after allocating the first native span.
        if text.geometry.batch_size != 1 {
            return Err(metadata.error(format_args!(
                "composite prepared input requires one sequence per request",
            )));
        }
        Ok(Self {
            text,
            config,
            inspector,
            metadata: context.cloned(),
        })
    }

    /// Carries the full prompt identity to the final committed span only.
    pub fn with_cache_identity(self, identity: PreparedInputCacheIdentity) -> Self {
        self.with_shared_cache_identity(SharedPreparedInputCacheIdentity::new(identity))
    }

    /// Carries the existing shared prompt identity through the final span without
    /// cloning its payload or rebuilding its semantic fingerprint.
    pub fn with_shared_cache_identity(
        mut self,
        identity: SharedPreparedInputCacheIdentity,
    ) -> Self {
        self.text = self.text.with_shared_cache_identity(identity);
        self
    }
}

/// Owned prepared tokens and the exact architecture admission derived from them.
pub struct CompositeTextPrefillChunk<T, P> {
    prepared: eredu_runtime::PreparedModelInput<T>,
    admitted: crate::media_plan::AdmittedCompositeInput<P>,
    metadata: Option<WorkspaceContext>,
    packet: Option<crate::speculative_execution::EmbeddedPredictionTensor<T>>,
}

impl<A, B, S, I> PreparedPrefillSource<PreparedCompositeArchitecture<A>, B, S>
    for PreparedCompositeTextPrefill<A::AdmissionConfig, B::Tensor, I>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: eredu_runtime::PreparedInputInspector<B::Tensor>,
{
    type Chunk = CompositeTextPrefillChunk<B::Tensor, A::InputPartPlan>;

    fn geometry(&self) -> InferenceGeometry {
        self.text.geometry
    }

    fn prepare_chunk(
        &self,
        chunk: &PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, Error> {
        prepare_composite_chunk::<A, B, S, I>(self, chunk, context, None)
    }

    fn input<'a>(
        &'a self,
        chunk: &'a Self::Chunk,
    ) -> PreparedCompositeInput<'a, B::Tensor, A::InputPartPlan> {
        // Both fields are private and prepared together from this exact span.
        PreparedCompositeInput::new(&chunk.prepared, &chunk.admitted)
            .expect("composite text chunk retains its exact admission")
            .with_metadata_loan(chunk.metadata.as_ref())
    }

    fn shared_cache_identity(&self) -> Option<SharedPreparedInputCacheIdentity> {
        self.text.identity.clone()
    }
}

fn prepare_composite_chunk<A, B, S, I>(
    source: &PreparedCompositeTextPrefill<A::AdmissionConfig, B::Tensor, I>,
    chunk: &PrefillChunk,
    context: &<B::Tensor as Tensor>::Context,
    supplied: Option<(
        B::Tensor,
        crate::speculative_execution::EmbeddedPredictionTensor<B::Tensor>,
    )>,
) -> Result<CompositeTextPrefillChunk<B::Tensor, A::InputPartPlan>, Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: eredu_runtime::PreparedInputInspector<B::Tensor>,
{
    let metadata = Metadata::new(source.metadata.as_ref());
    metadata.controls::<(
        CompositeTextPrefillChunk<B::Tensor, A::InputPartPlan>,
        eredu_runtime::PreparedInputPart<B::Tensor>,
        Option<(
            B::Tensor,
            crate::speculative_execution::EmbeddedPredictionTensor<B::Tensor>,
        )>,
        Option<crate::speculative_execution::EmbeddedPredictionTensor<B::Tensor>>,
        &PreparedCompositeTextPrefill<A::AdmissionConfig, B::Tensor, I>,
        &PrefillChunk,
        &<B::Tensor as Tensor>::Context,
    )>()?;
    // Declare custody before all raw aliases so failed admission/unwind
    // retires the part vector and tensor before its completed packet/Q/H.
    let mut packet = None;
    // Reserve the owned part row before the native span operation.
    let mut parts = metadata.vector(1)?;
    let tokens = match supplied {
        Some(supplied) => {
            let prepared = source.text.prepare_completed_span(chunk, supplied)?;
            {
                packet = prepared.packet;
                prepared.tokens
            }
        }
        None => {
            let prepared = source
                .text
                .prepare_span_with_metadata(chunk, context, metadata)?;
            prepared.tokens
        }
    };
    let part = eredu_runtime::PreparedInputPart::new(
        eredu_core::InputModality::Text,
        eredu_runtime::PreparedInputPayload::TokenIds(tokens),
        [],
    )
    .map_err(|cause| metadata.source(cause))?;
    parts.push(part);
    let prepared = match metadata.context() {
        Some(context) => {
            eredu_runtime::PreparedModelInput::new_with_metadata(parts, context, |tensor| {
                source.inspector.identity_with_metadata(tensor, context)
            })?
        }
        None => eredu_runtime::PreparedModelInput::new(parts, |tensor| {
            source.inspector.identity(tensor)
        })
        .map_err(Error::backend_retained_source)?,
    };
    let admitted = match metadata.context() {
        Some(context) => A::admit_prepared_input_with_metadata(
            &source.config,
            &prepared,
            &source.inspector,
            context,
        )?,
        None => A::admit_prepared_input(&source.config, &prepared, &source.inspector)
            .map_err(Error::backend_retained_source)?,
    };
    if admitted.decoder_shape()
        != [
            source.text.geometry.batch_size,
            chunk.input.end - chunk.input.start,
        ]
    {
        return Err(metadata.error(format_args!(
            "composite text admission changed the scheduled decoder coordinates",
        )));
    }
    if let Some(context) = metadata.context() {
        // Validate and pay the later infallible borrowed-input view while
        // refusal can still propagate from this chunk constructor.
        PreparedCompositeInput::new_with_metadata(&prepared, &admitted, context)?;
    }
    Ok(CompositeTextPrefillChunk {
        prepared,
        admitted,
        metadata: source.metadata.clone(),
        packet,
    })
}

#[cfg(test)]
mod identity_tests;

impl<A, B, S> crate::speculative_execution::PredictionPrefillSource<A, B, S>
    for PreparedTextPrefill<B::Tensor>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S, Error = Error>,
{
    fn original_prepared_input(
        &self,
    ) -> Option<&eredu_runtime::input::PreparedModelInputOwner<B::Tensor>> {
        self.original_prepared_input()
    }
    fn validate_completed_token_span(&self, chunk: &PrefillChunk) -> Result<(), Error> {
        self.validate_completed_span(chunk)
    }
    fn prepare_chunk_from_completed_tokens(
        &self,
        chunk: &PrefillChunk,
        supplied: (
            B::Tensor,
            crate::speculative_execution::EmbeddedPredictionTensor<B::Tensor>,
        ),
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, Error> {
        self.prepare_completed_span(chunk, supplied)
    }
    fn completed_token_packet<'a>(
        &self,
        chunk: &'a Self::Chunk,
    ) -> Option<&'a crate::speculative_execution::EmbeddedPredictionTensor<B::Tensor>> {
        chunk.packet.as_ref()
    }
    fn tokens<'a>(&self, chunk: &'a Self::Chunk) -> &'a B::Tensor {
        &chunk.tokens
    }
}
impl<A, B, S, I>
    crate::speculative_execution::PredictionPrefillSource<PreparedCompositeArchitecture<A>, B, S>
    for PreparedCompositeTextPrefill<A::AdmissionConfig, B::Tensor, I>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    I: eredu_runtime::PreparedInputInspector<B::Tensor>,
{
    fn original_prepared_input(
        &self,
    ) -> Option<&eredu_runtime::input::PreparedModelInputOwner<B::Tensor>> {
        self.text.original_prepared_input()
    }
    fn validate_completed_token_span(&self, chunk: &PrefillChunk) -> Result<(), Error> {
        self.text.validate_completed_span(chunk)
    }
    fn prepare_chunk_from_completed_tokens(
        &self,
        chunk: &PrefillChunk,
        supplied: (
            B::Tensor,
            crate::speculative_execution::EmbeddedPredictionTensor<B::Tensor>,
        ),
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, Error> {
        prepare_composite_chunk::<A, B, S, I>(self, chunk, context, Some(supplied))
    }
    fn completed_token_packet<'a>(
        &self,
        chunk: &'a Self::Chunk,
    ) -> Option<&'a crate::speculative_execution::EmbeddedPredictionTensor<B::Tensor>> {
        chunk.packet.as_ref()
    }
    fn tokens<'a>(&self, chunk: &'a Self::Chunk) -> &'a B::Tensor {
        // The constructor above makes exactly one TokenIds part from the same
        // chunk tensor; no projection or new tensor occurs at this borrow.
        match chunk.prepared.parts()[0].payload() {
            eredu_runtime::PreparedInputPayload::TokenIds(tokens) => tokens,
            _ => unreachable!("private token-only chunk construction"),
        }
    }
}
