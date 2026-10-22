//! Typed prepared-input ingress for replicated composite architectures.

use eredu_core::AttentionPolicy;
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{LayeredArchitecture, LayeredForwardState, ParallelLayeredArchitecture};

use crate::media_plan::AdmittedCompositeInput;

mod prefill;
pub(crate) use prefill::DecoderPrefillChunk;
pub use prefill::DecoderPrefillRequest;

/// One request-bounded tensor collective in a composite pipeline wave.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CompositeTensorCollective {
    /// Sum one tensor with this exact logical shape across the selected TP group.
    Sum {
        /// Exact logical shape of the active or zero-work tensor.
        shape: Vec<i32>,
    },
    /// Gather rank-local slices into an exact complete tensor shape.
    Gather {
        /// Complete output shape; the submitted axis uses this rank's count.
        shape: Vec<i32>,
        /// Exact per-rank counts on the selected axis.
        counts: Vec<usize>,
        /// Concatenation axis.
        axis: usize,
    },
}

/// Exact external-assistant capture requested from one ordinary target pass.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExternalPredictionCaptureRequest {
    /// Final decoder hidden state and every architecture-published shared K/V class.
    Gemma4SharedAttention {
        /// Exact final decoder unit-output observation path.
        final_hidden_path: String,
    },
    /// Ordered Muse decoder-unit outputs consumed by a trained assistant.
    Muse {
        /// Exact zero-based decoder layers, in assistant encoder order.
        target_layers: Box<[usize]>,
        /// Exact unit-output observation paths in the same order.
        target_paths: Box<[String]>,
    },
}

/// Architecture-owned values captured from one committed ordinary target pass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExternalPredictionTargetCapture<T> {
    /// Gemma target hidden state plus shared attention publications.
    Gemma4 {
        /// Final decoder activation before vocabulary projection.
        hidden: T,
        /// Shared K/V values keyed by their exact attention policy.
        shared_kv: Vec<(AttentionPolicy, T, T)>,
    },
    /// Muse decoder outputs in the requested target-layer order.
    Muse {
        /// Exact ordered target-layer activations.
        target_states: Vec<T>,
    },
}

/// Target-owned static operation needed by an external assistant.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExternalPredictionTargetOperation<'a, T> {
    /// Applies the ordinary target token embedding to assistant proposal IDs.
    TokenEmbeddings(&'a T),
    /// Applies the ordinary target vocabulary projection to assistant states.
    ProjectLogits(&'a T),
    /// Looks up the target token table without input normalization or scaling.
    RawTokenEmbeddings(&'a T),
    /// Applies the target vocabulary head without final normalization, scaling or capping.
    /// Tensor-parallel vocabulary shards are gathered before returning.
    RawVocabularyProjection(&'a T),
}

/// Semantic identity of a target-owned static equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExternalPredictionTargetOperationKind {
    /// Ordinary normalized token embeddings.
    TokenEmbeddings,
    /// Ordinary final-normalized and transformed readout.
    ProjectLogits,
    /// Raw token table lookup.
    RawTokenEmbeddings,
    /// Raw vocabulary matrix product.
    RawVocabularyProjection,
}
impl ExternalPredictionTargetOperationKind {
    /// Whether this equation consumes token IDs rather than decoder hidden states.
    pub const fn is_lookup(self) -> bool {
        matches!(self, Self::TokenEmbeddings | Self::RawTokenEmbeddings)
    }
}
impl<T> ExternalPredictionTargetOperation<'_, T> {
    /// Stable equation identity independent of native tensor representation.
    pub const fn kind(&self) -> ExternalPredictionTargetOperationKind {
        match self {
            Self::TokenEmbeddings(_) => ExternalPredictionTargetOperationKind::TokenEmbeddings,
            Self::ProjectLogits(_) => ExternalPredictionTargetOperationKind::ProjectLogits,
            Self::RawTokenEmbeddings(_) => {
                ExternalPredictionTargetOperationKind::RawTokenEmbeddings
            }
            Self::RawVocabularyProjection(_) => {
                ExternalPredictionTargetOperationKind::RawVocabularyProjection
            }
        }
    }
    /// Request tensor whose side-effect-free geometry resolves the static plan.
    pub fn input(&self) -> &T {
        match self {
            Self::TokenEmbeddings(v)
            | Self::ProjectLogits(v)
            | Self::RawTokenEmbeddings(v)
            | Self::RawVocabularyProjection(v) => v,
        }
    }
}

impl<T> Copy for ExternalPredictionTargetOperation<'_, T> {}
impl<T> Clone for ExternalPredictionTargetOperation<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl CompositeTensorCollective {
    /// Exact sum shape or complete gathered output shape.
    pub fn shape(&self) -> &[i32] {
        match self {
            Self::Sum { shape } | Self::Gather { shape, .. } => shape,
        }
    }
}

/// Embedding sums for independently looked-up token segments. Media positions
/// are supplied by their own ingress and must not enlarge these native waves.
pub(crate) fn segmented_token_ingress_collectives(
    positions: impl IntoIterator<Item = u64>,
    hidden_width: i32,
    tensor_partitions: usize,
) -> Result<Option<Vec<CompositeTensorCollective>>, String> {
    if tensor_partitions <= 1 {
        return Ok(None);
    }
    positions
        .into_iter()
        .map(|positions| {
            i32::try_from(positions)
                .map(|positions| CompositeTensorCollective::Sum {
                    shape: vec![1, positions, hidden_width],
                })
                .map_err(|_| "composite token ingress positions exceed i32".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// One typed component invocation, either native low-level input or admitted prepared input.
pub enum CompositeModelInput<'a, T, P, N> {
    /// Family-native component operation.
    Native(N),
    /// Prepared tensors coupled to their exact architecture admission.
    Prepared(PreparedCompositeInput<'a, T, P>),
}
impl<'a, T, P, N> From<N> for CompositeModelInput<'a, T, P, N> {
    fn from(input: N) -> Self {
        Self::Native(input)
    }
}

/// Exact prepared tensors paired with their architecture-owned admission proof.
pub struct PreparedCompositeInput<'a, T, P> {
    prepared: &'a eredu_runtime::PreparedModelInput<T>,
    admitted: &'a AdmittedCompositeInput<P>,
    refinement: Option<Option<&'a T>>,
    invocation: Option<eredu_core::decoding::AttentionInvocation>,
    decoder_chunk: Option<(usize, usize, Option<&'a T>)>,
}

impl<T, P> Clone for PreparedCompositeInput<'_, T, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, P> Copy for PreparedCompositeInput<'_, T, P> {}

impl<'a, T, P> PreparedCompositeInput<'a, T, P> {
    /// Couples exact prepared tensors to an admission derived from those tensors.
    pub fn new(
        prepared: &'a eredu_runtime::PreparedModelInput<T>,
        admitted: &'a AdmittedCompositeInput<P>,
    ) -> Result<Self, String> {
        if prepared.identity() != admitted.identity() {
            return Err("prepared composite input identity differs from admission".into());
        }
        if prepared.len() != admitted.parts().len() {
            return Err("prepared composite part count differs from admission".into());
        }
        Ok(Self {
            prepared,
            admitted,
            refinement: None,
            invocation: None,
            decoder_chunk: None,
        })
    }

    /// Selects a tentative-block invocation over already admitted token rows.
    /// The outer option distinguishes context from refinement without feedback.
    pub fn with_refinement(mut self, feedback: Option<&'a T>) -> Self {
        self.refinement = Some(feedback);
        self
    }

    /// Feedback authority for an explicit tentative-block invocation.
    pub fn refinement(&self) -> Option<Option<&'a T>> {
        self.refinement
    }

    /// Supplies explicit request-local attention and adapter semantics to a
    /// shared-backbone family. Admission and prepared handles remain unchanged.
    pub fn with_attention_invocation(
        mut self,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> Self {
        self.invocation = Some(invocation);
        self
    }

    /// Explicit invocation authority, when supplied by the generation driver.
    pub fn attention_invocation(&self) -> Option<eredu_core::decoding::AttentionInvocation> {
        self.invocation
    }

    pub(crate) fn decoder_chunk(&self) -> Option<DecoderPrefillChunk<'a, T>> {
        self.decoder_chunk
            .map(|(start, end, assembled)| DecoderPrefillChunk {
                range: start..end,
                assembled,
            })
    }

    /// Exact backend-native prepared tensor handles.
    pub const fn prepared(&self) -> &'a eredu_runtime::PreparedModelInput<T> {
        self.prepared
    }

    /// Exact ordered architecture admission.
    pub const fn admitted(&self) -> &'a AdmittedCompositeInput<P> {
        self.admitted
    }

    /// Derives prompt-cache identity from the exact prepared input paired with this admission.
    pub fn cache_identity(
        &self,
        semantic_content_fingerprint: impl Into<String>,
    ) -> Result<
        eredu_runtime::PreparedInputCacheIdentity,
        eredu_runtime::PreparedInputCacheIdentityError,
    > {
        self.prepared.cache_identity(semantic_content_fingerprint)
    }
}

/// Builds only semantic token parts from an exact admission. The architecture
/// supplies placeholder identities; media tensors are neither copied nor encoded.
pub(crate) fn prepared_token_parts<T: Tensor, P>(
    input: PreparedCompositeInput<'_, T, P>,
    context: &T::Context,
    placeholder: impl Fn(&P) -> Option<(u32, u64)>,
) -> Result<Vec<T>, eredu_nn::Error> {
    input
        .prepared()
        .parts()
        .iter()
        .zip(input.admitted().parts())
        .map(|(part, plan)| {
            if let Some((token, positions)) = placeholder(plan) {
                let token = i32::try_from(token).map_err(eredu_nn::Error::backend)?;
                let positions = i32::try_from(positions).map_err(eredu_nn::Error::backend)?;
                T::full_i32(token, &[1, positions], context)
            } else {
                match (part.modality(), part.payload()) {
                    (
                        eredu_core::InputModality::Text,
                        eredu_runtime::PreparedInputPayload::TokenIds(tokens),
                    ) => Ok(tokens.clone()),
                    _ => Err(eredu_nn::Error::backend(
                        "composite prediction input has no declared token identity",
                    )),
                }
            }
        })
        .collect()
}

/// One indivisible prepared composite invocation. Architectures that have no
/// chunk continuation contract retain their existing whole-request semantics.
/// Clones share the immutable prepared tensors and admission rather than copying
/// payloads; observation and state-copy budgets remain with the session.
#[derive(Clone)]
pub struct SingleCompositePrefillRequest<T, P> {
    input: std::sync::Arc<(
        eredu_runtime::PreparedModelInput<T>,
        AdmittedCompositeInput<P>,
    )>,
    tokens: usize,
}

impl<T, P> SingleCompositePrefillRequest<T, P> {
    /// Couples owned prepared tensors to their exact admission.
    pub fn new(
        prepared: eredu_runtime::PreparedModelInput<T>,
        admitted: AdmittedCompositeInput<P>,
    ) -> Result<Self, eredu_nn::Error> {
        PreparedCompositeInput::new(&prepared, &admitted).map_err(eredu_nn::Error::backend)?;
        let tokens =
            usize::try_from(admitted.decoder_positions()).map_err(eredu_nn::Error::backend)?;
        Ok(Self {
            input: std::sync::Arc::new((prepared, admitted)),
            tokens,
        })
    }
}

impl<A, B, S> eredu_runtime::prefill::ChunkedPrefillRequest<A, B, S>
    for SingleCompositePrefillRequest<B::Tensor, A::InputPartPlan>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S>,
    A::Error: From<eredu_nn::Error>,
{
    type Continuation = ();
    fn token_count(&self) -> usize {
        self.tokens
    }
    fn maximum_chunk_tokens(&self) -> usize {
        self.tokens
    }
    fn with_chunk<R>(
        &self,
        range: std::ops::Range<usize>,
        _continuation: Option<&Self::Continuation>,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> R,
    ) -> Result<R, A::Error> {
        if range != (0..self.tokens) {
            return Err(eredu_nn::Error::backend(
                "composite request has no partial invocation contract",
            )
            .into());
        }
        let paired = PreparedCompositeInput::new(&self.input.0, &self.input.1)
            .map_err(eredu_nn::Error::backend)?;
        Ok(operation(A::prepared_input(paired)))
    }
    fn request_values(&self) -> Vec<&B::Tensor> {
        self.input
            .0
            .storage_values()
            .map(|(_, value)| value)
            .collect()
    }
    fn continuation(&self, _completed: &A::ForwardContext) {}
    fn retained_values<'a>(&self, _continuation: &'a ()) -> Vec<&'a B::Tensor> {
        Vec::new()
    }
}

/// Request-specific encoder geometry at an intra-group pipeline cut.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CompositeContinuationGeometry {
    /// Every internal cut has the same frame count and width.
    Uniform {
        /// Prepared encoder frames.
        sequence: i32,
        /// Encoder activation width.
        width: i32,
    },
    /// Shape after each source unit, including reductions between encoder stages.
    AfterUnits(Vec<(i32, i32)>),
}

impl From<(i32, i32)> for CompositeContinuationGeometry {
    fn from((sequence, width): (i32, i32)) -> Self {
        Self::Uniform { sequence, width }
    }
}

impl CompositeContinuationGeometry {
    /// Resolves the exact shape after a positive exclusive source-unit frontier.
    pub fn after_unit_end(&self, end: usize) -> Result<(i32, i32), String> {
        if end == 0 {
            return Err("encoder continuation has no completed source unit".into());
        }
        let shape = match self {
            Self::Uniform { sequence, width } => (*sequence, *width),
            Self::AfterUnits(shapes) => *shapes
                .get(end - 1)
                .ok_or("encoder continuation exceeds its source units")?,
        };
        if shape.0 <= 0 || shape.1 <= 0 {
            return Err("encoder continuation geometry must be positive".into());
        }
        Ok(shape)
    }

    /// Checks complete unit coverage before beginning any native operation.
    pub fn validate_units(&self, units: usize) -> Result<(), String> {
        if units == 0 || matches!(self,Self::AfterUnits(shapes) if shapes.len() != units) {
            return Err("encoder continuation geometry differs from its unit count".into());
        }
        for end in 1..=units {
            self.after_unit_end(end)?;
        }
        Ok(())
    }
}

/// Architecture-owned interpretation of admitted prepared input.
pub trait CompositeArchitecture<B, S>: LayeredArchitecture<B, S> + Sized + 'static
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Binds admitted input to the same component execution and resource lifecycle.
    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a>;
    /// Recovers the admission proof retained by a prepared invocation.
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>>;

    /// Iterative output geometry, selected independently from prepared media.
    fn block_refinement_geometry(
        _config: &Self::AdmissionConfig,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        None
    }
    /// Exact request-local attention equations over prepared token rows.
    fn supports_attention_invocation(
        config: &Self::AdmissionConfig,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> bool {
        invocation == eredu_core::decoding::AttentionInvocation::CAUSAL
            || (invocation == eredu_core::decoding::AttentionInvocation::DIFFUSION
                && Self::block_refinement_geometry(config).is_some())
    }
    /// Prepared requests implement partial invocations with retained continuations.
    fn supports_retained_prefill() -> bool {
        false
    }

    /// Completed immutable assembled decoder ingress retained by a prefill cursor.
    fn prefill_decoder_continuation(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        let _ = forward;
        None
    }

    /// Ordinary unobserved plain-text passes preserve state across prefix boundaries.
    /// This does not authorize splitting prepared media or observed invocations.
    fn supports_chunked_text_prefill() -> bool {
        false
    }

    /// One architecture-specific plan for each ordered input part, with a neutral
    /// accounting projection of the same admitted geometry.
    type InputPartPlan: Clone + Into<crate::media_plan::PreparedInputPartPlan> + 'static;
    /// Owns request preparation and immutable encoder continuation while the
    /// ordinary runtime session owns traversal, commitment and observation.
    type PrefillRequest: Clone + eredu_runtime::prefill::ChunkedPrefillRequest<Self, B, S>;

    /// Declares an invocation-specific publication width when it differs from
    /// the selected output projection, such as an authoritative encoder result.
    fn prepared_publication_width(
        &self,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<i32>, String> {
        Ok(None)
    }

    /// Maximum target invocation for an admitted request, when the architecture
    /// has a bounded continuation contract. `None` retains one invocation.
    fn prefill_chunk_limit(
        _admitted: &AdmittedCompositeInput<Self::InputPartPlan>,
    ) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    /// Creates the architecture-selected request cursor input from exact admitted
    /// tensors. Families with shared encoders prepare their request products once.
    fn prepare_prefill_request(
        prepared: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Self::Error>;
    /// Minimal normalized configuration retained for repeated input admission.
    type AdmissionConfig: Clone;

    /// Clones the normalized architecture facts required by input admission.
    fn admission_config(&self) -> Self::AdmissionConfig;

    /// Publishes the target facts required to prove external-assistant compatibility.
    fn external_assistant_target_profile(
        _config: &Self::AdmissionConfig,
    ) -> Option<crate::external_assistant::ExternalAssistantTargetProfile> {
        None
    }

    /// Admits exact prepared tensor identity and derives ordered ingress plans.
    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<Self::InputPartPlan>, eredu_core::CapabilityError>;

    /// Reconstructs the target's exact semantic token sequence for a prediction
    /// lane, including architecture-declared media placeholders. The default
    /// accepts ordinary text token parts; media-aware families share their
    /// ingress token construction with this operation.
    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        B::Tensor::concatenate(&prepared_token_parts(input, context, |_| None)?, 1, context)
    }

    /// Returns whether one request-optional execution group is active for the
    /// exact admitted input.
    ///
    /// This decision intentionally consumes the architecture-owned part plan,
    /// rather than modality presence alone: projected media embeddings retain
    /// their semantic modality but must not execute the corresponding native
    /// media tower.
    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool;

    /// Exact sequence extent emitted across one prepared group boundary.
    ///
    /// Most composite groups preserve the complete decoder extent. Media
    /// towers whose projected output occupies only their admitted placeholder
    /// span override this before the prepared input is consumed.
    fn prepared_group_boundary_sequence(
        &self,
        _group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = input
            .decoder_chunk()
            .map_or(input.admitted().decoder_positions(), |chunk| {
                chunk.range.len() as u64
            });
        i32::try_from(positions)
            .map_err(|_| "prepared composite boundary sequence exceeds i32".to_owned())
    }

    /// Exact intermediate activation geometry for a pipeline continuation
    /// within one execution group.
    ///
    /// `None` means the group preserves the ordinary decoder-width
    /// `[batch, sequence, hidden]` boundary. Media towers whose internal
    /// workspace differs from their projected output declare its exact
    /// sequence and width here.
    fn prepared_group_continuation_geometry(
        &self,
        _group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<CompositeContinuationGeometry>, String> {
        Ok(None)
    }

    /// Whether transport adds a leading batch axis to an internal group activation.
    ///
    /// `true` expands before encoding and squeezes before decoding. Architectures
    /// whose activation already has its wire-ready batch axis, or whose encoding
    /// owns that reshape, return `false`.
    fn prepared_group_continuation_batched(&self, _group: usize) -> bool {
        true
    }

    /// Packs an internal group activation into its declared continuation shape.
    /// The default preserves the ordinary wire-ready activation without work.
    fn encode_group_continuation(
        &self,
        _group: usize,
        hidden: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(hidden)
    }

    /// Restores the internal shape using the retained, admitted request context.
    fn decode_group_continuation(
        &self,
        _group: usize,
        hidden: B::Tensor,
        _forward: &Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(hidden)
    }

    /// Exact TP collective sequence for every PP stage of one prepared group.
    ///
    /// `None` means the architecture declares no shared-world schedule for the
    /// group. `stage_units` carries exact retained group-local ownership for every
    /// pipeline stage, including empty ranges. A tensor-sharded optional group
    /// active under TP+PP returns one entry per stage and uses these ranges to
    /// describe its native operation order.
    fn prepared_group_collective_waves(
        &self,
        _group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        _tensor_partitions: usize,
        _stage_units: &[std::ops::Range<usize>],
    ) -> Result<Option<Vec<Vec<CompositeTensorCollective>>>, String> {
        Ok(None)
    }

    /// Exact TP collectives emitted while the primary decoder ingress is assembled.
    ///
    /// `None` preserves the ordinary single decoder-extent embedding sum. Composite
    /// architectures which look up independently segmented token parts override this
    /// with their request-bounded operation shapes and order.
    fn prepared_primary_ingress_collectives(
        &self,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        _tensor_partitions: usize,
    ) -> Result<Option<Vec<CompositeTensorCollective>>, String> {
        Ok(None)
    }

    /// Whether a retained context still has decoder-ingress collectives to run.
    ///
    /// Media may create a context before deferred token lookups are resolved.
    /// Inactive pipeline ranks must mirror those lookups in the primary wave.
    /// The default describes architectures which resolve ingress at context creation.
    fn primary_ingress_collectives_pending(&self, _forward: &Self::ForwardContext) -> bool {
        false
    }

    /// Exact TP reductions surrounding one routed decoder unit.
    ///
    /// The returned typed before/after order is carried unchanged into inactive
    /// pipeline waves. It must describe the concrete family equation rather
    /// than a generic transformer default.
    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Self::Error> {
        Ok(eredu_runtime::RoutedTensorReductions::hidden(1, 1))
    }

    /// Whether primary token embedding and final vocabulary use TP collectives.
    /// Replicated vocabulary equations omit those otherwise implicit idle waves.
    fn routed_tensor_vocabulary_sharded(&self) -> bool {
        true
    }

    /// Physical vocabulary width gathered by the routed TP output equation.
    ///
    /// `None` means the published logical width is also the physical sharded
    /// width. Architectures which gather padded rows before trimming expose
    /// that checkpoint width here so inactive PP ranks submit the exact same
    /// collective shape.
    fn routed_tensor_output_width(&self) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    /// Resolves the exact typed boundary for one continuation or dependency edge.
    ///
    /// `None` declares the ordinary primary-activation-only edge. Families with
    /// learned request context produced inside a partitioned optional root declare
    /// every additional role and its invocation geometry here.
    fn partition_boundary_schema(
        &self,
        _source_group: usize,
        _destination_group: usize,
        _selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        _batch: i32,
        _source_sequence: i32,
        _group_sequences: &[i32],
        _continuation: Option<(i32, i32)>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Self::Error> {
        Ok(None)
    }

    /// Encodes architecture-owned context for one continuation or dependency edge.
    fn partition_boundary_values(
        &self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        _hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>, Self::Error> {
        Ok(None)
    }

    /// Installs a typed continuation or dependency before its destination begins.
    fn accept_partition_boundary(
        &mut self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        _values: Vec<B::Tensor>,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Returns the exact unit-output paths required by an external-assistant capture.
    ///
    /// The default keeps prediction unavailable. Implementations must reject a request whose
    /// family or geometry does not match this target; callers never infer paths from family names.
    fn external_prediction_capture_paths(
        _request: &ExternalPredictionCaptureRequest,
    ) -> Result<Option<Vec<String>>, Self::Error> {
        Ok(None)
    }

    /// Forms a typed assistant capture from architecture-retained context and exact observed paths.
    fn external_prediction_capture(
        _request: &ExternalPredictionCaptureRequest,
        _forward: &Self::ForwardContext,
        _observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        Ok(None)
    }

    /// Applies one complete target-owned static operation. Contributions are
    /// combined before the architecture's finishing equations run.
    fn external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        match self.external_prediction_target_contribution(operation, parallel, context)? {
            Some(value) => {
                self.finish_external_prediction_target_operation(operation, value, context)
            }
            None => Ok(None),
        }
    }

    /// Computes raw lookup/projection values. With no parallel context a
    /// vocabulary-sharded module emits only its local contribution; normalization
    /// of embeddings and output transforms belong to the finishing stage.
    /// An ordinary parallel context combines vocabulary shards before returning.
    fn external_prediction_target_contribution(
        &mut self,
        _operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Finishes a complete combined value. This does not look up or project
    /// parameters, and must not receive an uncombined contribution.
    fn finish_external_prediction_target_operation(
        &self,
        _operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        _combined: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Builds architecture-native ingress and enters the ordinary graph lifecycle.
    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_forward(
            <Self as CompositeArchitecture<B, S>>::prepared_input(input),
            state,
            context,
        )
    }

    /// Observes architecture-owned ingress while entering the same graph.
    fn begin_composite_forward_observed<'a, O>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.begin_forward_observed(
            <Self as CompositeArchitecture<B, S>>::prepared_input(input),
            state,
            context,
            observer,
        )
    }
}

/// Composite ingress whose static operations honor the declared tensor partition.
/// Ordinary composite admission does not imply distributed execution support.
pub trait ParallelCompositeArchitecture<B, S>:
    CompositeArchitecture<B, S> + ParallelLayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Enters the same graph lifecycle through the selected parallel ingress.
    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_forward_parallel(
            <Self as CompositeArchitecture<B, S>>::prepared_input(input),
            state,
            parallel,
            context,
        )
    }

    /// Observes ingress under the retained tensor partition.
    fn begin_composite_forward_parallel_observed<'a, O>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.begin_forward_parallel_observed(
            <Self as CompositeArchitecture<B, S>>::prepared_input(input),
            state,
            parallel,
            context,
            observer,
        )
    }
}
