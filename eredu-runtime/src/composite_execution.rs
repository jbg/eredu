//! Typed prepared-input ingress for replicated composite architectures.

use crate::{
    prefill::ChunkedPrefillRequest, LayeredArchitecture, LayeredForwardState,
    ParallelLayeredArchitecture, RuntimeState,
};
use eredu_core::input::PreparedInputInspector;
use eredu_nn::{NeuralBackend, Tensor};

mod kernel;
pub use kernel::{CompositeComponentKernel, ParallelCompositeComponentKernel};

mod prefill;
pub use prefill::{DecoderPrefillRequest, SingleCompositePrefillRequest};

use eredu_core::{CapabilityError, InputModalities, InputModality, PreparedInputIdentity};

use eredu_core::{PreparedInputPart, PreparedModelInput};

/// Checks prepared inputs against the exact representation selected for the session.
pub fn validate_selected_composite_input<T>(
    processor: &crate::SelectedProcessorExecution,
    parts: &[PreparedInputPart<T>],
) -> Result<(), String> {
    if let Some(modality) = parts
        .iter()
        .map(|part| part.modality())
        .find(|modality| !processor.modalities().contains(modality))
    {
        return Err(format!(
            "prepared input modality {} is outside the selected composite modalities {:?}",
            modality.as_str(),
            processor.modalities()
        ));
    }
    if !processor.prepared_tensors()
        && parts
            .iter()
            .any(|part| part.modality() != InputModality::Text)
    {
        return Err("prepared media tensors were not admitted by processor selection".into());
    }
    if let Some(modality) = parts.iter().find_map(|part| {
        matches!(
            part.payload(),
            eredu_core::PreparedInputPayload::Embeddings(_)
        )
        .then_some(part.modality())
        .filter(|modality| !processor.projected_modalities().contains(modality))
    }) {
        return Err(format!(
            "projected {} embeddings were not admitted by processor selection",
            modality.as_str()
        ));
    }
    Ok(())
}

/// Input rejection from the portable composite admission path.
#[derive(Debug, thiserror::Error)]
pub enum SelectedCompositeInputError {
    /// The submitted representation differs from the selected processor contract.
    #[error("{0}")]
    Selection(String),
    /// Observing native text token IDs failed.
    #[error(transparent)]
    TokenRead(#[from] eredu_core::BackendFailure),
    /// The submitted input cannot be associated with this cache identity.
    #[error(transparent)]
    CacheIdentity(#[from] crate::PreparedInputCacheIdentityError),
    /// Architecture ingress rejected the prepared tensors.
    #[error(transparent)]
    Admission(#[from] CapabilityError),
}

/// Admits prepared tensors and resolves their cache identity in request order.
pub fn admit_selected_composite_input<A, B, S>(
    config: &A::AdmissionConfig,
    processor: &crate::SelectedProcessorExecution,
    prepared: &PreparedModelInput<B::Tensor>,
    supplied_identity: Option<&crate::PreparedInputCacheIdentity>,
    fingerprint_text: bool,
    inspector: &impl PreparedInputInspector<B::Tensor>,
) -> Result<
    (
        AdmittedCompositeInput<A::InputPartPlan>,
        Option<crate::PreparedInputCacheIdentity>,
    ),
    SelectedCompositeInputError,
>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S>,
{
    validate_selected_composite_input(processor, prepared.parts())
        .map_err(SelectedCompositeInputError::Selection)?;
    if supplied_identity.is_some_and(|identity| identity.prepared() != prepared.identity()) {
        return Err(SelectedCompositeInputError::Selection(
            "prepared-input cache identity differs from the submitted tensors".into(),
        ));
    }
    let admitted = A::admit_prepared_input(config, prepared, inspector)?;
    let identity = if supplied_identity.is_none()
        && fingerprint_text
        && prepared.parts().iter().all(|part| {
            part.modality() == InputModality::Text
                && matches!(
                    part.payload(),
                    eredu_core::PreparedInputPayload::TokenIds(_)
                )
        }) {
        let mut ids = Vec::new();
        for part in prepared.parts() {
            ids.extend(inspector.token_ids(part.payload().value())?);
        }
        Some(crate::PreparedInputCacheIdentity::new(
            prepared.identity().clone(),
            eredu_core::cache::prompt_cache_token_fingerprint(&ids),
        )?)
    } else {
        supplied_identity.cloned()
    };
    Ok((admitted, identity))
}

/// Whole-request architecture admission coupled to exact prepared tensor identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedCompositeInput<P> {
    identity: PreparedInputIdentity,
    parts: Vec<P>,
    decoder_positions: u64,
    active_modalities: InputModalities,
}

impl<P> AdmittedCompositeInput<P> {
    /// Attaches architecture-owned execution context while preserving the admitted
    /// request identity and neutral accounting.
    pub fn map_parts<Q>(self, mut map: impl FnMut(usize, P) -> Q) -> AdmittedCompositeInput<Q> {
        AdmittedCompositeInput {
            identity: self.identity,
            parts: self
                .parts
                .into_iter()
                .enumerate()
                .map(|(index, part)| map(index, part))
                .collect(),
            decoder_positions: self.decoder_positions,
            active_modalities: self.active_modalities,
        }
    }

    /// Identity recomputed from the exact tensor handles admitted by the architecture.
    pub const fn identity(&self) -> &PreparedInputIdentity {
        &self.identity
    }

    /// Ordered architecture-owned plan for every prepared input part.
    pub fn parts(&self) -> &[P] {
        &self.parts
    }

    /// Admitted decoder batch and sequence dimensions after assembling all parts.
    /// Current composite ingress contracts validate one sequence per request.
    pub const fn decoder_shape(&self) -> [u64; 2] {
        [1, self.decoder_positions]
    }

    /// Total decoder positions occupied by all ordered parts.
    pub const fn decoder_positions(&self) -> u64 {
        self.decoder_positions
    }

    /// Modalities that activate request-optional roots for this exact input.
    pub const fn active_modalities(&self) -> InputModalities {
        self.active_modalities
    }
}

/// Accounting needed to combine admitted parts into one composite request.
pub trait CompositePartPlan {
    /// Decoder positions owned by the part.
    fn decoder_positions(&self) -> u64;
}

/// Validates exact prepared input identity and accounts for architecture-admitted parts.
pub fn admit_composite_input<Tensor, P>(
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
    mut admit_part: impl FnMut(&PreparedInputPart<Tensor>) -> Result<P, CapabilityError>,
) -> Result<AdmittedCompositeInput<P>, CapabilityError>
where
    P: CompositePartPlan,
{
    let descriptors = input
        .parts()
        .iter()
        .map(|part| part.descriptor(&|tensor| inspector.identity(tensor)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let identity = PreparedInputIdentity::new(descriptors)
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    if &identity != input.identity() {
        return Err(CapabilityError::Observation(
            "prepared-input tensor identity changed before architecture admission".into(),
        ));
    }
    let mut active_modalities = InputModalities {
        text: false,
        image: false,
        audio: false,
        video: false,
    };
    let mut decoder_positions = 0u64;
    let mut parts = Vec::with_capacity(input.len());
    for part in input.parts() {
        let plan = admit_part(part)?;
        decoder_positions = decoder_positions
            .checked_add(plan.decoder_positions())
            .ok_or(CapabilityError::ArithmeticOverflow {
                operation: "composite decoder-position total",
            })?;
        match part.modality() {
            InputModality::Text => active_modalities.text = true,
            InputModality::Image => active_modalities.image = true,
            InputModality::Audio => active_modalities.audio = true,
            InputModality::Video => active_modalities.video = true,
            _ => {}
        }
        parts.push(plan);
    }
    if decoder_positions == 0 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: "replicated composite".into(),
            reason: "prepared input occupies no decoder positions".into(),
        });
    }
    Ok(AdmittedCompositeInput {
        identity,
        parts,
        decoder_positions,
        active_modalities,
    })
}

/// A decoder span over the same exact admitted composite request.
pub struct DecoderPrefillChunk<'a, T> {
    /// Position range of this decoder invocation.
    pub range: std::ops::Range<usize>,
    /// Retained decoder ingress when already assembled.
    pub assembled: Option<&'a T>,
}

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

pub use crate::speculative::{
    ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    ExternalPredictionTargetOperation, ExternalPredictionTargetOperationKind,
};

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
pub fn segmented_token_ingress_collectives(
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
    prepared: &'a eredu_core::PreparedModelInput<T>,
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
        prepared: &'a eredu_core::PreparedModelInput<T>,
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

    /// Selects a bounded span over the exact admitted request and retained ingress.
    pub fn with_decoder_chunk(
        mut self,
        range: std::ops::Range<usize>,
        continuation: Option<&'a Option<T>>,
    ) -> Result<Self, String>
    where
        T: Tensor,
    {
        let tokens = usize::try_from(self.admitted.decoder_positions())
            .map_err(|error| error.to_string())?;
        let assembled = continuation.and_then(Option::as_ref);
        if range.start >= range.end
            || range.end > tokens
            || (range.start > 0 && continuation.is_none())
            || assembled
                .is_some_and(|value| value.shape().len() != 3 || value.dim(1) as usize != tokens)
        {
            return Err("invalid retained composite prefill span or continuation".into());
        }
        self.decoder_chunk = Some((range.start, range.end, assembled));
        Ok(self)
    }

    /// Returns the active bounded decoder span and any retained input continuation.
    pub fn decoder_chunk(&self) -> Option<DecoderPrefillChunk<'a, T>> {
        self.decoder_chunk
            .map(|(start, end, assembled)| DecoderPrefillChunk {
                range: start..end,
                assembled,
            })
    }

    /// Decoder positions for this invocation, including chunked prefill.
    /// Returns the exact decoder positions for the current invocation.
    pub fn decoder_sequence(&self) -> Result<i32, String> {
        let positions = self
            .decoder_chunk()
            .map_or(self.admitted().decoder_positions(), |chunk| {
                chunk.range.len() as u64
            });
        i32::try_from(positions)
            .map_err(|_| "prepared composite boundary sequence exceeds i32".into())
    }

    /// Exact backend-native prepared tensor handles.
    pub const fn prepared(&self) -> &'a eredu_core::PreparedModelInput<T> {
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
    ) -> Result<crate::PreparedInputCacheIdentity, crate::PreparedInputCacheIdentityError> {
        crate::PreparedInputCacheIdentity::new(
            self.prepared.identity().clone(),
            semantic_content_fingerprint,
        )
    }
}

/// Builds only semantic token parts from an exact admission. The architecture
/// supplies placeholder identities; media tensors are neither copied nor encoded.
pub fn prepared_token_parts<T: Tensor, P>(
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
                        eredu_core::PreparedInputPayload::TokenIds(tokens),
                    ) => Ok(tokens.clone()),
                    _ => Err(eredu_nn::Error::backend(
                        "composite prediction input has no declared token identity",
                    )),
                }
            }
        })
        .collect()
}

/// Native encoder activation shapes at intra-group pipeline cuts.
/// The last axis is the feature width. Transport packs all preceding axes
/// into one sequence on a batch-one wire.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CompositeContinuationGeometry {
    /// Every source unit has the same native shape.
    Uniform(Vec<i32>),
    /// Native shape after each source unit, in execution order.
    AfterUnits(Vec<Vec<i32>>),
}

impl CompositeContinuationGeometry {
    /// Native shape after a positive exclusive source-unit frontier.
    pub fn native_shape(&self, end: usize) -> Result<&[i32], String> {
        let index = end
            .checked_sub(1)
            .ok_or("encoder continuation has no completed source unit")?;
        match self {
            Self::Uniform(shape) => Ok(shape),
            Self::AfterUnits(shapes) => shapes
                .get(index)
                .map(Vec::as_slice)
                .ok_or_else(|| "encoder continuation exceeds its source units".into()),
        }
    }

    /// Resolves the packed sequence and feature width.
    pub fn after_unit_end(&self, end: usize) -> Result<(i32, i32), String> {
        let (&width, axes) = self
            .native_shape(end)?
            .split_last()
            .filter(|(width, axes)| **width > 0 && !axes.is_empty())
            .ok_or("encoder continuation requires positive sequence and width axes")?;
        let sequence = axes
            .iter()
            .try_fold(1i32, |product, &extent| {
                (extent > 0).then(|| product.checked_mul(extent)).flatten()
            })
            .ok_or("encoder continuation has invalid native extents")?;
        Ok((sequence, width))
    }

    /// Checks complete unit coverage before beginning any native operation.
    pub fn validate_units(&self, units: usize) -> Result<(), String> {
        if units == 0 || matches!(self, Self::AfterUnits(shapes) if shapes.len() != units) {
            return Err("encoder continuation geometry differs from its unit count".into());
        }
        for end in 1..=units {
            self.after_unit_end(end)?;
        }
        Ok(())
    }
}

/// Whether a group produces a new value for this invocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CompositeGroupUse {
    /// Execute the group's equations for the admitted input.
    Execute,
    /// Reuse context carried by the architecture from an earlier decoder chunk.
    Retained,
    /// The request supplies no work for this group, including caller-projected inputs.
    Bypassed,
}

impl CompositeGroupUse {
    /// Whether the group runs and participates in activation transfers.
    pub fn is_active(self) -> bool {
        self == Self::Execute
    }
}

/// Request-specific execution and transport contract for one composite group.
pub struct CompositeGroupInvocation {
    /// How the current request uses the group.
    pub mode: CompositeGroupUse,
    /// Sequence extent emitted to other groups.
    pub boundary_sequence: i32,
    /// Per-cut geometry when an internal activation differs from the output.
    pub continuation: Option<CompositeContinuationGeometry>,
    /// Exact operations by pipeline stage, including explicitly empty waves.
    pub collective_waves: Option<Vec<Vec<CompositeTensorCollective>>>,
}

/// Architecture-owned interpretation of admitted prepared input.
pub trait CompositeArchitecture<B, S>: LayeredArchitecture<B, S> + Sized + 'static
where
    B: NeuralBackend,
    S: crate::RuntimeState<B>,
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
    ) -> Option<crate::generation::refinement::architecture::BlockRefinementGeometry> {
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
    type InputPartPlan: Clone
        + Into<crate::executable_binding::retained_session::PreparedInputPartPlan>
        + 'static;
    /// Owns request preparation and immutable encoder continuation while the
    /// ordinary runtime session owns traversal, commitment and observation.
    type PrefillRequest: Clone + crate::prefill::ChunkedPrefillRequest<Self, B, S>;

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
        prepared: eredu_core::PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Self::Error>;
    /// Minimal normalized configuration retained for repeated input admission.
    type AdmissionConfig: Clone;

    /// Clones the normalized architecture facts required by input admission.
    fn admission_config(&self) -> Self::AdmissionConfig;

    /// Whether this target exposes the invocation contract used by an external assistant.
    fn supports_external_assistant(_config: &Self::AdmissionConfig) -> bool {
        false
    }

    /// Admits exact prepared tensor identity and derives ordered ingress plans.
    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_core::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_core::input::PreparedInputInspector<B::Tensor>,
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

    /// Plans one group from admitted input and selected group-local stage ranges.
    fn prepared_group_invocation(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<CompositeGroupInvocation, String>;

    /// Decoder batch extent. Media-first inputs can have a patch or frame axis
    /// instead of a batch axis on their first payload.
    fn prepared_decoder_batch_size(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        input
            .prepared()
            .parts()
            .first()
            .map(|part| part.payload().value().dim(0))
            .ok_or_else(|| "composite request has no prepared parts".into())
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
    ) -> Result<crate::RoutedTensorReductions, Self::Error> {
        Ok(crate::RoutedTensorReductions::hidden(1, 1))
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
        _selected: &crate::ResolvedBoundaryWireSchema,
        _batch: i32,
        _source_sequence: i32,
        _groups: &[CompositeGroupInvocation],
        _continuation: Option<(i32, i32)>,
    ) -> Result<Option<crate::ResolvedBoundaryWireSchema>, Self::Error> {
        Ok(None)
    }

    /// Encodes architecture-owned context for one continuation or dependency edge.
    fn partition_boundary_values(
        &self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &crate::ResolvedBoundaryWireSchema,
        _hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>>, Self::Error> {
        Ok(None)
    }

    /// Installs a typed continuation or dependency before its destination begins.
    fn accept_partition_boundary(
        &mut self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &crate::ResolvedBoundaryWireSchema,
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
        O: crate::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
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
    S: crate::RuntimeState<B>,
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
        O: crate::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
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
