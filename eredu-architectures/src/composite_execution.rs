//! Typed prepared-input ingress for replicated composite architectures.

mod media_prefill;
pub(crate) mod graph;
mod description;
pub(crate) mod prediction_tokens;
pub use prediction_tokens::PredictionTokenPart;
pub use media_prefill::CompositeMediaIngressArchitecture;

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_core::AttentionPolicy;
use eredu_nn::{GroupedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    ArchitectureParameters, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    ParallelLayeredArchitecture, PartitionedLayeredArchitecture, RoutedExpertProvider,
    RoutedLayeredArchitecture, StaticParameterVisitor, StaticParameterVisitorMut,
};

use crate::media_plan::AdmittedCompositeInput;

/// One request-bounded tensor collective in a composite pipeline wave.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CompositeTensorCollective {
    /// Sum one tensor with this exact logical shape across the selected TP group.
    Sum {
        /// Exact logical shape of the active or zero-work tensor.
        shape: Vec<i32>,
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
    /// Ordered decoder-unit outputs consumed by a DFlash assistant.
    MuseGlimmerDFlash {
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
    /// Muse-Glimmer decoder outputs in the requested target-layer order.
    MuseGlimmerDFlash {
        /// Exact ordered target-layer activations.
        target_states: Vec<T>,
    },
}

impl<T> ExternalPredictionTargetCapture<T> {
    /// Borrows every actual captured root in its semantic order, including
    /// shared key/value publications; aliases remain distinct logical roots.
    pub fn visit_values(&self, visitor: &mut dyn FnMut(&T)) {
        match self {
            Self::Gemma4 { hidden, shared_kv } => {
                visitor(hidden);
                for (_, keys, values) in shared_kv {
                    visitor(keys);
                    visitor(values);
                }
            }
            Self::MuseGlimmerDFlash { target_states } => {
                for value in target_states { visitor(value); }
            }
        }
    }
}

impl ExternalPredictionCaptureRequest {
    pub(crate) fn collect_paths(
        &self,
        metadata: crate::decoder::ModuleMetadata<'_>,
    ) -> Result<Vec<String>, eredu_nn::Error> {
        metadata.controls::<(&Self, Vec<String>)>()?;
        match self {
            Self::Gemma4SharedAttention { final_hidden_path } => {
                let mut paths = metadata.vector(1)?;
                paths.push(metadata.text(format_args!("{final_hidden_path}"))?);
                Ok(paths)
            }
            Self::MuseGlimmerDFlash { target_layers, target_paths } => {
                if target_layers.is_empty() || target_paths.len() != target_layers.len() {
                    return Err(metadata.error(format_args!(
                        "Muse-Glimmer DFlash capture paths differ from its nonempty target layers"
                    )));
                }
                let mut paths = metadata.vector(target_paths.len())?;
                for path in target_paths { paths.push(metadata.text(format_args!("{path}"))?); }
                Ok(paths)
            }
        }
    }
}

/// Target-owned static operation needed by an external assistant.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExternalPredictionTargetOperation<'a, T> {
    /// Applies the ordinary target token embedding to assistant proposal IDs.
    TokenEmbeddings(&'a T),
    /// Applies the ordinary target vocabulary projection to assistant states.
    ProjectLogits(&'a T),
}

impl<T> Copy for ExternalPredictionTargetOperation<'_,T> {}
impl<T> Clone for ExternalPredictionTargetOperation<'_,T> {
    fn clone(&self)->Self { *self }
}

impl CompositeTensorCollective {
    /// Exact logical tensor shape submitted by active and zero-work ranks.
    pub fn shape(&self) -> &[i32] {
        match self {
            Self::Sum { shape } => shape,
        }
    }
}

/// Embedding sums for independently looked-up token segments. Media positions
/// are supplied by their own ingress and must not enlarge these native waves.
pub(crate) fn segmented_token_ingress_collectives(
    positions: impl IntoIterator<Item = u64>, hidden_width: i32, tensor_partitions: usize,
) -> Result<Option<Vec<CompositeTensorCollective>>, String> {
    segmented_token_ingress_collectives_in(positions, hidden_width, tensor_partitions,
        graph::Destination(None)).map_err(|cause| cause.to_string())
}

pub(crate) fn segmented_token_ingress_collectives_in(
    positions: impl IntoIterator<Item = u64>, hidden_width: i32, tensor_partitions: usize,
    destination: graph::Destination<'_>,
) -> Result<Option<Vec<CompositeTensorCollective>>, eredu_nn::Error> {
    destination.controls::<(Option<Vec<CompositeTensorCollective>>, i32, usize)>()?;
    if tensor_partitions <= 1 { return Ok(None); }
    destination.try_collect(positions.into_iter().map(|positions| {
        let positions = i32::try_from(positions).map_err(|_| destination.error(format_args!(
            "composite token ingress positions exceed i32")))?;
        Ok(CompositeTensorCollective::Sum {
            shape: destination.collect([1, positions, hidden_width])?,
        })
    })).map(Some)
}

/// Exact prepared tensors paired with their architecture-owned admission proof.
pub struct PreparedCompositeInput<'a, T, P> {
    prepared: &'a eredu_runtime::PreparedModelInput<T>,
    admission: CompositeAdmissionRef<'a, P>,
    metadata: Option<&'a eredu_nn::workspace::WorkspaceContext>,
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
        Self::new_with_diagnostic(prepared, admitted, str::to_owned)
    }

    pub(crate) fn new_with_diagnostic<E>(
        prepared: &'a eredu_runtime::PreparedModelInput<T>,
        admitted: &'a AdmittedCompositeInput<P>,
        diagnostic: impl FnOnce(&'static str) -> E,
    ) -> Result<Self, E> {
        if prepared.identity() != admitted.identity() {
            return Err(diagnostic(
                "prepared composite input identity differs from admission",
            ));
        }
        if prepared.len() != admitted.parts().len() {
            return Err(diagnostic(
                "prepared composite part count differs from admission",
            ));
        }
        Ok(Self {
            prepared,
            metadata: None,
            admission: CompositeAdmissionRef {
                ordinary: Some(admitted),
                original: None,
            },
        })
    }

    /// Couples the same validated source and admission with their live metadata
    /// destination. The caller retains its funding through dependent outputs.
    pub fn new_with_metadata(
        prepared: &'a eredu_runtime::PreparedModelInput<T>,
        admitted: &'a AdmittedCompositeInput<P>,
        context: &'a eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(Self, Result<Self, eredu_nn::Error>)>())?;
        let mut input = Self::new_with_diagnostic(prepared, admitted, |message| {
            context.metadata_error(format_args!("{message}"))
        })?;
        input.metadata = Some(context);
        Ok(input)
    }

    // Used only after the same private source has validated and paid this pair
    // in its fallible chunk constructor. Attaching the loan allocates nothing.
    pub(crate) fn with_metadata_loan(
        mut self,
        context: Option<&'a eredu_nn::workspace::WorkspaceContext>,
    ) -> Self {
        self.metadata = context;
        self
    }

    pub(crate) fn metadata(&self) -> Option<&'a eredu_nn::workspace::WorkspaceContext> {
        self.metadata
    }

    /// Exact backend-native prepared tensor handles.
    pub const fn prepared(&self) -> &'a eredu_runtime::PreparedModelInput<T> {
        self.prepared
    }

    /// Borrows either the original ordinary admission or its closed compiled
    /// representation. Common scalar geometry never materializes ordinary parts.
    pub const fn admitted(&self) -> CompositeAdmissionRef<'a, P> {
        self.admission
    }

    pub(crate) fn from_original(
        prepared: &'a eredu_runtime::PreparedModelInput<T>,
        original: &'a crate::media_plan::BoundPreparedMediaSemantics,
    ) -> Result<Self, String> {
        Self::from_original_with_diagnostic(prepared, original, str::to_owned)
    }

    pub(crate) fn from_original_with_diagnostic<E>(
        prepared: &'a eredu_runtime::PreparedModelInput<T>,
        original: &'a crate::media_plan::BoundPreparedMediaSemantics,
        diagnostic: impl FnOnce(&'static str) -> E,
    ) -> Result<Self, E> {
        if prepared.len() != original.records().len() {
            return Err(diagnostic("compiled prepared input part count differs from original source"));
        }
        Ok(Self {
            prepared,
            metadata: None,
            admission: CompositeAdmissionRef {
                ordinary: None,
                original: Some(original),
            },
        })
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

/// Borrowed common admission geometry with explicit ordinary-part access.
pub struct CompositeAdmissionRef<'a, P> {
    ordinary: Option<&'a AdmittedCompositeInput<P>>,
    original: Option<&'a crate::media_plan::BoundPreparedMediaSemantics>,
}
impl<P> Copy for CompositeAdmissionRef<'_, P> {}
impl<P> Clone for CompositeAdmissionRef<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, P> CompositeAdmissionRef<'a, P> {
    /// Existing ordinary part storage, when this is an ordinary admission.
    pub const fn ordinary(self) -> Option<&'a AdmittedCompositeInput<P>> {
        self.ordinary
    }
    pub fn decoder_positions(self) -> u64 {
        match self.ordinary {
            Some(value) => value.decoder_positions(),
            None => self.original.expect("closed admission").decoder_positions() as u64,
        }
    }
    pub fn decoder_shape(self) -> [u64; 2] {
        self.ordinary.map_or_else(
            || [1, self.decoder_positions()],
            |ordinary| ordinary.decoder_shape(),
        )
    }
    pub fn active_modalities(self) -> eredu_core::InputModalities {
        if let Some(ordinary) = self.ordinary {
            return ordinary.active_modalities();
        }
        let mut result = eredu_core::InputModalities {
            text: false,
            image: false,
            video: false,
            audio: false,
        };
        for part in self.original.expect("closed admission").records() {
            match part.modality {
                eredu_core::InputModality::Text => result.text = true,
                eredu_core::InputModality::Image => result.image = true,
                eredu_core::InputModality::Video => result.video = true,
                eredu_core::InputModality::Audio => result.audio = true,
                _ => {}
            }
        }
        result
    }
    fn len(self) -> usize {
        match self.ordinary {
            Some(value) => value.parts().len(),
            None => self.original.expect("closed admission").records().len(),
        }
    }
}

impl<'a> CompositeAdmissionRef<'a, crate::media_plan::Gemma4InputPartPlan> {
    /// Fixed ordinary plans or equivalent scalar projections from the exact
    /// original source. No metadata Vec, native read or admission is repeated.
    pub(crate) fn gemma_parts(self) -> impl ExactSizeIterator<Item = crate::media_plan::Gemma4InputPartPlan> + Clone + 'a {
        (0..self.len()).map(move |index| match self.ordinary {
            Some(value) => value.parts()[index].clone(),
            None => self.original.expect("closed Gemma admission").gemma_part(index),
        })
    }
}

/// Builds only semantic token parts from an exact admission. The architecture
/// supplies placeholder identities; media tensors are neither copied nor encoded.
pub(crate) fn prepared_token_parts<T: Tensor, P>(
    input: PreparedCompositeInput<'_, T, P>,
    context: &T::Context,
    placeholder: impl Fn(&P) -> Option<(u32, u64)>,
) -> Result<Vec<T>, eredu_nn::Error> {
    let metadata = crate::decoder::identity::Metadata::new(input.metadata());
    let mut parts = metadata.vector(input.prepared().len())?;
    prediction_tokens::visit(input, placeholder, &mut |part| {
        parts.push(prediction_tokens::materialize(part, context, metadata)?);
        Ok(())
    })?;
    Ok(parts)
}

/// Architecture-owned interpretation of admitted prepared input.
pub trait CompositeArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// One architecture-specific plan for each ordered input part, with a neutral
    /// accounting projection of the same admitted geometry.
    type InputPartPlan: Clone
        + Into<crate::media_plan::PreparedInputPartPlan>
        + crate::media_plan::CompositePartPlan;
    /// Minimal normalized configuration retained for repeated input admission.
    type AdmissionConfig: Clone;

    /// Clones the normalized architecture facts required by input admission.
    fn admission_config(&self) -> Self::AdmissionConfig;

    /// Retains actual immutable admission configuration with paid control
    /// storage. A checked backend must not silently deep-clone ordinary config.
    fn retain_admission_config_with_metadata(
        _config: &Self::AdmissionConfig,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::AdmissionConfig, eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Borrows the target facts directly from retained admission configuration.
    /// Missing fixed source support stays unavailable to original startup.
    fn external_assistant_target_profile_ref(
        _config:&Self::AdmissionConfig,
    )->Option<crate::external_assistant::ExternalAssistantTargetProfileRef<'_>> { None }

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

    /// Revalidates the same actual prepared identities and family policy using
    /// counted descriptor/inspection destinations. No ordinary allocation is
    /// permitted as a fallback when this constructor is not yet qualified.
    fn admit_prepared_input_with_metadata(
        _config: &Self::AdmissionConfig,
        _input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        _inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<AdmittedCompositeInput<Self::InputPartPlan>, eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Visits the exact semantic sequence in admitted decoder order. Tensor
    /// parts are borrowed from this input; repeated values are family-declared
    /// placeholders. The same visit drives ordinary and funded construction.
    fn visit_prepared_prediction_tokens(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        visitor: &mut dyn FnMut(PredictionTokenPart<'_, B::Tensor>) -> Result<(), eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        prediction_tokens::visit(input, |_| None, visitor)
    }

    /// Constructs the architecture's exact semantic token sequence, including
    /// its admitted media placeholders, through the shared visit above.
    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        let mut parts = metadata.vector(input.prepared().len())?;
        Self::visit_prepared_prediction_tokens(input, &mut |part| {
            parts.push(prediction_tokens::materialize(part, context, metadata)?);
            Ok(())
        })?;
        B::Tensor::concatenate(&parts, 1, context)
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
        i32::try_from(input.admitted().decoder_positions())
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
    ) -> Result<Option<(i32, i32)>, String> {
        Ok(None)
    }

    /// Resolves an exact selected source cut using the already retained output
    /// extent. Uniform towers preserve their existing prepared geometry. A tower
    /// with shape-changing units may require the actual exclusive unit endpoint;
    /// neither source rank arithmetic nor hidden width identifies that endpoint.
    fn group_continuation_geometry_at(
        &self,
        _group: usize,
        _source_unit_end: Option<usize>,
        _source_sequence: i32,
        prepared: Option<(i32, i32)>,
    ) -> Result<Option<(i32, i32)>, Self::Error> {
        Ok(prepared)
    }

    /// Packs the activation after an exact source cut. Existing continuation
    /// hooks remain the default; the shared source-completion step follows this
    /// packing before values may be transferred.
    fn encode_group_continuation_at(
        &self,
        group: usize,
        _source_unit_end: Option<usize>,
        _source_sequence: i32,
        hidden: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.encode_group_continuation(group, hidden, context)
    }

    /// Restores the activation at the actual destination's first unit, before
    /// that unit executes. No different rank or width may substitute for the cut.
    fn decode_group_continuation_at(
        &self,
        group: usize,
        _source_unit_end: usize,
        _source_sequence: i32,
        hidden: B::Tensor,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.decode_group_continuation(group, hidden, forward, context)
    }

    /// Whether a group-local continuation transports its internal activation
    /// with an explicit leading batch dimension.
    ///
    /// Shared Qwen vision blocks normally use batched activations. Conditional
    /// Qwen keeps its flattened patch matrix unbatched between vision owners.
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
    /// group. A tensor-sharded optional group that is active under TP+PP must
    /// return one stage entry (possibly empty) for every pipeline stage.
    fn prepared_group_collective_waves(
        &self,
        _group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        _tensor_partitions: usize,
        _pipeline_stages: usize,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<Vec<CompositeTensorCollective>>>, eredu_nn::Error> {
        graph::Destination(metadata).controls::<(&Self,
            PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>, usize, usize, usize,
            Option<Vec<Vec<CompositeTensorCollective>>>)>()?;
        Ok(None)
    }

    /// Exact TP collectives emitted while the primary decoder ingress is assembled.
    /// `None` preserves the ordinary single decoder-extent embedding sum. The
    /// supplied destination pays the actual segmented producer when overridden.
    fn prepared_primary_ingress_collectives(
        &self,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        _tensor_partitions: usize,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<CompositeTensorCollective>>, eredu_nn::Error> {
        graph::Destination(metadata).controls::<(&Self,
            PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>, usize,
            Option<Vec<CompositeTensorCollective>>)>()?;
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
    /// The returned `(before, after)` order is carried unchanged into inactive
    /// pipeline waves. It must describe the concrete family equation rather
    /// than a generic transformer default.
    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<(usize, usize), Self::Error> {
        Ok((1, 1))
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
        _metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
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
        _metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
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

    /// The same exact path selection using the caller's paid host destination.
    fn external_prediction_capture_paths_with_metadata(
        _request: &ExternalPredictionCaptureRequest,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<Vec<String>>, eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Forms a typed assistant capture from architecture-retained context and exact observed paths.
    fn external_prediction_capture(
        _request: &ExternalPredictionCaptureRequest,
        _forward: &Self::ForwardContext,
        _observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        Ok(None)
    }

    /// Constructs the same semantic capture while charging its owned containers
    /// and tensor-header destinations to the existing metadata account.
    fn external_prediction_capture_with_metadata(
        _request: &ExternalPredictionCaptureRequest,
        _forward: &Self::ForwardContext,
        _observed: Vec<B::Tensor>,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Applies one target-owned static operation without transferring ordinary target ownership.
    ///
    /// Implementations preserve architecture geometry, parameter topology, and observation
    /// declarations through success, errors, and unwinding. Runtime adapters retain the
    /// existing observation binding while this operation mutates target execution state.
    fn external_prediction_target_operation(
        &mut self,
        _operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Borrows the exact target value retained for an embedded prediction extension.
    fn prediction_target_capture(_forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        None
    }

    /// Declares the exact target-capture placeholder on a non-output pipeline rank.
    fn prediction_target_placeholder_shape(
        &self,
        _forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(None)
    }

    /// Builds architecture-native ingress and enters the ordinary graph lifecycle.
    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Builds architecture-native ingress through the selected tensor-parallel
    /// embedding boundary, then enters the same graph lifecycle.
    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend;
}

/// Additive input adapter over the same architecture modules and graph lifecycle.
pub struct PreparedCompositeArchitecture<A> {
    inner: A,
    // Aliases only the exact selection in the matched validated module/contract.
    // No graph payload or authority is constructed by this descriptive carrier.
    prepared_graph: PreparedCompositeGraph,
    // Closed description from the same completed contract/source. Mutable
    // architecture access invalidates it along with the graph projection.
    construction_parameters: Option<crate::routed_text::RetainedRoutedDescription>,
}

enum PreparedCompositeGraph {
    Unprepared,
    Validated(eredu_runtime::SelectedReplicatedTextRealization),
    Invalidated,
}

impl<A> PreparedCompositeArchitecture<A> {
    /// Wraps one constructed composite architecture.
    pub const fn new(inner: A) -> Self {
        Self {
            inner,
            prepared_graph: PreparedCompositeGraph::Unprepared,
            construction_parameters: None,
        }
    }

    /// Borrows the underlying architecture.
    pub const fn inner(&self) -> &A {
        &self.inner
    }

    /// Mutably borrows the underlying architecture for generic partition execution.
    pub fn inner_mut(&mut self) -> &mut A {
        self.construction_parameters = None;
        if !matches!(self.prepared_graph, PreparedCompositeGraph::Unprepared) {
            self.prepared_graph = PreparedCompositeGraph::Invalidated;
        }
        &mut self.inner
    }

    pub(crate) fn retain_prepared_graph(
        &mut self,
        selected: eredu_runtime::SelectedReplicatedTextRealization,
    ) {
        self.prepared_graph = PreparedCompositeGraph::Validated(selected);
    }

    /// Consumes the adapter.
    pub fn into_inner(self) -> A {
        self.inner
    }
}

impl<A, B> ArchitectureParameters<B> for PreparedCompositeArchitecture<A>
where
    B: NeuralBackend,
    A: ArchitectureParameters<B>,
{
    type DefinitionError = A::DefinitionError;


    fn state_layout(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::StateLayout, Self::DefinitionError> {
match context { Some(context) => {
        self.inner.state_layout(Some(context))
    }, None => {
        self.inner.state_layout(None)
    } }
}


    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
match context { Some(context) => {
        self.inner.state_identity(state, topology, Some(context))
    }, None => {
        self.inner.state_identity(state, topology, None)
    } }
}


    fn parameter_description(
        &self, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, eredu_runtime::ArchitectureParameterDescription>, Self::DefinitionError> {
        match &self.construction_parameters {
            Some(source) => Ok(std::borrow::Cow::Borrowed(&**source)),
            None => self.inner.parameter_description(context),
        }
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<std::collections::BTreeMap<String, DerivedWeightRecipe>, String> {
        self.inner.static_parameter_recipes(source)
    }

    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        eredu_runtime::ArchitectureParameters::retained_static_value_slot_bound(&self.inner)
    }

    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        eredu_runtime::ArchitectureParameters::visit_retained_static_values(&self.inner, visitor)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<B>,
    {
        self.inner.visit_static_parameters(visitor)
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<B>,
    {
        self.inner.visit_static_parameters_mut(visitor)
    }
}

impl<A, B, S> LayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    type Input<'a>
        = PreparedCompositeInput<'a, B::Tensor, A::InputPartPlan>
    where
        Self: 'a;
    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        Ok(Some(input.admitted().decoder_shape()))
    }

    type StaticModules = A::StaticModules;
    type Unit = A::Unit;
    type ForwardContext = A::ForwardContext;
    type RetainedContextValues<'a>
        = A::RetainedContextValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = A::Error;

    fn prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        self.inner.prefill_observation_declarations(metadata_context)
    }

    fn media_prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        self.inner.media_prefill_observation_declarations(metadata_context)
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        self.inner.observation_hooks()
    }

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        self.inner.group_transport(group)
    }

    fn group_transport_matches(
        &self,
        group: usize,
        expected: &eredu_runtime::ArchitectureGroupTransport,
    ) -> bool {
        self.inner.group_transport_matches(group, expected)
    }

    fn primary_execution_group(&self) -> &str {
        self.inner.primary_execution_group()
    }

    fn prediction_execution_groups(&self) -> Vec<String> {
        self.inner.prediction_execution_groups()
    }

    fn prediction_target_capture(context: &Self::ForwardContext) -> Option<&B::Tensor> {
        <A as CompositeArchitecture<B, S>>::prediction_target_capture(context)
    }

    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        <A as CompositeArchitecture<B, S>>::prediction_target_placeholder_shape(
            &self.inner,
            forward,
        )
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.inner.state_partition_plan(layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        self.inner.execution_graph()
    }

    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Self::Error> {

        self.inner.group_unit_count(group, metadata_context)
    }



    fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Self::Error> {

        self.inner.unit_path(group, index, metadata_context)
    }




    fn observes_unit_boundaries(&self, group: usize, index: usize) -> bool {
        self.inner.observes_unit_boundaries(group, index)
    }

    fn group_input_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Self::Error> {

        self.inner.group_input_observation_path(group, metadata_context)
    }

    fn group_output_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Self::Error> {

        self.inner.group_output_observation_path(group, metadata_context)
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self.inner.static_modules()
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self.inner.static_modules_mut()
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        self.inner.build_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner.begin_composite_forward(input, state, context)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .begin_execution_group(group, initial, dependencies, state, forward, context)
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        self.inner.should_execute_group(group, forward)
    }

    fn state_ordinal(&self, group: usize, index: usize, ordinal: usize) -> usize {
        self.inner.state_ordinal(group, index, ordinal)
    }

    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        ordinal: usize,
    ) -> std::ops::Range<usize> {
        self.inner.retained_state_ordinals(group, index, ordinal)
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .forward_unit(group, index, unit, hidden, state, forward, context)
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .complete_execution_group(group, hidden, state, forward, context)
    }

    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.inner.forward_unit_observed(
            group, index, unit, hidden, state, forward, context, observer,
        )
    }

    fn select_readout_positions(
        &self,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        self.inner
            .select_readout_positions(hidden, forward, demand, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner.finish_forward(hidden, state, forward, context)
    }

    fn forward_metadata(
        &self,
        forward: &Self::ForwardContext,
    ) -> Option<eredu_runtime::layered::LayeredMetadata<Self::Error>> {
        <A as LayeredArchitecture<B, S>>::forward_metadata(&self.inner, forward)
    }

    fn visit_retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
        visitor: &mut dyn FnMut(&'a B::Tensor),
    ) {
        <A as LayeredArchitecture<B, S>>::visit_retained_context_values(
            &self.inner, forward, group, index, visitor,
        );
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a> {
        self.inner.retained_context_values(forward, group, index)
    }

    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.inner
            .finish_forward_observed(hidden, state, forward, context, observer)
    }
}

impl<A, B, S> RoutedLayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: GroupedNeuralBackend,
    S: LayerRuntimeState<B>,
    A: CompositeArchitecture<B, S> + RoutedLayeredArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    fn routed_unit_observations(&self) -> bool {
        self.inner.routed_unit_observations()
    }

    fn routed_sparse_observations(&self) -> bool {
        self.inner.routed_sparse_observations()
    }

    fn routed_observation_points(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Self::Error> {
        self.inner.routed_observation_points(group, index)
    }

    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.inner.forward_unit_with_provider(
            group, index, unit, hidden, state, forward, pass, provider, context,
        )
    }

    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.inner.forward_unit_observed_with_provider(
            group, index, unit, hidden, state, forward, pass, provider, context, observer,
        )
    }
}

impl<A, B, S> ParallelLayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S> + ParallelLayeredArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        self.inner.parallel_observation_hooks()
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner
            .begin_composite_forward_parallel(input, state, parallel, context)
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner.forward_unit_parallel(
            group, index, unit, hidden, state, forward, parallel, context,
        )
    }

    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.inner.forward_unit_parallel_observed(
            group, index, unit, hidden, state, forward, parallel, context, observer,
        )
    }

    fn begin_execution_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner.begin_execution_group_parallel(
            group,
            initial,
            dependencies,
            state,
            forward,
            parallel,
            context,
        )
    }

    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .complete_execution_group_parallel(group, hidden, state, forward, parallel, context)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .finish_forward_parallel(hidden, state, forward, parallel, context)
    }

    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.inner
            .finish_forward_parallel_observed(hidden, state, forward, parallel, context, observer)
    }
}

impl<A, B, S> PartitionedLayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S> + PartitionedLayeredArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    type Boundary = A::Boundary;

    fn partition_observation_hooks(
        &self,
        tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        self.inner.partition_observation_hooks(tensor_parallel)
    }

    fn boundary_schema(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Self::Boundary, Self::Error> {
        self.inner.boundary_schema(metadata)
    }

    fn begin_partition<'a>(
        &mut self,
        input: eredu_runtime::LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &eredu_runtime::StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner
            .begin_partition(input, mask, state, expected, first_state_ordinal, context)
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: eredu_runtime::LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &eredu_runtime::StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner.begin_partition_parallel(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            parallel,
            context,
        )
    }

    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .enter_partition_group(group, initial, state, forward, parallel, context)
    }

    fn leave_partition_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .leave_partition_group(group, hidden, state, forward, parallel, context)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        eredu_runtime::LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    > {
        self.inner
            .finish_partition(hidden, state, forward, owns_output, parallel, context)
    }

    fn partition_prediction_capture(
        &self,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .partition_prediction_capture(hidden, forward, context)
    }

    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        eredu_runtime::LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.inner.finish_partition_observed(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            observer,
        )
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenPartView<'a> {
    pub role: crate::media_plan::qwen::QwenPartRole,
    pub modality: eredu_core::InputModality,
    pub positions: u64,
    pub placeholder: u32,
    pub grid: crate::qwen::vl::positions::GridRows<'a>,
    pub workspace_scalars: u64,
}
impl<'a> QwenPartView<'a> {
    fn original(value: crate::media_plan::qwen::QwenPartRef<'a>) -> Self {
        Self {
            role: value.role,
            modality: value.modality,
            positions: value.positions,
            placeholder: value.placeholder,
            grid: crate::qwen::vl::positions::GridRows::Arrays(value.grid),
            workspace_scalars: value.workspace_scalars,
        }
    }
}
impl<'a, T> PreparedCompositeInput<'a, T, crate::media_plan::QwenVlInputPartPlan> {
    pub(crate) fn qwen_parts(self) -> impl ExactSizeIterator<Item = QwenPartView<'a>> + Clone {
        use crate::media_plan::{qwen::QwenPartRole as R, QwenVlInputPartPlan as P};
        (0..self.admission.len()).map(move |index| {
            let Some(ordinary) = self.admission.ordinary else {
                return QwenPartView::original(
                    self.admission
                        .original
                        .expect("closed admission")
                        .part(index),
                );
            };
            let (role, positions, placeholder, grid, workspace_scalars) =
                match &ordinary.parts()[index] {
                    P::TextTokens { positions } => (R::Tokens, *positions, 0, &[][..], 0),
                    P::ProjectedText { positions } => (R::Projected, *positions, 0, &[][..], 0),
                    P::Media { ingress, shape } => (
                        R::Encoded,
                        shape.decoder_positions,
                        ingress.placeholder_token_id,
                        ingress.patch_grid.as_slice(),
                        shape.execution_workspace_scalars,
                    ),
                };
            QwenPartView {
                role,
                positions,
                placeholder,
                grid: crate::qwen::vl::positions::GridRows::Tuples(grid),
                workspace_scalars,
                modality: self.prepared.parts()[index].modality(),
            }
        })
    }
}
impl<'a, T> PreparedCompositeInput<'a, T, crate::media_plan::QwenHybridInputPartPlan> {
    pub(crate) fn qwen_parts(self) -> impl ExactSizeIterator<Item = QwenPartView<'a>> + Clone {
        use crate::media_plan::{qwen::QwenPartRole as R, QwenHybridInputPartPlan as P};
        (0..self.admission.len()).map(move |index| {
            let Some(ordinary) = self.admission.ordinary else {
                return QwenPartView::original(
                    self.admission
                        .original
                        .expect("closed admission")
                        .part(index),
                );
            };
            let (role, positions, placeholder, grid, workspace_scalars) =
                match &ordinary.parts()[index] {
                    P::TextTokens { positions } => (R::Tokens, *positions, 0, &[][..], 0),
                    P::Projected { positions, .. } => (R::Projected, *positions, 0, &[][..], 0),
                    P::Media { ingress, shape } => (
                        R::Encoded,
                        shape.decoder_positions,
                        ingress.placeholder_token_id,
                        ingress.patch_grid.as_slice(),
                        shape.execution_workspace_scalars,
                    ),
                };
            QwenPartView {
                role,
                positions,
                placeholder,
                grid: crate::qwen::vl::positions::GridRows::Tuples(grid),
                workspace_scalars,
                modality: self.prepared.parts()[index].modality(),
            }
        })
    }
}

impl<A, B> crate::routed_text::RoutedConstructionParameters<B> for PreparedCompositeArchitecture<A>
where B: NeuralBackend, A: ArchitectureParameters<B, DefinitionError = eredu_nn::Error> {
    fn install_construction_parameters(&mut self, source: crate::routed_text::RetainedRoutedDescription) {
        self.construction_parameters = Some(source);
    }
}
