//! Backend-neutral mutable model-state contracts.
//!
//! Architectures declare state geometry through [`StateLayout`]. Runtime
//! policies select a residency realization, while concrete backends retain
//! their native layer-state and tensor types.

use std::{marker::PhantomData, ops::Range};

use eredu_core::{
    cache::{
        LayerCachePolicy, PromptCacheError, PromptCacheModelIdentity, PromptCacheStateSegment,
        PromptCacheTopology, StateComponentPolicy, StateTensorRole,
    },
    LayerSchedule,
};
use eredu_nn::NeuralBackend;

/// Architecture-declared placement of one contiguous mutable-state range.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArchitectureStatePlacement {
    /// Partition the state range in lockstep with one execution group's units.
    GroupUnits {
        /// Canonical execution-group slot.
        group: usize,
    },
    /// Attach the complete state range to the realized architecture output owner.
    OutputOwner,
}

/// One architecture-authored rule in a mutable-state partition plan.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureStatePartitionRule {
    layers: Range<usize>,
    placement: ArchitectureStatePlacement,
}

impl ArchitectureStatePartitionRule {
    /// Aligns a state range one-for-one with an execution group's unit indices.
    pub fn group_units(group: usize, layers: Range<usize>) -> Self {
        Self {
            layers,
            placement: ArchitectureStatePlacement::GroupUnits { group },
        }
    }

    /// Attaches a complete state range to the architecture output owner.
    pub fn output_owner(layers: Range<usize>) -> Self {
        Self {
            layers,
            placement: ArchitectureStatePlacement::OutputOwner,
        }
    }

    /// Returns the architecture-global state-layer range governed by this rule.
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Returns the rule's neutral placement semantics.
    pub const fn placement(&self) -> ArchitectureStatePlacement {
        self.placement
    }
}

/// Complete architecture-authored mutable-state partition policy.
///
/// Resolution validates that the rules cover the supplied [`StateLayout`]
/// exactly once and that unit-aligned ranges match their execution groups.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureStatePartitionPlan {
    rules: Vec<ArchitectureStatePartitionRule>,
}

impl ArchitectureStatePartitionPlan {
    /// Collects the architecture's state placement rules in declaration order.
    pub fn new(rules: impl IntoIterator<Item = ArchitectureStatePartitionRule>) -> Self {
        Self {
            rules: rules.into_iter().collect(),
        }
    }

    /// Returns the declared state placement rules.
    pub fn rules(&self) -> &[ArchitectureStatePartitionRule] {
        &self.rules
    }
}

/// Invalid architecture-authored mutable-state partition policy.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArchitectureStatePartitionError {
    /// The plan contains no placement rules.
    #[error("architecture state partition plan must contain at least one rule")]
    EmptyPlan,
    /// A rule selected no state layers.
    #[error("architecture state partition rule has empty range {start}..{end}")]
    EmptyRange {
        /// Inclusive invalid range start.
        start: usize,
        /// Exclusive invalid range end.
        end: usize,
    },
    /// A rule selected state layers outside the complete layout.
    #[error("architecture state partition range {start}..{end} exceeds the {layers}-layer layout")]
    RangeOutOfBounds {
        /// Inclusive invalid range start.
        start: usize,
        /// Exclusive invalid range end.
        end: usize,
        /// Complete state-layout length.
        layers: usize,
    },
    /// Two rules selected the same state layer.
    #[error(
        "architecture state partition range starts at {start}, before prior frontier {frontier}"
    )]
    OverlappingRange {
        /// Inclusive overlapping range start.
        start: usize,
        /// End of the preceding declared range.
        frontier: usize,
    },
    /// No rule selected one or more state layers.
    #[error("architecture state layer {layer} is not assigned by the partition plan")]
    UnassignedLayer {
        /// First state layer without a rule.
        layer: usize,
    },
    /// A unit-aligned rule named a nonexistent execution group.
    #[error("architecture state partition references unknown execution group {group}")]
    UnknownGroup {
        /// Missing canonical execution-group slot.
        group: usize,
    },
    /// A unit-aligned state range and its execution group have different lengths.
    #[error(
        "architecture state range {start}..{end} has {} layers but group {group} has {units} units",
        end - start
    )]
    GroupLengthMismatch {
        /// Canonical execution-group slot.
        group: usize,
        /// Inclusive state-range start.
        start: usize,
        /// Exclusive state-range end.
        end: usize,
        /// Execution-group unit count.
        units: usize,
    },
    /// This partition's selected state ranges cannot use the contiguous state representation.
    #[error(
        "architecture state partition selects discontiguous ranges ending at {frontier} and starting at {start}"
    )]
    DiscontiguousSelection {
        /// End of the preceding selected range.
        frontier: usize,
        /// Start of the next selected range.
        start: usize,
    },
    /// The selected state layout could not be sliced from the complete layout.
    #[error("architecture state partition layout is invalid: {0}")]
    InvalidLayout(String),
}

/// Stable name assigned to the implicit segment of a simple state layout.
pub const DEFAULT_STATE_SEGMENT_ID: &str = "state";

/// Validated stable identity of one contiguous mutable-state segment.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct StateSegmentId(String);

impl StateSegmentId {
    /// Creates a non-empty stable segment identity.
    pub fn new(id: impl Into<String>) -> Result<Self, StateError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(StateError::EmptySegmentId);
        }
        Ok(Self(id))
    }

    /// Returns the stable segment name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StateSegmentId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Lifetime policy attached to a named mutable-state segment.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum StateSegmentLifetime {
    /// State survives from one model input or frame to the next.
    Persistent,
    /// State is reused within one frame and reset at the frame boundary.
    FrameLocal,
}

/// One named contiguous range in an architecture's state layout.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StateSegmentSpec {
    id: StateSegmentId,
    layers: Range<usize>,
    lifetime: StateSegmentLifetime,
    processed_token_offset: i32,
}

impl StateSegmentSpec {
    /// Creates a non-empty segment range.
    pub fn new(
        id: impl Into<String>,
        layers: Range<usize>,
        lifetime: StateSegmentLifetime,
        processed_token_offset: i32,
    ) -> Result<Self, StateError> {
        let id = StateSegmentId::new(id)?;
        if layers.is_empty() {
            return Err(StateError::EmptySegmentRange {
                segment: id,
                start: layers.start,
                end: layers.end,
            });
        }
        if processed_token_offset > 0 {
            return Err(StateError::PositiveSegmentOffset {
                segment: id,
                offset: processed_token_offset,
            });
        }
        Ok(Self {
            id,
            layers,
            lifetime,
            processed_token_offset,
        })
    }

    /// Returns the stable segment identity.
    pub const fn id(&self) -> &StateSegmentId {
        &self.id
    }

    /// Returns the architecture-global state-layer range.
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Returns whether the segment persists or resets at a frame boundary.
    pub const fn lifetime(&self) -> StateSegmentLifetime {
        self.lifetime
    }

    /// Returns the segment's processed-token delta from the persisted prefix.
    pub const fn processed_token_offset(&self) -> i32 {
        self.processed_token_offset
    }
}

/// Complete ordered mutable-state geometry owned by one model instance.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StateLayout {
    layers: LayerSchedule<LayerCachePolicy>,
    components: Vec<Vec<StateComponentPolicy>>,
    segments: Vec<StateSegmentSpec>,
}

impl StateLayout {
    /// Creates and validates a simple ordered layout with one persistent
    /// segment named [`DEFAULT_STATE_SEGMENT_ID`].
    pub fn new(layers: LayerSchedule<LayerCachePolicy>) -> Result<Self, StateError> {
        if layers.is_empty() {
            return Err(StateError::EmptyLayout);
        }
        let count = layers.len();
        Self::segmented(
            layers,
            [StateSegmentSpec::new(
                DEFAULT_STATE_SEGMENT_ID,
                0..count,
                StateSegmentLifetime::Persistent,
                0,
            )?],
        )
    }

    /// Creates an ordered layout partitioned into named contiguous segments.
    ///
    /// Segment declarations are sorted into layer order and must form an exact,
    /// non-overlapping partition of every state-bearing layer. Stable segment
    /// identity and lifetime therefore participate in layout equality and
    /// runtime-state compatibility checks.
    pub fn segmented(
        layers: LayerSchedule<LayerCachePolicy>,
        segments: impl IntoIterator<Item = StateSegmentSpec>,
    ) -> Result<Self, StateError> {
        if layers.is_empty() {
            return Err(StateError::EmptyLayout);
        }
        for (layer, policy) in layers.iter().enumerate() {
            policy
                .validate()
                .map_err(|error| StateError::InvalidLayer {
                    layer,
                    reason: error.to_string(),
                })?;
        }
        let components = layers.iter().map(LayerCachePolicy::components).collect();
        let mut segments = segments.into_iter().collect::<Vec<_>>();
        if segments.is_empty() {
            return Err(StateError::EmptySegments);
        }
        segments.sort_by(|left, right| {
            left.layers
                .start
                .cmp(&right.layers.start)
                .then_with(|| left.layers.end.cmp(&right.layers.end))
                .then_with(|| left.id.cmp(&right.id))
        });
        let mut identities = std::collections::BTreeSet::new();
        let mut frontier = 0usize;
        for segment in &segments {
            if !identities.insert(segment.id.clone()) {
                return Err(StateError::DuplicateSegment {
                    segment: segment.id.clone(),
                });
            }
            if segment.layers.end > layers.len() {
                return Err(StateError::SegmentOutOfBounds {
                    segment: segment.id.clone(),
                    start: segment.layers.start,
                    end: segment.layers.end,
                    layers: layers.len(),
                });
            }
            if segment.layers.start < frontier {
                return Err(StateError::OverlappingSegment {
                    segment: segment.id.clone(),
                    start: segment.layers.start,
                    frontier,
                });
            }
            if segment.layers.start > frontier {
                return Err(StateError::UnassignedStateLayer { layer: frontier });
            }
            frontier = segment.layers.end;
        }
        if frontier != layers.len() {
            return Err(StateError::UnassignedStateLayer { layer: frontier });
        }
        Ok(Self {
            layers,
            components,
            segments,
        })
    }

    /// Returns the number of architecture-global layers represented here.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns whether this layout has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns one layer's exact state policy.
    pub fn layer(&self, layer: usize) -> Option<&LayerCachePolicy> {
        self.layers.get(layer)
    }

    /// Borrows the portable ordered layer schedule.
    pub const fn layers(&self) -> &LayerSchedule<LayerCachePolicy> {
        &self.layers
    }

    /// Returns ordered named semantic components for one layer.
    pub fn components(&self, layer: usize) -> Option<&[StateComponentPolicy]> {
        self.components.get(layer).map(Vec::as_slice)
    }

    /// Returns named state segments in deterministic layer order.
    pub fn segments(&self) -> &[StateSegmentSpec] {
        &self.segments
    }

    /// Resolves one named state segment.
    pub fn segment(&self, id: &StateSegmentId) -> Option<&StateSegmentSpec> {
        self.segments.iter().find(|segment| segment.id() == id)
    }

    /// Resolves the unique named segment containing one state layer.
    pub fn segment_for_layer(&self, layer: usize) -> Option<&StateSegmentSpec> {
        self.segments
            .iter()
            .find(|segment| segment.layers.contains(&layer))
    }

    /// Expands architecture-declared segment frontiers into layer order.
    pub fn layer_prefix_offsets(&self) -> Vec<i32> {
        let mut offsets = Vec::with_capacity(self.len());
        for segment in &self.segments {
            offsets.extend(std::iter::repeat_n(
                segment.processed_token_offset(),
                segment.layers.len(),
            ));
        }
        offsets
    }

    /// Selects a contiguous architecture-global range while preserving the
    /// intersecting segment identities, lifetimes, and token frontiers.
    pub fn slice(&self, layers: Range<usize>) -> Result<Self, StateError> {
        if layers.is_empty() || layers.end > self.len() {
            return Err(StateError::InvalidLayoutSlice {
                start: layers.start,
                end: layers.end,
                layers: self.len(),
            });
        }
        let policies = self
            .layers
            .iter()
            .skip(layers.start)
            .take(layers.len())
            .cloned()
            .collect::<Vec<_>>();
        let mut segments = Vec::new();
        for segment in &self.segments {
            let start = segment.layers.start.max(layers.start);
            let end = segment.layers.end.min(layers.end);
            if start < end {
                segments.push(StateSegmentSpec::new(
                    segment.id.as_str(),
                    start - layers.start..end - layers.start,
                    segment.lifetime,
                    segment.processed_token_offset,
                )?);
            }
        }
        Self::segmented(
            LayerSchedule::new(policies.len(), policies)
                .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
            segments,
        )
    }
}

/// Concrete layer state capable of exposing backend-native retained tensors.
pub trait RuntimeLayerState<B: NeuralBackend> {
    /// Allocation-free iterator returned for one layer.
    type RetainedValues<'a>: Iterator<Item = &'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;

    /// Borrows tensors that must remain alive through this layer's submission.
    fn retained_values(&self) -> Self::RetainedValues<'_>;
}

/// Reset capability for one concrete backend-native layer state.
///
/// Reset drops the semantic contents of the layer state without replacing its
/// concrete cache type or inspecting backend-native values on the host.
pub trait ResettableRuntimeLayerState<B: NeuralBackend>: RuntimeLayerState<B> {
    /// Clears this layer state to its initial empty value.
    fn reset(&mut self) -> Result<(), StateError>;
}

/// Mutable access to architecture-declared fixed state components.
///
/// Operators address semantic roles rather than backend storage. Concrete
/// realizations keep native tensors and may combine these slots with an
/// append-only attention cache in the same layer state.
pub trait RuntimeStateComponents<B: NeuralBackend>: RuntimeLayerState<B> {
    /// Current absolute token frontier for this layer.
    fn position(&self) -> i32;

    /// Borrows the optional tensor slot for one declared fixed component.
    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<B::Tensor>, StateError>;

    /// Advances a fixed-state-only layer after a successful operator call.
    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError>;
}

/// Mutable state realization consumed by generic resident and layerwise engines.
pub trait RuntimeState<B: NeuralBackend> {
    /// Concrete iterator retaining native values for one execution unit.
    type RetainedValues<'a>: Iterator<Item = &'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;

    /// Returns the exact layout used to create this realization.
    fn layout(&self) -> &StateLayout;

    /// Borrows tensors retained by one execution unit without cloning handles.
    ///
    /// The flat ordinal addresses policy storage while `address` preserves
    /// architecture-group semantics for composite and shared state.
    fn retained_values(
        &self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError>;
}

/// Additive backend mechanism for realizing an architecture-declared state layout.
///
/// Ordinary key/value architectures need not implement this extension: it is
/// selected only by composition that requires a distinct concrete state
/// representation, such as a layout combining attention and fixed components.
pub trait ArchitectureStateFactory<B: NeuralBackend> {
    /// Concrete state returned by this realization mechanism.
    type State: RuntimeState<B>;
    /// Backend-specific construction failure.
    type Error;

    /// Allocates native state for the exact selected architecture layout.
    fn realize(&mut self, layout: &StateLayout) -> Result<Self::State, Self::Error>;
}

/// Failure while selecting and realizing architecture-authored mutable state.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ArchitectureStateRealizationError<ArchitectureError, FactoryError> {
    /// The architecture could not derive its authoritative state layout.
    #[error("architecture state layout selection failed")]
    Architecture(#[source] ArchitectureError),
    /// The selected backend mechanism could not realize the layout.
    #[error("backend state realization failed")]
    Factory(#[source] FactoryError),
    /// The backend returned state whose layout differs from the selected value.
    #[error("backend state realization changed the selected architecture layout")]
    LayoutMismatch,
}

/// Selects the architecture's exact state layout and realizes it through an
/// explicitly supplied additive mechanism.
pub fn realize_architecture_state<B, M, F>(
    architecture: &M,
    factory: &mut F,
) -> Result<F::State, ArchitectureStateRealizationError<M::DefinitionError, F::Error>>
where
    B: NeuralBackend,
    M: crate::ArchitectureParameters<B>,
    F: ArchitectureStateFactory<B>,
{
    let layout = architecture
        .state_layout()
        .map_err(ArchitectureStateRealizationError::Architecture)?;
    let state = factory
        .realize(&layout)
        .map_err(ArchitectureStateRealizationError::Factory)?;
    if state.layout() != &layout {
        return Err(ArchitectureStateRealizationError::LayoutMismatch);
    }
    Ok(state)
}

/// Named-segment reset supported by a concrete runtime-state realization.
pub trait ResettableRuntimeState<B: NeuralBackend>: RuntimeState<B> {
    /// Resets every layer in exactly one declared state segment.
    fn reset_segment(&mut self, segment: &StateSegmentId) -> Result<(), StateError>;
}

/// Optional capability for architectures with one indexed state per layer.
pub trait LayerRuntimeState<B: NeuralBackend>: RuntimeState<B> {
    /// Concrete monomorphized layer state used by the architecture.
    type LayerState: RuntimeLayerState<B>;

    /// Mutably borrows one architecture-global layer state.
    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError>;
}

/// Fully device-resident state with one concrete value per architecture layer.
#[derive(Debug)]
pub struct DeviceState<B: NeuralBackend, L> {
    layout: StateLayout,
    layers: Vec<L>,
    backend: PhantomData<fn() -> B>,
}

impl<B: NeuralBackend, L: Clone> Clone for DeviceState<B, L> {
    fn clone(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            layers: self.layers.clone(),
            backend: PhantomData,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.layout.clone_from(&source.layout);
        self.layers.clone_from(&source.layers);
    }
}

impl<B: NeuralBackend, L> DeviceState<B, L> {
    /// Realizes every layer through a backend-specific construction closure.
    pub fn create<E>(
        layout: StateLayout,
        mut create: impl FnMut(usize, &LayerCachePolicy) -> Result<L, E>,
    ) -> Result<Self, E> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| create(layer, policy))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout,
            layers,
            backend: PhantomData,
        })
    }
}

impl<B, L> RuntimeState<B> for DeviceState<B, L>
where
    B: NeuralBackend,
    L: RuntimeLayerState<B>,
{
    type RetainedValues<'a>
        = L::RetainedValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn retained_values(
        &self,
        _ordinal: usize,
        address: crate::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        let layer = address.index();
        self.layers
            .get(layer)
            .map(|layer| layer.retained_values())
            .ok_or(StateError::UnknownLayer {
                layer,
                count: self.layers.len(),
            })
    }
}

impl<B, L> LayerRuntimeState<B> for DeviceState<B, L>
where
    B: NeuralBackend,
    L: RuntimeLayerState<B>,
{
    type LayerState = L;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }
}

impl<B, L> ResettableRuntimeState<B> for DeviceState<B, L>
where
    B: NeuralBackend,
    L: ResettableRuntimeLayerState<B>,
{
    fn reset_segment(&mut self, segment: &StateSegmentId) -> Result<(), StateError> {
        let range = self
            .layout
            .segment(segment)
            .map(StateSegmentSpec::layers)
            .ok_or_else(|| StateError::UnknownSegment {
                segment: segment.clone(),
            })?;
        for layer in &mut self.layers[range] {
            layer.reset()?;
        }
        Ok(())
    }
}

impl<B: NeuralBackend, L> AsRef<[L]> for DeviceState<B, L> {
    fn as_ref(&self) -> &[L] {
        &self.layers
    }
}

impl<B: NeuralBackend, L> AsMut<[L]> for DeviceState<B, L> {
    fn as_mut(&mut self) -> &mut [L] {
        &mut self.layers
    }
}

/// Architecture and placement identity used to derive persistence identity.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModelStateIdentity {
    /// Stable architecture family.
    model_family: String,
    /// Effective normalized model type.
    effective_model_type: String,
    /// Cache-relevant architecture fingerprint.
    architecture_fingerprint: String,
    /// Total architecture layer count.
    layer_count: usize,
    /// Inclusive first global layer owned by this runtime instance.
    global_layer_start: usize,
    /// Attention sink or pinned-prefix token count.
    sink_tokens: usize,
    /// Rank-local distributed placement.
    topology: PromptCacheTopology,
}

impl ModelStateIdentity {
    /// Creates a validated architecture and placement identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_family: impl Into<String>,
        effective_model_type: impl Into<String>,
        architecture_fingerprint: impl Into<String>,
        layer_count: usize,
        global_layer_start: usize,
        sink_tokens: usize,
        topology: PromptCacheTopology,
    ) -> Result<Self, PromptCacheError> {
        let model_family = model_family.into();
        let effective_model_type = effective_model_type.into();
        let architecture_fingerprint = architecture_fingerprint.into();
        if model_family.trim().is_empty()
            || effective_model_type.trim().is_empty()
            || architecture_fingerprint.trim().is_empty()
        {
            return Err(PromptCacheError::Malformed(
                "model-state identity strings must be non-empty".into(),
            ));
        }
        if layer_count == 0 || global_layer_start > layer_count {
            return Err(PromptCacheError::Malformed(format!(
                "model-state layer start {global_layer_start} is invalid for {layer_count} layers"
            )));
        }
        topology.validate()?;
        Ok(Self {
            model_family,
            effective_model_type,
            architecture_fingerprint,
            layer_count,
            global_layer_start,
            sink_tokens,
            topology,
        })
    }

    /// Total architecture layer count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Inclusive first global layer owned by this runtime instance.
    pub const fn global_layer_start(&self) -> usize {
        self.global_layer_start
    }

    /// Combines architecture identity, placement, and exact state geometry.
    pub fn prompt_cache_identity(
        &self,
        layout: &StateLayout,
    ) -> Result<PromptCacheModelIdentity, PromptCacheError> {
        let global_layer_end = self
            .global_layer_start
            .checked_add(layout.len())
            .ok_or_else(|| PromptCacheError::Malformed("owned layer range overflowed".into()))?;
        PromptCacheModelIdentity::new(
            self.model_family.clone(),
            self.effective_model_type.clone(),
            self.architecture_fingerprint.clone(),
            self.layer_count,
            self.global_layer_start,
            global_layer_end,
            self.sink_tokens,
            self.topology.clone(),
            layout.layers().clone(),
            layout.layer_prefix_offsets(),
            layout
                .segments()
                .iter()
                .map(|segment| {
                    PromptCacheStateSegment::new(segment.id().as_str(), segment.layers())
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}

/// Invalid architecture state geometry or runtime access.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum StateError {
    /// A model declared no state-bearing layer slots.
    #[error("runtime state layout must contain at least one layer")]
    EmptyLayout,
    /// A composite layout declared no named state segments.
    #[error("runtime state layout must contain at least one named segment")]
    EmptySegments,
    /// A state segment identity was empty or whitespace-only.
    #[error("runtime state segment identity must not be empty")]
    EmptySegmentId,
    /// A state segment range contained no layers.
    #[error("runtime state segment {segment:?} has empty layer range {start}..{end}")]
    EmptySegmentRange {
        /// Invalid segment identity.
        segment: StateSegmentId,
        /// Inclusive range start.
        start: usize,
        /// Exclusive range end.
        end: usize,
    },
    /// A state segment claimed to be ahead of the persisted token prefix.
    #[error("runtime state segment {segment:?} has positive processed-token offset {offset}")]
    PositiveSegmentOffset {
        /// Invalid segment identity.
        segment: StateSegmentId,
        /// Invalid positive processed-token offset.
        offset: i32,
    },
    /// A requested sub-layout range was empty or outside the source layout.
    #[error("runtime state layout cannot select range {start}..{end} from {layers} layers")]
    InvalidLayoutSlice {
        /// Inclusive requested layer start.
        start: usize,
        /// Exclusive requested layer end.
        end: usize,
        /// Available source layer count.
        layers: usize,
    },
    /// Two state segments used the same stable identity.
    #[error("runtime state segment {segment:?} is declared more than once")]
    DuplicateSegment {
        /// Duplicated identity.
        segment: StateSegmentId,
    },
    /// A state segment addressed layers outside the layout.
    #[error(
        "runtime state segment {segment:?} range {start}..{end} exceeds {layers} layout layers"
    )]
    SegmentOutOfBounds {
        /// Invalid segment identity.
        segment: StateSegmentId,
        /// Inclusive range start.
        start: usize,
        /// Exclusive range end.
        end: usize,
        /// Total layout layer count.
        layers: usize,
    },
    /// A state segment overlapped an earlier segment in layer order.
    #[error(
        "runtime state segment {segment:?} starts at layer {start}, before prior frontier {frontier}"
    )]
    OverlappingSegment {
        /// Overlapping segment identity.
        segment: StateSegmentId,
        /// Inclusive range start.
        start: usize,
        /// End of the prior segment.
        frontier: usize,
    },
    /// No named segment owned one layer in the state layout.
    #[error("runtime state layer {layer} is not assigned to a named segment")]
    UnassignedStateLayer {
        /// First unassigned layer.
        layer: usize,
    },
    /// A reset requested a segment absent from the realized layout.
    #[error("runtime state layout has no segment {segment:?}")]
    UnknownSegment {
        /// Requested segment identity.
        segment: StateSegmentId,
    },
    /// A concrete backend failed while clearing a declared state segment.
    #[error("runtime state reset failed: {0}")]
    ResetFailed(String),
    /// One layer supplied an invalid portable state policy.
    #[error("invalid runtime state policy for layer {layer}: {reason}")]
    InvalidLayer {
        /// Invalid global layer index.
        layer: usize,
        /// Validation detail.
        reason: String,
    },
    /// A residency plan violates a finite-resource invariant.
    #[error("invalid runtime state residency plan: {0}")]
    InvalidResidency(String),
    /// A runtime requested a layer outside the realized layout.
    #[error("runtime state layer {layer} is outside the {count}-layer layout")]
    UnknownLayer {
        /// Requested layer.
        layer: usize,
        /// Realized layer count.
        count: usize,
    },
    /// A layer state does not declare the requested fixed component.
    #[error("runtime state layer does not declare fixed component {role:?}")]
    UnknownComponent {
        /// Requested semantic component.
        role: StateTensorRole,
    },
    /// A fixed-state token frontier could not be advanced safely.
    #[error("invalid fixed-state advance: {0}")]
    InvalidAdvance(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{AttentionPolicy, LayerSchedule};

    fn layout() -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(
                2,
                vec![
                    LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
                    LayerCachePolicy::key_value(
                        AttentionPolicy::from_sliding_window(Some(16)).unwrap(),
                        2,
                        8,
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn prompt_identity_is_derived_from_layout_and_placement() {
        let layout = layout();
        let identity = ModelStateIdentity {
            model_family: "fixture".into(),
            effective_model_type: "fixture-v1".into(),
            architecture_fingerprint: "geometry-1".into(),
            layer_count: 4,
            global_layer_start: 1,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        }
        .prompt_cache_identity(&layout)
        .unwrap();
        assert_eq!(identity.global_layer_start(), 1);
        assert_eq!(identity.global_layer_end(), 3);
        assert_eq!(identity.layer_layout(), layout.layers());
    }

    #[test]
    fn prompt_identity_derives_offsets_from_segments() {
        let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap();
        let layout = StateLayout::segmented(
            LayerSchedule::new(2, vec![policy.clone(), policy]).unwrap(),
            [
                StateSegmentSpec::new("target", 0..1, StateSegmentLifetime::Persistent, 0).unwrap(),
                StateSegmentSpec::new("prediction", 1..2, StateSegmentLifetime::Persistent, -1)
                    .unwrap(),
            ],
        )
        .unwrap();
        let identity = ModelStateIdentity {
            model_family: "fixture".into(),
            effective_model_type: "fixture-v1".into(),
            architecture_fingerprint: "geometry-1".into(),
            layer_count: 2,
            global_layer_start: 0,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        }
        .prompt_cache_identity(&layout)
        .unwrap();
        assert_eq!(identity.layer_prefix_offsets(), [0, -1]);
        assert_eq!(identity.state_segments().len(), 2);
        assert_eq!(identity.state_segments()[0].id(), "target");
        assert_eq!(identity.state_segments()[0].layers(), 0..1);
        assert_eq!(identity.state_segments()[1].id(), "prediction");
        assert_eq!(identity.state_segments()[1].layers(), 1..2);

        let prediction = identity.select_state_segment("prediction").unwrap();
        assert_eq!(prediction.global_layer_start(), 1);
        assert_eq!(prediction.global_layer_end(), 2);
        assert_eq!(prediction.layer_prefix_offsets(), [-1]);
        assert_eq!(prediction.state_segments()[0].id(), "prediction");
        assert_eq!(prediction.state_segments()[0].layers(), 0..1);
    }

    #[test]
    fn state_layout_exposes_stable_semantic_component_names() {
        let layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![
                    LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 16, 8)
                        .unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let names = layout
            .components(0)
            .unwrap()
            .iter()
            .map(|component| component.role().stable_name())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["attention.compressed_latent", "attention.rotary_keys"]
        );
    }

    fn four_layer_schedule() -> LayerSchedule<LayerCachePolicy> {
        LayerSchedule::new(
            4,
            (0..4)
                .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap())
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn composite_state_segments_are_canonical_and_cover_every_layer() {
        let layout = StateLayout::segmented(
            four_layer_schedule(),
            [
                StateSegmentSpec::new("depth", 2..4, StateSegmentLifetime::FrameLocal, 0).unwrap(),
                StateSegmentSpec::new("temporal", 0..2, StateSegmentLifetime::Persistent, 0)
                    .unwrap(),
            ],
        )
        .unwrap();

        assert_eq!(
            layout
                .segments()
                .iter()
                .map(|segment| (segment.id().as_str(), segment.layers(), segment.lifetime()))
                .collect::<Vec<_>>(),
            [
                ("temporal", 0..2, StateSegmentLifetime::Persistent),
                ("depth", 2..4, StateSegmentLifetime::FrameLocal),
            ]
        );
        assert_eq!(
            layout.segment_for_layer(0).unwrap().id().as_str(),
            "temporal"
        );
        assert_eq!(layout.segment_for_layer(3).unwrap().id().as_str(), "depth");
        assert!(layout.segment_for_layer(4).is_none());
    }

    #[test]
    fn state_layout_slice_preserves_and_rebases_segment_frontiers() {
        let layout = StateLayout::segmented(
            four_layer_schedule(),
            [
                StateSegmentSpec::new("target", 0..2, StateSegmentLifetime::Persistent, 0).unwrap(),
                StateSegmentSpec::new("prediction", 2..4, StateSegmentLifetime::Persistent, -1)
                    .unwrap(),
            ],
        )
        .unwrap();

        let sliced = layout.slice(1..4).unwrap();
        assert_eq!(sliced.segments()[0].layers(), 0..1);
        assert_eq!(sliced.segments()[1].layers(), 1..3);
        assert_eq!(sliced.layer_prefix_offsets(), [0, -1, -1]);
    }

    #[test]
    fn segment_identity_lifetime_and_offset_participate_in_layout_equality() {
        let layout = |depth_name, lifetime, offset| {
            StateLayout::segmented(
                four_layer_schedule(),
                [
                    StateSegmentSpec::new("temporal", 0..2, StateSegmentLifetime::Persistent, 0)
                        .unwrap(),
                    StateSegmentSpec::new(depth_name, 2..4, lifetime, offset).unwrap(),
                ],
            )
            .unwrap()
        };
        let canonical = layout("depth", StateSegmentLifetime::FrameLocal, 0);
        assert_ne!(
            canonical,
            layout("predictor", StateSegmentLifetime::FrameLocal, 0)
        );
        assert_ne!(
            canonical,
            layout("depth", StateSegmentLifetime::Persistent, 0)
        );
        assert_ne!(
            canonical,
            layout("depth", StateSegmentLifetime::FrameLocal, -1)
        );
    }

    #[test]
    fn malformed_segment_partitions_fail_closed() {
        let duplicate = StateLayout::segmented(
            four_layer_schedule(),
            [
                StateSegmentSpec::new("cache", 0..2, StateSegmentLifetime::Persistent, 0).unwrap(),
                StateSegmentSpec::new("cache", 2..4, StateSegmentLifetime::FrameLocal, 0).unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(duplicate, StateError::DuplicateSegment { .. }));

        let overlap = StateLayout::segmented(
            four_layer_schedule(),
            [
                StateSegmentSpec::new("left", 0..3, StateSegmentLifetime::Persistent, 0).unwrap(),
                StateSegmentSpec::new("right", 2..4, StateSegmentLifetime::FrameLocal, 0).unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(overlap, StateError::OverlappingSegment { .. }));

        let gap = StateLayout::segmented(
            four_layer_schedule(),
            [
                StateSegmentSpec::new("left", 0..1, StateSegmentLifetime::Persistent, 0).unwrap(),
                StateSegmentSpec::new("right", 2..4, StateSegmentLifetime::FrameLocal, 0).unwrap(),
            ],
        )
        .unwrap_err();
        assert_eq!(gap, StateError::UnassignedStateLayer { layer: 1 });

        let outside = StateLayout::segmented(
            four_layer_schedule(),
            [StateSegmentSpec::new("all", 0..5, StateSegmentLifetime::Persistent, 0).unwrap()],
        )
        .unwrap_err();
        assert!(matches!(outside, StateError::SegmentOutOfBounds { .. }));
    }
}
