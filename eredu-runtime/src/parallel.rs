//! Backend-neutral semantic parameter sharding and rank-local layouts.
//!
//! Architectures describe physical checkpoint members in logical groups. An
//! execution backend may then realize the resulting placement without knowing
//! projection names, attention geometry, or other model-family semantics.

mod construction;
mod linear_expansion;
pub use linear_expansion::expand_linear_format_parameter_groups_with_metadata;
pub use construction::{
    aligned_partition_units_with_metadata, module_parameter_group_with_metadata,
    partition_parameter_group_chunks_with_metadata,
    partitioned_module_parameter_group_with_metadata, partitioned_projection_group_with_metadata,
    projection_parameter_group_with_metadata, segmented_projection_group_with_metadata,
};

use std::{
    collections::BTreeMap,
    ops::Range,
};

use eredu_checkpoint::LinearFormat;
use eredu_nn::{LinearFormatSpec, ParameterMetadataView, ParameterVisitor, Parameterized, Tensor};

/// Architecture-neutral information for one rank-local parallel model.
#[derive(Debug, Clone)]
pub struct ParallelModelInfo<T> {
    topology: T,
    effective_model_type: String,
    owned_tensors: Vec<String>,
    local_parameter_bytes: u64,
    global_parameter_bytes: u64,
    pinned_device_parameter_bytes: u64,
    maximum_device_parameter_bytes: u64,
}

impl<T> ParallelModelInfo<T> {
    /// Creates a complete rank-local parallel model summary.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        topology: T,
        effective_model_type: impl Into<String>,
        owned_tensors: Vec<String>,
        local_parameter_bytes: u64,
        global_parameter_bytes: u64,
        pinned_device_parameter_bytes: u64,
        maximum_device_parameter_bytes: u64,
    ) -> Self {
        Self {
            topology,
            effective_model_type: effective_model_type.into(),
            owned_tensors,
            local_parameter_bytes,
            global_parameter_bytes,
            pinned_device_parameter_bytes,
            maximum_device_parameter_bytes,
        }
    }

    /// Returns the backend's concrete topology value unchanged.
    pub fn topology(&self) -> T
    where
        T: Clone,
    {
        self.topology.clone()
    }

    /// Returns the parsed implementation or nested text-model type.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Returns exact checkpoint targets owned or replicated by this rank.
    pub fn owned_tensors(&self) -> &[String] {
        &self.owned_tensors
    }

    /// Returns planned rank-local parameter bytes across static and execution units.
    pub const fn local_parameter_bytes(&self) -> u64 {
        self.local_parameter_bytes
    }

    /// Returns the unsharded model parameter bytes represented by this checkpoint.
    pub const fn global_parameter_bytes(&self) -> u64 {
        self.global_parameter_bytes
    }

    /// Returns rank-local parameter bytes permanently pinned on the execution device.
    pub const fn pinned_device_parameter_bytes(&self) -> u64 {
        self.pinned_device_parameter_bytes
    }

    /// Returns the maximum planned rank-local parameter footprint on device.
    pub const fn maximum_device_parameter_bytes(&self) -> u64 {
        self.maximum_device_parameter_bytes
    }
}

/// Semantic role of a logical parameter group.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ParameterRole {
    /// Small or otherwise non-partitioned state.
    Replicated,
    /// Projection whose output features are rank-local.
    ColumnProjection,
    /// Projection whose input features are rank-local and whose output is reduced.
    RowProjection,
    /// Token embedding or output projection partitioned by vocabulary.
    Vocabulary,
    /// Query, key, or value heads.
    AttentionHeads,
    /// Dense feed-forward intermediate channels shared by input and output projections.
    FeedForwardIntermediate,
    /// Routed expert intermediate channels partitioned over the expert axis.
    ExpertIntermediate,
    /// Expert-parallel packed projections whose complete output rows are
    /// owned by tensor ranks, while every projection retains its full input.
    ExpertOutput,
    /// Always-on expert intermediate channels replicated over the expert axis.
    SharedExpertIntermediate,
    /// State-space, convolution, or recurrent channels.
    Channels,
    /// A fused tensor containing independently partitioned segments.
    Segmented,
}

/// Logical sharding behavior for a parameterized affine projection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProjectionSharding {
    /// Keep every projection parameter complete on every rank.
    Replicated,
    /// Partition projection output features.
    Column,
    /// Partition projection input features and replicate output bias.
    Row,
}

/// Rank-local selection rule for one physical checkpoint tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MemberSharding {
    /// Materialize the complete member on every tensor-parallel rank.
    Replicated,
    /// Split an axis into equal contiguous shards.
    Equal {
        /// Source tensor axis to partition.
        axis: usize,
    },
    /// Split an axis into balanced, potentially uneven contiguous ranges.
    Balanced {
        /// Source tensor axis to partition.
        axis: usize,
    },
    /// Map the group's logical partition onto one physical tensor axis.
    Partitioned {
        /// Source tensor axis to partition.
        axis: usize,
    },
    /// Map shared logical units to physical chunks, retaining a short final chunk.
    PartitionedChunks {
        /// Physical source axis.
        axis: usize,
        /// Physical elements in each complete logical unit.
        chunk_size: usize,
    },
    /// Map the same group-level logical range into each supplied source segment.
    PartitionedSegments {
        /// Source tensor axis containing the fused segments.
        axis: usize,
        /// Ordered, non-overlapping physical source ranges.
        segments: Vec<Range<usize>>,
    },
    /// Apply shared logical chunks independently to each fused source segment.
    PartitionedChunkSegments {
        /// Physical source axis.
        axis: usize,
        /// Ordered, non-overlapping physical source ranges.
        segments: Vec<Range<usize>>,
        /// Elements per complete logical unit within each segment.
        chunk_size: usize,
    },
    /// Partition each supplied source range independently.
    Segmented {
        /// Source tensor axis containing the fused segments.
        axis: usize,
        /// Ordered, non-overlapping source ranges.
        segments: Vec<Range<usize>>,
    },
}

/// One physical tensor belonging to a logical parameter group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterMemberSpec {
    target: String,
    global_shape: Vec<usize>,
    sharding: MemberSharding,
    linear_companion: Option<eredu_nn::LinearCompanionRole>,
    linear_companion_of: Option<String>,
    linear_row_layout: eredu_nn::LinearRowLayout,
}

impl ParameterMemberSpec {
    /// Creates a member with an exact pre-selection checkpoint shape.
    pub fn new(
        target: impl Into<String>,
        global_shape: impl Into<Vec<usize>>,
        sharding: MemberSharding,
    ) -> Self {
        Self {
            target: target.into(),
            global_shape: global_shape.into(),
            sharding,
            linear_companion: None,
            linear_companion_of: None,
            linear_row_layout: eredu_nn::LinearRowLayout::Contiguous,
        }
    }

    fn with_sharding(mut self, sharding: MemberSharding) -> Self {
        self.sharding = sharding;
        self
    }

    fn with_linear_row_layout(mut self, layout: eredu_nn::LinearRowLayout) -> Self {
        self.linear_row_layout = layout;
        self
    }

    /// Independent row-block origins of the primary encoded weight.
    pub const fn linear_row_layout(&self) -> eredu_nn::LinearRowLayout {
        self.linear_row_layout
    }

    fn with_linear_companion(mut self, role: eredu_nn::LinearCompanionRole, primary: &str) -> Self {
        self.linear_companion = Some(role);
        self.linear_companion_of = Some(primary.to_owned());
        self
    }

    /// Returns the rewritten checkpoint target.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Returns the complete source shape.
    pub fn global_shape(&self) -> &[usize] {
        &self.global_shape
    }

    /// Returns the requested rank-local selection.
    pub const fn sharding(&self) -> &MemberSharding {
        &self.sharding
    }

    /// Returns this member's encoded-linear companion role, when present.
    pub const fn linear_companion(&self) -> Option<eredu_nn::LinearCompanionRole> {
        self.linear_companion
    }

    /// Returns the primary linear weight owning this companion.
    pub fn linear_companion_of(&self) -> Option<&str> {
        self.linear_companion_of.as_deref()
    }
}

/// Atomic logical parameter and all of its physical checkpoint companions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterGroupSpec {
    logical_name: String,
    role: ParameterRole,
    partition_units: Option<usize>,
    members: Vec<ParameterMemberSpec>,
}

impl ParameterGroupSpec {
    /// Copies this already validated immutable declaration through exact owned destinations.
    pub fn clone_with_metadata(&self,context:&eredu_nn::workspace::WorkspaceContext)->Result<Self,eredu_nn::Error>{
        linear_expansion::copy_group(self,context)
    }

    /// Creates a non-empty logical group.
    pub fn new(
        logical_name: impl Into<String>,
        role: ParameterRole,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, ParallelPlanError> {
        Self::build(logical_name.into(), role, None, members)
    }

    /// Creates a group whose partitioned members share one logical domain.
    pub fn partitioned(
        logical_name: impl Into<String>,
        role: ParameterRole,
        units: usize,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, ParallelPlanError> {
        if units == 0 {
            return Err(ParallelPlanError::InvalidGroup(
                "parallel logical partition must contain at least one unit".into(),
            ));
        }
        Self::build(logical_name.into(), role, Some(units), members)
    }

    /// Validates and moves already funded declarations through the shared group worker.
    pub fn from_owned_with_metadata(
        logical_name: String,
        role: ParameterRole,
        units: Option<usize>,
        members: Vec<ParameterMemberSpec>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        construction::finish_owned(logical_name, role, units, members, context)
    }

    /// Reuses this group's owned metadata with an exact shared partition.
    /// The same validation as `partitioned` runs before the result is returned.
    pub fn into_partitioned(self, units: usize) -> Result<Self, ParallelPlanError> {
        Self::partitioned(self.logical_name, self.role, units, self.members)
    }

    /// Moves this group's metadata into an admitted checked partition result.
    pub fn into_partitioned_with_metadata(
        self,
        units: usize,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        construction::finish_owned(
            self.logical_name,
            self.role,
            Some(units),
            self.members,
            context,
        )
    }

    fn build(
        logical_name: String,
        role: ParameterRole,
        partition_units: Option<usize>,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, ParallelPlanError> {
        construction::validate_name(&logical_name).map_err(construction::GroupIssue::ordinary)?;
        let members = members.into_iter().collect::<Vec<_>>();
        construction::validate_members(
            &logical_name,
            partition_units,
            &members,
            construction::Ordinary,
        )?;
        Ok(Self {
            logical_name,
            role,
            partition_units,
            members,
        })
    }

    /// Returns the stable logical name.
    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    /// Returns the semantic role.
    pub const fn role(&self) -> ParameterRole {
        self.role
    }

    /// Returns the shared logical-unit count, when the group is partitioned.
    pub const fn partition_units(&self) -> Option<usize> {
        self.partition_units
    }

    /// Returns physical checkpoint members.
    pub fn members(&self) -> &[ParameterMemberSpec] {
        &self.members
    }
}

/// Describes every parameter in a neutral module as one logical group.
pub fn module_parameter_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    module: &M,
    mut sharding: impl FnMut(ParameterMetadataView<'_>, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    struct Collector<'a, F> {
        members: Vec<ParameterMemberSpec>,
        sharding: &'a mut F,
        error: Option<ParallelPlanError>,
    }

    impl<'a, 'tensor, T, F> ParameterVisitor<'tensor, T> for Collector<'a, F>
    where
        T: Tensor,
        F: FnMut(ParameterMetadataView<'_>, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
    {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'tensor T) {
            if self.error.is_some() {
                return;
            }
            match construction::member(
                construction::MemberSource::borrowed(metadata),
                value.shape(),
                construction::Ordinary,
                |shape| (self.sharding)(metadata, shape),
            ) {
                Ok(member) => self.members.push(member),
                Err(cause) => self.error = Some(cause),
            }
        }
    }

    let mut collector = Collector {
        members: Vec::new(),
        sharding: &mut sharding,
        error: None,
    };
    module.visit_parameters(&mut collector)?;
    if let Some(error) = collector.error {
        return Err(error);
    }
    ParameterGroupSpec::new(logical_name, role, collector.members)
}

/// Describes every parameter in a neutral module as one shared logical partition.
pub fn partitioned_module_parameter_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    preferred_units: usize,
    module: &M,
    mut sharding: impl FnMut(ParameterMetadataView<'_>, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    if preferred_units == 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned module group has zero preferred units".into(),
        ));
    }
    struct Collector<'a, F> {
        members: Vec<ParameterMemberSpec>,
        sharding: &'a mut F,
        error: Option<ParallelPlanError>,
    }
    impl<'a, 'tensor, T, F> ParameterVisitor<'tensor, T> for Collector<'a, F>
    where
        T: Tensor,
        F: FnMut(ParameterMetadataView<'_>, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
    {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'tensor T) {
            if self.error.is_some() {
                return;
            }
            match construction::member(
                construction::MemberSource::borrowed(metadata),
                value.shape(),
                construction::Ordinary,
                |shape| (self.sharding)(metadata, shape),
            ) {
                Ok(member) => self.members.push(member),
                Err(cause) => self.error = Some(cause),
            }
        }
    }
    let mut collector = Collector {
        members: Vec::new(),
        sharding: &mut sharding,
        error: None,
    };
    module.visit_parameters(&mut collector)?;
    if let Some(error) = collector.error {
        return Err(error);
    }
    partitioned_group_with_preferred_units(logical_name, role, preferred_units, collector.members)
}

/// Describes one affine projection and all encoding companions.
pub fn projection_parameter_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    module: &M,
    placement: ProjectionSharding,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    module_parameter_group(logical_name, role, module, |metadata, shape| {
        construction::projection_sharding(placement, metadata.id().as_str(), shape)
            .map_err(construction::GroupIssue::ordinary)
    })
}

/// Describes projections that consume one shared logical partition.
pub fn partitioned_projection_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    projections: &[(&M, ProjectionSharding)],
    preferred_units: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    if preferred_units == 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned projection group has zero preferred units".into(),
        ));
    }
    let mut members = Vec::new();
    for (module, placement) in projections {
        let group = projection_parameter_group::<T, M>("projection", role, *module, *placement)?;
        for member in group.members {
            let sharding =
                construction::partitioned_sharding(*placement, member.global_shape.len());
            members.push(member.with_sharding(sharding));
        }
    }
    partitioned_group_with_preferred_units(logical_name, role, preferred_units, members)
}

/// Describes a component-major fused column projection and its row-parallel
/// output as one shared logical partition.
///
/// The same ordered segment selection is attached to the fused weight and all
/// encoding companions exposed by the module. The row projection consumes the
/// corresponding local hidden partition and is reduced once by the backend.
pub fn segmented_projection_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    fused: &M,
    row: &M,
    segments: Vec<Range<usize>>,
    preferred_units: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    let previous_end = construction::segment_width(preferred_units, &segments)
        .map_err(construction::GroupIssue::ordinary)?;

    let fused_group =
        projection_parameter_group::<T, M>("fused", role, fused, ProjectionSharding::Column)?;
    let row_group = projection_parameter_group::<T, M>("row", role, row, ProjectionSharding::Row)?;
    assemble_segmented_projection_group(
        logical_name,
        role,
        fused_group,
        row_group,
        segments,
        preferred_units,
        previous_end,
    )
}

#[allow(clippy::too_many_arguments)]
fn assemble_segmented_projection_group(
    logical_name: impl Into<String>,
    role: ParameterRole,
    fused_group: ParameterGroupSpec,
    row_group: ParameterGroupSpec,
    segments: Vec<Range<usize>>,
    units: usize,
    expected_fused_width: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let members = construction::segmented_members(
        fused_group,
        row_group,
        &segments,
        expected_fused_width,
        construction::Ordinary,
    )?;
    partitioned_group_with_preferred_units(logical_name, role, units, members)
}

/// Returns the finest legal logical-unit count for an aligned partition.
pub fn aligned_partition_units(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    required_alignment: usize,
) -> Result<usize, ParallelPlanError> {
    partition_units(
        name,
        semantic_units,
        elements_per_unit,
        required_alignment,
        false,
    )
}

/// Returns aligned logical units for an encoding that permits a partial final
/// block in the complete tensor. An unaligned total is retained as one unit:
/// equal multi-rank slices would otherwise start inside an encoded block.
pub fn aligned_partition_units_with_tail(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    required_alignment: usize,
) -> Result<usize, ParallelPlanError> {
    partition_units(
        name,
        semantic_units,
        elements_per_unit,
        required_alignment,
        true,
    )
}

fn partition_units(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    required_alignment: usize,
    allow_tail: bool,
) -> Result<usize, ParallelPlanError> {
    partition_units_with(
        name,
        semantic_units,
        elements_per_unit,
        required_alignment,
        allow_tail,
        |args| ParallelPlanError::InvalidGroup(args.to_string()),
    )
}

fn partition_units_with<E>(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    required_alignment: usize,
    allow_tail: bool,
    mut invalid: impl FnMut(std::fmt::Arguments<'_>) -> E,
) -> Result<usize, E> {
    if semantic_units == 0 || elements_per_unit == 0 || required_alignment == 0 {
        return Err(invalid(format_args!(
            "{name} aligned partition dimensions must be positive, got units={semantic_units}, width={elements_per_unit}, alignment={required_alignment}"
        )));
    }
    let units_per_partition =
        required_alignment / greatest_common_divisor(elements_per_unit, required_alignment);
    if !semantic_units.is_multiple_of(units_per_partition) {
        if allow_tail {
            return Ok(1);
        }
        return Err(invalid(format_args!(
            "{name} has {semantic_units} semantic units of width {elements_per_unit}, which cannot form complete alignment-{required_alignment} partitions"
        )));
    }
    Ok(semantic_units / units_per_partition)
}

/// Replaces uniform physical units with exact per-member chunk widths.
/// Replicated parameters remain outside the partition. Every supplied physical
/// width must describe the same logical unit count, including its final tail.
pub fn partition_parameter_group_chunks(
    group: ParameterGroupSpec,
    units: usize,
    mut width: impl FnMut(&ParameterMemberSpec) -> Result<usize, ParallelPlanError>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    partition_parameter_group_chunks_with_source(group, units, |member, _| width(member))
}

/// Converts uniform members while borrowing the complete unchanged source group.
/// The callback can resolve a companion's primary without cloning its metadata.
/// It runs exactly once for each partitioned member, in source order. Widths are
/// retained until validation of every source member succeeds; member/name/shape
/// and segment owners are then moved into the destination.
pub fn partition_parameter_group_chunks_with_source(
    group: ParameterGroupSpec,
    units: usize,
    width: impl FnMut(&ParameterMemberSpec, &[ParameterMemberSpec]) -> Result<usize, ParallelPlanError>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let widths = construction::chunk_widths(&group, construction::Ordinary, width)?;
    construction::apply_chunk_widths(group, widths).into_partitioned(units)
}

/// Fixed physical-chunk validation failure; no diagnostic allocation occurs.
#[derive(Debug,Clone,Copy,Eq,PartialEq,thiserror::Error)]
pub enum PartitionChunkRangeError {
    /// The physical extent, width or interval is invalid.
    #[error("invalid physical chunk range")] Invalid,
    /// The logical interval exceeds the physical chunk population.
    #[error("logical chunk range exceeds physical extent")] Extent,
    /// A physical boundary cannot be represented.
    #[error("physical chunk boundary overflows")] Overflow,
}

/// Maps a logical chunk interval to exact physical coordinates without padding.
pub fn partition_chunk_range(
    extent: usize,
    chunk_size: usize,
    logical: Range<usize>,
) -> Result<Range<usize>, PartitionChunkRangeError> {
    if extent == 0 || chunk_size == 0 || logical.start > logical.end {
        return Err(PartitionChunkRangeError::Invalid);
    }
    let units = extent.div_ceil(chunk_size);
    if logical.end > units {
        return Err(PartitionChunkRangeError::Extent);
    }
    let boundary = |index: usize| {
        if index == units {
            Ok(extent)
        } else {
            index.checked_mul(chunk_size).ok_or_else(|| {
                PartitionChunkRangeError::Overflow
            })
        }
    };
    Ok(boundary(logical.start)?..boundary(logical.end)?)
}

/// Rewrites semantic dense matrix declarations into their authoritative
/// physical checkpoint representation and publishes every required companion
/// in the same atomic parameter group.
pub fn expand_linear_format_parameter_groups(
    groups: Vec<ParameterGroupSpec>,
    declaration: impl Fn(&ParameterMemberSpec) -> Result<Option<LinearFormatSpec>, ParallelPlanError>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    linear_expansion::ordinary(groups, declaration)
}

fn partitioned_group_with_preferred_units(
    logical_name: impl Into<String>,
    role: ParameterRole,
    preferred_units: usize,
    members: Vec<ParameterMemberSpec>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let units = construction::preferred_units(&members, preferred_units)
        .map_err(construction::GroupIssue::ordinary)?;
    ParameterGroupSpec::partitioned(logical_name, role, units, members)
}

const fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Behavior when a requested shard is not legal for the current TP size.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum ShardingPolicy {
    /// Reject the complete plan with a precise shape/alignment error.
    #[default]
    Require,
    /// Replicate the complete logical parameter group.
    ReplicateUnsupported,
}

/// Backend-neutral placement decision for one physical tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TensorPlacement {
    /// Materialize the complete tensor on every rank.
    Replicated,
    /// Materialize the complete tensor on this rank.
    Local,
    /// Intentionally omit this tensor on this rank.
    Omit,
    /// Materialize the complete tensor only on one global rank.
    Rank {
        /// Owning global rank.
        rank: usize,
    },
    /// Materialize an equal contiguous source-tensor slice.
    Shard {
        /// Source tensor axis being sharded.
        axis: usize,
        /// Shard index.
        index: usize,
        /// Total shard count.
        parts: usize,
    },
    /// Materialize an explicit contiguous source-tensor range.
    Range {
        /// Source tensor axis being sliced.
        axis: usize,
        /// Inclusive element offset on `axis`.
        start: usize,
        /// Exclusive element offset on `axis`.
        end: usize,
    },
    /// Materialize selected source-tensor indices in the supplied order.
    Indices {
        /// Source tensor axis being selected.
        axis: usize,
        /// Distinct source indices in local output order.
        indices: Vec<usize>,
    },
}

/// Fixed admission/capacity failures and the actual host vector refusal.
#[derive(Debug,thiserror::Error)]
pub enum ParallelLayoutStorageError {
    /// The original construction account refused a producer.
    #[error("{0}")] Funding(#[from] eredu_core::HostMetadataFundingError),
    /// The host allocator refused the admitted vector layout.
    #[error("parallel layout allocation: {0}")] Allocation(#[from] std::collections::TryReserveError),
    /// The actual vector capacity differs from the qualified request.
    #[error("parallel layout capacity differs from its request")] Capacity,
}

/// Rank-local shape and placement for one planned physical tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalTensorLayout<P = TensorPlacement> {
    logical_name: String,
    role: ParameterRole,
    global_shape: Vec<usize>,
    local_shape: Vec<usize>,
    placement: P,
    additional_placements: Vec<TensorPlacement>,
    logical_units: Option<usize>,
    logical_range: Option<Range<usize>>,
    partition_chunk_size: Option<usize>,
    fell_back_to_replication: bool,
}

impl<P> LocalTensorLayout<P> {
    /// Creates one validated-planner output entry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        logical_name: impl Into<String>,
        role: ParameterRole,
        global_shape: Vec<usize>,
        local_shape: Vec<usize>,
        placement: P,
        logical_units: Option<usize>,
        logical_range: Option<Range<usize>>,
        fell_back_to_replication: bool,
    ) -> Self {
        Self {
            logical_name: logical_name.into(),
            role,
            global_shape,
            local_shape,
            placement,
            additional_placements: Vec::new(),
            logical_units,
            logical_range,
            partition_chunk_size: None,
            fell_back_to_replication,
        }
    }

    /// Returns the logical parameter group name.
    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    /// Returns the semantic parameter role.
    pub const fn role(&self) -> ParameterRole {
        self.role
    }

    /// Returns the checkpoint-global shape.
    pub fn global_shape(&self) -> &[usize] {
        &self.global_shape
    }

    /// Returns the shape materialized on this rank.
    pub fn local_shape(&self) -> &[usize] {
        &self.local_shape
    }

    /// Returns the backend-realized placement.
    pub const fn placement(&self) -> &P {
        &self.placement
    }

    /// Returns independent checkpoint-global selections applied before the
    /// primary placement.
    ///
    /// This is used when distinct parallel axes own distinct tensor axes, such
    /// as EP selection of packed experts followed by TP selection of each
    /// expert matrix. Empty means the primary placement is complete.
    pub fn additional_placements(&self) -> &[TensorPlacement] {
        &self.additional_placements
    }

    /// Adds one exact checkpoint-global selection preceding the primary
    /// placement.
    pub fn with_additional_placement(self, placement: TensorPlacement) -> Self {
        self.additional_placement(placement, None).expect("ordinary additional placement allocation")
    }

    /// Adds one selection through the same vector producer with prospective
    /// backing/control funding. The source's enclosing owner retains the payer.
    pub fn try_with_additional_placement(self, placement: TensorPlacement, funding: &eredu_core::HostMetadataFunding)
        -> Result<Self, ParallelLayoutStorageError> {
        self.additional_placement(placement, Some(funding))
    }
    fn additional_placement(mut self, placement: TensorPlacement, funding: Option<&eredu_core::HostMetadataFunding>)
        -> Result<Self, ParallelLayoutStorageError> {
        use eredu_core::HostMetadataFundingError as E;
        let required = self.additional_placements.len().checked_add(1).ok_or(E::Overflow)?;
        if let Some(funding) = funding {
            funding.reserve_metadata(std::mem::size_of::<(Self, TensorPlacement, Option<&eredu_core::HostMetadataFunding>,
                usize, Result<Self,ParallelLayoutStorageError>, std::collections::TryReserveError)>())?;
        }
        if required > self.additional_placements.capacity() {
            let target = required.max(4).checked_next_power_of_two().ok_or(E::Overflow)?;
            if let Some(funding) = funding {
                funding.reserve_metadata(std::alloc::Layout::array::<TensorPlacement>(target).map_err(|_|E::Overflow)?.size())?;
            }
            self.additional_placements.try_reserve_exact(target-self.additional_placements.len())?;
            if self.additional_placements.capacity()!=target {return Err(ParallelLayoutStorageError::Capacity);}
        }
        self.additional_placements.push(placement);
        Ok(self)
    }

    /// Returns the rank-local range in the parameter group's semantic domain.
    pub fn logical_range(&self) -> Option<&Range<usize>> {
        self.logical_range.as_ref()
    }

    /// Returns the size of the complete semantic partition domain.
    pub const fn logical_units(&self) -> Option<usize> {
        self.logical_units
    }

    /// Expands uniform partition units into an architecture-supplied logical
    /// width. Encoded weights and companions may share one partition unit per
    /// quantization block, rather than one unit per scalar component.
    ///
    /// Missing ranges and explicit chunk mappings are rejected: neither tensor
    /// shape nor a nonuniform physical partition determines this correspondence.
    pub fn expanded_logical_range(
        &self,
        global_width: usize,
    ) -> Result<Range<usize>, ParallelPlanError> {
        let invalid =
            |reason| ParallelPlanError::InvalidTensor(format!("{}: {reason}", self.logical_name));
        let units = self
            .logical_units
            .filter(|&units| units > 0 && global_width > 0 && global_width.is_multiple_of(units))
            .ok_or_else(|| invalid("partition units do not uniformly divide the logical width"))?;
        if self.partition_chunk_size.is_some() {
            return Err(invalid(
                "explicit partition chunks require their own logical mapping",
            ));
        }
        let range = self
            .logical_range
            .as_ref()
            .filter(|range| !range.is_empty() && range.end <= units)
            .ok_or_else(|| invalid("missing or invalid exact partition range"))?;
        let width = global_width / units;
        // Both endpoints are bounded by `units`; the expanded endpoints cannot
        // exceed the caller's representable global width.
        Ok(range.start * width..range.end * width)
    }

    /// Retains the physical width of each complete logical partition chunk.
    /// `None` denotes the existing uniform-unit mapping.
    pub fn with_partition_chunk_size(mut self, width: Option<usize>) -> Self {
        self.partition_chunk_size = width;
        self
    }

    /// Physical chunk width on the primary partition axis, when declared.
    pub const fn partition_chunk_size(&self) -> Option<usize> {
        self.partition_chunk_size
    }

    /// Returns whether permissive planning replicated an unsupported shard.
    pub const fn fell_back_to_replication(&self) -> bool {
        self.fell_back_to_replication
    }
}

/// Complete rank-local model geometry produced alongside checkpoint placement.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalModelLayout<P = TensorPlacement> {
    tensors: eredu_collections::ordered_map::Map<String, LocalTensorLayout<P>>,
}

mod transform_source;
pub use transform_source::derive_transform_source_layout;

impl<P> Default for LocalModelLayout<P> {
    fn default() -> Self {
        Self {
            tensors: eredu_collections::ordered_map::Map::new(),
        }
    }
}

impl<P> LocalModelLayout<P> {
    /// Returns whether a physical target has already been planned.
    pub fn contains(&self, target: &str) -> bool {
        self.tensors.contains_key(target)
    }

    /// Inserts one planner-produced physical layout.
    pub fn insert(&mut self, target: String, layout: LocalTensorLayout<P>) {
        self.tensors.insert(target, layout);
    }

    /// Inserts a source-produced row after admitting its exact ordered node.
    /// Nested target/layout storage and the enclosing source custody belong to
    /// the caller. Existing rows are replaced without allocating another node.
    pub fn try_insert_with_funding(&mut self, target: String, layout: LocalTensorLayout<P>,
        funding: &eredu_core::HostMetadataFunding) -> Result<(), eredu_core::HostMetadataFundingError> {
        use eredu_core::HostMetadataFundingError as E;
        funding.reserve_metadata(self.tensors.insertion_control_bytes(&target).ok_or(E::Overflow)?)?;
        self.tensors.try_insert_with(target, layout, |layout| funding.reserve_metadata(layout.size()))
            .map(|_|()).map_err(|cause| match cause {
                eredu_collections::ordered_map::TryInsertError::SizeOverflow=>E::Overflow,
                eredu_collections::ordered_map::TryInsertError::Funding(cause)=>cause,
            })
    }

    /// Returns one physical tensor layout by rewritten target name.
    pub fn tensor(&self, target: &str) -> Option<&LocalTensorLayout<P>> {
        self.tensors.get(target)
    }

    /// Iterates physical layouts in deterministic target-name order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &LocalTensorLayout<P>)> {
        self.tensors
            .iter()
            .map(|(target, layout)| (target.as_str(), layout))
    }

    /// Returns the number of planned physical tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether no physical tensors were planned.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Invalid architecture-declared parallel semantics.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ParallelPlanError {
    /// Exact physical-chunk failure from the allocation-free source worker.
    #[error("{0}")]
    Chunk(#[from] PartitionChunkRangeError),

    /// Canonical source traversal could not establish complete parameter coverage.
    #[error(transparent)]
    Source(#[from] eredu_nn::ParameterSourceError),

    /// A logical group is empty, ambiguous, or internally inconsistent.
    #[error("{0}")]
    InvalidGroup(String),
    /// A backend-native parameter exposed invalid logical geometry.
    #[error("{0}")]
    InvalidTensor(String),
}

#[cfg(test)]
mod tests {
    use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding};

    use super::*;

    #[test]
    fn logical_range_expansion_preserves_encoded_component_ownership() {
        let layout = |units, range| {
            LocalTensorLayout::new(
                "expert.intermediate",
                ParameterRole::ExpertIntermediate,
                vec![4, 64, 8],
                vec![4, 32, 8],
                TensorPlacement::Range {
                    axis: 1,
                    start: 32,
                    end: 64,
                },
                units,
                range,
                false,
            )
        };
        for units in [2, 4, 64] {
            assert_eq!(
                layout(Some(units), Some(units / 2..units))
                    .expanded_logical_range(64)
                    .unwrap(),
                32..64,
            );
        }
        for (units, range) in [
            (None, None),
            (Some(0), Some(0..0)),
            (Some(3), Some(0..1)),
            (Some(2), None),
            (Some(2), Some(1..1)),
            (Some(2), Some(1..3)),
            (Some(2), Some(2..1)),
        ] {
            assert!(layout(units, range).expanded_logical_range(64).is_err());
        }
        assert!(layout(Some(2), Some(0..1))
            .expanded_logical_range(0)
            .is_err());
        assert!(layout(Some(2), Some(0..1))
            .with_partition_chunk_size(Some(32))
            .expanded_logical_range(64)
            .is_err());
        assert_eq!(
            layout(Some(1), Some(0..1))
                .expanded_logical_range(usize::MAX)
                .unwrap(),
            0..usize::MAX,
        );
    }

    #[test]
    fn fused_fp8_tails_keep_each_scale_partition_and_metadata() {
        let layout = eredu_nn::LinearRowLayout::equal_partitions(2).unwrap();
        let group = ParameterGroupSpec::partitioned(
            "fused",
            ParameterRole::ExpertIntermediate,
            3,
            [ParameterMemberSpec::new(
                "matrix",
                [3, 518, 130],
                MemberSharding::PartitionedChunkSegments {
                    axis: 1,
                    segments: vec![0..259, 259..518],
                    chunk_size: 128,
                },
            )],
        )
        .unwrap();
        let expanded = expand_linear_format_parameter_groups(vec![group], |_| {
            Ok(Some(
                LinearFormatSpec::scaled(
                    LinearFormat::E4M3BlockFp8(
                        BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint)
                            .unwrap(),
                    ),
                    eredu_nn::ParameterSpec::trainable("scale").unwrap(),
                )
                .unwrap()
                .with_row_layout(layout)
                .unwrap(),
            ))
        })
        .unwrap();
        assert_eq!(expanded[0].partition_units(), Some(3));
        assert_eq!(expanded[0].members()[0].linear_row_layout(), layout);
        assert_eq!(expanded[0].members()[1].global_shape(), [3, 6, 2]);
        assert_eq!(
            expanded[0].members()[1].sharding(),
            &MemberSharding::PartitionedChunkSegments {
                axis: 1,
                segments: vec![0..3, 3..6],
                chunk_size: 1,
            }
        );
        let spec = eredu_nn::GroupedProjectionSpec::new(
            eredu_nn::ParameterSpec::trainable("matrix").unwrap(),
            None,
            LinearFormatSpec::scaled(
                LinearFormat::E4M3BlockFp8(
                    BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
                ),
                eredu_nn::ParameterSpec::trainable("scale").unwrap(),
            )
            .unwrap()
            .with_row_layout(layout)
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            eredu_nn::ParameterMetadata::from_spec(spec.weight(), true).linear_row_layout,
            layout
        );
    }

    #[test]
    fn chunk_ranges_retain_short_tails_without_padding_or_overflow() {
        assert_eq!(partition_chunk_range(259, 128, 0..2).unwrap(), 0..256);
        assert_eq!(partition_chunk_range(259, 128, 2..3).unwrap(), 256..259);
        assert_eq!(partition_chunk_range(259, 128, 3..3).unwrap(), 259..259);
        let units = usize::MAX.div_ceil(128);
        assert_eq!(
            partition_chunk_range(usize::MAX, 128, units - 1..units).unwrap(),
            usize::MAX - 127..usize::MAX
        );
        for (extent, width, range) in [
            (0, 128, 0..1),
            (259, 0, 0..1),
            (259, 128, 2..1),
            (259, 128, 0..4),
        ] {
            assert!(partition_chunk_range(extent, width, range).is_err());
        }
    }

    #[test]
    fn fp8_chunk_expansion_preserves_shared_weight_and_scale_ownership() {
        for encoding in [
            BlockFp8ScaleEncoding::FloatingPoint,
            BlockFp8ScaleEncoding::Ue8m0,
        ] {
            let group = ParameterGroupSpec::partitioned(
                "ffn",
                ParameterRole::FeedForwardIntermediate,
                1,
                [
                    ParameterMemberSpec::new(
                        "read",
                        [259, 130],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    ParameterMemberSpec::new(
                        "write",
                        [130, 259],
                        MemberSharding::Partitioned { axis: 1 },
                    ),
                ],
            )
            .unwrap();
            let group = partition_parameter_group_chunks(group, 3, |_| Ok(128)).unwrap();
            let format =
                LinearFormat::E4M3BlockFp8(BlockFp8Format::new(128, 128, encoding).unwrap());
            let groups = expand_linear_format_parameter_groups(vec![group], |member| {
                Ok(Some(
                    LinearFormatSpec::scaled(
                        format,
                        eredu_nn::ParameterSpec::trainable(format!("{}_scale", member.target()))
                            .unwrap(),
                    )
                    .unwrap(),
                ))
            })
            .unwrap();
            assert_eq!(groups[0].partition_units(), Some(3));
            let members = groups[0].members();
            assert_eq!(members[1].global_shape(), [3, 2]);
            assert_eq!(members[3].global_shape(), [2, 3]);
            for (index, axis, width) in [(0, 0, 128), (1, 0, 1), (2, 1, 128), (3, 1, 1)] {
                assert_eq!(
                    members[index].sharding(),
                    &MemberSharding::PartitionedChunks {
                        axis,
                        chunk_size: width
                    }
                );
                let extent = members[index].global_shape()[axis];
                let first = partition_chunk_range(extent, width, 0..2).unwrap();
                let last = partition_chunk_range(extent, width, 2..3).unwrap();
                assert_eq!(first.end, last.start);
                assert_eq!(last.end, extent);
            }
            assert_eq!(members[1].linear_companion_of(), Some("read"));
            assert_eq!(members[3].linear_companion_of(), Some("write"));
        }
    }

    #[test]
    fn chunk_groups_reject_mismatched_companions_and_segments() {
        for sharding in [
            MemberSharding::PartitionedChunks {
                axis: 0,
                chunk_size: 0,
            },
            MemberSharding::PartitionedChunks {
                axis: 2,
                chunk_size: 128,
            },
            MemberSharding::PartitionedChunkSegments {
                axis: 0,
                segments: vec![],
                chunk_size: 128,
            },
            MemberSharding::PartitionedChunkSegments {
                axis: 0,
                segments: vec![0..130, 129..259],
                chunk_size: 128,
            },
            MemberSharding::PartitionedChunkSegments {
                axis: 0,
                segments: vec![0..260],
                chunk_size: 128,
            },
        ] {
            assert!(ParameterGroupSpec::partitioned(
                "bad",
                ParameterRole::FeedForwardIntermediate,
                3,
                [ParameterMemberSpec::new("weight", [259, 130], sharding)]
            )
            .is_err());
        }
        assert!(ParameterGroupSpec::partitioned(
            "bad",
            ParameterRole::FeedForwardIntermediate,
            3,
            [
                ParameterMemberSpec::new(
                    "weight",
                    [259, 130],
                    MemberSharding::PartitionedChunks {
                        axis: 0,
                        chunk_size: 128
                    }
                ),
                ParameterMemberSpec::new(
                    "scale",
                    [2, 2],
                    MemberSharding::PartitionedChunks {
                        axis: 0,
                        chunk_size: 1
                    }
                ),
            ]
        )
        .is_err());
    }

    #[test]
    fn gguf_chunk_expansion_uses_byte_coordinates_for_complete_encoded_blocks() {
        for (width, sharding, expected) in [
            (
                288,
                MemberSharding::PartitionedChunks {
                    axis: 1,
                    chunk_size: 128,
                },
                MemberSharding::PartitionedChunks {
                    axis: 1,
                    chunk_size: 136,
                },
            ),
            (
                576,
                MemberSharding::PartitionedChunkSegments {
                    axis: 1,
                    segments: vec![0..288, 288..576],
                    chunk_size: 128,
                },
                MemberSharding::PartitionedChunkSegments {
                    axis: 1,
                    segments: vec![0..306, 306..612],
                    chunk_size: 136,
                },
            ),
        ] {
            let group = ParameterGroupSpec::partitioned(
                "packed",
                ParameterRole::RowProjection,
                3,
                [ParameterMemberSpec::new("matrix", [4, width], sharding)],
            )
            .unwrap();
            let groups = expand_linear_format_parameter_groups(vec![group], |_| {
                Ok(Some(
                    LinearFormatSpec::unscaled(LinearFormat::GgufIQuant {
                        ggml_type: eredu_gguf::GgmlType::Q8_0,
                        endian: eredu_gguf::Endian::Little,
                    })
                    .unwrap(),
                ))
            })
            .unwrap();
            assert_eq!(groups[0].partition_units(), Some(3));
            assert_eq!(groups[0].members()[0].global_shape(), [4, width / 32 * 34]);
            assert_eq!(groups[0].members()[0].sharding(), &expected);
            assert_eq!(partition_chunk_range(306, 136, 2..3).unwrap(), 272..306);
        }
    }

    #[test]
    fn encoded_partial_blocks_remain_whole_partition_units() {
        for (units, width, alignment, expected) in [
            (2, 8, 128, 1),
            (129, 1, 128, 1),
            (6, 64, 128, 3),
            (3, 128, 128, 3),
            (usize::MAX, 128, 128, usize::MAX),
        ] {
            assert_eq!(
                aligned_partition_units_with_tail("encoded", units, width, alignment).unwrap(),
                expected
            );
        }
        assert!(aligned_partition_units("packed", 129, 1, 128).is_err());
        for (units, width, alignment) in [(0, 1, 128), (1, 0, 128), (1, 1, 0)] {
            assert!(aligned_partition_units_with_tail("encoded", units, width, alignment).is_err());
        }
        let groups = vec![ParameterGroupSpec::partitioned(
            "partial",
            ParameterRole::FeedForwardIntermediate,
            aligned_partition_units_with_tail("partial", 130, 1, 128).unwrap(),
            [
                ParameterMemberSpec::new(
                    "read",
                    [130, 129],
                    MemberSharding::Partitioned { axis: 0 },
                ),
                ParameterMemberSpec::new(
                    "write",
                    [129, 130],
                    MemberSharding::Partitioned { axis: 1 },
                ),
            ],
        )
        .unwrap()];
        let format = LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
        );
        let expanded = expand_linear_format_parameter_groups(groups, |member| {
            Ok(Some(
                LinearFormatSpec::scaled(
                    format,
                    eredu_nn::ParameterSpec::trainable(format!("{}_scale", member.target()))
                        .unwrap(),
                )
                .unwrap(),
            ))
        })
        .unwrap();
        assert_eq!(expanded[0].partition_units(), Some(1));
        assert_eq!(expanded[0].members()[0].global_shape(), [130, 129]);
        assert_eq!(expanded[0].members()[1].global_shape(), [2, 2]);
        assert_eq!(expanded[0].members()[2].global_shape(), [129, 130]);
        assert_eq!(expanded[0].members()[3].global_shape(), [2, 2]);
    }

    #[test]
    fn groups_reject_duplicate_physical_targets() {
        let error = ParameterGroupSpec::new(
            "attention",
            ParameterRole::AttentionHeads,
            [
                ParameterMemberSpec::new("q.weight", [8, 8], MemberSharding::Replicated),
                ParameterMemberSpec::new("q.weight", [8, 8], MemberSharding::Replicated),
            ],
        )
        .unwrap_err();
        assert!(error.to_string().contains("repeats tensor target"));
    }

    #[test]
    fn group_partition_contract_is_explicit() {
        assert!(ParameterGroupSpec::new(
            "query",
            ParameterRole::AttentionHeads,
            [ParameterMemberSpec::new(
                "q.weight",
                [8, 8],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )
        .is_err());
        assert!(ParameterGroupSpec::partitioned(
            "query",
            ParameterRole::AttentionHeads,
            4,
            [ParameterMemberSpec::new(
                "q.weight",
                [8, 8],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )
        .is_ok());
    }

    #[test]
    fn preferred_partition_units_follow_every_physical_companion() {
        let group = partitioned_group_with_preferred_units(
            "experts.intermediate",
            ParameterRole::ExpertIntermediate,
            64,
            vec![
                ParameterMemberSpec::new(
                    "experts.up_proj",
                    [4, 64, 16],
                    MemberSharding::Partitioned { axis: 1 },
                ),
                ParameterMemberSpec::new(
                    "experts.down_proj",
                    [4, 16, 8],
                    MemberSharding::Partitioned { axis: 2 },
                ),
                ParameterMemberSpec::new(
                    "experts.down_proj_scales",
                    [4, 16, 2],
                    MemberSharding::Partitioned { axis: 2 },
                ),
            ],
        )
        .unwrap();

        assert_eq!(group.partition_units(), Some(2));
    }

    #[test]
    fn fp8_expansion_uses_architecture_declared_companion_identity() {
        let groups = vec![ParameterGroupSpec::partitioned(
            "query",
            ParameterRole::AttentionHeads,
            8,
            [ParameterMemberSpec::new(
                "opaque_matrix",
                [256, 256],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )
        .unwrap()];
        let format = LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0).unwrap(),
        );

        let expanded = expand_linear_format_parameter_groups(groups, |_| {
            Ok(Some(
                LinearFormatSpec::scaled(
                    format,
                    eredu_nn::ParameterSpec::trainable("opaque_scale").unwrap(),
                )
                .unwrap(),
            ))
        })
        .unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].partition_units(), Some(2));
        assert_eq!(expanded[0].members().len(), 2);
        assert_eq!(
            expanded[0]
                .members()
                .iter()
                .map(ParameterMemberSpec::target)
                .collect::<Vec<_>>(),
            ["opaque_matrix", "opaque_scale"]
        );
        assert_eq!(expanded[0].members()[1].global_shape(), [2, 2]);
        assert_eq!(
            expanded[0].members()[1].sharding(),
            &MemberSharding::Partitioned { axis: 0 }
        );
    }

    #[test]
    fn segmented_projection_applies_identical_ranges_to_every_fused_companion() {
        let fused = ParameterGroupSpec::new(
            "fused",
            ParameterRole::FeedForwardIntermediate,
            [
                ParameterMemberSpec::new(
                    "gate_up.weight",
                    [12, 8],
                    MemberSharding::Equal { axis: 0 },
                ),
                ParameterMemberSpec::new(
                    "gate_up.scales",
                    [12, 2],
                    MemberSharding::Equal { axis: 0 },
                ),
                ParameterMemberSpec::new(
                    "gate_up.biases",
                    [12, 2],
                    MemberSharding::Equal { axis: 0 },
                ),
            ],
        )
        .unwrap();
        let row = ParameterGroupSpec::new(
            "row",
            ParameterRole::FeedForwardIntermediate,
            [
                ParameterMemberSpec::new("down.weight", [8, 6], MemberSharding::Equal { axis: 1 }),
                ParameterMemberSpec::new("down.scales", [8, 2], MemberSharding::Equal { axis: 1 }),
                ParameterMemberSpec::new("down.bias", [8], MemberSharding::Replicated),
            ],
        )
        .unwrap();
        let segments = vec![0..4, 4..8, 8..12];
        let group = assemble_segmented_projection_group(
            "mlp.projections",
            ParameterRole::FeedForwardIntermediate,
            fused,
            row,
            segments.clone(),
            2,
            12,
        )
        .unwrap();
        assert_eq!(group.partition_units(), Some(2));
        for member in &group.members()[..3] {
            assert_eq!(
                member.sharding(),
                &MemberSharding::PartitionedSegments {
                    axis: 0,
                    segments: segments.clone(),
                }
            );
        }
        assert_eq!(
            group.members()[3].sharding(),
            &MemberSharding::Partitioned { axis: 1 }
        );
        assert_eq!(
            group.members()[4].sharding(),
            &MemberSharding::Partitioned { axis: 1 }
        );
        assert_eq!(group.members()[5].sharding(), &MemberSharding::Replicated);
    }

    #[test]
    fn segmented_projection_rejects_one_misaligned_companion_atomically() {
        let fused = ParameterGroupSpec::new(
            "fused",
            ParameterRole::AttentionHeads,
            [
                ParameterMemberSpec::new("qkv.weight", [12, 8], MemberSharding::Equal { axis: 0 }),
                ParameterMemberSpec::new("qkv.scales", [11, 2], MemberSharding::Equal { axis: 0 }),
            ],
        )
        .unwrap();
        let row = ParameterGroupSpec::new(
            "row",
            ParameterRole::AttentionHeads,
            [ParameterMemberSpec::new(
                "output.weight",
                [8, 4],
                MemberSharding::Equal { axis: 1 },
            )],
        )
        .unwrap();
        assert!(matches!(
            assemble_segmented_projection_group(
                "attention.projections",
                ParameterRole::AttentionHeads,
                fused,
                row,
                vec![0..4, 4..8, 8..12],
                2,
                12,
            ),
            Err(ParallelPlanError::InvalidTensor(_))
        ));
    }

    #[test]
    fn parallel_model_info_preserves_opaque_topology_and_accounting() {
        let info = ParallelModelInfo::new(
            (2usize, 1usize),
            "generic",
            vec!["layer.weight".into()],
            10,
            20,
            4,
            8,
        );
        assert_eq!(info.topology(), (2, 1));
        assert_eq!(info.effective_model_type(), "generic");
        assert_eq!(info.owned_tensors(), ["layer.weight"]);
        assert_eq!(info.local_parameter_bytes(), 10);
        assert_eq!(info.global_parameter_bytes(), 20);
        assert_eq!(info.pinned_device_parameter_bytes(), 4);
        assert_eq!(info.maximum_device_parameter_bytes(), 8);
    }
}
