//! Backend-neutral semantic parameter sharding and rank-local layouts.
//!
//! Architectures describe physical checkpoint members in logical groups. An
//! execution backend may then realize the resulting placement without knowing
//! projection names, attention geometry, or other model-family semantics.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use eredu_checkpoint::LinearFormat;
use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized, Tensor};

/// Architecture-neutral information for one rank-local parallel model.
#[derive(Debug, Clone)]
pub struct ParallelModelInfo<T> {
    topology: T,
    model_type: String,
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
        model_type: impl Into<String>,
        owned_tensors: Vec<String>,
        local_parameter_bytes: u64,
        global_parameter_bytes: u64,
        pinned_device_parameter_bytes: u64,
        maximum_device_parameter_bytes: u64,
    ) -> Self {
        Self {
            topology,
            model_type: model_type.into(),
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

    /// Returns the architecture's normalized model type.
    pub fn model_type(&self) -> &str {
        &self.model_type
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
    /// Routed or shared expert intermediate channels.
    ExpertIntermediate,
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
    /// Map the same group-level logical range into each supplied source segment.
    PartitionedSegments {
        /// Source tensor axis containing the fused segments.
        axis: usize,
        /// Ordered, non-overlapping physical source ranges.
        segments: Vec<Range<usize>>,
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
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum LinearFormatCompanions {
    None,
    Scaled { scale: String, bias: Option<String> },
}

/// Architecture declaration of a linear matrix's physical checkpoint format.
///
/// Runtime uses this declaration only for format geometry. Architectures own
/// eligibility and the exact identities of every physical companion.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LinearFormatParameter {
    format: LinearFormat,
    companions: LinearFormatCompanions,
}

impl LinearFormatParameter {
    /// Declares an encoding with no separate physical companions.
    pub const fn unscaled(format: LinearFormat) -> Self {
        Self {
            format,
            companions: LinearFormatCompanions::None,
        }
    }

    /// Declares an encoding with an exact scale target.
    pub fn scaled(format: LinearFormat, scale: impl Into<String>) -> Self {
        Self {
            format,
            companions: LinearFormatCompanions::Scaled {
                scale: scale.into(),
                bias: None,
            },
        }
    }

    /// Declares an encoding with exact scale and affine-bias targets.
    pub fn affine(format: LinearFormat, scale: impl Into<String>, bias: impl Into<String>) -> Self {
        Self {
            format,
            companions: LinearFormatCompanions::Scaled {
                scale: scale.into(),
                bias: Some(bias.into()),
            },
        }
    }

    const fn format(&self) -> LinearFormat {
        self.format
    }

    const fn companions(&self) -> &LinearFormatCompanions {
        &self.companions
    }
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
        }
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

    fn build(
        logical_name: String,
        role: ParameterRole,
        partition_units: Option<usize>,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, ParallelPlanError> {
        if logical_name.trim().is_empty() {
            return Err(ParallelPlanError::InvalidGroup(
                "parallel parameter logical name must not be empty".into(),
            ));
        }
        let members = members.into_iter().collect::<Vec<_>>();
        if members.is_empty() {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "parallel parameter group {logical_name:?} must contain at least one tensor"
            )));
        }
        let mut targets = BTreeSet::new();
        let mut has_partitioned_member = false;
        for member in &members {
            if member.target.trim().is_empty() {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "parallel parameter group {logical_name:?} contains an empty tensor target"
                )));
            }
            if !targets.insert(member.target.clone()) {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "parallel parameter group {logical_name:?} repeats tensor target {:?}",
                    member.target
                )));
            }
            has_partitioned_member |= matches!(
                member.sharding,
                MemberSharding::Partitioned { .. } | MemberSharding::PartitionedSegments { .. }
            );
        }
        if has_partitioned_member != partition_units.is_some() {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "parallel parameter group {logical_name:?} must declare exactly one group-level logical partition for its partitioned members"
            )));
        }
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
    mut sharding: impl FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
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
        F: FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
    {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'tensor T) {
            if self.error.is_some() {
                return;
            }
            let shape = value
                .shape()
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension).map_err(|_| {
                        ParallelPlanError::InvalidTensor(format!(
                            "parameter {} has negative dimension {dimension}",
                            metadata.id.as_str()
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>();
            let shape = match shape {
                Ok(shape) => shape,
                Err(error) => {
                    self.error = Some(error);
                    return;
                }
            };
            match (self.sharding)(&metadata, &shape) {
                Ok(sharding) => self.members.push(ParameterMemberSpec::new(
                    metadata.id.as_str(),
                    shape,
                    sharding,
                )),
                Err(error) => self.error = Some(error),
            }
        }
    }

    let mut collector = Collector {
        members: Vec::new(),
        sharding: &mut sharding,
        error: None,
    };
    module.visit_parameters(&mut collector);
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
    mut sharding: impl FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
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
        F: FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
    {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'tensor T) {
            if self.error.is_some() {
                return;
            }
            let shape = value
                .shape()
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension).map_err(|_| {
                        ParallelPlanError::InvalidTensor(format!(
                            "parameter {} has negative dimension {dimension}",
                            metadata.id.as_str()
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>();
            match shape.and_then(|shape| {
                (self.sharding)(&metadata, &shape)
                    .map(|sharding| ParameterMemberSpec::new(metadata.id.as_str(), shape, sharding))
            }) {
                Ok(member) => self.members.push(member),
                Err(error) => self.error = Some(error),
            }
        }
    }
    let mut collector = Collector {
        members: Vec::new(),
        sharding: &mut sharding,
        error: None,
    };
    module.visit_parameters(&mut collector);
    if let Some(error) = collector.error {
        return Err(error);
    }
    ParameterGroupSpec::partitioned(logical_name, role, preferred_units, collector.members)
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
    module_parameter_group(
        logical_name,
        role,
        module,
        |metadata, shape| match placement {
            ProjectionSharding::Replicated => Ok(MemberSharding::Replicated),
            ProjectionSharding::Column if shape.is_empty() => {
                Err(ParallelPlanError::InvalidTensor(format!(
                    "column projection parameter {} is scalar",
                    metadata.id.as_str()
                )))
            }
            ProjectionSharding::Column => Ok(MemberSharding::Equal { axis: 0 }),
            ProjectionSharding::Row if shape.len() >= 2 => Ok(MemberSharding::Equal { axis: 1 }),
            ProjectionSharding::Row => Ok(MemberSharding::Replicated),
        },
    )
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
    let mut units = preferred_units;
    let mut members = Vec::new();
    for (module, placement) in projections {
        let group = projection_parameter_group::<T, M>("projection", role, *module, *placement)?;
        for member in group.members {
            let sharding = match (placement, member.global_shape.len()) {
                (ProjectionSharding::Replicated, _) | (ProjectionSharding::Row, 0 | 1) => {
                    MemberSharding::Replicated
                }
                (ProjectionSharding::Column, 0) => unreachable!("validated above"),
                (ProjectionSharding::Column, _) => {
                    units = greatest_common_divisor(units, member.global_shape[0]);
                    MemberSharding::Partitioned { axis: 0 }
                }
                (ProjectionSharding::Row, _) => {
                    units = greatest_common_divisor(units, member.global_shape[1]);
                    MemberSharding::Partitioned { axis: 1 }
                }
            };
            members.push(ParameterMemberSpec::new(
                member.target,
                member.global_shape,
                sharding,
            ));
        }
    }
    ParameterGroupSpec::partitioned(logical_name, role, units, members)
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
    if preferred_units == 0 || segments.is_empty() {
        return Err(ParallelPlanError::InvalidGroup(
            "segmented projection requires positive logical units and at least one segment".into(),
        ));
    }
    let mut previous_end = 0usize;
    for segment in &segments {
        if segment.start != previous_end || segment.start >= segment.end {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "segmented projection ranges must be positive, contiguous, and ordered, got {segments:?}"
            )));
        }
        previous_end = segment.end;
    }

    let mut units = preferred_units;
    for segment in &segments {
        units = greatest_common_divisor(units, segment.len());
    }
    let fused_group =
        projection_parameter_group::<T, M>("fused", role, fused, ProjectionSharding::Column)?;
    let row_group = projection_parameter_group::<T, M>("row", role, row, ProjectionSharding::Row)?;
    assemble_segmented_projection_group(
        logical_name,
        role,
        fused_group,
        row_group,
        segments,
        units,
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
    mut units: usize,
    expected_fused_width: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let mut members = Vec::new();
    for member in fused_group.members {
        let dimension = member.global_shape.first().copied().ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "segmented projection parameter {} is scalar",
                member.target
            ))
        })?;
        if dimension != expected_fused_width {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "segmented projection parameter {} has output dimension {dimension}, expected {expected_fused_width}",
                member.target
            )));
        }
        members.push(ParameterMemberSpec::new(
            member.target,
            member.global_shape,
            MemberSharding::PartitionedSegments {
                axis: 0,
                segments: segments.clone(),
            },
        ));
    }
    for member in row_group.members {
        let sharding = if member.global_shape.len() >= 2 {
            units = greatest_common_divisor(units, member.global_shape[1]);
            MemberSharding::Partitioned { axis: 1 }
        } else {
            MemberSharding::Replicated
        };
        members.push(ParameterMemberSpec::new(
            member.target,
            member.global_shape,
            sharding,
        ));
    }
    if units == 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "segmented projection has no common logical partition".into(),
        ));
    }
    ParameterGroupSpec::partitioned(logical_name, role, units, members)
}

/// Returns the finest legal logical-unit count for an aligned partition.
pub fn aligned_partition_units(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    required_alignment: usize,
) -> Result<usize, ParallelPlanError> {
    if semantic_units == 0 || elements_per_unit == 0 || required_alignment == 0 {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "{name} aligned partition dimensions must be positive, got units={semantic_units}, width={elements_per_unit}, alignment={required_alignment}"
        )));
    }
    let units_per_partition =
        required_alignment / greatest_common_divisor(elements_per_unit, required_alignment);
    if !semantic_units.is_multiple_of(units_per_partition) {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "{name} has {semantic_units} semantic units of width {elements_per_unit}, which cannot form complete alignment-{required_alignment} partitions"
        )));
    }
    Ok(semantic_units / units_per_partition)
}

/// Rewrites semantic dense matrix declarations into their authoritative
/// physical checkpoint representation and publishes every required companion
/// in the same atomic parameter group.
pub fn expand_linear_format_parameter_groups(
    groups: Vec<ParameterGroupSpec>,
    declaration: impl Fn(&ParameterMemberSpec) -> Option<LinearFormatParameter>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    groups
        .into_iter()
        .map(|group| {
            let mut members = Vec::new();
            for source in group.members() {
                members.extend(match declaration(source) {
                    Some(declaration) => expand_linear_format_member(source, &declaration)?,
                    None => vec![source.clone()],
                });
            }
            match group.partition_units() {
                Some(mut units) => {
                    for member in &members {
                        match member.sharding() {
                            MemberSharding::Partitioned { axis } => {
                                units =
                                    greatest_common_divisor(units, member.global_shape()[*axis]);
                            }
                            MemberSharding::PartitionedSegments { segments, .. }
                            | MemberSharding::Segmented { segments, .. } => {
                                for segment in segments {
                                    units = greatest_common_divisor(units, segment.len());
                                }
                            }
                            _ => {}
                        }
                    }
                    ParameterGroupSpec::partitioned(
                        group.logical_name(),
                        group.role(),
                        units,
                        members,
                    )
                }
                None => ParameterGroupSpec::new(group.logical_name(), group.role(), members),
            }
        })
        .collect()
}

fn remap_linear_segments(
    sharding: &MemberSharding,
    axis: usize,
    divisor: usize,
    name: &str,
) -> Result<MemberSharding, ParallelPlanError> {
    let remap = |segments: &[Range<usize>]| {
        segments
            .iter()
            .map(|segment| {
                if !segment.start.is_multiple_of(divisor) || !segment.end.is_multiple_of(divisor) {
                    return Err(ParallelPlanError::InvalidTensor(format!(
                        "packed companion {name} segment {segment:?} is not aligned to {divisor}"
                    )));
                }
                Ok(segment.start / divisor..segment.end / divisor)
            })
            .collect::<Result<Vec<_>, _>>()
    };
    match sharding {
        MemberSharding::PartitionedSegments {
            axis: selected,
            segments,
        } if *selected == axis => Ok(MemberSharding::PartitionedSegments {
            axis: *selected,
            segments: remap(segments)?,
        }),
        MemberSharding::Segmented {
            axis: selected,
            segments,
        } if *selected == axis => Ok(MemberSharding::Segmented {
            axis: *selected,
            segments: remap(segments)?,
        }),
        other => Ok(other.clone()),
    }
}

fn expand_linear_format_member(
    source: &ParameterMemberSpec,
    declaration: &LinearFormatParameter,
) -> Result<Vec<ParameterMemberSpec>, ParallelPlanError> {
    let name = source.target();
    let shape = source.global_shape();
    let format = declaration.format();
    if format == LinearFormat::Dense {
        return match declaration.companions() {
            LinearFormatCompanions::None => Ok(vec![source.clone()]),
            LinearFormatCompanions::Scaled { .. } => Err(ParallelPlanError::InvalidGroup(format!(
                "dense linear parameter {name} declares physical companions"
            ))),
        };
    }
    if shape.len() < 2 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "encoded linear parameter {name} must have at least two dimensions"
        )));
    }
    let row_axis = shape.len() - 2;
    let column_axis = shape.len() - 1;
    let invalid = |detail: String| ParallelPlanError::InvalidTensor(detail);
    match format {
        LinearFormat::Dense => unreachable!(),
        LinearFormat::E4M3BlockFp8(fp8) => {
            let LinearFormatCompanions::Scaled { scale, bias: None } = declaration.companions()
            else {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "block-FP8 linear parameter {name} must declare exactly one scale companion"
                )));
            };
            fp8.validate().map_err(|error| invalid(error.to_string()))?;
            let rows = usize::try_from(fp8.block_rows)
                .map_err(|_| invalid(format!("invalid block rows for {name}")))?;
            let columns = usize::try_from(fp8.block_columns)
                .map_err(|_| invalid(format!("invalid block columns for {name}")))?;
            let mut scale_shape = shape.to_vec();
            scale_shape[row_axis] = scale_shape[row_axis].div_ceil(rows);
            scale_shape[column_axis] = scale_shape[column_axis].div_ceil(columns);
            let scale_sharding = remap_linear_segments(source.sharding(), row_axis, rows, name)
                .and_then(|value| remap_linear_segments(&value, column_axis, columns, name))?;
            Ok(vec![
                source.clone(),
                ParameterMemberSpec::new(scale, scale_shape, scale_sharding),
            ])
        }
        LinearFormat::GgufIQuant { ggml_type, .. } => {
            if declaration.companions() != &LinearFormatCompanions::None {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "GGUF linear parameter {name} must not declare companion tensors"
                )));
            }
            let (block_values, block_bytes) = ggml_type
                .block_and_bytes()
                .map_err(|error| invalid(error.to_string()))?;
            let block_values = usize::try_from(block_values)
                .map_err(|_| invalid(format!("GGUF block width for {name} exceeds usize")))?;
            let block_bytes = usize::try_from(block_bytes)
                .map_err(|_| invalid(format!("GGUF block bytes for {name} exceeds usize")))?;
            let input = shape[column_axis];
            if !input.is_multiple_of(block_values) {
                return Err(invalid(format!(
                    "GGUF matrix {name} input {input} is not aligned to block {block_values}"
                )));
            }
            let mut packed = shape.to_vec();
            packed[column_axis] = input / block_values * block_bytes;
            Ok(vec![ParameterMemberSpec::new(
                name,
                packed,
                remap_linear_segments(source.sharding(), column_axis, block_values, name)?,
            )])
        }
        LinearFormat::Affine(_) | LinearFormat::MxFp4 => {
            let quantization = format.weight_quantization().expect("packed format");
            let LinearFormatCompanions::Scaled { scale, bias } = declaration.companions() else {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "packed linear parameter {name} must declare a scale companion"
                )));
            };
            if quantization.has_biases() != bias.is_some() {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "packed linear parameter {name} declares companions inconsistent with its format"
                )));
            }
            let bits = usize::try_from(quantization.bits())
                .map_err(|_| invalid(format!("packed bit width for {name} exceeds usize")))?;
            let group = usize::try_from(quantization.group_size())
                .map_err(|_| invalid(format!("packed group width for {name} exceeds usize")))?;
            let input = shape[column_axis];
            let packed_bits = input
                .checked_mul(bits)
                .ok_or_else(|| invalid(format!("packed matrix {name} overflows")))?;
            if group == 0 || !input.is_multiple_of(group) || !packed_bits.is_multiple_of(32) {
                return Err(invalid(format!(
                    "packed matrix {name} input {input} is incompatible with group {group} and {bits} bits"
                )));
            }
            let mut packed = shape.to_vec();
            packed[column_axis] = packed_bits / 32;
            let mut companion = shape.to_vec();
            companion[column_axis] = input / group;
            let mut members = vec![ParameterMemberSpec::new(
                name,
                packed,
                remap_linear_segments(source.sharding(), column_axis, 32 / bits, name)?,
            )];
            let companion_sharding =
                remap_linear_segments(source.sharding(), column_axis, group, name)?;
            members.push(ParameterMemberSpec::new(
                scale,
                companion.clone(),
                companion_sharding.clone(),
            ));
            if let Some(bias) = bias {
                members.push(ParameterMemberSpec::new(
                    bias,
                    companion,
                    companion_sharding,
                ));
            }
            Ok(members)
        }
    }
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
    /// Materialize the complete tensor only on one pipeline stage.
    PipelineStage {
        /// Owning pipeline-stage coordinate.
        stage: usize,
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

/// Rank-local shape and placement for one planned physical tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalTensorLayout<P = TensorPlacement> {
    logical_name: String,
    role: ParameterRole,
    global_shape: Vec<usize>,
    local_shape: Vec<usize>,
    placement: P,
    logical_units: Option<usize>,
    logical_range: Option<Range<usize>>,
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
            logical_units,
            logical_range,
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

    /// Returns the rank-local range in the parameter group's semantic domain.
    pub fn logical_range(&self) -> Option<&Range<usize>> {
        self.logical_range.as_ref()
    }

    /// Returns the size of the complete semantic partition domain.
    pub const fn logical_units(&self) -> Option<usize> {
        self.logical_units
    }

    /// Returns whether permissive planning replicated an unsupported shard.
    pub const fn fell_back_to_replication(&self) -> bool {
        self.fell_back_to_replication
    }
}

/// Complete rank-local model geometry produced alongside checkpoint placement.
#[derive(Debug, Clone)]
pub struct LocalModelLayout<P = TensorPlacement> {
    tensors: BTreeMap<String, LocalTensorLayout<P>>,
}

impl<P> Default for LocalModelLayout<P> {
    fn default() -> Self {
        Self {
            tensors: BTreeMap::new(),
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
            Some(LinearFormatParameter::scaled(format, "opaque_scale"))
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
        assert_eq!(info.model_type(), "generic");
        assert_eq!(info.owned_tensors(), ["layer.weight"]);
        assert_eq!(info.local_parameter_bytes(), 10);
        assert_eq!(info.global_parameter_bytes(), 20);
        assert_eq!(info.pinned_device_parameter_bytes(), 4);
        assert_eq!(info.maximum_device_parameter_bytes(), 8);
    }
}
