//! Shared member/validation workers and their checked metadata destination.

use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};
use eredu_nn::{ParameterMetadataView, ParameterSourceError, ParameterSourceVisitor};

#[derive(Debug, thiserror::Error)]
pub(super) enum GroupIssue<'a> {
    #[error("chunk conversion requires a uniform group-level partition")]
    UniformChunks,
    #[error("segmented projection requires positive logical units and at least one segment")]
    SegmentedPrelude,
    #[error("segmented projection ranges must be positive, contiguous, and ordered, got {0:?}")]
    SegmentOrder(&'a [Range<usize>]),
    #[error("segmented projection parameter {0} is scalar")]
    FusedScalar(&'a str),
    #[error(
        "segmented projection parameter {target} has output dimension {dimension}, expected {expected}"
    )]
    FusedDimension {
        target: &'a str,
        dimension: usize,
        expected: usize,
    },
    #[error("column projection parameter {0} is scalar")]
    ColumnScalar(&'a str),
    #[error("partitioned parameter {target} has no axis {axis}")]
    PartitionAxis { target: &'a str, axis: usize },
    #[error("segmented parameter {target} has no axis {axis}")]
    SegmentedAxis { target: &'a str, axis: usize },
    #[error("parallel parameter logical name must not be empty")]
    Name,
    #[error("parallel parameter group {0:?} must contain at least one tensor")]
    Empty(&'a str),
    #[error("parallel parameter group {0:?} contains an empty tensor target")]
    EmptyTarget(&'a str),
    #[error("parallel parameter group {group:?} repeats tensor target {target:?}")]
    Duplicate { group: &'a str, target: &'a str },
    #[error("chunked partition axis is absent")]
    ChunkAxis,
    #[error("chunked segment axis is absent")]
    SegmentAxis,
    #[error("invalid chunked partition segments")]
    Segments,
    #[error("physical chunks do not match the shared logical partition")]
    Chunks,
    #[error(
        "parallel parameter group {0:?} must declare exactly one group-level logical partition for its partitioned members"
    )]
    Partition(&'a str),
}
impl GroupIssue<'_> {
    pub(super) fn ordinary(self) -> ParallelPlanError {
        let tensor = matches!(
            self,
            Self::FusedScalar(_)
                | Self::FusedDimension { .. }
                | Self::ChunkAxis
                | Self::SegmentAxis
                | Self::Segments
                | Self::ColumnScalar(_)
                | Self::PartitionAxis { .. }
                | Self::SegmentedAxis { .. }
        );
        if tensor {
            ParallelPlanError::InvalidTensor(self.to_string())
        } else {
            ParallelPlanError::InvalidGroup(self.to_string())
        }
    }
}
pub(super) fn validate_name(name: &str) -> Result<(), GroupIssue<'_>> {
    if name.trim().is_empty() {
        Err(GroupIssue::Name)
    } else {
        Ok(())
    }
}
fn chunks_match(width: usize, units: Option<usize>, extents: impl Iterator<Item = usize>) -> bool {
    if width == 0 {
        return false;
    }
    let mut any = false;
    for extent in extents {
        any = true;
        if extent == 0 || Some(extent.div_ceil(width)) != units {
            return false;
        }
    }
    any
}
pub(super) fn validate_members<D: Destination>(
    name: &str,
    units: Option<usize>,
    members: &[ParameterMemberSpec],
    destination: D,
) -> Result<(), D::Error> {
    if members.is_empty() {
        return Err(destination.issue(GroupIssue::Empty(name)));
    }
    // Preserve the first repeated source ordinal while sorting only indices.
    // Semantic member validation below still runs in its original order, so an
    // earlier malformed member takes precedence over a later repeated target.
    let mut order = destination.vector::<usize>(members.len())?;
    order.extend(0..members.len());
    order.sort_unstable_by(|first, second| {
        members[*first]
            .target
            .cmp(&members[*second].target)
            .then(first.cmp(second))
    });
    let duplicate = order
        .windows(2)
        .filter(|pair| members[pair[0]].target == members[pair[1]].target)
        .map(|pair| pair[1])
        .min();
    let mut partitioned = false;
    for (index, member) in members.iter().enumerate() {
        if member.target.trim().is_empty() {
            return Err(destination.issue(GroupIssue::EmptyTarget(name)));
        }
        if duplicate == Some(index) {
            return Err(destination.issue(GroupIssue::Duplicate {
                group: name,
                target: &member.target,
            }));
        }
        partitioned |= matches!(
            member.sharding,
            MemberSharding::Partitioned { .. }
                | MemberSharding::PartitionedSegments { .. }
                | MemberSharding::PartitionedChunks { .. }
                | MemberSharding::PartitionedChunkSegments { .. }
        );
        match member.sharding() {
            MemberSharding::PartitionedChunks { axis, chunk_size } => {
                let extent = member
                    .global_shape()
                    .get(*axis)
                    .ok_or_else(|| destination.issue(GroupIssue::ChunkAxis))?;
                if !chunks_match(*chunk_size, units, std::iter::once(*extent)) {
                    return Err(destination.issue(GroupIssue::Chunks));
                }
            }
            MemberSharding::PartitionedChunkSegments {
                axis,
                segments,
                chunk_size,
            } => {
                let extent = member
                    .global_shape()
                    .get(*axis)
                    .ok_or_else(|| destination.issue(GroupIssue::SegmentAxis))?;
                let mut previous = 0;
                for segment in segments {
                    if segment.start < previous
                        || segment.start >= segment.end
                        || segment.end > *extent
                    {
                        return Err(destination.issue(GroupIssue::Segments));
                    }
                    previous = segment.end;
                }
                if !chunks_match(
                    *chunk_size,
                    units,
                    segments.iter().map(|segment| segment.len()),
                ) {
                    return Err(destination.issue(GroupIssue::Chunks));
                }
            }
            _ => {}
        }
    }
    if partitioned != units.is_some() {
        return Err(destination.issue(GroupIssue::Partition(name)));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct MemberSource<'a> {
    id: &'a str,
    companion: Option<eredu_nn::LinearCompanionRole>,
    primary: Option<&'a str>,
    rows: eredu_nn::LinearRowLayout,
}
impl<'a> MemberSource<'a> {
    pub(super) fn ordinary(value: &'a ParameterMetadata) -> Self {
        Self {
            id: value.id.as_str(),
            companion: value.linear_companion,
            primary: value.linear_companion_of.as_ref().map(|id| id.as_str()),
            rows: value.linear_row_layout,
        }
    }
    fn borrowed(value: ParameterMetadataView<'a>) -> Self {
        Self {
            id: value.id().as_str(),
            companion: value.linear_companion(),
            primary: value.linear_companion_of().map(|id| id.as_str()),
            rows: value.linear_row_layout(),
        }
    }
}
pub(super) trait Destination: Copy {
    type Error;
    fn vector<T>(self, count: usize) -> Result<Vec<T>, Self::Error>;
    fn string(self, args: std::fmt::Arguments<'_>) -> Result<String, Self::Error>;
    fn invalid_tensor(self, args: std::fmt::Arguments<'_>) -> Self::Error;
    fn issue(self, cause: GroupIssue<'_>) -> Self::Error;
    fn overflow(self) -> Self::Error;
}
#[derive(Clone, Copy)]
pub(super) struct Ordinary;
impl Destination for Ordinary {
    type Error = ParallelPlanError;
    fn vector<T>(self, count: usize) -> Result<Vec<T>, Self::Error> {
        Ok(Vec::with_capacity(count))
    }
    fn string(self, args: std::fmt::Arguments<'_>) -> Result<String, Self::Error> {
        Ok(args.to_string())
    }
    fn invalid_tensor(self, args: std::fmt::Arguments<'_>) -> Self::Error {
        ParallelPlanError::InvalidTensor(args.to_string())
    }
    fn issue(self, cause: GroupIssue<'_>) -> Self::Error {
        cause.ordinary()
    }
    fn overflow(self) -> Self::Error {
        ParallelPlanError::InvalidTensor("parameter group member count overflowed usize".into())
    }
}
#[derive(Clone, Copy)]
struct Checked<'a>(&'a WorkspaceContext);
impl Destination for Checked<'_> {
    type Error = eredu_nn::Error;
    fn vector<T>(self, count: usize) -> Result<Vec<T>, Self::Error> {
        self.0.metadata_vec(count)
    }
    fn string(self, args: std::fmt::Arguments<'_>) -> Result<String, Self::Error> {
        self.0.metadata_string(args)
    }
    fn invalid_tensor(self, args: std::fmt::Arguments<'_>) -> Self::Error {
        self.0.metadata_error(args)
    }
    fn issue(self, cause: GroupIssue<'_>) -> Self::Error {
        self.0.metadata_error(format_args!("{cause}"))
    }
    fn overflow(self) -> Self::Error {
        WorkspaceMetadataError::Overflow.into()
    }
}
pub(super) fn member<D: Destination>(
    source: MemberSource<'_>,
    dimensions: &[i32],
    destination: D,
    sharding: impl FnOnce(&[usize]) -> Result<MemberSharding, D::Error>,
) -> Result<ParameterMemberSpec, D::Error> {
    let mut shape = destination.vector(dimensions.len())?;
    for dimension in dimensions {
        shape.push(usize::try_from(*dimension).map_err(|_| {
            destination.invalid_tensor(format_args!(
                "parameter {} has negative dimension {dimension}",
                source.id
            ))
        })?);
    }
    let sharding = sharding(&shape)?;
    Ok(ParameterMemberSpec {
        target: destination.string(format_args!("{}", source.id))?,
        global_shape: shape,
        sharding,
        linear_companion: source.companion,
        linear_companion_of: source
            .primary
            .map(|primary| destination.string(format_args!("{primary}")))
            .transpose()?,
        linear_row_layout: source.rows,
    })
}

/// Builds a module group through the strict borrowed parameter traversal.
/// The caller retains the Context metadata custody through the group's lifetime.
/// Sharding uses the same supplied semantic policy; its callback must use the
/// supplied metadata destination for any owning output or diagnostic.
pub fn module_parameter_group_with_metadata<T, M>(
    logical_name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    module: &M,
    context: &WorkspaceContext,
    sharding: impl FnMut(ParameterMetadataView<'_>, &[usize]) -> Result<MemberSharding, eredu_nn::Error>,
) -> Result<ParameterGroupSpec, eredu_nn::Error>
where
    T: Tensor,
    M: Parameterized<T>,
{
    let members = members_with_metadata::<T, M>(module, context, sharding)?;
    finish_group(logical_name, role, None, members, context)
}

fn members_with_metadata<T, M>(
    module: &M,
    context: &WorkspaceContext,
    mut sharding: impl FnMut(
        ParameterMetadataView<'_>,
        &[usize],
    ) -> Result<MemberSharding, eredu_nn::Error>,
) -> Result<Vec<ParameterMemberSpec>, eredu_nn::Error>
where
    T: Tensor,
    M: Parameterized<T>,
{
    struct Count(Option<usize>);
    impl<'a, T: 'a> ParameterSourceVisitor<'a, T> for Count {
        fn parameter(&mut self, _: ParameterMetadataView<'a>, _: &'a T) {
            self.0 = self.0.and_then(|count| count.checked_add(1));
        }
        fn retained(&mut self, _: &'a T) {}
    }
    struct Collector<'c, F> {
        members: Vec<ParameterMemberSpec>,
        count: usize,
        context: &'c WorkspaceContext,
        sharding: &'c mut F,
        error: Option<eredu_nn::Error>,
    }
    impl<'a, T, F> ParameterSourceVisitor<'a, T> for Collector<'_, F>
    where
        T: Tensor,
        F: FnMut(ParameterMetadataView<'_>, &[usize]) -> Result<MemberSharding, eredu_nn::Error>,
    {
        fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a T) {
            if self.error.is_some() {
                return;
            }
            if self.members.len() == self.count {
                self.error = Some(
                    self.context
                        .metadata_source(ParameterSourceError::TopologyMismatch {
                            slot: self.count,
                        }),
                );
                return;
            }
            match member(
                MemberSource::borrowed(metadata),
                value.shape(),
                Checked(self.context),
                |shape| (self.sharding)(metadata, shape),
            ) {
                Ok(member) => self.members.push(member),
                Err(cause) => self.error = Some(cause),
            }
        }
        fn retained(&mut self, _: &'a T) {}
    }
    let controls = [
        size_of::<Count>(),
        size_of::<MemberSource<'_>>(),
        size_of::<GroupIssue<'_>>(),
        size_of::<ParameterMetadataView<'_>>(),
        size_of::<ParameterGroupSpec>(),
        size_of::<ParameterMemberSpec>(),
        size_of::<Result<ParameterMemberSpec, eredu_nn::Error>>(),
        size_of::<std::fmt::Arguments<'_>>(),
        size_of::<Result<ParameterGroupSpec, eredu_nn::Error>>(),
        size_of::<Result<(), ParameterSourceError>>(),
    ];
    let sharding_controls = size_of_val(&sharding);
    let mut collector = Collector {
        members: Vec::new(),
        count: 0,
        context,
        sharding: &mut sharding,
        error: None,
    };
    // Query the actual closure-bearing frame without allocating or invoking it.
    let bytes = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .and_then(|bytes| bytes.checked_add(size_of_val(&collector)))
        .and_then(|bytes| bytes.checked_add(sharding_controls))
        .ok_or(WorkspaceMetadataError::Overflow)?;
    context.charge_metadata(bytes)?;
    let mut count = Count(Some(0));
    module
        .visit_parameter_sources(&mut count)
        .map_err(|cause| context.metadata_source(cause))?;
    collector.count = count.0.ok_or(WorkspaceMetadataError::Overflow)?;
    collector.members = context.metadata_vec(collector.count)?;
    let visited = module.visit_parameter_sources(&mut collector);
    if let Some(cause) = collector.error {
        return Err(cause);
    }
    visited.map_err(|cause| context.metadata_source(cause))?;
    if collector.members.len() != collector.count {
        return Err(
            context.metadata_source(ParameterSourceError::TopologyMismatch {
                slot: collector.members.len(),
            }),
        );
    }
    Ok(collector.members)
}

fn finish_group(
    logical_name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    units: Option<usize>,
    members: Vec<ParameterMemberSpec>,
    context: &WorkspaceContext,
) -> Result<ParameterGroupSpec, eredu_nn::Error> {
    let logical_name = context.metadata_string(logical_name)?;
    finish_owned(logical_name, role, units, members, context)
}

pub(super) fn finish_owned(
    logical_name: String,
    role: ParameterRole,
    units: Option<usize>,
    members: Vec<ParameterMemberSpec>,
    context: &WorkspaceContext,
) -> Result<ParameterGroupSpec, eredu_nn::Error> {
    context.charge_metadata(
        size_of::<ParameterGroupSpec>()
            + size_of::<Result<ParameterGroupSpec, eredu_nn::Error>>()
            + size_of::<GroupIssue<'_>>(),
    )?;
    if units == Some(0) {
        return Err(context.metadata_error(format_args!(
            "parallel logical partition must contain at least one unit"
        )));
    }
    validate_name(&logical_name)
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    validate_members(&logical_name, units, &members, Checked(context))?;
    Ok(ParameterGroupSpec {
        logical_name,
        role,
        partition_units: units,
        members,
    })
}

pub(super) fn projection_sharding<'a>(
    placement: ProjectionSharding,
    name: &'a str,
    shape: &[usize],
) -> Result<MemberSharding, GroupIssue<'a>> {
    match placement {
        ProjectionSharding::Replicated => Ok(MemberSharding::Replicated),
        ProjectionSharding::Column if shape.is_empty() => Err(GroupIssue::ColumnScalar(name)),
        ProjectionSharding::Column => Ok(MemberSharding::Equal { axis: 0 }),
        ProjectionSharding::Row if shape.len() >= 2 => Ok(MemberSharding::Equal { axis: 1 }),
        ProjectionSharding::Row => Ok(MemberSharding::Replicated),
    }
}

pub(super) fn partitioned_sharding(placement: ProjectionSharding, rank: usize) -> MemberSharding {
    match (placement, rank) {
        (ProjectionSharding::Replicated, _) | (ProjectionSharding::Row, 0 | 1) => {
            MemberSharding::Replicated
        }
        (ProjectionSharding::Column, 0) => {
            unreachable!("column rank validated before partitioning")
        }
        (ProjectionSharding::Column, _) => MemberSharding::Partitioned { axis: 0 },
        (ProjectionSharding::Row, _) => MemberSharding::Partitioned { axis: 1 },
    }
}

pub(super) fn preferred_units(
    members: &[ParameterMemberSpec],
    preferred: usize,
) -> Result<usize, GroupIssue<'_>> {
    let mut units = preferred;
    for member in members {
        match member.sharding() {
            MemberSharding::Partitioned { axis } => {
                let dimension =
                    member
                        .global_shape()
                        .get(*axis)
                        .ok_or(GroupIssue::PartitionAxis {
                            target: member.target(),
                            axis: *axis,
                        })?;
                units = greatest_common_divisor(units, *dimension);
            }
            MemberSharding::PartitionedSegments { axis, segments }
            | MemberSharding::Segmented { axis, segments } => {
                if member.global_shape().get(*axis).is_none() {
                    return Err(GroupIssue::SegmentedAxis {
                        target: member.target(),
                        axis: *axis,
                    });
                }
                for segment in segments {
                    units = greatest_common_divisor(units, segment.len());
                }
            }
            _ => {}
        }
    }
    Ok(units)
}

/// Checked affine projection declaration using the shared placement rule.
pub fn projection_parameter_group_with_metadata<T, M>(
    name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    module: &M,
    placement: ProjectionSharding,
    context: &WorkspaceContext,
) -> Result<ParameterGroupSpec, eredu_nn::Error>
where
    T: Tensor,
    M: Parameterized<T>,
{
    module_parameter_group_with_metadata::<T, M>(name, role, module, context, |metadata, shape| {
        projection_sharding(placement, metadata.id().as_str(), shape)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))
    })
}

/// Checked shared partition over the same ordered projection sources.
pub fn partitioned_projection_group_with_metadata<T, M>(
    name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    projections: &[(&M, ProjectionSharding)],
    preferred: usize,
    context: &WorkspaceContext,
) -> Result<ParameterGroupSpec, eredu_nn::Error>
where
    T: Tensor,
    M: Parameterized<T>,
{
    context.charge_metadata(
        size_of::<Vec<ParameterMemberSpec>>()
            + size_of::<ParameterGroupSpec>()
            + size_of::<Result<ParameterGroupSpec, eredu_nn::Error>>(),
    )?;
    if preferred == 0 {
        return Err(context.metadata_error(format_args!(
            "partitioned projection group has zero preferred units"
        )));
    }
    let mut members = Vec::new();
    for (module, placement) in projections {
        let group = projection_parameter_group_with_metadata::<T, M>(
            format_args!("projection"),
            role,
            *module,
            *placement,
            context,
        )?;
        context.reserve_metadata_vec(&mut members, group.members.len())?;
        for member in group.members {
            let sharding = partitioned_sharding(*placement, member.global_shape.len());
            members.push(member.with_sharding(sharding));
        }
    }
    let units = preferred_units(&members, preferred)
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    finish_group(name, role, Some(units), members, context)
}

/// Checked arbitrary module over one shared logical partition.
pub fn partitioned_module_parameter_group_with_metadata<T, M>(
    name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    preferred: usize,
    module: &M,
    context: &WorkspaceContext,
    sharding: impl FnMut(ParameterMetadataView<'_>, &[usize]) -> Result<MemberSharding, eredu_nn::Error>,
) -> Result<ParameterGroupSpec, eredu_nn::Error>
where
    T: Tensor,
    M: Parameterized<T>,
{
    if preferred == 0 {
        return Err(context.metadata_error(format_args!(
            "partitioned module group has zero preferred units"
        )));
    }
    let members = members_with_metadata::<T, M>(module, context, sharding)?;
    let units = preferred_units(&members, preferred)
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    finish_group(name, role, Some(units), members, context)
}

pub(super) fn segment_width(
    preferred: usize,
    segments: &[Range<usize>],
) -> Result<usize, GroupIssue<'_>> {
    if preferred == 0 || segments.is_empty() {
        return Err(GroupIssue::SegmentedPrelude);
    }
    let mut previous = 0;
    for segment in segments {
        if segment.start != previous || segment.start >= segment.end {
            return Err(GroupIssue::SegmentOrder(segments));
        }
        previous = segment.end;
    }
    Ok(previous)
}

pub(super) fn segmented_members<D: Destination>(
    fused: ParameterGroupSpec,
    row: ParameterGroupSpec,
    segments: &[Range<usize>],
    expected: usize,
    destination: D,
) -> Result<Vec<ParameterMemberSpec>, D::Error> {
    // Each source member is transferred exactly once into this destination.
    let mut members = destination.vector(
        fused
            .members
            .len()
            .checked_add(row.members.len())
            .ok_or_else(|| destination.overflow())?,
    )?;
    for member in fused.members {
        let dimension = member
            .global_shape
            .first()
            .copied()
            .ok_or_else(|| destination.issue(GroupIssue::FusedScalar(&member.target)))?;
        if dimension != expected {
            return Err(destination.issue(GroupIssue::FusedDimension {
                target: &member.target,
                dimension,
                expected,
            }));
        }
        let mut owned_segments = destination.vector(segments.len())?;
        owned_segments.extend_from_slice(segments);
        members.push(member.with_sharding(MemberSharding::PartitionedSegments {
            axis: 0,
            segments: owned_segments,
        }));
    }
    for member in row.members {
        let sharding = if member.global_shape.len() >= 2 {
            MemberSharding::Partitioned { axis: 1 }
        } else {
            MemberSharding::Replicated
        };
        members.push(member.with_sharding(sharding));
    }
    Ok(members)
}

/// Checked fused-column and row-output declaration using the shared segment rule.
pub fn segmented_projection_group_with_metadata<T, M>(
    name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    fused: &M,
    row: &M,
    segments: Vec<Range<usize>>,
    preferred: usize,
    context: &WorkspaceContext,
) -> Result<ParameterGroupSpec, eredu_nn::Error>
where
    T: Tensor,
    M: Parameterized<T>,
{
    context.charge_metadata(
        size_of::<Vec<ParameterMemberSpec>>()
            + 2 * size_of::<ParameterGroupSpec>()
            + size_of::<Result<ParameterGroupSpec, eredu_nn::Error>>(),
    )?;
    let width = segment_width(preferred, &segments)
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    let fused = projection_parameter_group_with_metadata::<T, M>(
        format_args!("fused"),
        role,
        fused,
        ProjectionSharding::Column,
        context,
    )?;
    let row = projection_parameter_group_with_metadata::<T, M>(
        format_args!("row"),
        role,
        row,
        ProjectionSharding::Row,
        context,
    )?;
    let members = segmented_members(fused, row, &segments, width, Checked(context))?;
    let units = preferred_units(&members, preferred)
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    finish_group(name, role, Some(units), members, context)
}

pub(super) fn chunk_widths<D: Destination>(
    group: &ParameterGroupSpec,
    destination: D,
    mut width: impl FnMut(&ParameterMemberSpec, &[ParameterMemberSpec]) -> Result<usize, D::Error>,
) -> Result<Vec<Option<usize>>, D::Error> {
    let mut widths = destination.vector(group.members.len())?;
    for member in &group.members {
        let value = match member.sharding() {
            MemberSharding::Partitioned { .. } | MemberSharding::PartitionedSegments { .. } => {
                Some(width(member, &group.members)?)
            }
            MemberSharding::Replicated => None,
            _ => return Err(destination.issue(GroupIssue::UniformChunks)),
        };
        widths.push(value);
    }
    Ok(widths)
}

pub(super) fn apply_chunk_widths(
    mut group: ParameterGroupSpec,
    widths: Vec<Option<usize>>,
) -> ParameterGroupSpec {
    for (member, width) in group.members.iter_mut().zip(widths) {
        let Some(chunk_size) = width else {
            continue;
        };
        member.sharding = match std::mem::replace(&mut member.sharding, MemberSharding::Replicated)
        {
            MemberSharding::Partitioned { axis } => {
                MemberSharding::PartitionedChunks { axis, chunk_size }
            }
            MemberSharding::PartitionedSegments { axis, segments } => {
                MemberSharding::PartitionedChunkSegments {
                    axis,
                    segments,
                    chunk_size,
                }
            }
            _ => unreachable!("source sharding was validated before mutation"),
        };
    }
    group
}

/// Checked chunk conversion with immutable access to the complete source group.
pub fn partition_parameter_group_chunks_with_metadata(
    group: ParameterGroupSpec,
    units: usize,
    context: &WorkspaceContext,
    width: impl FnMut(&ParameterMemberSpec, &[ParameterMemberSpec]) -> Result<usize, eredu_nn::Error>,
) -> Result<ParameterGroupSpec, eredu_nn::Error> {
    context.charge_metadata(
        size_of::<ParameterGroupSpec>() + size_of::<Vec<Option<usize>>>() + size_of_val(&width),
    )?;
    let widths = chunk_widths(&group, Checked(context), width)?;
    apply_chunk_widths(group, widths).into_partitioned_with_metadata(units, context)
}

/// Checked diagnostic destination for the same exact alignment-unit equation.
pub fn aligned_partition_units_with_metadata(
    name: &str,
    units: usize,
    width: usize,
    alignment: usize,
    allow_tail: bool,
    context: &WorkspaceContext,
) -> Result<usize, eredu_nn::Error> {
    context
        .charge_metadata(size_of::<Result<usize, eredu_nn::Error>>() + 5 * size_of::<usize>())?;
    partition_units_with(name, units, width, alignment, allow_tail, |args| {
        context.metadata_error(args)
    })
}
