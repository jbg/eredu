//! Shared description validation over borrowed names and owners.

use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};

#[derive(Debug, thiserror::Error)]
pub(super) enum LayoutIssue<'a> {
    #[error("execution graph contains {graph} groups but its unit layout contains {layout}")]
    Count { graph: usize, layout: usize },
    #[error("execution group {index} is {graph:?} in the graph but {layout:?} in the unit layout")]
    Identity {
        index: usize,
        graph: &'a str,
        layout: &'a str,
    },
}
impl LayoutIssue<'_> {
    pub(super) fn into_owned(self) -> ArchitecturePartitionError {
        match self {
            Self::Count { graph, layout } => {
                ArchitecturePartitionError::LayoutGroupCountMismatch { graph, layout }
            }
            Self::Identity {
                index,
                graph,
                layout,
            } => ArchitecturePartitionError::LayoutGroupMismatch {
                index,
                graph: graph.to_owned(),
                layout: layout.to_owned(),
            },
        }
    }
}

pub(super) fn canonical<'a>(
    graph: &'a ExecutionGraph,
    layout: &'a ExecutionUnitLayout,
) -> Result<(), LayoutIssue<'a>> {
    if graph.groups().len() != layout.group_count() {
        return Err(LayoutIssue::Count {
            graph: graph.groups().len(),
            layout: layout.group_count(),
        });
    }
    for (index, group) in graph.groups().iter().enumerate() {
        let layout_group = layout
            .group_id(index)
            .expect("matching group counts provide every layout identity");
        if layout_group.as_str() != group.id() {
            return Err(LayoutIssue::Identity {
                index,
                graph: group.id(),
                layout: layout_group.as_str(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub(super) enum Issue<'a> {
    #[error("invalid architecture parameter layout: {0}")]
    Layout(LayoutIssue<'a>),
    #[error("architecture parameter static role must not be empty")]
    EmptyStaticRole,
    #[error("architecture shared parameter owner repeats a static role")]
    DuplicateStaticRole,
    #[error("architecture pinned parameter has no unit consumer")]
    EmptyStaticConsumers,
    #[error("architecture pinned parameter repeats a unit consumer")]
    DuplicateStaticConsumer,
    #[error("architecture parameter owner names unknown execution group {0:?}")]
    UnknownExecutionGroup(&'a str),
    #[error("architecture parameter owner {group}:{global_unit} exceeds {available} units")]
    UnitOutOfRange {
        group: &'a str,
        global_unit: usize,
        available: usize,
    },
    #[error("expected parameter target {target:?} appears in both {first:?} and {second:?}")]
    DuplicateExpectedTarget {
        target: &'a str,
        first: &'a str,
        second: &'a str,
    },
    #[error("parameter target {target:?} is owned by both {first:?} and {second:?}")]
    DuplicateOwnership {
        target: &'a str,
        first: &'a ParameterGroupOwner,
        second: &'a ParameterGroupOwner,
    },
    #[error("parameter target {0:?} has no architecture owner")]
    MissingOwnership(&'a str),
    #[error("parameter target {0:?} is not present in the authoritative parameter groups")]
    UnexpectedOwnership(&'a str),
}
impl Issue<'_> {
    pub(super) fn into_owned(self) -> ArchitectureParameterError {
        match self {
            Self::Layout(cause) => ArchitectureParameterError::InvalidLayout(cause.to_string()),
            Self::EmptyStaticRole => ArchitectureParameterError::EmptyStaticRole,
            Self::DuplicateStaticRole => ArchitectureParameterError::DuplicateStaticRole,
            Self::EmptyStaticConsumers => ArchitectureParameterError::EmptyStaticConsumers,
            Self::DuplicateStaticConsumer => ArchitectureParameterError::DuplicateStaticConsumer,
            Self::UnknownExecutionGroup(group) => {
                ArchitectureParameterError::UnknownExecutionGroup(group.to_owned())
            }
            Self::UnitOutOfRange {
                group,
                global_unit,
                available,
            } => ArchitectureParameterError::UnitOutOfRange {
                group: group.to_owned(),
                global_unit,
                available,
            },
            Self::DuplicateExpectedTarget {
                target,
                first,
                second,
            } => ArchitectureParameterError::DuplicateExpectedTarget {
                target: target.to_owned(),
                first: first.to_owned(),
                second: second.to_owned(),
            },
            Self::DuplicateOwnership {
                target,
                first,
                second,
            } => ArchitectureParameterError::DuplicateOwnership {
                target: target.to_owned(),
                first: first.clone(),
                second: second.clone(),
            },
            Self::MissingOwnership(target) => {
                ArchitectureParameterError::MissingOwnership(target.to_owned())
            }
            Self::UnexpectedOwnership(target) => {
                ArchitectureParameterError::UnexpectedOwnership(target.to_owned())
            }
        }
    }
}

pub(super) fn owner<'a>(
    owner: &'a ParameterGroupOwner,
    graph: &ExecutionGraph,
    layout: &ExecutionUnitLayout,
) -> Result<(), Issue<'a>> {
    match owner {
        ParameterGroupOwner::StaticRole(role) => {
            if role.trim().is_empty() {
                return Err(Issue::EmptyStaticRole);
            }
        }
        ParameterGroupOwner::StaticAnyOf(roles) => {
            if roles.is_empty() || roles.iter().any(|role| role.trim().is_empty()) {
                return Err(Issue::EmptyStaticRole);
            }
            if roles
                .iter()
                .enumerate()
                .any(|(index, role)| roles[..index].contains(role))
            {
                return Err(Issue::DuplicateStaticRole);
            }
        }
        ParameterGroupOwner::StaticUnitConsumers { role, consumers } => {
            if role.trim().is_empty() {
                return Err(Issue::EmptyStaticRole);
            }
            if consumers.is_empty() {
                return Err(Issue::EmptyStaticConsumers);
            }
            if consumers
                .iter()
                .enumerate()
                .any(|(index, value)| consumers[..index].contains(value))
            {
                return Err(Issue::DuplicateStaticConsumer);
            }
        }
        ParameterGroupOwner::ExecutionUnit { .. } => {}
    }
    let validate = |group: &'a ExecutionGroupId, global_unit: usize| {
        let Some(index) = graph
            .groups()
            .iter()
            .position(|candidate| candidate.id() == group.as_str())
        else {
            return Err(Issue::UnknownExecutionGroup(group.as_str()));
        };
        let available = layout
            .group_range(index)
            .expect("validated canonical layout contains every group")
            .len();
        if global_unit >= available {
            return Err(Issue::UnitOutOfRange {
                group: group.as_str(),
                global_unit,
                available,
            });
        }
        Ok(())
    };
    match owner {
        ParameterGroupOwner::StaticUnitConsumers { consumers, .. } => {
            for (group, unit) in consumers {
                validate(group, *unit)?;
            }
        }
        ParameterGroupOwner::ExecutionUnit { group, global_unit } => validate(group, *global_unit)?,
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct Destination<'a>(pub(super) Option<&'a WorkspaceContext>);
pub(super) enum Failure {
    Ordinary(ArchitectureParameterError),
    Metadata(eredu_nn::Error),
}
impl Failure {
    pub(super) fn ordinary(self) -> ArchitectureParameterError {
        match self {
            Self::Ordinary(cause) => cause,
            Self::Metadata(cause) => ArchitectureParameterError::InvalidLayout(cause.to_string()),
        }
    }
    pub(super) fn metadata(self) -> eredu_nn::Error {
        match self {
            Self::Metadata(cause) => cause,
            Self::Ordinary(_) => {
                unreachable!("checked description always uses its metadata destination")
            }
        }
    }
}
impl Destination<'_> {
    pub(super) fn issue(self, cause: Issue<'_>) -> Failure {
        match self.0 {
            Some(context) => Failure::Metadata(context.metadata_error(format_args!("{cause}"))),
            None => Failure::Ordinary(cause.into_owned()),
        }
    }
    fn overflow(self) -> Failure {
        Failure::Metadata(WorkspaceMetadataError::Overflow.into())
    }
    pub(super) fn controls<T>(self) -> Result<(), Failure> {
        if let Some(context) = self.0 {
            let bytes = [
                size_of::<T>(),
                size_of::<Self>(),
                size_of::<Failure>(),
                size_of::<Issue<'_>>(),
                size_of::<Result<T, Failure>>(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(|| self.overflow())?;
            context
                .charge_metadata(bytes)
                .map_err(|cause| Failure::Metadata(cause.into()))?;
        }
        Ok(())
    }
    fn vector<T>(self, count: usize) -> Result<Vec<T>, Failure> {
        match self.0 {
            Some(context) => context.metadata_vec(count).map_err(Failure::Metadata),
            None => Ok(Vec::with_capacity(count)),
        }
    }
}

pub(super) fn expected<'a>(
    groups: impl Iterator<Item = &'a ParameterGroupSpec> + Clone,
    destination: Destination<'_>,
) -> Result<Vec<(&'a str, &'a str, usize)>, Failure> {
    type Row<'a> = (&'a str, &'a str, usize);
    destination.controls::<Vec<Row<'_>>>()?;
    let count = groups
        .clone()
        .try_fold(0usize, |count, group| {
            count.checked_add(group.members().len())
        })
        .ok_or_else(|| destination.overflow())?;
    let mut rows: Vec<Row<'a>> = destination.vector(count)?;
    for group in groups {
        for member in group.members() {
            rows.push((member.target(), group.logical_name(), rows.len()));
        }
    }
    rows.sort_unstable_by(|left, right| (left.0, left.2).cmp(&(right.0, right.2)));
    // Sorting includes source order within each target. Among duplicate targets,
    // the smallest second ordinal is exactly the original traversal's first error.
    if let Some(pair) = rows
        .windows(2)
        .filter(|pair| pair[0].0 == pair[1].0)
        .min_by_key(|pair| pair[1].2)
    {
        return Err(destination.issue(Issue::DuplicateExpectedTarget {
            target: pair[1].0,
            first: pair[0].1,
            second: pair[1].1,
        }));
    }
    Ok(rows)
}

pub(super) fn owned(
    graph: &ExecutionGraph,
    layout: &ExecutionUnitLayout,
    groups: &[OwnedParameterGroupSpec],
    expected: &[(&str, &str, usize)],
    destination: Destination<'_>,
) -> Result<(), Failure> {
    type Row<'a> = (&'a str, &'a ParameterGroupOwner, usize);
    destination.controls::<(Vec<Row<'_>>, Option<Issue<'_>>)>()?;
    let count = groups
        .iter()
        .try_fold(0usize, |count, group| {
            count.checked_add(group.members().len())
        })
        .ok_or_else(|| destination.overflow())?;
    let mut rows: Vec<Row<'_>> = destination.vector(count)?;
    let mut owner_failure = None;
    for tagged in groups {
        if let Err(cause) = owner(tagged.owner(), graph, layout) {
            owner_failure = Some(cause);
            break;
        }
        for member in tagged.members() {
            rows.push((member.target(), tagged.owner(), rows.len()));
        }
    }
    rows.sort_unstable_by(|left, right| (left.0, left.2).cmp(&(right.0, right.2)));
    // A duplicate in the completed prefix precedes the next owner's validation.
    // Never inspect rows belonging to the first invalid owner or later groups.
    if let Some(pair) = rows
        .windows(2)
        .filter(|pair| pair[0].0 == pair[1].0)
        .min_by_key(|pair| pair[1].2)
    {
        return Err(destination.issue(Issue::DuplicateOwnership {
            target: pair[1].0,
            first: pair[0].1,
            second: pair[1].1,
        }));
    }
    if let Some(cause) = owner_failure {
        return Err(destination.issue(cause));
    }
    if let Some(&(target, _, _)) = expected
        .iter()
        .find(|(target, _, _)| rows.binary_search_by_key(target, |row| row.0).is_err())
    {
        return Err(destination.issue(Issue::MissingOwnership(target)));
    }
    if let Some(&(target, _, _)) = rows
        .iter()
        .find(|(target, _, _)| expected.binary_search_by_key(target, |row| row.0).is_err())
    {
        return Err(destination.issue(Issue::UnexpectedOwnership(target)));
    }
    Ok(())
}
