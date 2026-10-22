//! Backend-neutral pipeline owners and dependency transport endpoints.
use crate::{
    ArchitectureGroupPlacement, ArchitectureGroupTransport, ExecutionGraph, ExecutionUnitLayout,
};
use std::ops::Range;

/// Invalid global invocation geometry or pipeline ownership declaration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PipelinePlacementError {
    /// Pipeline stages or transport declarations do not cover the graph.
    #[error("pipeline placement differs from its declared groups")]
    GroupCoverage,
    /// The layout is foreign or already projected to sparse/global-offset units.
    #[error("pipeline group {group} requires its canonical global invocation layout")]
    NonCanonicalLayout {
        /// Graph group whose identity or invocation sequence differs.
        group: usize,
    },
    /// Sequential groups do not form one ordered dependency chain.
    #[error("pipeline sequence group {group} breaks the ordered dependency chain")]
    SequenceDependencies {
        /// First inconsistent sequential group.
        group: usize,
    },
    /// A sequential group has no executable invocations.
    #[error("pipeline sequence group {group} contains no execution units")]
    EmptySequence {
        /// Empty sequential group.
        group: usize,
    },
    /// Concatenated invocation counts overflowed.
    #[error("pipeline sequence unit count overflowed")]
    UnitCountOverflow,
    /// A sequential stack cannot place at least one invocation on every stage.
    #[error("pipeline sequence has {units} execution units for {stages} stages")]
    InsufficientSequence {
        /// Total sequential invocations.
        units: usize,
        /// Requested pipeline stages.
        stages: usize,
    },
    /// A nonempty invocation set has no owner coordinates.
    #[error("pipeline owner range {start}..{end} is empty")]
    EmptyOwnerRange {
        /// First requested pipeline owner.
        start: usize,
        /// Exclusive owner frontier.
        end: usize,
    },
}

/// One validated global placement, shared by local ownership and transport plans.
/// This retains no native group, checkpoint source or execution authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelinePlacementPlan<'a> {
    graph: &'a ExecutionGraph,
    owners: Vec<Vec<(usize, Range<usize>)>>,
}
impl<'a> PipelinePlacementPlan<'a> {
    /// Projects complete canonical invocation geometry using declared placements.
    /// Rank-local subsets are selected after this global ownership projection.
    pub fn new(
        graph: &'a ExecutionGraph,
        layout: &ExecutionUnitLayout,
        transports: &[ArchitectureGroupTransport],
        stages: usize,
    ) -> Result<Self, PipelinePlacementError> {
        if layout.group_count() != graph.groups().len() {
            return Err(PipelinePlacementError::GroupCoverage);
        }
        for (group, spec) in graph.groups().iter().enumerate() {
            let range = layout
                .group_range(group)
                .ok_or(PipelinePlacementError::NonCanonicalLayout { group })?;
            if layout.group_id(group).map(|id| id.as_str()) != Some(spec.id())
                || range.enumerate().any(|(index, ordinal)| {
                    layout
                        .address(ordinal)
                        .is_none_or(|address| address.group() != group || address.index() != index)
                })
            {
                return Err(PipelinePlacementError::NonCanonicalLayout { group });
            }
        }
        let owners = pipeline_group_owners(graph, layout, transports, stages)?;
        Ok(Self { graph, owners })
    }
    /// Group-indexed pipeline coordinates and group-global invocation intervals.
    pub fn owners(&self) -> &[Vec<(usize, Range<usize>)>] {
        &self.owners
    }
    /// Projects ordered cross-stage producer endpoints from the retained owners.
    /// Ownership-only callers allocate no route metadata.
    pub fn routes(&self) -> Vec<PipelineRoute> {
        semantic_pipeline_routes(self.graph, &self.owners)
    }
}

/// Semantic producer endpoint for a declared cross-stage transport.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PipelineRoute {
    /// Producer execution group.
    pub source_group: usize,
    /// Consumer execution group.
    pub destination_group: usize,
    /// Pipeline coordinate where the dependency value exists.
    pub source_pipeline: usize,
    /// Pipeline coordinate where the value is consumed.
    pub destination_pipeline: usize,
    /// Exclusive producer invocation frontier in the source group.
    pub source_unit_end: usize,
}

fn semantic_pipeline_routes(
    graph: &ExecutionGraph,
    owners: &[Vec<(usize, Range<usize>)>],
) -> Vec<PipelineRoute> {
    let mut semantic_routes = Vec::new();
    for (group, group_owners) in owners.iter().enumerate() {
        for pair in group_owners.windows(2) {
            semantic_routes.push(PipelineRoute {
                source_group: group,
                destination_group: group,
                source_pipeline: pair[0].0,
                destination_pipeline: pair[1].0,
                source_unit_end: pair[0].1.end,
            });
        }
        let destination = group_owners.first().map_or(0, |owner| owner.0);
        for dependency in graph.dependencies(group).into_iter().flatten() {
            let source_owners = &owners[*dependency];
            // A dependency value exists where its final unit executed.  The
            // merge-destination policy names where that value is consumed; it
            // never relocates the value by itself.  Preserve the terminal
            // producer endpoint here so the selected route performs that move.
            let source = source_owners.last().map_or(0, |owner| owner.0);
            if source != destination {
                semantic_routes.push(PipelineRoute {
                    source_group: *dependency,
                    destination_group: group,
                    source_pipeline: source,
                    destination_pipeline: destination,
                    source_unit_end: source_owners.last().map_or(0, |owner| owner.1.end),
                });
            }
        }
    }
    semantic_routes
}

/// One retained placement drives both route endpoints and rank-local parameter/state ownership.
fn pipeline_group_owners(
    graph: &crate::ExecutionGraph,
    layout: &crate::ExecutionUnitLayout,
    transports: &[crate::ArchitectureGroupTransport],
    stages: usize,
) -> Result<Vec<Vec<(usize, Range<usize>)>>, PipelinePlacementError> {
    if stages == 0 || transports.len() != graph.groups().len() {
        return Err(PipelinePlacementError::GroupCoverage);
    }
    let mut owners = vec![Vec::new(); transports.len()];
    let sequence = graph
        .execution_order()
        .iter()
        .copied()
        .filter(|&group| {
            transports[group].placement == ArchitectureGroupPlacement::PipelineSequence
        })
        .collect::<Vec<_>>();
    let mut total = 0usize;
    let mut spans = Vec::new();
    for (index, &group) in sequence.iter().enumerate() {
        let dependencies = graph
            .dependencies(group)
            .ok_or(PipelinePlacementError::NonCanonicalLayout { group })?;
        if (index == 0 && !dependencies.is_empty())
            || (index > 0 && dependencies != [sequence[index - 1]])
        {
            return Err(PipelinePlacementError::SequenceDependencies { group });
        }
        let count = layout
            .group_range(group)
            .ok_or(PipelinePlacementError::NonCanonicalLayout { group })?
            .len();
        if count == 0 {
            return Err(PipelinePlacementError::EmptySequence { group });
        }
        let end = total
            .checked_add(count)
            .ok_or(PipelinePlacementError::UnitCountOverflow)?;
        spans.push((group, total..end));
        total = end;
    }
    if !sequence.is_empty() {
        if total < stages {
            return Err(PipelinePlacementError::InsufficientSequence {
                units: total,
                stages,
            });
        }
        let balanced = balanced_pipeline_ranges(total, 0..stages)?;
        for (group, span) in spans {
            owners[group] = balanced
                .iter()
                .filter_map(|(owner, range)| {
                    let start = range.start.max(span.start);
                    let end = range.end.min(span.end);
                    (start < end).then(|| (*owner, start - span.start..end - span.start))
                })
                .collect();
        }
    }
    for (group, transport) in transports.iter().enumerate() {
        let count = layout
            .group_range(group)
            .ok_or(PipelinePlacementError::NonCanonicalLayout { group })?
            .len();
        match transport.placement {
            ArchitectureGroupPlacement::PipelineSequence => {}
            ArchitectureGroupPlacement::Pipeline => {
                owners[group] = balanced_pipeline_ranges(count, 0..stages)?
            }
            ArchitectureGroupPlacement::OutputOwner => {
                owners[group] = balanced_pipeline_ranges(count, stages - 1..stages)?
            }
        }
    }
    Ok(owners)
}

/// Balances indivisible invocations over a contiguous set of pipeline owners.
/// Excess owners receive no interval; zero invocations require no owner.
pub fn balanced_pipeline_ranges(
    unit_count: usize,
    ranks: Range<usize>,
) -> Result<Vec<(usize, Range<usize>)>, PipelinePlacementError> {
    if unit_count == 0 {
        return Ok(Vec::new());
    }
    if ranks.is_empty() {
        return Err(PipelinePlacementError::EmptyOwnerRange {
            start: ranks.start,
            end: ranks.end,
        });
    }
    let active = ranks.len().min(unit_count);
    let base = unit_count / active;
    let remainder = unit_count % active;
    let mut start = 0;
    Ok(ranks
        .take(active)
        .enumerate()
        .map(|(index, rank)| {
            let end = start + base + usize::from(index < remainder);
            let value = (rank, start..end);
            start = end;
            value
        })
        .collect())
}
