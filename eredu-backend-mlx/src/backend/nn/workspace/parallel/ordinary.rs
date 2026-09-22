//! Ordinary caller and worker allowances from retained communication sources.
//!
//! These are descriptive projections of the selected workers. They contain no
//! Original scope, bank, reservation, invocation counter or execution authority.
use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OrdinaryParallelControls {
    pub(crate) calls: OrdinaryCallControls,
    pub(crate) native: OrdinaryNativeControls,
    pub(crate) cpu: Option<resident_recipe::OrdinaryCpuPopulation>,
    pub(crate) completion_roots: usize,
    pub(crate) completions: usize,
}

impl LogicalCollectiveQuote {
    pub(crate) fn ordinary_controls(&self) -> Option<OrdinaryParallelControls> {
        use crate::backend::nn::logical_collective::{self, Native};
        use logical::LogicalCollectiveStages as S;
        let mut result = OrdinaryParallelControls::default();
        match &self.stages {
            S::Exchange {
                pairs,
                peer_recipe,
                result_recipe,
                ..
            } => {
                for pair in pairs {
                    result = result
                        .append(OrdinaryParallelControls::group(pair.ordinary_controls()?)?)?
                        .append(OrdinaryParallelControls::numerical(*peer_recipe)?)?;
                }
                result = result
                    .append(OrdinaryParallelControls::numerical(*result_recipe)?)?
                    .metadata(
                        logical_collective::control_bytes::<Native<'_>>()?
                            .checked_mul(pairs.len().checked_add(2)?)?,
                    )?;
            }
            S::RoutedPeer {
                pairs, peer_recipe, ..
            } => {
                for pair in pairs {
                    result = result
                        .append(OrdinaryParallelControls::group(pair.ordinary_controls()?)?)?;
                }
                result = result
                    .append(OrdinaryParallelControls::numerical(*peer_recipe)?)?
                    .metadata(logical_collective::control_bytes::<Native<'_>>()?)?;
            }
            S::Packed(value) => {
                result = result
                    .append(OrdinaryParallelControls::numerical(value.pack_recipe)?)?
                    .append(OrdinaryParallelControls::group(
                        value.world.ordinary_controls()?,
                    )?)?
                    .append(OrdinaryParallelControls::numerical(value.result_recipe)?)?
                    .metadata(
                        logical_collective::packed::controls::<Native<'_>>(
                            self.input.shape().len(),
                        )?
                        .checked_add(logical_collective::packed::controls::<Native<'_>>(
                            self.input.shape().len().checked_add(1)?,
                        )?)?
                        .checked_add(
                            logical_collective::packed::gather_controls::<Native<'_>>(
                                value.members,
                            )?,
                        )?,
                    )?;
                result.calls = result.calls.append(value.ordinary_completion?)?;
            }
            S::Routed(value) => {
                for route in &value.values {
                    for step in &route.steps {
                        result = result.append(step.source.value().ordinary_controls()?)?;
                    }
                }
                result = result
                    .append(OrdinaryParallelControls::numerical(value.result_recipe)?)?
                    .metadata(logical_collective::routed::result_controls::<Native<'_>>(
                        value.values.len(),
                    )?)?;
            }
        }
        // Actual ordinary wrappers retain shape comparisons and an optional
        // gathered output reshape; these are caller cells, never tensor data.
        result.metadata(
            std::mem::size_of::<Vec<i32>>()
                .checked_mul(2)?
                .checked_add(
                    std::mem::size_of::<i32>()
                        .checked_mul(self.input.shape().len().checked_add(1)?)?
                        .checked_mul(2)?,
                )?,
        )
    }
}

impl PipelineBoundaryQuote {
    pub(crate) fn ordinary_controls(&self) -> Option<OrdinaryParallelControls> {
        let mut result = OrdinaryParallelControls::numerical(self.encoding)?;
        for pair in &self.rounds {
            result = result.append(OrdinaryParallelControls::group(pair.ordinary_controls()?)?)?;
        }
        if let Some(decoded) = self.decoding {
            result = result.append(OrdinaryParallelControls::numerical(decoded)?)?;
        }
        result = result.append(OrdinaryParallelControls::constructed(
            self.ordinary_header?,
        )?)?;
        result.calls = result.calls.append(self.ordinary_boundary?)?;
        result.completion_roots = result.completion_roots.max(self.ordinary_completion_roots);
        result.completions = result.completions.checked_add(1)?;
        Some(result)
    }
}

impl ExpertLocalQuote {
    fn ordinary_controls(&self) -> Option<OrdinaryParallelControls> {
        let mut result = self
            .ordinary_local?
            .append(
                self.movement
                    .as_ref()?
                    .ordinary_controls(self.declaration.as_view().movement.counts())?,
            )?
            .append(self.transport.as_ref()?.ordinary_controls()?)?
            .append(self.counts.as_ref()?.ordinary?)?
            .append(self.provider.as_ref()?.ordinary?)?;
        if self.ordinary_addressable.is_some() {
            result = result
                .metadata(crate::backend::nn::shared::ordinary_expert_local_control_bytes()?)?;
        }
        Some(result)
    }
}

/// Selects only occurrences retained from this exact trace/source. The
/// enclosing ordinary program separately owns indexed local child graphs.
pub(in crate::backend::nn::workspace) fn controls(
    invocation: &crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation,
    ordinal: usize,
    operation: WorkspaceOperationView<'_>,
    mechanism: ResidentExecutionMechanisms,
) -> Result<Option<OrdinaryParallelControls>, Error> {
    let mut result = match operation.kind {
        WorkspaceOperationKindView::Collective(_) => {
            if let Some(boundary) = invocation.boundary_occurrence(ordinal) {
                boundary.ordinary_controls()
            } else if let Some(logical) = invocation.logical_occurrence(ordinal) {
                logical.ordinary_controls()
            } else {
                invocation.occurrence(ordinal).and_then(|(native, _)| {
                    native
                        .ordinary_controls()
                        .and_then(OrdinaryParallelControls::group)
                })
            }
        }
        WorkspaceOperationKindView::ExpertRegion(_) => invocation
            .expert_occurrence(ordinal)
            .and_then(ExpertLocalQuote::ordinary_controls),
        WorkspaceOperationKindView::ExpertInactiveWave(_) => invocation
            .expert_inactive_wave_occurrence(ordinal)
            .and_then(|source| {
                source
                    .counts
                    .ordinary?
                    .append(source.transport.ordinary_controls()?)?
                    .append(source.provider.ordinary?)
            }),
        WorkspaceOperationKindView::ExpertProviderWave(_) => invocation
            .expert_provider_wave_occurrence(ordinal)
            .and_then(|source| source.provider.ordinary),
        _ => None,
    };
    if matches!(operation.kind, WorkspaceOperationKindView::Collective(_))
        && invocation.boundary_occurrence(ordinal).is_none()
    {
        result = match (
            result,
            invocation.ordinary_collective_completion_controls(ordinal),
        ) {
            (Some(mut value), Some(calls)) => {
                value.calls = value
                    .calls
                    .append(calls)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                value.completion_roots = value.completion_roots.max(1);
                value.completions = value
                    .completions
                    .checked_add(1)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                Some(value)
            }
            _ => None,
        };
    }
    if let Some(aggregate) = invocation.expert_aggregate(ordinal) {
        for (operation, count) in std::iter::once((&aggregate.empty_slice, aggregate.empty_slices))
            .chain(
                aggregate
                    .extra_parents
                    .iter()
                    .map(|operation| (operation, 1)),
            )
        {
            let calls = match mechanism {
                ResidentExecutionMechanisms::Cpu { cpu, .. } => {
                    cpu.ordinary_call_controls(operation.as_view())
                }
                ResidentExecutionMechanisms::Metal(metal) => {
                    metal.ordinary_call_controls(operation.as_view())
                }
            }
            .map_err(MlxWorkspaceFactError::ordinary)?;
            result = match (result, calls) {
                (Some(mut result), Some(calls)) => {
                    result.calls = result
                        .calls
                        .append(
                            calls
                                .repeat(count)
                                .ok_or(WorkspaceMetadataError::Overflow)?,
                        )
                        .ok_or(WorkspaceMetadataError::Overflow)?;
                    Some(result)
                }
                _ => None,
            };
        }
        if let Some(value) = &mut result {
            value.calls=value.calls.append(
                crate::backend::nn::shared::MlxNeuralBackend::ordinary_completion_call_controls(4)
                    .and_then(|calls|calls.repeat(aggregate.parent_completions))
                    .ok_or(WorkspaceMetadataError::Unqualified)?,
            ).ok_or(WorkspaceMetadataError::Overflow)?;
        }
    }
    Ok(result)
}
impl OrdinaryParallelControls {
    pub(crate) fn communication(
        mut self,
        group: &crate::backend::runtime::distributed::Group,
        count_lengths: &[usize],
    ) -> Option<Self> {
        self.calls=self.calls.append(crate::backend::runtime::distributed::completion::MlxCommunicationCompletion::ordinary_collective_completion_call_controls(group,count_lengths)?)?;
        self.completion_roots = self.completion_roots.max(1);
        self.completions = self.completions.checked_add(1)?;
        Some(self)
    }
    pub(crate) fn metadata(mut self, bytes: usize) -> Option<Self> {
        self.calls = self.calls.metadata(bytes)?;
        Some(self)
    }
    pub(crate) fn group(source: safemlx::distributed::OrdinaryGroupControls) -> Option<Self> {
        let (primitives, arrays, edges) = source.graph_population();
        let (completion_roots, completions) = source.completion_frontier();
        // The model stream and selected CPU communication stream are distinct
        // sources. Two is a conservative union even for an already completed
        // source or a caller which is itself on the communication stream.
        let base = safemlx::OperationEvent::eval_record_layout(
            primitives.checked_add(1)?,
            2,
            primitives.checked_add(1)?,
        )?;
        let mut calls = OrdinaryCallControls::default().metadata(source.caller_control_bytes())?;
        if completions != 0 {
            calls = calls.append(
                crate::backend::nn::shared::MlxNeuralBackend::ordinary_completion_call_controls(
                    completion_roots,
                )?
                .metadata(safemlx::Event::ordinary_synchronize_control_bytes()?)?
                .repeat(completions)?,
            )?;
        }
        Some(Self {
            calls,
            native: OrdinaryNativeControls::default(),
            cpu: Some(resident_recipe::OrdinaryCpuPopulation {
                // The exact group constructor extents below already include
                // its descriptors, primitive and ordinary C result shell.
                construction_entries: 0,
                primitives,
                seeds: 0,
                maximum_rank: source.maximum_rank(),
                maximum_operands: 4,
                parameter_shells: 0,
                array_nodes: arrays,
                input_edges: edges,
                streams: 2,
                captures: source.captures().max(base.capture_slots()),
                nested_completions: 0,
                nested_roots: 0,
                dispatch_graph_extents: source.graph_extents(),
            }),
            completion_roots,
            completions,
        })
    }
    pub(crate) fn append(self, other: Self) -> Option<Self> {
        Some(Self {
            calls: self.calls.append(other.calls)?,
            native: self.native.append(other.native)?,
            cpu: match (self.cpu, other.cpu) {
                (Some(a), Some(b)) => Some(a.append(b)?),
                (a, b) => a.or(b),
            },
            completion_roots: self.completion_roots.max(other.completion_roots),
            completions: self.completions.checked_add(other.completions)?,
        })
    }
    pub(crate) fn repeat(self, count: usize) -> Option<Self> {
        Some(Self {
            calls: self.calls.repeat(count)?,
            native: self.native.repeat(count)?,
            cpu: match self.cpu {
                Some(population) => Some(population.repeat(count)?),
                None => None,
            },
            completion_roots: self.completion_roots,
            completions: self.completions.checked_mul(count)?,
        })
    }
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            calls: self.calls.union(other.calls),
            native: self.native.union(other.native),
            cpu: match (self.cpu, other.cpu) {
                (Some(a), Some(b)) => Some(a.union(b)),
                (a, b) => a.or(b),
            },
            completion_roots: self.completion_roots.max(other.completion_roots),
            completions: self.completions.max(other.completions),
        }
    }
    /// A completed numerical stage uses the same equation census and concrete
    /// completion caller as the enclosing model. Payload storage is separate.
    pub(crate) fn numerical(recipe: SpeculativeNumericalRecipe) -> Option<Self> {
        use crate::backend::nn::shared::MlxNeuralBackend;
        let completion = recipe.completion;
        let mut result = Self::constructed(recipe)?;
        result.calls = result
            .calls
            .append(MlxNeuralBackend::ordinary_completion_call_controls(
                completion.traversal.roots(),
            )?)?;
        if completion.nested_completions != 0 {
            result.calls = result.calls.append(
                MlxNeuralBackend::ordinary_completion_call_controls(
                    completion.nested_traversal()?.roots(),
                )?
                .repeat(completion.nested_completions)?,
            )?;
        }
        result.completion_roots =
            completion
                .traversal
                .roots()
                .max(if completion.nested_completions == 0 {
                    0
                } else {
                    completion.nested_traversal()?.roots()
                });
        result.completions = completion.nested_completions.checked_add(1)?;
        Some(result)
    }
    /// A lazy constructor joins its consumer's reachable graph. It supplies no
    /// independent completion frontier or execution authority.
    pub(crate) fn constructed(recipe: SpeculativeNumericalRecipe) -> Option<Self> {
        let completion = recipe.completion;
        let cpu = resident_recipe::OrdinaryCpuPopulation::from_completion(completion);
        let native = if cpu.is_some() {
            OrdinaryNativeControls::default()
        } else {
            completion.ordinary_controls()?
        };
        Some(Self {
            calls: recipe.ordinary_calls?,
            native,
            cpu,
            completion_roots: 0,
            completions: 0,
        })
    }
}
