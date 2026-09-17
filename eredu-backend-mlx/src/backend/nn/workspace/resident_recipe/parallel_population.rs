//! One census for exact retained collectives on either model device.
use super::*;

#[derive(Default)]
pub(super) struct ParallelPopulation {
    pub(super) arrays: usize,
    pub(super) primitives: usize,
    pub(super) edges: usize,
    pub(super) births: usize,
    pub(super) child_births: usize,
    pub(super) bytes: u64,
    pub(super) graph_extents: usize,
    pub(super) controls: usize,
    pub(super) streams: usize,
    pub(super) nested: usize,
}
impl ResidentRecipeRecorder {
    pub(super) fn parallel_population(
        &self,
        index: usize,
        operation: &WorkspaceOperation,
        invocation: Option<&parallel::OriginalParallelInvocation>,
    ) -> Result<Option<ParallelPopulation>, Error> {
        if let Some(context) = &self.context {
            context.charge_metadata(std::mem::size_of::<(
                &Self,
                usize,
                &WorkspaceOperation,
                Option<&parallel::OriginalParallelInvocation>,
                ParallelPopulation,
                Result<Option<ParallelPopulation>, Error>,
            )>())?;
        }
        let invalid = || self.metadata_error("retained parallel population overflow");
        if matches!(
            operation.kind,
            WorkspaceOperationKind::Collective(
                eredu_nn::workspace::WorkspaceCollective::Boundary { .. }
            )
        ) {
            let Some(quote) = invocation.and_then(|p| p.boundary_occurrence(index)) else {
                return Ok(None);
            };
            if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
                return Ok(None);
            }
            return Ok(Some(ParallelPopulation {
                arrays: usize::from(quote.receiving),
                child_births: quote.maximum_backing_births().ok_or_else(invalid)?,
                bytes: quote
                    .output
                    .unwrap_or(0)
                    .checked_add(quote.scratch)
                    .ok_or_else(invalid)?,
                streams: 1,
                ..Default::default()
            }));
        }
        if let Some(quote) = invocation.and_then(|p| p.logical_occurrence(index)) {
            return Ok(Some(ParallelPopulation {
                arrays: 1,
                child_births: quote.maximum_backing_births().ok_or_else(invalid)?,
                bytes: quote
                    .output
                    .checked_add(quote.scratch)
                    .ok_or_else(invalid)?,
                streams: 1,
                nested: 1,
                ..Default::default()
            }));
        }
        let Some((native, backing)) = invocation.and_then(|p| p.occurrence(index)) else {
            return Ok(None);
        };
        let (primitives, edges) = native.graph_population();
        if primitives != 1
            || edges != 1
            || !(1..=2).contains(&backing.births)
            || operation.outputs.len() != 1
        {
            return Ok(None);
        }
        Ok(Some(ParallelPopulation {
            arrays: primitives
                .checked_add(operation.inputs.len())
                .ok_or_else(invalid)?,
            primitives,
            edges,
            births: backing.births,
            child_births: 0,
            bytes: backing
                .output
                .checked_add(backing.scratch)
                .ok_or_else(invalid)?,
            graph_extents: native.graph_extent().ok_or_else(invalid)?,
            controls: native.execution_control_bytes().ok_or_else(invalid)?,
            streams: 2,
            nested: if matches!(
                operation.kind,
                WorkspaceOperationKind::Collective(
                    eredu_nn::workspace::WorkspaceCollective::Broadcast { .. }
                )
            ) {
                2
            } else {
                0
            },
        }))
    }
}
