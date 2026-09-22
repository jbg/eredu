//! Every actual explicit-route Pair/Peer and the shared final result source.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::{
    parallel::RetainedLogicalCollective, OriginalCommunicationSource,
};

pub(crate) struct RoutedCollectiveQuote {
    pub(crate) values: Vec<RoutedValueQuote>,
    pub(crate) result_recipe: SpeculativeNumericalRecipe,
    pub(crate) result_scratch: Option<WorkspaceAllocationPopulation>,
    pub(crate) result_capacity: BoundaryStageCapacity,
}
pub(crate) struct RoutedValueQuote {
    pub(crate) source_rank: usize,
    pub(crate) steps: Vec<RoutedStepQuote>,
}
pub(crate) struct RoutedStepQuote {
    pub(crate) step: usize,
    pub(crate) source: RetainedLogicalCollective,
}
impl RoutedCollectiveQuote {
    pub(super) fn prepare(
        source: &OriginalParallelSource,
        actual: &OriginalCommunicationSource<'_>,
        order: usize,
        kind: LogicalCollectiveKind,
        layout: &WorkspaceLayout,
        dtype: safemlx::Dtype,
        mechanism: ResidentExecutionMechanisms,
        protocol: eredu_runtime::CommunicationOperation,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(size_of::<(
            Self,
            Result<Self, Error>,
            Vec<RoutedValueQuote>,
            Vec<RoutedStepQuote>,
            Vec<(usize, WorkspaceTensor)>,
            LogicalCollectiveQuote,
            WorkspaceTraceReport,
            SpeculativeNumericalRecipe,
            Option<WorkspaceAllocationPopulation>,
            BoundaryStageCapacity,
            Vec<bool>,
            std::ops::Range<usize>,
            Option<usize>,
            usize,
            bool,
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "routed model source differs from its retained plan"
            ))
        };
        let group = actual.group(order).ok_or_else(invalid)?.0;
        let plan = group
            .logical_routed_plan()
            .map_err(|_| invalid())?
            .ok_or_else(invalid)?;
        if plan.len() != group.size() || plan.len() == 0 {
            return Err(invalid());
        }
        let mut values = context.metadata_vec(plan.len())?;
        let mut results = context.metadata_vec(plan.len())?;
        let mut ranks = context.metadata_vec(plan.len())?;
        ranks.resize(plan.len(), false);
        for (value, route) in plan.values().enumerate() {
            let occupied = ranks.get_mut(route.source_rank()).ok_or_else(invalid)?;
            if *occupied {
                return Err(invalid());
            }
            *occupied = true;
            let count = (0..route.steps())
                .filter(|step| route.exchange(*step).is_some())
                .count();
            let mut steps = context.metadata_vec(count)?;
            for step in 0..route.steps() {
                if route.exchange(step).is_none() {
                    continue;
                }
                let mut quote = LogicalCollectiveQuote::prepare_routed_peer_protocol(
                    source,
                    order,
                    value,
                    step,
                    layout.as_view(),
                    dtype,
                    mechanism,
                    protocol,
                )?;
                quote.kind = kind;
                let source = RetainedLogicalCollective::retain_prepared(source, quote)
                    .map_err(|cause| source.neural_error(cause))?;
                steps.push(RoutedStepQuote { step, source });
            }
            values.push(RoutedValueQuote {
                source_rank: route.source_rank(),
                steps,
            });
            // Each routed result is a completed, separately retained descriptor.
            results.push((
                route.source_rank(),
                WorkspaceTensor::existing(layout.clone(), context)?,
            ));
        }
        context.begin_span();
        let ops = logical_collective::Workspace(context);
        let result = match kind {
            LogicalCollectiveKind::Sum => logical_collective::routed::sum(&ops, results)?,
            LogicalCollectiveKind::Gather => {
                logical_collective::routed::gather(&ops, results, layout.shape())?
            }
        };
        let result_scratch = context.new_allocation_scratch()?;
        let report = context.finish_report(&[result])?;
        let result_recipe = super::super::numerical(&report, 1, mechanism, context)?;
        let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
        let result_capacity = boundary::capacity(result_recipe, runtime, context)?;
        Ok(Self {
            values,
            result_recipe,
            result_scratch,
            result_capacity,
        })
    }
    pub(super) fn scratch(&self) -> Option<usize> {
        self.values.iter().try_fold(0usize, |total, value| {
            value.steps.iter().try_fold(total, |total, step| {
                total
                    .checked_add(usize::try_from(step.source.value().output).ok()?)?
                    .checked_add(usize::try_from(step.source.value().scratch).ok()?)
            })
        })
    }
    pub(super) fn maximum_births(&self) -> Option<usize> {
        self.values.iter().try_fold(
            self.result_recipe.storage.maximum_births(),
            |total, value| {
                value.steps.iter().try_fold(total, |total, step| {
                    total.checked_add(step.source.value().maximum_backing_births()?)
                })
            },
        )
    }
}
