//! Actual bounded-policy group frontiers reuse the enclosing equation recipe.
//! Materialization/source-copy arenas remain a distinct required producer.
use super::*;
use crate::backend::error::Error;
use eredu_runtime::working_memory::WorkingMemoryError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryCompletion {
    RetainedEvent,
    SynchronousPrediction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NeuralBoundaries {
    per_forward: usize,
    consumers: usize,
    waits: usize,
    completion: BoundaryCompletion,
}
impl ResidentNativeRecipe {
    /// Only the selected native policy supplies these call-site populations.
    /// Every same-role group can reach the full enclosing equation DAG. Using
    /// that ceiling once per attempted boundary preserves asynchronous overlap
    /// and gives no early-retirement credit or physical payload duplication.
    pub(crate) fn bind_neural_boundaries(
        &mut self,
        geometry: InferenceGeometry,
        per_forward: usize,
        consumers: usize,
    ) -> Result<(), Error> {
        self.bind_neural_boundaries_with_roots(
            geometry, per_forward, consumers, None, BoundaryCompletion::RetainedEvent,
        )
    }
    /// Consumes an already paid exact root-list inventory from one prediction
    /// equation. This inventory is descriptive and carries no source authority.
    pub(super) fn bind_prediction_boundaries(
        &mut self, geometry: InferenceGeometry, roots: Vec<usize>, consumers: usize,
    ) -> Result<(), Error> {
        if self.records.len() != 1 || roots.is_empty() || roots.contains(&0) || consumers != 0 {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        self.bind_neural_boundaries_with_roots(
            geometry, roots.len(), consumers, Some(roots), BoundaryCompletion::SynchronousPrediction,
        )
    }
    fn bind_neural_boundaries_with_roots(
        &mut self, geometry: InferenceGeometry, per_forward: usize, consumers: usize,
        roots: Option<Vec<usize>>,
        completion: BoundaryCompletion,
    ) -> Result<(), Error> {
        let error = |cause| Error::PrefillControl(cause).at_speculative_stage(match completion {
            BoundaryCompletion::RetainedEvent => "retained-event neural boundary",
            BoundaryCompletion::SynchronousPrediction => "synchronous prediction boundary",
        });
        if self.plan.geometry() != geometry || self.neural.is_some() || per_forward == 0 {
            return Err(error(WorkingMemoryError::IdentityMismatch));
        }
        let waits = per_forward
            .checked_mul(consumers)
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        let source_controls = if completion == BoundaryCompletion::RetainedEvent {
            ResidentDispatchPopulation::completion_stream_control_bytes()
        } else { 0 };
        if source_controls != 0 {
            if let Some(funding) = &self.planning_metadata {
                funding.reserve_metadata(source_controls).map_err(Error::WorkspacePlanning)?;
            }
        }
        // Both workers use run_nested_graph_event. Mixed CPU/GPU callbacks
        // settle before its suspended resident bank is restored; GPU-only
        // retained events keep the ordinary asynchronous completion ownership.
        // Preserve every actual source gate, and identify the first absent
        // prerequisite instead of conflating it with the completion mode.
        for (record, row) in self.records.iter_mut().enumerate() {
            let requirement = if row.first_missing_operation.is_some() {
                Some("equation producer")
            } else if let Some(owner) = row.unqualified_kernel_owner {
                Some(owner.reason())
            } else if row.traversal.is_none() {
                Some("Eval traversal source")
            } else if row.dispatch.is_none() {
                Some("Eval dispatch source")
            } else if row.graph.is_none() {
                Some("Graph constructor source")
            } else if row.mutable_storage.is_none() {
                Some("native backing source")
            } else {
                let traversal = row.traversal.expect("checked traversal");
                let dispatch = row.dispatch.expect("checked dispatch");
                // The same native device may own more than one source stream.
                // Keep the exact model/router/collective classification shared
                // with source-copy expansion and native Graph construction.
                (completion == BoundaryCompletion::RetainedEvent
                    && dispatch.completion_streams() != Some(traversal.limits().streams))
                    .then_some("completion stream population")
            };
            if let Some(requirement) = requirement {
                return Err(Error::NeuralBoundarySource {
                    boundary: match completion {
                        BoundaryCompletion::RetainedEvent => "retained event",
                        BoundaryCompletion::SynchronousPrediction => "synchronous prediction",
                    },
                    record,
                    requirement,
                    operation: row.first_missing_operation,
                    detail: row.missing_operation_detail.take(),
                    cause: WorkingMemoryError::UnknownBound,
                });
            }
            row.nested_completions
                .checked_add(per_forward)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
            if source_controls != 0 && self.planning_metadata.is_none() {
                row.query_controls = Some(row.query_controls
                    .ok_or_else(|| error(WorkingMemoryError::UnknownBound))?
                    .checked_add(source_controls)
                    .ok_or_else(|| error(WorkingMemoryError::Overflow))?);
            }
        }
        safemlx::OperationEvent::wait_record_layout(waits).ok_or_else(|| {
            Error::PrefillControl(WorkingMemoryError::UnknownBound)
                .at_speculative_stage("neural boundary consumer WaitRecord source")
        })?;
        if let Some(roots) = roots {
            self.records[0].prediction_roots = Some(roots);
        } else {
            for row in &mut self.records {
                row.nested_completions += per_forward;
            }
        }
        self.neural = Some(NeuralBoundaries {
            per_forward,
            consumers,
            waits,
            completion,
        });
        Ok(())
    }
    pub(crate) fn matches_neural_boundaries(
        &self,
        geometry: InferenceGeometry,
        submissions: usize,
        consumers: usize,
    ) -> bool {
        self.matches_boundaries(geometry, submissions, consumers, BoundaryCompletion::RetainedEvent)
    }
    pub(super) fn matches_prediction_boundaries(
        &self, geometry: InferenceGeometry, submissions: usize, consumers: usize,
    ) -> bool {
        self.matches_boundaries(geometry, submissions, consumers, BoundaryCompletion::SynchronousPrediction)
    }
    fn matches_boundaries(
        &self, geometry: InferenceGeometry, submissions: usize, consumers: usize,
        completion: BoundaryCompletion,
    ) -> bool {
        self.plan.geometry() == geometry
            && self.neural.is_some_and(|n| {
                n.completion == completion
                    && n.consumers == consumers
                    && self
                        .plan
                        .generation_forward_count()
                        .and_then(|forwards| n.per_forward.checked_mul(forwards))
                        == Some(submissions)
            })
    }
    pub(super) fn retains_neural_bank_through_cpu_completion(&self) -> bool {
        // Both bound variants use run_nested_graph_event, which settles mixed
        // callbacks before its suspended resident bank can be restored.
        self.neural.is_some()
    }
    pub(super) fn neural_waits_per_forward(&self) -> usize {
        self.neural.map_or(0, |n| n.waits)
    }
}
