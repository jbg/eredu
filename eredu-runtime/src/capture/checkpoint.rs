//! In-process checkpoints of the shared record owner, independent of native state.

use super::*;
use eredu_core::intervention::{
    AdmittedInterventionPlan, InterventionDiscovery, InterventionEstimator, InterventionPlan,
};
use std::sync::Arc;

/// Reusable checkpoint at a drained portable record boundary. This is only the
/// capture/intervention component of a generation snapshot; callers must also
/// establish native completion and save model, sampler, and semantic state.
///
/// It contains no estimator or native handle. Admission proofs remain opaque and
/// cannot be deserialized. Clones share only immutable host admission data.
#[derive(Clone)]
pub struct CaptureCheckpoint {
    owner: Arc<()>,
    artifact_identity: String,
    plan: AdmittedCapturePlan,
    intervention: Option<AdmittedInterventionPlan>,
    prediction: u64,
    phase: CapturePhase,
    has_step: bool,
    usage: CaptureUsage,
}

/// Explicit admission inputs for a child capture run. The backend supplies the
/// actual child's retained discovery and current estimator facts.
pub struct CaptureForkRequest<'a> {
    /// Child's actual loaded source, observation catalog and support.
    pub discovery: &'a CaptureDiscovery,
    /// Absolute output limit, including the inherited prefix.
    pub max_predictions: u64,
    /// Explicit child limits. Cumulative limits include inherited consumption.
    pub limits: CaptureLimits,
    /// Required when inheriting interventions; also permits adding a new plan.
    pub intervention: Option<InterventionForkRequest<'a>>,
}

/// Shared re-admission of inherited or prospectively replaced interventions.
pub struct InterventionForkRequest<'a> {
    /// Actual child's backend/source/target discovery.
    pub discovery: &'a InterventionDiscovery,
    /// Fresh child facade session identity, never the source admission identity.
    pub session_id: &'a str,
    /// None inherits the original operations. Some replaces future operations;
    /// an empty plan removes them. Earlier native state is not recomputed.
    pub replacement: Option<InterventionPlan>,
    /// Child's current side-effect-free native facts. Not part of the checkpoint.
    pub estimator: Arc<dyn InterventionEstimator>,
}

/// Validated, move-only same-run schedule restoration. Dropping this preparation
/// changes nothing. Commit is infallible so native and portable state can be
/// installed atomically after every fallible operation has succeeded.
pub struct PreparedCaptureRestore<'a> {
    run: &'a mut CaptureSession,
    prediction: u64,
    phase: CapturePhase,
    has_step: bool,
}

impl PreparedCaptureRestore<'_> {
    /// Installs only rewindable schedule/delivery state, leaving consumption intact.
    pub fn commit(self) {
        self.run.prediction = self.prediction;
        self.run.phase = self.phase;
        self.run.has_step = self.has_step;
        self.run.capture_seconds = 0.0;
        self.run.checkpoint_ready = true;
    }
}

impl CaptureSession {
    /// Known logical host storage for a copied portable checkpoint. Includes all
    /// admitted plan payloads and declarations, but no native estimator or handle.
    /// Callers reserve this before `checkpoint` clones the admission data.
    pub fn checkpoint_storage_bytes(&self, discovery: &CaptureDiscovery) -> Option<u64> {
        checkpoint_storage_bytes(
            &self.plan,
            self.interventions.as_ref().map(|run| &run.plan),
            &discovery.artifact_identity,
        )
    }
    /// Saves portable schedule position only after the current records have been
    /// delivered and all intervention finish checks succeeded. The initial run is
    /// also a valid boundary. The caller supplies retained loaded discovery.
    pub fn checkpoint(
        &self,
        discovery: &CaptureDiscovery,
    ) -> Result<CaptureCheckpoint, CaptureError> {
        if !self.checkpoint_ready
            || self.records.is_some()
            || self
                .interventions
                .as_ref()
                .is_some_and(|run| run.records.is_some() || run.routing_pending.is_some())
        {
            return Err(CaptureError::Invalid(
                "capture checkpoint requires a successful, drained record boundary".into(),
            ));
        }
        let checked = self.plan.plan().clone().admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            self.plan.request(),
        )?;
        if checked.identity() != self.plan.identity()
            || self
                .interventions
                .as_ref()
                .is_some_and(|run| run.plan.artifact_identity() != discovery.artifact_identity)
        {
            return Err(CaptureError::Invalid(
                "checkpoint source/admission mismatch".into(),
            ));
        }
        Ok(CaptureCheckpoint {
            owner: Arc::clone(&self.owner),
            artifact_identity: discovery.artifact_identity.clone(),
            plan: self.plan.clone(),
            intervention: self.interventions.as_ref().map(|run| run.plan.clone()),
            prediction: self.prediction,
            phase: self.phase,
            has_step: self.has_step,
            usage: self.ledger.total(),
        })
    }

    /// Checks a same-run restore before any native or portable mutation. Undelivered
    /// records must be consumed even when the previous operation failed.
    pub fn validate_restore(&self, checkpoint: &CaptureCheckpoint) -> Result<(), CaptureError> {
        if !Arc::ptr_eq(&self.owner, &checkpoint.owner)
            || self.plan.identity() != checkpoint.plan.identity()
            || self.interventions.as_ref().map(|run| run.plan.identity())
                != checkpoint.intervention.as_ref().map(|plan| plan.identity())
        {
            return Err(CaptureError::Invalid(
                "capture checkpoint belongs to another run".into(),
            ));
        }
        if self.records.is_some()
            || self
                .interventions
                .as_ref()
                .is_some_and(|run| run.records.is_some() || run.routing_pending.is_some())
        {
            return Err(CaptureError::Invalid(
                "restore requires drained records and resolved routing".into(),
            ));
        }
        Ok(())
    }

    /// Rewinds schedule state without refunding any consumed resource or publishing
    /// old records again. Native restoration must succeed before this is committed.
    pub fn restore(&mut self, checkpoint: &CaptureCheckpoint) -> Result<(), CaptureError> {
        self.prepare_restore(checkpoint)?.commit();
        Ok(())
    }

    /// Validates restoration while exclusively borrowing the run until commit.
    pub fn prepare_restore(
        &mut self,
        checkpoint: &CaptureCheckpoint,
    ) -> Result<PreparedCaptureRestore<'_>, CaptureError> {
        self.validate_restore(checkpoint)?;
        Ok(PreparedCaptureRestore {
            run: self,
            prediction: checkpoint.prediction,
            phase: checkpoint.phase,
            has_step: checkpoint.has_step,
        })
    }

    /// Cumulative usage of this run, including the explicitly inherited child base.
    pub fn cumulative_usage(&self) -> CaptureUsage {
        self.ledger.total()
    }
}

impl CaptureCheckpoint {
    /// Logical storage of this immutable admission/schedule checkpoint.
    pub fn logical_storage_bytes(&self) -> Option<u64> {
        checkpoint_storage_bytes(
            &self.plan,
            self.intervention.as_ref(),
            &self.artifact_identity,
        )
    }

    /// Conservative logical host storage for shared child re-admission, measured
    /// against actual child declarations before cloning plans or payloads.
    pub fn fork_storage_bytes(&self, request: &CaptureForkRequest<'_>) -> Option<u64> {
        use crate::execution_control::storage::heap_bytes;
        let mut bytes = self
            .logical_storage_bytes()?
            .checked_add(u64::try_from(std::mem::size_of::<CaptureSession>()).ok()?)?;
        for selection in &self.plan.plan().selections {
            let point = request
                .discovery
                .catalog
                .points
                .iter()
                .find(|point| point.path == selection.path)?;
            bytes = bytes
                .checked_add(u64::try_from(std::mem::size_of_val(point)).ok()?)?
                .checked_add(heap_bytes(point)?)?;
        }
        if let Some(child) = &request.intervention {
            let plan = child
                .replacement
                .as_ref()
                .or_else(|| self.intervention.as_ref().map(|plan| plan.plan()))?;
            bytes = bytes
                .checked_add(u64::try_from(std::mem::size_of::<AdmittedInterventionPlan>()).ok()?)?
                .checked_add(heap_bytes(plan)?)?
                .checked_add(u64::try_from(child.discovery.artifact_identity.len()).ok()?)?
                .checked_add(u64::try_from(child.session_id.len()).ok()?)?
                .checked_add(64)?;
            for operation in &plan.operations {
                let point = child
                    .discovery
                    .points
                    .iter()
                    .find(|point| point.path == operation.target)?;
                bytes = bytes
                    .checked_add(u64::try_from(std::mem::size_of_val(point)).ok()?)?
                    .checked_add(heap_bytes(point)?)?;
            }
        }
        Some(bytes)
    }
    /// Absolute next prediction; zero means prompt prefill has not executed.
    pub fn next_prediction(&self) -> u64 {
        // begin_step requires prediction < max_predictions, hence this cannot overflow.
        if self.has_step {
            self.prediction + 1
        } else {
            0
        }
    }

    /// Usage inherited by a child. Same-run restore never writes this to the ledger.
    pub fn inherited_usage(&self) -> CaptureUsage {
        self.usage
    }

    /// Original source provenance, retained when admitting child plans.
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }

    /// Original intervention admission for lineage/override provenance.
    pub fn intervention_plan(&self) -> Option<&AdmittedInterventionPlan> {
        self.intervention.as_ref()
    }

    /// Re-admits child plans with current discovery and estimator facts before
    /// installing a new shared run. Absolute schedules keep their original origin.
    /// Child budgets include consumption at this checkpoint; subsequent work in
    /// either parent or child is charged independently. No native state is copied.
    pub fn fork(
        &self,
        request: CaptureForkRequest<'_>,
        estimate: impl FnMut(
            &[u64],
            &CaptureSelection,
            &ResolvedCaptureSlice,
        ) -> Result<CaptureUsage, CaptureError>,
    ) -> Result<CaptureSession, CaptureError> {
        if request.discovery.artifact_identity != self.artifact_identity {
            return Err(CaptureError::Invalid(
                "child prepared source differs from checkpoint".into(),
            ));
        }
        let mut geometry = self.plan.request();
        geometry.max_predictions = request.max_predictions;
        let mut plan = self.plan.plan().clone();
        plan.limits = request.limits;
        let plan = plan.admit(
            &request.discovery.catalog,
            &request.discovery.support,
            &request.discovery.support.capture,
            geometry,
        )?;
        super::validate_continuation(
            &plan,
            request.discovery,
            self.next_prediction(),
            self.usage,
            estimate,
        )?;

        let intervention = match request.intervention {
            Some(child) => {
                if child.session_id.is_empty()
                    || self
                        .intervention
                        .as_ref()
                        .is_some_and(|parent| parent.session_id() == child.session_id)
                    || child.discovery.artifact_identity != self.artifact_identity
                {
                    return Err(CaptureError::Invalid(
                        "child intervention identity/source mismatch".into(),
                    ));
                }
                let operations = child
                    .replacement
                    .or_else(|| self.intervention.as_ref().map(|p| p.plan().clone()))
                    .ok_or_else(|| {
                        CaptureError::Invalid("child intervention plan is absent".into())
                    })?;
                let admitted = operations.admit(child.discovery, geometry, child.session_id)?;
                crate::intervention::validate_continuation(
                    &plan,
                    &admitted,
                    child.discovery,
                    child.estimator.as_ref(),
                    self.next_prediction(),
                    self.usage,
                )?;
                Some((admitted, child.estimator))
            }
            None if self.intervention.is_some() => {
                return Err(CaptureError::Invalid(
                    "inherited interventions require child discovery and re-admission".into(),
                ))
            }
            None => None,
        };
        // Installation stays with the same shared owner. A fully empty child still
        // carries its inherited ledger, so removing controls cannot erase usage.
        let mut child = CaptureSession::new(plan);
        if let Some((plan, estimator)) = intervention {
            child.enable_interventions(plan, estimator)?;
        }
        child.ledger = CaptureLedger::with_inherited_usage(&child.plan, self.usage)?;
        child.prediction = self.prediction;
        child.phase = self.phase;
        child.has_step = self.has_step;
        Ok(child)
    }
}

fn checkpoint_storage_bytes(
    plan: &AdmittedCapturePlan,
    intervention: Option<&AdmittedInterventionPlan>,
    artifact: &str,
) -> Option<u64> {
    use crate::execution_control::storage::heap_bytes;
    let mut total = u64::try_from(std::mem::size_of::<CaptureCheckpoint>())
        .ok()?
        .checked_add(u64::try_from(artifact.len()).ok()?)?
        .checked_add(heap_bytes(plan.plan())?)?
        .checked_add(heap_bytes(plan.points())?)?
        .checked_add(u64::try_from(plan.identity().len()).ok()?)?;
    if let Some(plan) = intervention {
        total = total
            .checked_add(heap_bytes(plan.plan())?)?
            .checked_add(heap_bytes(plan.points())?)?
            .checked_add(u64::try_from(plan.identity().len()).ok()?)?
            .checked_add(u64::try_from(plan.artifact_identity().len()).ok()?)?
            .checked_add(u64::try_from(plan.session_id().len()).ok()?)?;
    }
    Some(total)
}
