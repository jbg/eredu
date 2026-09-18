//! Immutable internal authority and atomic selection over one cumulative ledger.
use super::*;
use eredu_core::speculative::AdmittedSpeculativeActivations;
use std::sync::Arc;

/// Host-only saved authority. Native state and snapshot reservations remain with
/// the enclosing execution controller. Invocation IDs and consumption are not saved.
#[derive(Clone, Debug)]
pub struct SpeculativeActivationCheckpoint {
    pub(super) authority: Authority,
    // Last: copied authority retires before inherited host custody.
    pub(super) host_owner: Arc<crate::capture::CaptureHostOwner>,
    // Original shared shell and authority allocations retire before source H.
    pub(super) original_funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}

#[derive(Clone, Debug)]
pub(super) enum Authority {
    Ordinary { owner: Arc<()>, plan: AdmittedSpeculativeActivations },
    Original(super::original::control::Saved),
}


impl SpeculativeActivationCheckpoint {
    /// Retains already-acquired host custody for this checkpoint and source-run
    /// aliases. This is not fresh speculative destination admission or a bound.
    pub fn retain_host_preparation(
        &self,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Result<(), CaptureError> {
        self.host_owner.retain(authority)
    }
}

/// Fully validated authority replacement. Dropping this preparation changes
/// nothing; commit performs no native work and cannot fail.
pub trait PreparedSpeculativeActivationRestore {
    /// Installs saved authority after complete native replacement succeeds.
    fn commit(self: Box<Self>);
}

struct Prepared<'a, P: CaptureBackendProvider, F> {
    observer: &'a mut SpeculativeCaptureObserver<P, F>,
    plan: AdmittedSpeculativeActivations,
    intervention: Option<crate::intervention::InterventionRun>,
    identity: String,
    scopes: Vec<SpeculativeCaptureScope>,
}

impl<P: CaptureBackendProvider, F> PreparedSpeculativeActivationRestore for Prepared<'_, P, F> {
    fn commit(self: Box<Self>) {
        self.observer.session.interventions = self.intervention;
        self.observer.admission_identity = Some(self.identity);
        self.observer.intervention_scopes = self.scopes;
        self.observer.admitted = Some(self.plan);
        self.observer.routed_path = None;
    }
}

pub(super) fn storage(plan: &AdmittedSpeculativeActivations) -> Option<u64> {
    use crate::execution_control::storage::heap_bytes;
    let strings = [
        plan.identity(),
        plan.execution_identity(),
        plan.artifact_identity(),
        plan.session_identity(),
        plan.captures().identity(),
        plan.interventions().identity(),
        plan.interventions().intent_identity(),
        plan.interventions().artifact_identity(),
        plan.interventions().session_id(),
    ];
    let host = strings.into_iter().try_fold(
        std::mem::size_of::<SpeculativeActivationCheckpoint>() as u64,
        |sum, string| sum.checked_add(string.len() as u64),
    )?;
    [
        heap_bytes(plan.captures().plan())?,
        heap_bytes(plan.captures().points())?,
        heap_bytes(plan.interventions().plan())?,
        heap_bytes(plan.interventions().points())?,
        heap_bytes(plan.capture_scopes())?,
        heap_bytes(plan.intervention_scopes())?,
    ]
    .into_iter()
    .try_fold(host, |sum, n| sum.checked_add(n))?
    .checked_mul(2)
}

impl<P: CaptureBackendProvider, F> SpeculativeCaptureObserver<P, F> {
    pub(super) fn control_storage_bytes(&self) -> Option<u64> {
        self.drained_boundary().ok()?;
        if !self.session.checkpoint_ready || self.failure.borrow().is_some() {
            return None;
        }
        storage(self.admitted.as_ref()?)
    }

    pub(super) fn save_control(&self) -> Result<SpeculativeActivationCheckpoint, CaptureError> {
        self.control_storage_bytes().ok_or_else(|| {
            CaptureError::Unsupported("internal authority has no drained bounded checkpoint".into())
        })?;
        let _source_authority = self.session.owner.retained()?;
        Ok(SpeculativeActivationCheckpoint {
            authority: Authority::Ordinary { owner: Arc::clone(&self.control_owner),
            plan: self
                .admitted
                .as_ref()
                .expect("bounded admitted authority")
                .clone() },
            host_owner: Arc::clone(&self.session.owner),
            original_funding: None,
        })
    }

    fn validate_plan(
        &self,
        plan: &AdmittedSpeculativeActivations,
        inherited: CaptureUsage,
    ) -> Result<(), CaptureError> {
        self.drained_boundary()?;
        if !self.session.checkpoint_ready || self.failure.borrow().is_some() {
            return Err(CaptureError::Invalid(
                "internal edits require a successful drained boundary".into(),
            ));
        }
        let current = self.admitted.as_ref().ok_or_else(|| {
            CaptureError::Unsupported(
                "internal re-admission requires capture authority from run creation".into(),
            )
        })?;
        if plan.execution_identity() != current.execution_identity()
            || plan.artifact_identity() != current.artifact_identity()
            || plan.session_identity() != current.session_identity()
            || plan.captures().identity() != current.captures().identity()
            || plan.capture_scopes() != current.capture_scopes()
        {
            return Err(CaptureError::Invalid(
                "internal re-admission changed execution, capture selections or allowances".into(),
            ));
        }
        crate::intervention::preflight_continuation(
            plan.captures(),
            plan.interventions(),
            self.estimator
                .as_deref()
                .ok_or_else(|| CaptureError::Unsupported("missing internal estimator".into()))?,
            0,
            inherited,
        )
    }

    fn prepare_plan(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Box<dyn PreparedSpeculativeActivationRestore + '_> {
        let intervention =
            (!plan.interventions().is_empty()).then(|| crate::intervention::InterventionRun {
                plan: plan.interventions().clone(),
                records: None,
                routing_pending: None,
                estimator: Arc::clone(self.estimator.as_ref().expect("validated estimator")),
            });
        let identity = plan.identity().to_owned();
        let scopes = plan.intervention_scopes().to_vec();
        Box::new(Prepared {
            observer: self,
            plan,
            intervention,
            identity,
            scopes,
        })
    }

    pub(super) fn prepare_control(
        &mut self,
        saved: &SpeculativeActivationCheckpoint,
    ) -> Result<Box<dyn PreparedSpeculativeActivationRestore + '_>, CaptureError> {
        let Authority::Ordinary { owner, plan } = &saved.authority else {
            return Err(CaptureError::Invalid("internal checkpoint belongs to another collector".into()));
        };
        if !Arc::ptr_eq(&self.control_owner, owner) {
            return Err(CaptureError::Invalid(
                "internal checkpoint belongs to another collector".into(),
            ));
        }
        // Restoring authority alone performs no capture work. Its next invocation
        // must still reserve against the current, unreduced cumulative ledger.
        self.validate_plan(plan, CaptureUsage::default())?;
        let source_authority = saved.host_owner.retained()?;
        self.session.retain_host_preparation(&source_authority)?;
        Ok(self.prepare_plan(plan.clone()))
    }

    pub(super) fn readmit_control(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), CaptureError> {
        self.validate_plan(&plan, self.session.cumulative_usage())?;
        let bytes = storage(&plan).ok_or(CaptureError::Overflow)?;
        if let Some(CaptureSkipReason::Limit { budget, cumulative }) =
            self.session.ledger.reserve(CaptureUsage {
                host_bytes: bytes,
                ..Default::default()
            })?
        {
            return Err(CaptureError::Limit { budget, cumulative });
        }
        self.prepare_plan(plan).commit();
        Ok(())
    }
}
