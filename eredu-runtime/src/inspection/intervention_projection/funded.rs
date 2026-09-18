//! Paid selected declaration destinations from the ordinary support projection.
use super::*;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError, HostPreparationAuthority};
use eredu_core::capture::CapturePlanCopyError;
use std::mem::{size_of, size_of_val};

/// A complete discovery intermediate retaining the account that created it.
#[derive(Debug)]
pub struct FundedInterventionDiscovery {
    discovery: InterventionDiscovery,
    _funding: HostMetadataFunding,
}
impl FundedInterventionDiscovery {
    /// The exact selected declarations; this borrow grants no native authority.
    pub fn discovery(&self) -> &InterventionDiscovery { &self.discovery }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)] Host(#[from] HostMetadataFundingError),
    #[error(transparent)] Copy(#[from] CapturePlanCopyError),
    #[error("intervention discovery source or selected placement changed")] Identity,
}
/// A fixed refusal retaining any account already used by its producer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct InterventionDiscoveryPreparationError {
    #[source] cause: Cause,
    _funding: HostMetadataFunding,
}
impl InterventionDiscoveryPreparationError {
    /// Source owners use this before projection when retained identity differs.
    pub fn identity(funding: &HostMetadataFunding) -> Self {
        Self { cause: Cause::Identity, _funding: funding.clone() }
    }
}
/// Fixed census frames paid before any point's destination is inspected.
pub fn intervention_discovery_preparation_bytes() -> Option<usize> {
    let frames = [PreparedInterventionPointCopy::inspection_control_bytes()?,
        static_intervention_validation_control_bytes()?,
        size_of::<FundedInterventionDiscovery>(), size_of::<InterventionDiscovery>(),
        size_of::<InterventionDiscoveryPreparationError>(), size_of::<Cause>(),
        size_of::<Result<FundedInterventionDiscovery, InterventionDiscoveryPreparationError>>(),
        size_of::<[u8; 71]>(), size_of::<[usize; 4]>(),
        size_of::<[std::slice::Iter<'static, InterventionPoint>; 2]>(),
        size_of::<[std::slice::Iter<'static, InterventionOperation>; 2]>(),
        size_of::<[String; 2]>(), size_of::<std::collections::TryReserveError>(),
        HostMetadataFunding::reservation_control_bytes(),
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()?];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}
fn phase<'a>(value: &ProjectedStatus<'a>) -> InterventionPhaseView<'a> {
    match value {
        ProjectedStatus::Borrowed(status) => InterventionPhaseView::Retained(status),
        ProjectedStatus::Unavailable => InterventionPhaseView::Unavailable(UNAVAILABLE),
    }
}
fn text(source: &str) -> Result<String, CapturePlanCopyError> {
    let mut output = String::new();
    output.try_reserve_exact(source.len())?;
    if output.capacity() != source.len() { return Err(CapturePlanCopyError::Capacity); }
    output.push_str(source);
    Ok(output)
}
/// Copy only declarations addressed by the actual raw plan, preserving the same
/// point order, support selection and mechanism filtering as ordinary discovery.
/// Both the inspection frames and complete destinations are reserved before use.
pub fn prepare_intervention_discovery(
    plan: &InterventionPlan, points: &[InterventionPoint], support: &[ObservationSupport],
    facts: InterventionMechanismFacts<'_>, artifact: eredu_core::artifact::ArtifactIdentity,
    session: &str, funding: &HostMetadataFunding,
) -> Result<FundedInterventionDiscovery, InterventionDiscoveryPreparationError> {
    let result = (|| -> Result<_, Cause> {
        funding.reserve_metadata(intervention_discovery_preparation_bytes().ok_or(HostMetadataFundingError::Overflow)?)?;
        let selected = |point: &&InterventionPoint| plan.operations.iter().any(|operation| operation.target == point.path);
        let count = points.iter().filter(selected).count();
        let encoded = artifact.encoded();
        let mut bytes = std::alloc::Layout::array::<InterventionPoint>(count)
            .map_err(|_| HostMetadataFundingError::Overflow)?.size()
            .checked_add(encoded.len()).and_then(|n| n.checked_add(session.len()))
            .ok_or(HostMetadataFundingError::Overflow)?;
        for point in points.iter().filter(selected) {
            let projection = Projection::new(point, support_for(point, support), facts);
            bytes = bytes.checked_add(PreparedInterventionPointCopy::inspect(point, facts,
                phase(&projection.prefill), phase(&projection.decode))?.required_bytes())
                .ok_or(HostMetadataFundingError::Overflow)?;
        }
        funding.reserve_metadata(bytes)?;
        let authority = HostPreparationAuthority::retain(funding.clone());
        let mut copied = Vec::new();
        copied.try_reserve_exact(count).map_err(CapturePlanCopyError::from)?;
        if copied.capacity() != count { return Err(CapturePlanCopyError::Capacity.into()); }
        for point in points.iter().filter(selected) {
            let projection = Projection::new(point, support_for(point, support), facts);
            copied.push(PreparedInterventionPointCopy::inspect(point, facts,
                phase(&projection.prefill), phase(&projection.decode))?.copy(&authority)?);
        }
        Ok(FundedInterventionDiscovery {
            discovery: InterventionDiscovery { schema_version: INTERVENTION_SCHEMA_VERSION,
                artifact_identity: text(std::str::from_utf8(&encoded).expect("canonical ASCII identity"))?,
                session_identity: Some(text(session)?), points: copied },
            _funding: funding.clone(),
        })
    })();
    result.map_err(|cause| InterventionDiscoveryPreparationError { cause, _funding: funding.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{HostMetadataAccount, SymbolicDimension, TensorAxis};
    use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
    #[derive(Debug)]
    struct Account { remaining: Arc<Mutex<usize>>, retired: Arc<AtomicBool> }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            let mut remaining = self.remaining.lock().unwrap();
            *remaining = remaining.checked_sub(bytes).ok_or(HostMetadataFundingError::Capacity {
                required: bytes as u64, available: *remaining as u64 })?;
            Ok(())
        }
    }
    impl Drop for Account { fn drop(&mut self) { self.retired.store(true, Ordering::SeqCst); } }
    fn fixture() -> (InterventionPlan, Vec<InterventionPoint>, Vec<ObservationSupport>, InterventionMechanisms) {
        let point = InterventionPoint {
            path: "block.output".into(), node_id: "block".into(), stage: InterventionStage::Activation,
            axes: vec![TensorAxis { name: "component".into(), dimension: SymbolicDimension::Known(7) }],
            dtypes: vec![InterventionDtype::Float32, InterventionDtype::Float16],
            operations: vec![InterventionKind::Zero, InterventionKind::Scale], score_stages: vec![],
            prefill: Status::Unverified("architecture only".into()), decode: Status::Unverified("architecture only".into()),
            conditions: vec!["complete component axis".into()], routing: None, routed_units: None,
        };
        let mut ignored = point.clone(); ignored.path = "unused.output".into();
        let plan = InterventionPlan { schema_version: INTERVENTION_SCHEMA_VERSION, operations: vec![InterventionOperation {
            id: "edit-λ".into(), target: point.path.clone(), schedule: Default::default(), slices: vec![],
            action: InterventionAction::Scale { dtype: InterventionDtype::Float32, factor: -0.75 }, evidence: InterventionEvidence::None,
        }] };
        let support = vec![ObservationSupport { path: point.path.clone(),
            prefill: Status::Conditional("actual selected prefill".into()), decode: Status::Supported, floating_to_f32: true }];
        (plan, vec![ignored, point], support, InterventionMechanisms {
            operations: vec![InterventionKind::Scale], dtypes: vec![InterventionDtype::Float32], ..Default::default()
        })
    }
    #[test]
    fn selected_projection_retains_actual_sources_and_refusal_funding() {
        let (plan, points, support, mechanisms) = fixture();
        let artifact = eredu_core::artifact::fingerprint_artifact("fixture", [
            eredu_core::artifact::ArtifactMemberIdentity::new("weights", 3, [17; 32])
        ]).unwrap();
        for insufficient in [false, true] {
            let remaining = Arc::new(Mutex::new(1 << 20));
            let retired = Arc::new(AtomicBool::new(false));
            let funding = HostMetadataFunding::new(Account { remaining: remaining.clone(), retired: retired.clone() }).unwrap();
            if insufficient { *remaining.lock().unwrap() = intervention_discovery_preparation_bytes().unwrap(); }
            let result = prepare_intervention_discovery(&plan, &points, &support, mechanisms.borrowed(),
                artifact, "actual-backend", &funding);
            drop(funding);
            assert!(!retired.load(Ordering::SeqCst));
            if insufficient {
                let error = result.unwrap_err();
                assert!(matches!(error.cause, Cause::Host(HostMetadataFundingError::Capacity { .. })));
                assert_eq!(*remaining.lock().unwrap(), 0);
                drop(error);
            } else {
                let owner = result.unwrap();
                let discovery = owner.discovery();
                assert_eq!(discovery.artifact_identity, artifact.to_string());
                assert_eq!(discovery.session_identity.as_deref(), Some("actual-backend"));
                assert_eq!(discovery.points.len(), 1);
                let copied = &discovery.points[0];
                assert_eq!(copied.path, "block.output");
                assert_eq!(copied.operations, [InterventionKind::Scale]);
                assert_eq!(copied.dtypes, [InterventionDtype::Float32]);
                assert_eq!(copied.prefill, support[0].prefill);
                assert_eq!(copied.decode, Status::Supported);
                assert_ne!(copied.path.as_ptr(), points[1].path.as_ptr());
                drop(owner);
            }
            assert!(retired.load(Ordering::SeqCst));
        }
    }
}
