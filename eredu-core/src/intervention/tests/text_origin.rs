use super::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
struct Host(Arc<AtomicBool>);
impl Drop for Host {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[test]
fn cached_intervention_origin_survives_identity_copy_evidence_and_fixed_geometry() {
    let mut discovery = discovery(false);
    discovery.points[0].axes[2].dimension = D::Context;
    let mut operation = zero();
    operation.evidence = InterventionEvidence::Summary;
    let raw = plan(vec![operation]);
    let zero = raw.clone().admit(&discovery, request(), "session").unwrap();
    let explicit = raw
        .clone()
        .admit_with_text_origin(
            &discovery,
            request(),
            CaptureTextOrigin::default(),
            "session",
        )
        .unwrap();
    assert_eq!(zero.identity(), explicit.identity());
    assert_eq!(zero.intent_identity(), explicit.intent_identity());
    let origin = CaptureTextOrigin {
        cached_positions: 7,
    };
    let admitted = raw
        .clone()
        .admit_with_text_origin(&discovery, request(), origin, "session")
        .unwrap();
    assert_ne!(zero.identity(), admitted.identity());
    assert_ne!(zero.intent_identity(), admitted.intent_identity());
    assert_eq!(admitted.text_origin(), Some(origin));
    assert_eq!(
        admitted.readmit(&discovery).unwrap().identity(),
        admitted.identity()
    );
    admitted.validate_discovery(&discovery).unwrap();
    let mut changed = admitted.clone();
    changed.text_origin.cached_positions += 1;
    assert_eq!(
        changed.validate_discovery(&discovery),
        Err(InterventionSourceError::Identity)
    );
    for (phase, prediction, sequence, context) in [
        (CapturePhase::Prefill, 0, 3, 10),
        (CapturePhase::Decode, 1, 1, 11),
        (CapturePhase::Decode, 3, 1, 13),
    ] {
        let shape = [1, sequence, context];
        let ordinary = admitted
            .validate_at(
                0,
                phase,
                prediction,
                None,
                &shape,
                Some(InterventionDtype::Float32),
            )
            .unwrap();
        let mut fixed = ResolvedCaptureSlice {
            starts: vec![0; 3],
            ends: vec![0; 3],
            strides: vec![0; 3],
            shape: vec![0; 3],
        };
        admitted
            .resolve_prepared_at(
                0,
                phase,
                prediction,
                &shape,
                InterventionDtype::Float32,
                &mut fixed,
            )
            .unwrap();
        assert_eq!(ordinary, fixed);
        assert_eq!(
            admitted
                .geometry_at(phase, prediction, None)
                .unwrap()
                .context,
            Some(context)
        );
        assert_eq!(
            admitted
                .estimate_shape(
                    &discovery.points[0].observation_geometry(),
                    phase,
                    prediction
                )
                .unwrap(),
            Some(shape.to_vec())
        );
        let mut stale = shape;
        stale[2] -= 7;
        assert!(admitted
            .resolve_prepared_at(
                0,
                phase,
                prediction,
                &stale,
                InterventionDtype::Float32,
                &mut fixed
            )
            .is_err());
    }
    assert!(raw
        .admit_with_text_origin(
            &discovery,
            request(),
            CaptureTextOrigin {
                cached_positions: u64::MAX
            },
            "session"
        )
        .is_err());
    let retired = Arc::new(AtomicBool::new(false));
    let source = PreparedInterventionPlanCopy::inspect(&admitted)
        .unwrap()
        .copy(crate::HostPreparationAuthority::retain(Host(
            retired.clone(),
        )))
        .unwrap();
    source.admission().validate_discovery(&discovery).unwrap();
    assert_eq!(source.admission().text_origin(), Some(origin));
    let evidence = source.evidence(0).unwrap().shared_geometry_source().clone();
    assert_eq!(evidence.admission().text_origin(), Some(origin));
    assert_eq!(
        evidence
            .admission()
            .geometry_at(CapturePhase::Decode, 3, None)
            .unwrap()
            .context,
        Some(13)
    );
    drop(source);
    drop(admitted);
    assert!(!retired.load(Ordering::SeqCst));
    drop(evidence);
    assert!(retired.load(Ordering::SeqCst));
}
