use super::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

struct HostToken(Arc<AtomicBool>);
impl Drop for HostToken {
    fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); }
}

#[test]
fn evidence_geometry_alias_retains_original_copy_and_host_until_last_side() {
    for evidence in [InterventionEvidence::Preview { max_elements: 3 }, InterventionEvidence::Summary] {
        let mut operation = zero();
        operation.evidence = evidence;
        let admitted = admit(operation).unwrap();
        let copied = PreparedInterventionPlanCopy::inspect(&admitted).unwrap();
        let retired = Arc::new(AtomicBool::new(false));
        let source = copied.copy(crate::HostPreparationAuthority::retain(HostToken(retired.clone()))).unwrap();
        let companion = source.evidence(0).unwrap();
        let before = companion.shared_geometry_source().clone();
        let after = companion.shared_geometry_source().clone();
        assert!(before.same_storage(&after));
        assert!(std::ptr::eq(companion.geometry_source(), before.admission()));
        let original_path = before.admission().plan().selections[0].path.as_ptr();
        assert_eq!(before.admission().plan().selections.len(), 2);
        assert_ne!(before.admission().plan().selections[0].id, before.admission().plan().selections[1].id);
        drop(source);
        drop(admitted);
        assert!(!retired.load(Ordering::SeqCst));
        assert_eq!(after.admission().plan().selections[0].path.as_ptr(), original_path);
        assert_eq!(after.admission().request(), request());
        drop(before);
        assert!(!retired.load(Ordering::SeqCst));
        drop(after);
        assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn evidence_free_source_does_not_construct_geometry_owner() {
    let admitted = admit(zero()).unwrap();
    let source = PreparedInterventionPlanCopy::inspect(&admitted).unwrap()
        .copy(crate::HostPreparationAuthority::unmanaged()).unwrap();
    assert!(source.evidence(0).is_none());
}

#[test]
fn peer_evidence_geometry_agrees_without_sharing_session_or_storage_authority() {
    for evidence in [InterventionEvidence::Preview { max_elements: 3 }, InterventionEvidence::Summary] {
        let mut operation = zero();
        operation.evidence = evidence;
        let host = plan(vec![operation.clone()]);
        let local = discovery(false);
        let mut peer = local.clone();
        peer.session_identity = Some("peer-native-session".into());
        let a = host.clone().admit(&local, request(), "run-a").unwrap();
        let b = host.admit(&peer, request(), "run-b").unwrap();
        assert_ne!(a.identity(), b.identity());
        assert!(a.validate_discovery(&peer).is_err());
        let copy = |source: &AdmittedInterventionPlan| {
            PreparedInterventionPlanCopy::inspect(source).unwrap()
                .copy(crate::HostPreparationAuthority::unmanaged()).unwrap()
        };
        let a_source = copy(&a);
        let b_source = copy(&b);
        let a_geometry = a_source.evidence(0).unwrap().shared_geometry_source();
        let b_geometry = b_source.evidence(0).unwrap().shared_geometry_source();
        assert_eq!(a_geometry.admission().identity(), b_geometry.admission().identity());
        assert!(!a_geometry.same_storage(b_geometry));
        assert_eq!(a_geometry.admission().plan(), b_geometry.admission().plan());
        assert_eq!(a_geometry.admission().request(), b_geometry.admission().request());
        operation.action = InterventionAction::Scale {
            dtype: InterventionDtype::Float32, factor: 0.75,
        };
        let changed = plan(vec![operation]).admit(&peer, request(), "run-b").unwrap();
        let changed_source = copy(&changed);
        assert_ne!(a_geometry.admission().identity(),
            changed_source.evidence(0).unwrap().geometry_source().identity());
    }
}
