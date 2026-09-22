use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
#[derive(Debug)]
struct Meter(Arc<AtomicUsize>);
impl HostMetadataAccount for Meter {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }
}

fn invocation_sources() -> (SharedCapturePlan, AdmittedInterventionPlan) {
    let (text, edits) = sources_with_rows(true);
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 3,
        max_context: None,
        max_predictions: 4,
    };
    let capture = SharedCapturePlan::new(
        text.admission()
            .plan()
            .clone()
            .admit_invocations(
                &ObservationCatalog {
                    schema_version: 1,
                    points: vec![],
                    completeness: DescriptionCompleteness::Complete,
                },
                &ObservationSupportReport {
                    schema_version: 1,
                    capture: Default::default(),
                    points: vec![],
                },
                &CaptureCapabilities::default(),
                bounds,
            )
            .unwrap(),
    );
    let mut plan = edits.plan().clone();
    plan.operations.truncate(1);
    let discovery = InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "scheduled-original".into(),
        session_identity: Some("session".into()),
        points: vec![edits.points()[0].clone()],
    };
    (
        capture,
        plan.admit_invocations(&discovery, bounds, "session")
            .unwrap(),
    )
}

#[test]
fn partition_model_intervention_source_retains_actual_axes_window_and_failed_prefix() {
    for (phase, sequence, window) in [
        (CapturePhase::Prefill, 2, None),
        (CapturePhase::Decode, 3, None),
        (CapturePhase::Decode, 1, None),
        (
            CapturePhase::Prefill,
            2,
            Some(CaptureInvocationWindow {
                logical_sequence: 3,
                start: 1,
            }),
        ),
    ] {
        for rank in 0..2 {
            let (capture, admitted) = invocation_sources();
            let physical = CaptureInvocationShape {
                batch: 1,
                sequence,
                context: None,
            };
            let logical = window.map_or(physical, |window| window.validate(physical).unwrap());
            let pool = capture_test_ledger(1 << 26, 0).unwrap();
            let original = pool
                .compile_intervention_source(
                    PreparedInterventionPlanCopy::inspect(&admitted).unwrap(),
                )
                .unwrap();
            let meter = Arc::new(AtomicUsize::new(0));
            let funding = HostMetadataFunding::new(Meter(Arc::clone(&meter))).unwrap();
            let shape = [1, sequence, 4];
            let projection = PreparedPartitionInterventionProjection::prepare(
                &original,
                0,
                phase,
                1,
                Some(logical),
                &[1, logical.sequence, 4],
                2,
                &eredu_core::component::ComponentCoordinateMap::range(4, 0..4).unwrap(),
                None,
                4,
                funding.clone(),
            )
            .unwrap();
            let usage = CaptureUsage {
                retained_bytes: 32,
                host_bytes: 16,
                ..Default::default()
            };
            let projected = [CaptureUsage {
                host_bytes: 24,
                ..Default::default()
            }; 2];
            let dependencies = CaptureUsage {
                retained_bytes: 64,
                host_bytes: 8,
                ..Default::default()
            };
            let member = PartitionInterventionMemberSource {
                rank: 0,
                projection: &projection,
                shape: &shape,
                execution_identity: [7; 32],
                usage,
                projection_usage: projected,
                source_usage: dependencies,
            };
            let members = [member];
            let before = meter.load(Ordering::Relaxed);
            let required = PreparedPartitionInterventionSource::preparation_metadata_bytes(
                2,
                &[PartitionInterventionInvocationSource {
                    window: None,
                    members: &members,
                }],
            )
            .unwrap();
            assert_eq!(
                meter.load(Ordering::Relaxed),
                before,
                "a prospective query allocates and spends nothing"
            );
            let source = PreparedPartitionInterventionSource::new_invocation(
                &original, 0, phase, 1, 2, physical, window, &members, &funding,
            )
            .unwrap();
            assert_eq!(
                meter.load(Ordering::Relaxed) - before,
                required,
                "query and actual descriptor-copy worker agree"
            );
            if let Some(window) = window {
                let shifted = PreparedPartitionInterventionSource::new_invocation(
                    &original,
                    0,
                    phase,
                    1,
                    2,
                    physical,
                    Some(CaptureInvocationWindow { start: 0, ..window }),
                    &members,
                    &funding,
                )
                .unwrap();
                assert_ne!(
                    source.descriptor(),
                    shifted.descriptor(),
                    "equal physical widths do not identify distinct logical windows"
                );
                assert!(
                    PreparedPartitionInterventionSource::new_invocation(
                        &original,
                        0,
                        phase,
                        1,
                        2,
                        physical,
                        Some(CaptureInvocationWindow {
                            logical_sequence: 1,
                            start: 0
                        }),
                        &members,
                        &funding,
                    )
                    .is_err()
                );
            }
            assert_eq!(source.model_invocation(), Some((physical, window)));
            let transport = Transport {
                rank,
                reject_peer: false,
            };
            let mut ledger = CaptureLedger::new(capture.admission());
            let mut work = source
                .prepare(&transport, DistributedCommitEpoch::FIRST, &mut ledger)
                .unwrap();
            let global = work.global_reserved();
            if rank == 0 {
                let mut loan = work.begin(None, true).unwrap().unwrap();
                assert!(
                    loan.charge_execution(usage).is_err(),
                    "an unvalidated claim grants no edit credit"
                );
                drop(loan);
                assert!(
                    work.begin(None, true).is_err(),
                    "the exact failed prefix stays spent"
                );
                assert!(work.deliver(CaptureUsage::default()).is_err());
            } else {
                assert!(work.begin(None, true).unwrap().is_none());
            }
            drop(work);
            assert_eq!(ledger.total(), global);
            drop((projection, original, funding));
        }
    }
}
