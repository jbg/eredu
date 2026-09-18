use super::*;
use crate::{establish_communication_session, CommunicationManifest, ExpertPass};
use eredu_core::consensus::ConsensusTransport;
use eredu_core::{DescriptionCompleteness, ObservationCatalog, ObservationSupportReport};

struct Singleton;
impl ConsensusTransport for Singleton {
    type Error = std::convert::Infallible;
    fn participant_count(&self) -> usize {
        1
    }
    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        Ok(local.to_vec())
    }
}

#[test]
fn first_attempt_identity_survives_failure_restore_and_later_steps() {
    let setup = establish_communication_session(
        &Singleton,
        &CommunicationManifest::new(1, 0, vec![], vec![]).unwrap(),
        Some([1; 32]),
    )
    .unwrap()
    .identity();
    let discovery = CaptureDiscovery {
        artifact_identity: "artifact".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: vec![],
        },
        support: ObservationSupportReport {
            schema_version: 1,
            capture: Default::default(),
            points: vec![],
        },
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![],
        limits: CaptureLimits {
            per_step: crate::capture::policy::frame_usage().unwrap(),
            cumulative: crate::capture::policy::frame_usage().unwrap().checked_mul(2).unwrap(),
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 2,
            max_predictions: 2,
        },
    )
    .unwrap();
    let new_run = || {
        let mut run = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan.clone()));
        run.configure_partition_capture(
            PartitionCaptureIdentity::for_session(
                discovery.artifact_identity.clone(),
                "execution".into(),
                setup,
                None,
            )
            .unwrap(),
        )
        .unwrap();
        run
    };
    let mut run = new_run();
    let initial = run.checkpoint(&discovery).unwrap();
    let mut skipped_limits = plan.plan().limits.clone();
    skipped_limits.on_limit = CaptureLimitPolicy::Skip;
    assert!(initial
        .fork(
            crate::capture::CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 2,
                limits: skipped_limits,
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default())
        )
        .is_ok());
    let first = DistributedCommitEpoch::FIRST;
    assert!(run
        .prepare_step_transaction(first, ExpertPass::Prefill, 5)
        .is_err());
    assert!(
        run.restore(&initial).is_err(),
        "pending attempts cannot restore"
    );
    run.finish_transaction(first, false);
    assert!(
        run.take_step().is_none(),
        "range rejection allocated no records"
    );
    run.restore(&initial).unwrap();
    assert!(run
        .prepare_step_transaction(first, ExpertPass::Prefill, 0)
        .is_err());
    let second = first.next().unwrap();
    run.prepare_step_transaction(second, ExpertPass::Prefill, 0)
        .unwrap();
    let context = run.partition.as_ref().unwrap().identity.context(
        &plan,
        0,
        CapturePhase::Prefill,
        0,
        second,
        None,
    );
    assert_eq!(
        context.run_identity,
        format!("capture:{setup}:{}", first.value())
    );
    assert_eq!(context.forward_epoch, second.value());
    run.finish_transaction(second, false);
    assert!(run.take_step().unwrap().partitions.is_empty());
    run.restore(&initial).unwrap();

    let third = second.next().unwrap();
    run.prepare_step_transaction(third, ExpertPass::Prefill, 0)
        .unwrap();
    let mut later_run = initial
        .fork(
            crate::capture::CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 2,
                limits: plan.plan().limits.clone(),
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default()),
        )
        .unwrap();
    let identity =
        PartitionCaptureIdentity::for_session("artifact".into(), "execution".into(), setup, None)
            .unwrap();
    later_run
        .ensure_partition_capture(identity.clone())
        .unwrap();
    run.ensure_partition_capture(identity.clone()).unwrap();
    let mut foreign = identity;
    foreign.execution = "different loaded selection".into();
    assert!(later_run.ensure_partition_capture(foreign).is_err());
    assert!(run
        .validate_restore(&later_run.checkpoint(&discovery).unwrap())
        .is_err());
    later_run
        .prepare_step_transaction(third, ExpertPass::Prefill, 0)
        .unwrap();
    assert_eq!(
        run.partition.as_ref().unwrap().identity.run,
        context.run_identity
    );
    assert_eq!(
        later_run.partition.as_ref().unwrap().identity.run,
        format!("capture:{setup}:{}", third.value()),
    );
    assert_ne!(
        run.partition.as_ref().unwrap().identity.run,
        later_run.partition.as_ref().unwrap().identity.run,
    );
}
