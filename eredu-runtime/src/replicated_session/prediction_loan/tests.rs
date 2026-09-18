use super::*;
use crate::{
    DeviceState,
    working_memory::{InferenceStateRetention, WorkingMemoryPool, WorkingMemoryUnquotedLease},
};
use eredu_nn::workspace::WorkspaceBackend;

type State = DeviceState<WorkspaceBackend, ()>;
struct Owner {
    state: State,
}
fn slot(owner: &mut Owner) -> &mut State {
    &mut owner.state
}
fn valid(_: &mut Owner, _: &State) -> Result<(), &'static str> {
    Ok(())
}
fn owner(pool: &WorkingMemoryPool) -> State {
    let mut state = State::stateless();
    state
        .inference_retention_mut()
        .retain_unquoted(&pool.acquire_unquoted().unwrap());
    state
}

#[test]
fn placement_preserves_actual_revisions_and_backing_owners() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let mut installed = Owner {
        state: owner(&pool),
    };
    let mut lane = owner(&pool);
    let target_revision = installed.state.inference_retention().revision().clone();
    let lane_revision = lane.inference_retention().revision().clone();
    let (source, returned) = with_state(&mut installed, &mut lane, slot, valid, |session| {
        session
            .state
            .inference_retention()
            .validate_revision(&lane_revision)
            .unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
        Ok::<_, ()>(session.state.inference_retention().revision().clone())
    })
    .unwrap();
    returned.unwrap();
    lane.inference_retention()
        .validate_revision(&source.unwrap())
        .unwrap();
    installed
        .state
        .inference_retention()
        .validate_revision(&target_revision)
        .unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(lane);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(installed);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn publication_advances_actual_lane_revision_even_when_callback_fails() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let mut installed = Owner {
        state: owner(&pool),
    };
    let mut lane = owner(&pool);
    let original = installed.state.inference_retention().revision().clone();
    let before = lane.inference_retention().revision().clone();
    let (output, returned) = with_state(&mut installed, &mut lane, slot, valid, |session| {
        session.state.inference_retention_mut().commit_span(1);
        Err::<(), _>(session.state.inference_retention().revision().clone())
    })
    .unwrap();
    returned.unwrap();
    let published = output.unwrap_err();
    lane.inference_retention()
        .validate_revision(&published)
        .unwrap();
    assert!(
        lane.inference_retention()
            .validate_revision(&before)
            .is_err()
    );
    installed
        .state
        .inference_retention()
        .validate_revision(&original)
        .unwrap();
    let (output, returned) = with_state(&mut installed, &mut lane, slot, valid, |session| {
        session
            .state
            .inference_retention()
            .validate_revision(&published)
            .unwrap();
        assert!(
            session
                .state
                .inference_retention()
                .validate_revision(&before)
                .is_err()
        );
        Ok::<_, ()>(())
    })
    .unwrap();
    output.unwrap();
    returned.unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
}

#[test]
fn rejected_entry_and_failed_return_preserve_exact_error_custody() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let mut installed = Owner {
        state: owner(&pool),
    };
    let mut lane = owner(&pool);
    let original = installed.state.inference_retention().revision().clone();
    let before = lane.inference_retention().revision().clone();
    let rejected = with_state(
        &mut installed,
        &mut lane,
        slot,
        |_, _| Err("source mismatch"),
        |_| {
            panic!("invalid source must not enter callback");
            #[allow(unreachable_code)]
            Ok::<(), ()>(())
        },
    );
    assert_eq!(rejected.unwrap_err(), "source mismatch");
    installed
        .state
        .inference_retention()
        .validate_revision(&original)
        .unwrap();
    lane.inference_retention()
        .validate_revision(&before)
        .unwrap();
    let mut entered = false;
    let failure: WorkingMemoryUnquotedLease = pool.acquire_unquoted().unwrap();
    let recovery: WorkingMemoryUnquotedLease = pool.acquire_unquoted().unwrap();
    let mut recovery = Some(recovery);
    let (output, returned) = with_state(
        &mut installed,
        &mut lane,
        slot,
        |_, _| {
            if entered {
                Err(recovery.take().unwrap())
            } else {
                entered = true;
                Ok(())
            }
        },
        |_| Err::<(), _>(failure),
    )
    .unwrap();
    assert!(output.is_err());
    assert!(returned.is_err());
    installed
        .state
        .inference_retention()
        .validate_revision(&original)
        .unwrap();
    lane.inference_retention()
        .validate_revision(&before)
        .unwrap();
    drop((installed, lane));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(output);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(returned);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn unwind_restores_owners_without_restoring_a_stale_revision() {
    for return_panics in [false, true] {
        let pool = WorkingMemoryPool::new(100, 0).unwrap();
        let mut installed = Owner {
            state: owner(&pool),
        };
        let mut lane = owner(&pool);
        let original = installed.state.inference_retention().revision().clone();
        let before = lane.inference_retention().revision().clone();
        let mut entered = false;
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_state(
                &mut installed,
                &mut lane,
                slot,
                |_, _| {
                    assert!(!entered || !return_panics, "return boundary unwind");
                    entered = true;
                    Ok::<_, ()>(())
                },
                |session| {
                    session.state.inference_retention_mut().commit_span(1);
                    assert!(return_panics, "operation unwind");
                    Ok::<_, ()>(())
                },
            )
        }));
        assert!(unwind.is_err());
        installed
            .state
            .inference_retention()
            .validate_revision(&original)
            .unwrap();
        assert!(
            lane.inference_retention()
                .validate_revision(&before)
                .is_err()
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
        drop((installed, lane));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn boundary_error_preserves_the_original_fixed_metadata_source() {
    use std::error::Error as _;
    let error =
        PredictionStateLoanError::<std::io::Error>::Metadata(HostMetadataFundingError::Unavailable);
    assert!(matches!(
        error
            .source()
            .unwrap()
            .downcast_ref::<HostMetadataFundingError>(),
        Some(HostMetadataFundingError::Unavailable)
    ));
}
