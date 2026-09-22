use super::*;

#[test]
fn ordinary_communication_completion_retains_same_account_until_last_owner() {
    if !crate::tests::support::native_process::enter("ordinary-communication-custody") {
        return;
    }
    communication_custody(Scenario::MatchingHeader);
}

#[test]
fn ordinary_communication_header_refusal_retains_same_account() {
    if !crate::tests::support::native_process::enter("ordinary-communication-header-custody") {
        return;
    }
    communication_custody(Scenario::MismatchedHeader);
}

#[test]
fn ordinary_communication_deadline_refusal_retains_same_account() {
    if !crate::tests::support::native_process::enter("ordinary-communication-deadline-custody") {
        return;
    }
    communication_custody(Scenario::Deadline);
}

#[test]
fn ordinary_communication_cancellation_refusal_retains_same_account() {
    if !crate::tests::support::native_process::enter("ordinary-communication-cancellation-custody")
    {
        return;
    }
    communication_custody(Scenario::Cancellation);
}

#[derive(Clone, Copy)]
enum Scenario {
    MatchingHeader,
    MismatchedHeader,
    Deadline,
    Cancellation,
}

fn communication_custody(scenario: Scenario) {
    let header_matches = !matches!(scenario, Scenario::MismatchedHeader);
    let deadline = matches!(scenario, Scenario::Deadline);
    let cancellation = matches!(scenario, Scenario::Cancellation);
    let bounded_refusal = deadline || cancellation;
    use crate::backend::{
        nn::shared::{OrdinaryExecutionOwner, OrdinaryExecutionRegistration},
        runtime::distributed::completion::{
            MlxCommunicationCompletion, prepared::CompletionResourceLayout,
        },
    };
    use eredu_core::Completion;
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let old_cache = safemlx::memory::set_cache_limit(0).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(
        mechanism.allocation(),
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
    )
    .ordinary_storage();
    let context = WorkspaceContext::new(cpu);
    let a = WorkspaceTensor::existing(
        context.layout(&[1], WorkspaceDtype::Int32).unwrap(),
        &context,
    )
    .unwrap();
    let b = WorkspaceTensor::existing(
        context.layout(&[1], WorkspaceDtype::Int32).unwrap(),
        &context,
    )
    .unwrap();
    let header_layout = WorkspaceTensor::existing(
        context.layout(&[3], WorkspaceDtype::Uint8).unwrap(),
        &context,
    )
    .unwrap();
    context.begin_state_span([&a, &b, &header_layout]).unwrap();
    let result = a.add(&b, &context).unwrap();
    let report = context.finish_report(&[result, header_layout]).unwrap();
    let recipe =
        SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 2, mechanism, cpu, &context)
            .unwrap();
    let calls = cpu.ordinary_report_call_controls(&report).unwrap().unwrap();
    let submit = MlxCommunicationCompletion::ordinary_submit_call_controls(
        2,
        CompletionResourceLayout {
            arrays: 2,
            counts: &[],
            groups: 0,
            routes: 0,
            streams: 1,
        },
    )
    .unwrap();
    let controls = recipe
        .ordinary_cpu_controls()
        .unwrap()
        .append(calls.observed)
        .unwrap()
        .append(submit.observed)
        .unwrap();
    let wrappers = OrdinaryExecutionRegistration::control_bytes()
        .unwrap()
        .checked_add(usize::try_from(calls.metadata_bytes).unwrap())
        .unwrap()
        .checked_add(usize::try_from(submit.metadata_bytes).unwrap())
        .unwrap()
        .checked_add(MlxCommunicationCompletion::ordinary_scalar_result_control_bytes().unwrap())
        .unwrap()
        .checked_add(
            MlxCommunicationCompletion::ordinary_boundary_headers_control_bytes(1, 3).unwrap(),
        )
        .unwrap()
        .checked_add(3)
        .unwrap()
        .checked_add(Array::ordinary_clone_control_bytes().unwrap())
        .unwrap();
    let backing = recipe
        .storage
        .mutable_bytes()
        .checked_add(
            u64::try_from(recipe.storage.maximum_births())
                .unwrap()
                .checked_mul(
                    u64::try_from(safemlx::physical_backing_control_bytes())
                        .unwrap()
                        .checked_add(managed_memory::ordinary_root_metadata_bytes().unwrap())
                        .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    let allowance = controls
        .host_allowance_with_ledger_metadata()
        .unwrap()
        .checked_add(backing)
        .unwrap()
        .checked_add(
            u64::try_from(StorageMetadataFunding::host_owner_bytes(wrappers).unwrap()).unwrap(),
        )
        .unwrap();
    let left = Array::try_from_slice(&[1i32], &[1]).unwrap();
    let right = Array::try_from_slice(&[3i32], &[1]).unwrap();
    let header = Array::try_from_slice(&[1u8, 4, 9], &[3]).unwrap();
    header.evaluated().unwrap();
    reclaim(&stream);
    let baseline = current(&pool);
    let accounts = pool.snapshot().unwrap().funding_accounts;
    let (reservation, run) = funding(&pool, allowance);
    let mut scope = run.scope().unwrap();
    let metadata = scope.prepare_storage_metadata().unwrap();
    let host = metadata.prepare_host_owner(wrappers).unwrap();
    let observer_host = metadata
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    let payer = managed_memory::prepare_scoped_observer(&mut scope, observer_host).unwrap();
    let mut native = SubmissionScope::begin().unwrap();
    native.bind_physical_observer(&payer).unwrap();
    let registration = OrdinaryExecutionRegistration::new(OrdinaryExecutionOwner::new(
        host.clone(),
        payer.clone(),
    ))
    .unwrap();
    let output = left.add(&right, &stream).unwrap();
    let pending =
        crate::backend::runtime::distributed::completion::ordinary_completed_i32_scalar(&output);
    assert!(
        pending.is_err(),
        "a pending source is never evaluated by completed readback"
    );
    drop(pending);
    assert!(output.try_completed().is_err());
    if cancellation {
        crate::backend::runtime::distributed::completion::force_next_communication_pending();
    }
    let completion = MlxCommunicationCompletion::submit(
        [&output, &header],
        vec![output.clone(), header.clone()],
        vec![],
        vec![],
        vec![],
        vec![stream.clone()],
    )
    .unwrap();
    let (agreement, completion) = completion.with_failure_agreement(output, 4);
    let expected = if header_matches {
        vec![1u8, 4, 9]
    } else {
        vec![1u8, 4, 8]
    };
    let mut completion = Some(completion.with_boundary_headers([(header.clone(), expected)]));
    let refusal = if bounded_refusal {
        use eredu_core::{BoundedCompletion, BoundedCompletionWait, CompletionCancellationMode};
        let policy = BoundedCompletionWait::new(
            if deadline {
                std::time::Duration::MAX
            } else {
                std::time::Duration::from_millis(1)
            },
            if deadline {
                CompletionCancellationMode::QuarantineUntilComplete
            } else {
                CompletionCancellationMode::NativeCancel
            },
        )
        .unwrap();
        let error = completion.take().unwrap().wait_bounded(policy).unwrap_err();
        assert!(error.to_string().contains(if deadline {
            "deadline exceeds the host monotonic clock range"
        } else {
            "no native cancellation"
        }));
        Some(error)
    } else if header_matches {
        let completion = completion.as_ref().unwrap();
        completion.wait().unwrap();
        let before_poll = current(&pool);
        for _ in 0..5 {
            assert!(completion.is_complete().unwrap());
        }
        assert_eq!(
            current(&pool),
            before_poll,
            "completed polling allocates no new recovery or host owner"
        );
        None
    } else {
        let completion = completion.as_ref().unwrap();
        let error = completion.wait().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("completed readout differs from its source")
        );
        assert!(
            completion.resources_releasable(),
            "header refusal follows exact native completion"
        );
        Some(error)
    };
    native.seal();
    if cancellation {
        crate::backend::runtime::distributed::completion::release_forced_pending_orphans();
    }
    if bounded_refusal {
        reclaim(&stream);
    }
    assert!(native.progress().is_settled());
    drop((registration, native, payer, host, metadata));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert!(
        current(&pool) > baseline,
        "completion retains its genuine ordinary account"
    );
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts + 1);
    drop(completion);
    reclaim(&stream);
    assert!(
        current(&pool) > baseline,
        "escaped scalar result retains its actual payer"
    );
    if bounded_refusal {
        drop(agreement);
    } else {
        assert!(
            agreement.resolve().unwrap(),
            "nonzero native sum is exactly four"
        );
    }
    reclaim(&stream);
    if let Some(error) = refusal {
        assert!(
            current(&pool) > baseline,
            "escaped completed-read or deadline refusal retains its actual payer"
        );
        assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts + 1);
        drop(error);
        reclaim(&stream);
    }
    assert_eq!(current(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
    drop((left, right, header));
    reclaim(&stream);
    safemlx::memory::set_cache_limit(old_cache).unwrap();
}

#[test]
fn ordinary_completed_scalar_observes_signalled_event_and_refuses_pending_source() {
    if !crate::tests::support::native_process::enter("ordinary-scalar-readiness") {
        return;
    }
    let _pool = crate::tests::support::test_utils::initialize_original_sources();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    assert!(
        crate::backend::nn::shared::current_ordinary_execution_owner()
            .unwrap()
            .is_none()
    );
    for (left_value, right_value) in [(1i32, 3i32), (2, 5)] {
        let expected = left_value + right_value;
        let left = Array::try_from_slice(&[left_value], &[1]).unwrap();
        let right = Array::try_from_slice(&[right_value], &[1]).unwrap();
        let output = left.add(&right, &stream).unwrap();
        assert!(
            crate::backend::runtime::distributed::completion::ordinary_completed_i32_scalar(
                &output
            )
            .is_err()
        );
        assert!(
            output.try_completed().is_err(),
            "readback did not evaluate the lazy source"
        );
        let event = safemlx::transforms::async_eval_with_event([&output]).unwrap();
        event.synchronize().unwrap();
        assert!(
            output.try_completed().is_err(),
            "a waited event remains attached until availability is observed"
        );
        assert_eq!(
            crate::backend::runtime::distributed::completion::ordinary_completed_i32_scalar(
                &output
            )
            .unwrap(),
            expected,
        );
        assert!(output.try_completed().is_ok());
    }
}
