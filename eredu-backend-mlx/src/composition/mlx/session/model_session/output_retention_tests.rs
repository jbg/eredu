use super::*;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, InferenceRetention, MemoryLedger,
    WorkingMemoryError,
};

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

// Synthetic admission isolates charge ownership; it makes no native peak claim.
fn admission(bytes: u64) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "completed output ownership fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        1,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        },
    ))
    .unwrap();
    crate::memory_fixture::admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 2,
        state,
        incremental_required_bytes: Some(bytes),
    })
}

fn resources() -> (SessionAuthority, SubmissionResourcesOwner) {
    let mut session = SessionAuthority::new();
    let resources = SubmissionResources::new(
        session.begin_submission().unwrap(),
        Rc::new(Cell::new(false)),
    );
    (session, resources)
}

fn retain_request(
    resources: &SubmissionResources,
    pool: &MemoryLedger,
    bytes: u64,
) -> InferenceRequest {
    let request: InferenceRequest = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(bytes))
        .unwrap()
        .into();
    let mut retained = InferenceRetention::new();
    retained.retain(&request);
    resources.retain_inference(&retained);
    request
}

fn settle(stream: &Stream, pool: &MemoryLedger, bytes: u64, owners: usize) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.fixture_host_charge().unwrap() == bytes
            && pool.unquoted_owner_count().unwrap() == owners
    });
}

#[test]
fn completed_output_aliases_keep_all_request_charges_until_last_backing_retires() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(256, 0).unwrap();
    let (session, resources) = resources();
    let first = retain_request(&resources, &pool, 96);
    let second = retain_request(&resources, &pool, 32);
    resources.retain_inference(&resources.inference_retention());
    let output = Array::from_slice(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[8])
        .square(&stream)
        .unwrap();
    output.evaluated().unwrap();
    let identity = output.allocation_info().unwrap().unwrap();
    let alias = output.clone();
    let view = output.as_strided(&[4][..], &[2][..], 1, &stream).unwrap();
    view.evaluated().unwrap();
    assert_eq!(view.allocation_info().unwrap(), Some(identity));
    resources.retain_output_allocation(&output).unwrap();
    assert_eq!(output.allocation_info().unwrap(), Some(identity));

    resources.request_release();
    drop((resources, session, first, second, output));
    settle(&stream, &pool, 128, 0);
    assert_eq!(alias.evaluated().unwrap().as_slice::<i32>()[3], 16);
    drop(alias);
    settle(&stream, &pool, 128, 0);
    assert_eq!(
        view.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
        vec![4, 16, 36, 64]
    );
    assert_eq!(view.allocation_info().unwrap(), Some(identity));
    drop(view);
    settle(&stream, &pool, 0, 0);
    assert_eq!(pool.fixture_host_peak().unwrap(), 128);
    assert!(pool.acquire_unquoted().is_ok());
}

#[test]
fn zero_byte_request_still_excludes_unquoted_work_while_output_backing_survives() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let (session, resources) = resources();
    let request = retain_request(&resources, &pool, 0);
    let output = Array::from_slice(&[3_i32, 7], &[2]);
    resources.retain_output_allocation(&output).unwrap();
    let alias = output.clone();
    drop((request, resources, session, output));
    settle(&stream, &pool, 0, 0);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(alias);
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.acquire_unquoted().is_ok()
    });
}

#[test]
fn allocation_free_completed_output_adds_no_charge_owner() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(64, 0).unwrap();
    let (session, resources) = resources();
    let request = retain_request(&resources, &pool, 64);
    let output = Array::from_slice::<i32>(&[], &[0]);
    output.evaluated().unwrap();
    assert_eq!(output.allocation_info().unwrap().unwrap().bytes(), 0);
    resources.retain_output_allocation(&output).unwrap();
    drop((request, resources, session));
    settle(&stream, &pool, 0, 0);
    assert_eq!(output.size(), 0);
}

#[test]
fn unlimited_output_attachment_retains_ordinary_reservation_until_last_alias() {
    let stream = stream();
    let topology = crate::memory_fixture::topology();
    let pool = MemoryLedger::new(
        topology.clone(),
        eredu_core::MemoryLimits::unlimited(&topology),
        eredu_core::DomainMemoryRequirements::zero(&topology),
    )
    .unwrap();
    let (session, resources) = resources();
    let request = retain_request(&resources, &pool, 64);
    let source = Array::from_slice(&[2_i32, 5], &[2]);
    let output = source.square(&stream).unwrap();
    output.evaluated().unwrap();
    resources.retain_output_allocation(&output).unwrap();
    let alias = output.clone();
    drop((request, resources, session, output));
    settle(&stream, &pool, 64, 0);
    assert_eq!(alias.evaluated().unwrap().as_slice::<i32>(), &[4, 25]);
    drop(alias);
    settle(&stream, &pool, 0, 0);
}

#[test]
fn existing_unquoted_output_owners_still_follow_the_native_backing() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let (session, resources) = resources();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    resources.retain_memory_owner(Some(&owner));
    resources.retain_memory(&NativeMemoryRetention::from_owner(&owner));
    let output = Array::from_slice(&[11_i32, 13], &[2]);
    resources.retain_output_allocation(&output).unwrap();
    let alias = output.clone();
    drop((owner, resources, session, output));
    settle(&stream, &pool, 0, 1);
    drop(alias);
    settle(&stream, &pool, 0, 0);
}

#[test]
fn unfinished_backing_failure_preserves_submission_charge_for_retry() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(64, 0).unwrap();
    let (session, resources) = resources();
    let request = retain_request(&resources, &pool, 64);
    let source = Array::from_slice(&[2_i32, 5], &[2]);
    let lazy = source.square(&stream).unwrap();
    let error = resources.retain_output_allocation(&lazy).unwrap_err();
    assert!(matches!(error, Error::Exception(_)));
    assert_eq!(lazy.allocation_info().unwrap(), None);
    drop(request);
    settle(&stream, &pool, 64, 0);
    assert_eq!(resources.inference_retention().requests().len(), 1);

    lazy.evaluated().unwrap();
    resources.retain_output_allocation(&lazy).unwrap();
    drop((resources, session, source));
    settle(&stream, &pool, 64, 0);
    assert_eq!(lazy.evaluated().unwrap().as_slice::<i32>(), &[4, 25]);
    drop(lazy);
    settle(&stream, &pool, 0, 0);
}

#[test]
fn direct_session_outputs_inherit_state_reservations_without_a_sampler() {
    struct Retired(Rc<Cell<bool>>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    let stream = stream();
    for decode_output in [false, true] {
        // This isolated reservation is synthetic ownership evidence, not a
        // complete native bound or successful public managed-budget admission.
        // Model, prompt and operation allocations use another unquoted domain.
        let native_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(native_pool);
        let artifact =
            crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
        let model =
            eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let retired = Rc::new(Cell::new(false));
        runtime
            .session_mut()
            .set_retirement_probe(Box::new(Retired(retired.clone())));

        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 5,
            max_output_tokens: 2,
            prefill_chunk_positions: 2,
            output: OutputDemand::LastPosition,
        };
        let capability = runtime
            .session()
            .payload
            .model
            .erased()
            .capability_estimate();
        let bound = |bytes| {
            WorkspaceBound::bounded(
                bytes,
                "synthetic direct-session ownership fixture, not native peak certification",
            )
        };
        let state = eredu_core::estimate_runtime_state(
            capability.state_layout(),
            InputTokenCount::text(5),
            2,
            1,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap()
        .with_execution_workspace(crate::memory_fixture::workspace(
            ExecutionWorkspaceEstimate {
                physical_domains: None,
                geometry,
                activations: bound(1 << 24),
                attention: bound(0),
                vocabulary: bound(0),
                state_update: bound(0),
                materialization: bound(0),
                retained: bound(4096),
            },
        ))
        .unwrap();
        let eredu_core::AdmissionResult::Admitted(admitted) = eredu_core::apply_admission_policy(
            capability.capabilities(),
            eredu_core::AdmissionRequest {
                input: InputTokenCount::text(5),
                max_output_tokens: 2,
                batch_size: 1,
                additional_headroom: crate::memory_fixture::headroom(0),
                memory_limits: Default::default(),
            },
            state,
        )
        .unwrap() else {
            panic!("selected fixture geometry must admit");
        };
        let pool = crate::memory_fixture::ledger(1 << 26, 0).unwrap();
        let request: InferenceRequest = pool
            .reserve(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                &admitted,
            )
            .unwrap()
            .into();
        let charged = request
            .memory_reservation()
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3, 4, 5])
            .unwrap()
            .with_inference_request(request.clone());
        let prefill = runtime.prefill(prompt).unwrap().wait().unwrap();
        drop(request);
        let output = if decode_output {
            drop(prefill);
            crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
            safemlx::reclaim_allocation_owners();
            // Only retained model state supplies request authority to this
            // direct decode; there is no sampler or input request to copy it.
            runtime
                .decode(Array::from_slice(&[2_u32], &[1, 1]))
                .unwrap()
                .wait()
                .unwrap()
        } else {
            prefill
        };
        let escaped = output.into_logits().unwrap().into_array();
        let allocation = escaped.allocation_info().unwrap().unwrap();
        let values = escaped.evaluated().unwrap().as_slice::<f32>().to_vec();
        assert!(values.iter().any(|value| value.abs() > 1e-6));
        let view = escaped
            .as_strided(&[(values.len() / 2) as i32][..], &[2][..], 1, &stream)
            .unwrap();
        view.evaluated().unwrap();
        assert_eq!(view.allocation_info().unwrap(), Some(allocation));
        drop((escaped, runtime));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
            safemlx::reclaim_allocation_owners();
            retired.get()
        });
        settle(&stream, &pool, charged, 0);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
        assert_eq!(
            view.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            values.into_iter().skip(1).step_by(2).collect::<Vec<_>>()
        );
        drop(view);
        settle(&stream, &pool, 0, 0);
        assert!(pool.acquire_unquoted().is_ok());
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
