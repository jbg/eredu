//! Later requests must not acquire custody of surviving disk parameter cells.
use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::residency::MemoryTier;

#[test]
fn persistent_cpu_disk_window_releases_later_request_accounts() {
    const CAPACITY: u64 = 64 << 30;
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let artifact = disk::artifact_with_layers(2, false);
    let inspection =
        eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(
            artifact.path(),
        )
        .unwrap();
    let factory = crate::MlxBackendFactory::default();
    let plan = eredu_core::ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
    )
    .with_residency(eredu_core::ResidencyPlan::DenseDiskStream {
        device_budget_bytes: 1 << 20,
        host_budget_bytes: 1 << 20,
        host_lookahead: 1,
        background_queue: 1,
    });
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
    let target = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
    target.backend().validate_original_stream_owners().unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    // Account for persistent factory stream births before any model/request
    // exists. CPU wrapper controls retire with the loaded backend itself.
    let retiring = target.backend().retiring_stream_wrapper_control_bytes();
    let source_baseline = pool
        .fixture_host_charge()
        .unwrap()
        .checked_sub(retiring)
        .unwrap();
    let mut runtime = target.into_runtime().unwrap();
    let stream = runtime.backend().stream().clone();
    assert!(runtime.backend().memory_ledger().same_ledger(&pool));
    {
        let model = &runtime.session().payload.model;
        let workspace = model.layerwise_workspace().unwrap().unwrap();
        assert_eq!(workspace.layout().len(), 2);
        assert_eq!(
            workspace.destination_device_type(),
            safemlx::DeviceType::Cpu
        );
        assert!(
            workspace.disk_receipt().is_none(),
            "canonical loading retains the prepared foreground payload reader"
        );
        assert_eq!(
            model
                .inference_blueprint()
                .unwrap()
                .selected()
                .text_realization()
                .residency()
                .device_depth(2),
            2
        );
    }
    let mut previous_reads = fixture::PreparedResidencyFixture::actual_disk_read_bytes(&runtime);

    let mut expected = None;
    let mut source_charge = None;
    let mut retained_parameters = None;
    for request in 0..4 {
        let input = [2, 5, 7, 3, 11];
        let mut generation = ControlledTextGeneration::from_token_ids_with_sequence(
            &mut runtime,
            eredu_core::TokenIdsInputPlan::new(&input).unwrap(),
            disk::config(0.0, 2, CAPACITY),
            disk::Controller::default(),
            None,
            GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap();
        let mut sequence = generation
            .take_prepared_sequence()
            .unwrap()
            .prepare_storage()
            .unwrap();
        let mut tokens = Vec::new();
        for token in &mut generation {
            let token = token.unwrap_or_else(|error| panic!("request {request}: {error:?}"));
            sequence
                .commit(
                    token.token_id(),
                    eredu_core::TokenTerminalSignals::default(),
                )
                .unwrap();
            tokens.push(token.token_id());
        }
        assert_eq!(tokens.len(), 4);
        assert_eq!(sequence.tokens(), tokens.as_slice());
        drop((sequence, generation));
        runtime.synchronize().unwrap();

        let state = runtime
            .session()
            .payload
            .model
            .erased()
            .resident_reset_source()
            .unwrap();
        let numeric = state
            .state()
            .retained_arrays()
            .into_iter()
            .map(|value| {
                (
                    value.shape().to_vec(),
                    value.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert!(!numeric.is_empty());
        assert!(
            numeric
                .iter()
                .flat_map(|(_, values)| values)
                .all(|value| value.is_finite())
        );
        assert!(
            numeric
                .iter()
                .flat_map(|(_, values)| values)
                .any(|value| *value != 0.0)
        );
        assert!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .state_snapshot()
                .iter()
                .all(|(position, _)| *position == 8)
        );
        if let Some(expected) = &expected {
            assert_eq!(&(tokens, numeric), expected);
        } else {
            expected = Some((tokens, numeric));
        }
        drop(state);
        runtime
            .prepare_reset_ordinary()
            .unwrap()
            .reset_admitted(eredu_core::SessionResetLimits::new(
                crate::memory_fixture::limits(CAPACITY),
            ))
            .unwrap();
        runtime.synchronize().unwrap();
        disk::reclaim();

        let report = runtime.session().residency_report().unwrap().unwrap();
        let disk_units = report
            .units()
            .iter()
            .filter(|unit| unit.planned_tier() == MemoryTier::Disk)
            .collect::<Vec<_>>();
        assert_eq!(disk_units.len(), 2);
        assert_eq!(
            disk_units
                .iter()
                .filter(|unit| unit.device_resident())
                .count(),
            1,
            "the final unit survives both the final and next initial window"
        );
        // The workspace retains the actual prepared foreground payload reader.
        // Manager catalog diagnostics and direct-route transfer counters belong
        // to different sources; neither witnesses this reader's physical I/O.
        let reads = fixture::PreparedResidencyFixture::actual_disk_read_bytes(&runtime);
        if request == 0 {
            assert!(
                reads > previous_reads,
                "the cold request must read payload bytes: before={previous_reads}, after={reads}"
            );
        } else {
            // The final parameter unit survives in this two-unit fixture;
            // subsequent requests can also reuse its warm payload cache.
            // The parameter identities and payer retirement below are the
            // invariant under test; no cache eviction is requested.
            assert!(
                reads >= previous_reads,
                "request {request} must preserve cumulative physical-read telemetry"
            );
        }
        previous_reads = reads;

        // The scalar inventory does not retain any Array/cell owner across a
        // request. It proves the surviving physical parameter allocations are
        // unchanged, so retirement cannot be achieved by flushing that cache.
        let mut nonstate = RetainedStorage::default();
        let mut decoder = RetainedStorage::default();
        runtime
            .session()
            .payload
            .model
            .collect_retained_idle_storage(&mut nonstate, &mut decoder)
            .unwrap();
        let parameters = nonstate.array_allocation_facts();
        assert!(!parameters.is_empty());
        drop((nonstate, decoder));
        disk::reclaim();
        if let Some(original) = &retained_parameters {
            assert_eq!(&parameters, original);
            fixture::settle(&pool, source_charge.unwrap());
        } else {
            retained_parameters = Some(parameters);
            // The first request owns the real cached source. Later requests
            // must retire exactly while that source and its charge survive.
            source_charge = Some(pool.fixture_host_charge().unwrap());
            assert!(source_charge.unwrap() > source_baseline);
        }
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, source_baseline);
}
