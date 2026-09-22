//! Active replacements survive actual bounded unit retirement and reacquisition.
use super::*;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, text_quote::PreparedResidencyFixture,
};
use eredu_core::{
    ControlledTextGeneration, GenerationSequenceRequest, TextGeneration, TextPreparationOptions,
    TokenIdsInputPlan,
};
use eredu_runtime::parameter_operations::PreparedParameterLocation;

type Runtime = ModelRuntime<MlxBackend<'static>>;

fn limits() -> CaptureUsage {
    CaptureUsage {
        captures: 1024,
        retained_bytes: 1 << 30,
        host_bytes: 1 << 30,
        encoded_bytes: 1 << 30,
    }
}

fn predictions(runtime: &mut Runtime, controlled: bool) -> Vec<Vec<f32>> {
    runtime.reset().unwrap();
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let source = SharedCapturePlan::new(
        CapturePlan {
            schema_version: 1,
            selections: vec![CaptureSelection {
                id: "replacement-logits".into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule::default(),
                slices: vec![],
                transform: CaptureTransform::FullTensor,
            }],
            limits: CaptureLimits {
                per_step: limits(),
                cumulative: limits(),
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
        )
        .unwrap(),
    );
    let options = Some(TextPreparationOptions {
        capture: Some(source),
        interventions: None,
    });
    macro_rules! collect {
        ($generation:expr) => {{
            let mut generation = $generation.unwrap();
            let mut logits = Vec::new();
            for _ in 0..4 {
                drop(generation.next().unwrap().unwrap());
                let step = generation.take_captured_delivery().unwrap().unwrap();
                let record = step
                    .records
                    .iter()
                    .find(|record| record.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                    .unwrap();
                assert_eq!(record.outcome, CaptureOutcome::Captured);
                let tensor = record
                    .payload
                    .as_ref()
                    .and_then(CapturePayload::as_tensor)
                    .expect("exact logit tensor");
                let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                    panic!("F32 fixture logits");
                };
                assert!(values.iter().all(|value| value.is_finite()));
                logits.push(values.clone());
            }
            assert!(generation.next().is_none());
            logits
        }};
    }
    if controlled {
        collect!(ControlledTextGeneration::from_token_ids_with_sequence(
            runtime,
            TokenIdsInputPlan::new(&[1, 2, 3]).unwrap(),
            disk::config(0.0, 1, u64::MAX),
            disk::Controller::default(),
            options,
            GenerationSequenceRequest::new(4, &[]),
        ))
    } else {
        collect!(TextGeneration::from_token_ids_with_sequence(
            runtime,
            TokenIdsInputPlan::new(&[1, 2, 3]).unwrap(),
            disk::config(0.0, 1, u64::MAX),
            eredu_core::TokenFilter::All,
            options,
            GenerationSequenceRequest::new(4, &[]),
        ))
    }
}

fn same_predictions(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 3e-5 + 3e-5 * expected.abs(),
                "{actual} != {expected}"
            );
        }
    }
}

fn current_snapshot(
    pool: &eredu_runtime::working_memory::MemoryLedger,
) -> eredu_runtime::working_memory::MemoryLedgerSnapshot {
    let mut snapshot = pool.snapshot().unwrap();
    for domain in &mut snapshot.domains {
        domain.historical_peak_bytes = 0;
    }
    snapshot
}

#[test]
#[ignore = "requires the qualified native factory and bounded parameter sources"]
fn prepared_parameter_overlays_preserve_three_residencies_and_following_predictions() {
    if !crate::tests::support::native_process::enter("prepared parameter replacements") {
        return;
    }
    let fixture = PreparedResidencyFixture::new();
    let mut reference: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = None;
    for residency in 0..3 {
        let (mut runtime, _artifact, _) = fixture.load(residency, Some(128));
        let baseline = predictions(&mut runtime, false);
        runtime.reset().unwrap();
        let discovery = MlxBackend::parameter_discovery(&mut runtime).unwrap();
        let slots = runtime
            .session()
            .original_model_source()
            .unwrap()
            .erased()
            .prepared_parameter_slots();
        let ids = [0, 2].map(|selected| slots.iter().find(|slot| {
            matches!(slot.location, PreparedParameterLocation::Unit { ordinal, .. } if ordinal == selected)
                && slot.materialized.shape.len() == 2
                && slot.materialized.shape.iter().all(|&n| n >= 2)
        }).expect("a matrix in each selected real unit").parameter.id.as_str().to_owned());
        let edits = ids
            .iter()
            .enumerate()
            .map(|(ordinal, id)| {
                let parameter = discovery
                    .parameters
                    .iter()
                    .find(|parameter| &parameter.id == id)
                    .unwrap();
                let count = parameter
                    .shape
                    .iter()
                    .try_fold(1usize, |n, &extent| {
                        n.checked_mul(usize::try_from(extent).unwrap())
                    })
                    .unwrap();
                ParameterEdit {
                    id: format!("unit-matrix-{ordinal}"),
                    parameter: id.clone(),
                    parameter_shape: parameter.shape.clone(),
                    dtype: parameter.dtype.unwrap(),
                    region: ParameterRegion {
                        starts: vec![0, 0],
                        shape: parameter.shape.clone(),
                    },
                    update: ParameterUpdate::Replace {
                        values: (0..count).map(|n| ((n % 13) as f32 - 6.) * 0.25).collect(),
                    },
                }
            })
            .collect::<Vec<_>>();
        let original = MlxBackend::query_parameter(
            &mut runtime,
            &discovery.identity,
            &edits[0].parameter,
            edits[0].region.clone(),
            limits(),
        )
        .unwrap();
        assert!(original.values.iter().any(|value| *value != 0.));
        let overlay = AdmittedParameterOverlay::admit(
            ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: discovery.identity.clone(),
                provenance: "bounded source replacement parity".into(),
                edits: edits.clone(),
            },
            &discovery,
        )
        .unwrap();
        let active =
            MlxBackend::activate_parameter_overlay(&mut runtime, &overlay, limits()).unwrap();
        if residency == 2 {
            let state = &runtime.session().payload.parameter_state;
            let foreign = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let mut retained_host_aliases = 0;
            for value in state.originals.values() {
                let Some(witness) = value.as_array().inspect_host_transfer_alias().unwrap() else {
                    continue;
                };
                let receipt = state
                    .original_sources
                    .host_receipt(witness.allocation())
                    .unwrap()
                    .expect("displaced disk parameter preserves its completed host attachment");
                assert!(receipt.matches(witness.allocation(), &fixture.pool));
                assert!(!receipt.matches(witness.allocation(), &foreign));
                retained_host_aliases += 1;
            }
            assert_eq!(retained_host_aliases, edits.len());
        }
        let edited = MlxBackend::query_parameter(
            &mut runtime,
            &active.identity,
            &edits[0].parameter,
            edits[0].region.clone(),
            limits(),
        )
        .unwrap();
        assert_eq!(edited.values, edits[0].update.values());
        let alias = edited.clone();
        drop(edited);
        let changed = predictions(&mut runtime, false);
        assert!(
            changed
                .iter()
                .flatten()
                .zip(baseline.iter().flatten())
                .any(|(actual, old)| (actual - old).abs() > 1e-6),
            "the selected unit edit changes actual predictions"
        );
        same_predictions(&predictions(&mut runtime, true), &changed);
        runtime.reset().unwrap();
        runtime.synchronize().unwrap();
        disk::reclaim();
        let completed = runtime
            .session()
            .original_model_source()
            .unwrap()
            .parameter_sources()
            .sources()
            .map(|source| source.budget().clone())
            .collect::<Vec<_>>();
        assert!(!completed.is_empty());
        let occupied = completed
            .iter()
            .map(|source| source.occupied_bytes())
            .collect::<Vec<_>>();
        assert!(occupied.iter().all(|&bytes| bytes != 0));
        disk::reclaim();
        let retained = current_snapshot(&fixture.pool);
        assert!(retained.reservations >= completed.len());
        // Reusing these same physical roots must not append publisher sidecars
        // or keep an obsolete request's staging/control allowance alive.
        for controlled in [false, true, false] {
            same_predictions(&predictions(&mut runtime, controlled), &changed);
            runtime.reset().unwrap();
            runtime.synchronize().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(9);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                disk::reclaim();
                let current = current_snapshot(&fixture.pool);
                if std::time::Instant::now() >= deadline {
                    assert_eq!(
                        current, retained,
                        "residency={residency} controlled={controlled}: repeated publication retains the same live owners"
                    );
                }
                current == retained
            });
            assert_eq!(
                completed
                    .iter()
                    .map(|source| source.occupied_bytes())
                    .collect::<Vec<_>>(),
                occupied,
                "repeated publication preserves exactly the live numerical backing"
            );
        }
        if let Some((old, new)) = &reference {
            same_predictions(&baseline, old);
            same_predictions(&changed, new);
        } else {
            reference = Some((baseline.clone(), changed));
        }
        assert_eq!(alias.values, edits[0].update.values());
        assert_eq!(fixture.pool.unquoted_owner_count().unwrap(), 0);
        runtime.reset().unwrap();
        let before_removal = MlxBackend::parameter_discovery(&mut runtime).unwrap();
        let restored =
            MlxBackend::remove_parameter_overlay(&mut runtime, &before_removal.identity).unwrap();
        assert!(restored.usage.retained_bytes >= before_removal.usage.retained_bytes);
        let restored_values = MlxBackend::query_parameter(
            &mut runtime,
            &restored.identity,
            &edits[0].parameter,
            edits[0].region.clone(),
            limits(),
        )
        .unwrap();
        assert_eq!(restored_values.values, original.values);
        same_predictions(&predictions(&mut runtime, false), &baseline);
        assert_eq!(
            alias.values,
            edits[0].update.values(),
            "escaped results remain stable after removal"
        );
        assert_eq!(fixture.pool.unquoted_owner_count().unwrap(), 0);
        drop((runtime, original, alias, restored_values));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            disk::reclaim();
            completed.iter().all(|source| source.occupied_bytes() == 0)
        });
        drop(completed);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            disk::reclaim();
            let live = fixture.pool.snapshot().unwrap();
            live.reservations == 0 && live.unquoted_owners == 0
        });
    }
}
