use super::*;
use crate::working_memory::MemoryLedger;
use eredu_core::intervention::PreparedInterventionPlanCopy;
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
#[derive(Debug)]
struct Account {
    retired: Arc<AtomicBool>,
    refuse: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, _bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.refuse.load(Ordering::SeqCst) {
            Err(HostMetadataFundingError::Overflow)
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
fn funding() -> (HostMetadataFunding, Arc<AtomicBool>, Arc<AtomicBool>) {
    let retired = Arc::new(AtomicBool::new(false));
    let refuse = Arc::new(AtomicBool::new(false));
    (
        HostMetadataFunding::new(Account {
            retired: retired.clone(),
            refuse: refuse.clone(),
        })
        .unwrap(),
        retired,
        refuse,
    )
}
#[test]
fn original_projection_preserves_ordinary_payloads_coordinates_and_additive_sources() {
    let dtype = InterventionDtype::Float32;
    let tensor = || InterventionTensor {
        shape: vec![2, 4],
        values: InterventionValues::Float32(
            (0..8).map(|index| 17.0 - index as f32 * 1.25).collect(),
        ),
    };
    let actions = [
        (InterventionAction::Zero { dtype }, true),
        (
            InterventionAction::Scale {
                dtype,
                factor: -0.75,
            },
            true,
        ),
        (
            InterventionAction::Mask {
                dtype,
                shape: vec![2, 4],
                keep: vec![true, false, true, false, false, true, false, true],
            },
            true,
        ),
        (InterventionAction::Replace { tensor: tensor() }, true),
        (InterventionAction::Add { tensor: tensor() }, true),
        (
            InterventionAction::MaskComponents {
                dtype,
                indices: vec![6, 1, 4],
                keep_selected: true,
            },
            false,
        ),
        (
            InterventionAction::MaskComponents {
                dtype,
                indices: vec![6, 1, 4],
                keep_selected: false,
            },
            false,
        ),
        (
            InterventionAction::MaskLogits {
                dtype,
                token_ids: vec![4, 5, 6, 7],
            },
            false,
        ),
    ];
    let input = Value {
        shape: vec![3, 8],
        data: (0..24).map(|index| index as f32 * 0.3 - 2.7).collect(),
    };
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    for (action, partial) in actions {
        let plan = admitted(action, partial);
        let source = pool
            .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&plan).unwrap())
            .unwrap();
        for (axis, map, offset) in [
            (1, ComponentCoordinateMap::range(8, 0..4).unwrap(), None),
            (
                1,
                ComponentCoordinateMap::indices(8, vec![6, 1, 4]).unwrap(),
                None,
            ),
            (1, ComponentCoordinateMap::range(8, 0..0).unwrap(), None),
            (
                0,
                ComponentCoordinateMap::indices(3, vec![2, 0]).unwrap(),
                None,
            ),
            (
                1,
                ComponentCoordinateMap::range(8, 0..8).unwrap(),
                Some(false),
            ),
            (
                1,
                ComponentCoordinateMap::range(8, 0..8).unwrap(),
                Some(true),
            ),
        ] {
            let ordinary = PartitionActivationProjection::new(
                source.plan().admission(),
                0,
                CapturePhase::Prefill,
                0,
                &input.shape,
                axis,
                &map,
                16,
            )
            .unwrap();
            let ordinary = match offset {
                Some(offset) => match ordinary.as_sum_term(offset) {
                    Ok(value) => value,
                    Err(_) => {
                        assert!(matches!(
                            plan.plan().operations[0].action,
                            InterventionAction::MaskLogits { .. }
                        ));
                        continue;
                    }
                },
                None => ordinary,
            };
            let (funding, retired, _) = funding();
            let prepared = PreparedPartitionInterventionProjection::prepare(
                &source,
                0,
                CapturePhase::Prefill,
                0,
                None,
                &input.shape,
                axis,
                &map,
                offset,
                16,
                funding.clone(),
            )
            .unwrap();
            assert_eq!(ordinary.geometry_identity(), prepared.geometry_identity());
            assert_eq!(ordinary.local_shape(), prepared.local_shape());
            assert!(prepared.source().same_source(&source));
            assert!(prepared.matches_layout(axis, &map, offset));
            let local_input = local(&input, axis, &map);
            let expected = ordinary
                .reserve(&mut Budget::default(), &Estimates)
                .unwrap()
                .apply(&mut Backend::default(), &local_input)
                .unwrap()
                .unwrap_or_else(|| local_input.clone());
            let mut actual = local_input;
            for index in 0..prepared.update_count() {
                let update = prepared.update(index).unwrap();
                actual = crate::intervention::activation::apply_partition_activation(
                    &mut Backend::default(),
                    &actual,
                    update.action,
                    update.slice,
                )
                .unwrap();
            }
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(actual.data, expected.data);
            drop(funding);
            assert!(!retired.load(Ordering::SeqCst));
            drop(prepared);
            assert!(retired.load(Ordering::SeqCst));
        }
    }
    let plan = admitted(InterventionAction::Scale { dtype, factor: 2.0 }, true);
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&plan).unwrap())
        .unwrap();
    let (funding, retired, refuse) = funding();
    refuse.store(true, Ordering::SeqCst);
    let error = PreparedPartitionInterventionProjection::prepare(
        &source,
        0,
        CapturePhase::Prefill,
        0,
        None,
        &input.shape,
        1,
        &ComponentCoordinateMap::range(8, 0..4).unwrap(),
        None,
        16,
        funding.clone(),
    )
    .unwrap_err();
    drop(funding);
    assert!(!retired.load(Ordering::SeqCst));
    drop(error);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn partition_source_transcript_binds_every_original_window_member_and_update() {
    use crate::capture::partition::{
        PartitionInterventionInvocationSource, PartitionInterventionMemberSource,
        PreparedPartitionInterventionSource,
    };
    use crate::intervention::InterventionPrefillWindow;
    let plan = admitted(
        InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: 1.75,
        },
        false,
    );
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&plan).unwrap())
        .unwrap();
    let (funding, retired, _) = funding();
    let projections = [
        ComponentCoordinateMap::range(8, 0..4).unwrap(),
        ComponentCoordinateMap::indices(8, vec![6, 1, 4]).unwrap(),
        ComponentCoordinateMap::range(8, 8..8).unwrap(),
    ]
    .iter()
    .map(|map| {
        PreparedPartitionInterventionProjection::prepare(
            &source,
            0,
            CapturePhase::Prefill,
            0,
            None,
            &[3, 8],
            1,
            map,
            None,
            16,
            funding.clone(),
        )
        .unwrap()
    })
    .collect::<Vec<_>>();
    let inference = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: plan.request().max_predictions,
        prefill_chunk_positions: 2,
        output: eredu_core::OutputDemand::LastPosition,
    };
    let windows = [(0, 2), (2, 3)].map(|(start, end)| {
        InterventionPrefillWindow::new(
            &plan,
            inference,
            &crate::prefill::PrefillChunk {
                input: start..end,
                position: start,
                output: inference.output.for_chunk(end == 3),
            },
        )
        .unwrap()
    });
    let rows = |count, changed| {
        [0usize, 2, 3]
            .into_iter()
            .enumerate()
            .map(|(index, rank)| PartitionInterventionMemberSource {
                rank,
                projection: &projections[index],
                shape: match (count, index) {
                    (2, 0) => &[2, 4],
                    (2, 1) => &[2, 3],
                    (2, 2) => &[2, 0],
                    (1, 0) => &[1, 4],
                    (1, 1) => &[1, 3],
                    _ => &[1, 0],
                },
                execution_identity: [if index == 1 && changed { 5 } else { 4 }; 32],
                usage: CaptureUsage {
                    retained_bytes: 32,
                    host_bytes: 8,
                    ..Default::default()
                },
                projection_usage: [CaptureUsage {
                    host_bytes: 24,
                    ..Default::default()
                }; 2],
                source_usage: CaptureUsage {
                    retained_bytes: 48,
                    host_bytes: 16,
                    ..Default::default()
                },
            })
            .collect::<Vec<_>>()
    };
    let first = rows(2, false);
    let last = rows(1, false);
    let changed = rows(1, true);
    let invocations = [
        PartitionInterventionInvocationSource {
            window: Some(windows[0]),
            members: &first,
        },
        PartitionInterventionInvocationSource {
            window: Some(windows[1]),
            members: &last,
        },
    ];
    let prepared = PreparedPartitionInterventionSource::new(
        &source,
        0,
        CapturePhase::Prefill,
        0,
        4,
        &invocations,
        &funding,
    )
    .unwrap();
    assert_eq!(prepared.invocation_count(), 2);
    assert!(prepared.original().same_source(&source));
    let different = PreparedPartitionInterventionSource::new(
        &source,
        0,
        CapturePhase::Prefill,
        0,
        4,
        &[
            PartitionInterventionInvocationSource {
                window: Some(windows[0]),
                members: &first,
            },
            PartitionInterventionInvocationSource {
                window: Some(windows[1]),
                members: &changed,
            },
        ],
        &funding,
    )
    .unwrap();
    assert_ne!(
        prepared.descriptor(),
        different.descriptor(),
        "post-window updates belong to agreement"
    );
    assert!(PreparedPartitionInterventionSource::new(
        &source,
        0,
        CapturePhase::Prefill,
        0,
        4,
        &invocations[..1],
        &funding
    )
    .is_err());
    assert!(PreparedPartitionInterventionSource::new(
        &source,
        0,
        CapturePhase::Prefill,
        0,
        4,
        &[
            PartitionInterventionInvocationSource {
                window: Some(windows[1]),
                members: &last
            },
            PartitionInterventionInvocationSource {
                window: Some(windows[0]),
                members: &first
            }
        ],
        &funding
    )
    .is_err());
    let foreign = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&plan).unwrap())
        .unwrap();
    assert!(PreparedPartitionInterventionSource::new(
        &foreign,
        0,
        CapturePhase::Prefill,
        0,
        4,
        &invocations,
        &funding
    )
    .is_err());
    drop(different);
    drop(projections);
    drop(funding);
    assert!(!retired.load(Ordering::SeqCst));
    drop(prepared);
    assert!(retired.load(Ordering::SeqCst));
}
