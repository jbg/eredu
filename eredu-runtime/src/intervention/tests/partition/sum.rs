use super::*;

fn terms() -> [Value; 3] {
    std::array::from_fn(|rank| Value {
        shape: vec![3, 8],
        data: (0..24)
            .map(|i| match rank {
                0 => i as f32 * 0.5 - 3.0,
                1 => 5.0 - i as f32 * 0.25,
                _ => -2.0 + i as f32 * 0.125,
            })
            .collect(),
    })
}

fn sum(terms: &[Value; 3]) -> Value {
    Value {
        shape: terms[0].shape.clone(),
        data: (0..24)
            .map(|index| terms.iter().map(|term| term.data[index]).sum())
            .collect(),
    }
}

#[test]
fn summed_activation_actions_match_complete_edits_at_strided_positions_in_order() {
    let dtype = InterventionDtype::Float32;
    let tensor = || InterventionTensor {
        shape: vec![2, 4],
        values: InterventionValues::Float32((0..8).map(|i| 3.0 - i as f32 * 0.25).collect()),
    };
    let actions = [
        (InterventionAction::Add { tensor: tensor() }, true),
        (
            InterventionAction::Scale {
                dtype,
                factor: -0.5,
            },
            true,
        ),
        (
            InterventionAction::Mask {
                dtype,
                shape: vec![2, 4],
                keep: vec![true, false, true, true, false, true, false, true],
            },
            true,
        ),
        (InterventionAction::Replace { tensor: tensor() }, true),
        (
            InterventionAction::MaskComponents {
                dtype,
                indices: vec![6, 1, 4],
                keep_selected: true,
            },
            false,
        ),
        (InterventionAction::Add { tensor: tensor() }, true),
        (
            InterventionAction::MaskComponents {
                dtype,
                indices: vec![6, 1, 4],
                keep_selected: false,
            },
            false,
        ),
        (InterventionAction::Zero { dtype }, true),
    ];
    let map = ComponentCoordinateMap::range(8, 0..8).unwrap();
    // Each action is checked on nonzero original terms and in an overlapping
    // ordered chain. The designated offset owner changes between operations.
    for chained in [false, true] {
        for (sequence_start, sequence_stride) in [(1, 1), (0, 2)] {
            let mut sources = terms();
            let mut expected = sum(&sources);
            for (index, (action, partial)) in actions.iter().enumerate() {
                if !chained {
                    sources = terms();
                    expected = sum(&sources);
                }
                let plan = admitted_at(action.clone(), *partial, sequence_start, sequence_stride);
                let slice = plan
                    .validate_actual(0, CapturePhase::Prefill, 0, &expected.shape, Some(dtype))
                    .unwrap();
                expected =
                    apply_activation(&mut Backend::default(), &expected, action, &slice).unwrap();
                for (rank, source) in sources.iter_mut().enumerate() {
                    let projection = PartitionActivationProjection::new(
                        &plan,
                        0,
                        CapturePhase::Prefill,
                        0,
                        &source.shape,
                        1,
                        &map,
                        8,
                    )
                    .unwrap()
                    .as_sum_term(rank == index % 3)
                    .unwrap();
                    let mut budget = Budget::default();
                    let work = projection.reserve(&mut budget, &Estimates).unwrap();
                    let mut backend = Backend::default();
                    let changed = work.apply(&mut backend, source).unwrap();
                    if matches!(action, InterventionAction::Add { .. }) && rank != index % 3 {
                        assert!(changed.is_none());
                        assert_eq!(backend.applications, 0);
                        assert_eq!(budget.used.retained_bytes, 0);
                    } else {
                        assert!(backend.applications > 0);
                    }
                    if let Some(value) = changed {
                        *source = value;
                    }
                }
                assert_eq!(
                    sum(&sources).data,
                    expected.data,
                    "{action:?}, chained={chained}, positions={sequence_start}/{sequence_stride}"
                );
            }
        }
    }
}

#[test]
fn sum_projection_binds_offset_ownership_and_charges_only_projected_payloads() {
    let map = ComponentCoordinateMap::range(8, 0..8).unwrap();
    for replace in [false, true] {
        let tensor = InterventionTensor {
            shape: vec![2, 4],
            values: InterventionValues::Float32(vec![1.5; 8]),
        };
        let plan = admitted(
            if replace {
                InterventionAction::Replace { tensor }
            } else {
                InterventionAction::Add { tensor }
            },
            true,
        );
        let project = || {
            PartitionActivationProjection::new(
                &plan,
                0,
                CapturePhase::Prefill,
                0,
                &[3, 8],
                1,
                &map,
                8,
            )
            .unwrap()
        };
        let normal = project().geometry_identity();
        let owner = project().as_sum_term(true).unwrap().geometry_identity();
        let peer = project().as_sum_term(false).unwrap().geometry_identity();
        assert_ne!(normal, owner);
        assert_ne!(normal, peer);
        assert_ne!(owner, peer);
        assert!(project()
            .as_sum_term(false)
            .unwrap()
            .as_sum_term(true)
            .is_err());

        let mut owner_budget = Budget::default();
        let owner_work = project()
            .as_sum_term(true)
            .unwrap()
            .reserve(&mut owner_budget, &Estimates)
            .unwrap();
        let mut peer_budget = Budget::default();
        let peer_work = project()
            .as_sum_term(false)
            .unwrap()
            .reserve(&mut peer_budget, &Estimates)
            .unwrap();
        assert!(owner_budget.used.host_bytes > peer_budget.used.host_bytes);
        if replace {
            assert_eq!(
                owner_budget.used.host_bytes - peer_budget.used.host_bytes,
                32
            );
            assert_eq!(
                owner_budget.used.retained_bytes,
                peer_budget.used.retained_bytes
            );
        } else {
            assert!(peer_work.is_empty());
            assert_eq!(peer_budget.used.retained_bytes, 0);
        }
        let spent = owner_budget.used;
        drop(owner_work);
        assert_eq!(owner_budget.used, spent);
        let mut backend = Backend {
            dtype: Some(InterventionDtype::Bfloat16),
            ..Default::default()
        };
        assert!(peer_work.apply(&mut backend, &terms()[0]).is_err());
        assert_eq!(backend.applications, 0);

        let mut budget = Budget::default();
        budget.limit.retained_bytes = 0;
        assert!(project()
            .as_sum_term(true)
            .unwrap()
            .reserve(&mut budget, &Estimates)
            .is_err());
        assert!(budget.used.host_bytes > 0);
        assert_eq!(budget.used.retained_bytes, 0);
    }
}

#[test]
fn sum_projection_rejects_shards_permutations_and_vocabulary_sentinels() {
    let plan = admitted(
        InterventionAction::Zero {
            dtype: InterventionDtype::Float32,
        },
        true,
    );
    for map in [
        ComponentCoordinateMap::range(8, 0..4).unwrap(),
        ComponentCoordinateMap::range(8, 0..0).unwrap(),
        ComponentCoordinateMap::indices(8, vec![7, 1, 6, 2, 5, 3, 4, 0]).unwrap(),
    ] {
        assert!(PartitionActivationProjection::new(
            &plan,
            0,
            CapturePhase::Prefill,
            0,
            &[3, 8],
            1,
            &map,
            8,
        )
        .unwrap()
        .as_sum_term(true)
        .is_err());
    }
    let logits = admitted(
        InterventionAction::MaskLogits {
            dtype: InterventionDtype::Float32,
            token_ids: vec![1, 4],
        },
        false,
    );
    let map = ComponentCoordinateMap::range(8, 0..8).unwrap();
    assert!(matches!(
        PartitionActivationProjection::new(
            &logits,
            0,
            CapturePhase::Prefill,
            0,
            &[3, 8],
            1,
            &map,
            8,
        )
        .unwrap()
        .as_sum_term(true),
        Err(CaptureError::Unsupported(_))
    ));
}
