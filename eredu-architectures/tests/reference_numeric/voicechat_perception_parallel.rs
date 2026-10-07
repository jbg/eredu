//! NVIDIA perception outputs and full-width caches through actual TP collectives.
use super::voicechat_perception_residency::{
    assert_state, assert_tensor, dimensions, parameters, tensor,
};
use super::*;
use eredu_architectures::voicechat::{
    perception::{PerceptionError, PerceptionState},
    perception_layered::{
        perception_parameter_description, LayerwisePerception, PerceptionArchitecture,
    },
};
use eredu_core::checkpoint::TensorDtype;
use serde_json::Value;

#[test]
fn voicechat_perception_parallel_tp2_resident_and_bounded_match_nvidia_caches() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let dimensions = dimensions(&fixture);
    let description = perception_parameter_description(dimensions).unwrap();
    for resident in [true, false] {
        let world = Arc::new(NumericPartitionWorld::default());
        std::thread::scope(|scope| {
            let workers = (0..2)
                .map(|rank| {
                    let world = world.clone();
                    let fixture = &fixture;
                    let description = &description;
                    scope.spawn(move || {
                        let topology = ParallelRankTopology::new(
                            ParallelTopology::new(2, 1, 1, 1).unwrap(),
                            rank,
                        )
                        .unwrap();
                        let layout =
                            eredu_runtime::derive_partitioned_local_layout(description, topology)
                                .unwrap();
                        let context =
                            NumericContext::with_partition(layout.clone(), rank, world.clone());
                        let group = CollectiveGroupId::new(1);
                        let manifest = CommunicationManifest::new(
                            2,
                            rank,
                            vec![CommunicationGroupDescriptor::new(
                                group,
                                0,
                                vec![0, 1],
                                Some(rank),
                                CommunicationGroupRequirements::new([
                                    CommunicationOperationRequirement::tensors(
                                        CommunicationOperation::AllReduceSum,
                                        vec![TensorDtype::F32],
                                        CommunicationTensorLimits::new(1, 3, 512, None).unwrap(),
                                        true,
                                    )
                                    .unwrap(),
                                ])
                                .unwrap(),
                            )
                            .unwrap()],
                            vec![],
                        )
                        .unwrap()
                        .with_completion_policy(
                            CommunicationCompletionPolicy::new(
                                std::time::Duration::from_secs(10),
                                CompletionCancellationMode::QuarantineUntilComplete,
                            )
                            .unwrap(),
                        );
                        world.realize_manifest(&manifest).unwrap();
                        let parallel = numeric_parallel_context(&world, &manifest, group).unwrap();
                        let values: BTreeMap<_, _> = parameters(fixture)
                            .into_iter()
                            .map(|(name, value)| {
                                let selected = select_parameter(
                                    &value,
                                    layout.tensor(&name).unwrap().placement(),
                                )
                                .unwrap();
                                (name, selected)
                            })
                            .collect();
                        let architecture = PerceptionArchitecture::<NumericBackend>::new_parallel(
                            dimensions, &layout,
                        )
                        .unwrap();
                        let policy = if resident {
                            let mut units = Vec::new();
                            for group in 0..3 {
                                for index in 0..architecture.group_unit_count(group).unwrap() {
                                    let mut unit =
                                        architecture.build_unit(group, index, &context).unwrap();
                                    replace_numeric_parameters(&mut unit, &values);
                                    units.push(unit);
                                }
                            }
                            NumericReplicatedPolicy::resident(units)
                        } else {
                            let mut policy = NumericReplicatedPolicy::bounded();
                            policy.parameter_overrides = values;
                            policy
                        };
                        let mut model = LayerwisePerception::new(architecture, policy);
                        let full = model
                            .forward_parallel(
                                &tensor(&fixture["subsampling"]["input"]),
                                0,
                                &PerceptionState::default(),
                                &parallel,
                                &context,
                            )
                            .unwrap();
                        assert_tensor(
                            &full.encoder,
                            &fixture["perception"]["encoder_output"],
                            fixture,
                        );
                        assert_tensor(
                            &full.projected,
                            &fixture["perception"]["projected_output"],
                            fixture,
                        );
                        let windows = fixture["subsampling"]["steps"].as_array().unwrap();
                        let references = fixture["perception"]["steps"].as_array().unwrap();
                        assert_state(
                            &full.state,
                            references.last().unwrap(),
                            6,
                            fixture,
                            &context,
                        );
                        let mut previous = PerceptionState::default();
                        for (index, (window, reference)) in
                            windows.iter().zip(references).enumerate()
                        {
                            let input = tensor(&window["input"]);
                            if index == 2 {
                                let wrong_parallel =
                                    NumericParallelContext::new(0, NumericParallelGroup::new(1));
                                assert!(matches!(
                                    model.forward_parallel(
                                        &input,
                                        window["drop"].as_u64().unwrap() as usize,
                                        &previous,
                                        &wrong_parallel,
                                        &context,
                                    ),
                                    Err(PerceptionError::Configuration(_))
                                ));
                                // Rejection follows subsampling lease acquisition. The committed
                                // full-width caches remain usable by a subsequent collective step.
                                assert!(matches!(
                                    model.forward_parallel(
                                        &input, 100, &previous, &parallel, &context
                                    ),
                                    Err(PerceptionError::State(_))
                                ));
                                assert_state(
                                    &previous,
                                    &references[index - 1],
                                    index as u64,
                                    fixture,
                                    &context,
                                );
                            }
                            let next = model
                                .forward_parallel(
                                    &input,
                                    window["drop"].as_u64().unwrap() as usize,
                                    &previous,
                                    &parallel,
                                    &context,
                                )
                                .unwrap();
                            assert_tensor(&next.encoder, &reference["encoder_output"], fixture);
                            assert_tensor(&next.projected, &reference["projected_output"], fixture);
                            assert_state(
                                &next.state,
                                reference,
                                index as u64 + 1,
                                fixture,
                                &context,
                            );
                            assert_eq!(previous.frames(), index as u64);
                            previous = next.state;
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        assert!(world.tensor_collective_calls() > 0);
    }
}
