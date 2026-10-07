//! Pinned transformer equations across block and final-normalization pipeline cuts.
use super::voicechat_transformer::{geometry, CausalCache};
use super::voicechat_tts_generator::{compare, compare_cache, tensor};
use super::*;
use eredu_architectures::voicechat::{
    tts_transformer::TransformerConfig,
    tts_transformer_layered::{
        tts_transformer_parameter_description, OwnedTransformerRuntimeState,
        TransformerArchitecture, TransformerBoundary, TransformerBoundarySchema,
        TransformerStateMarker, TransformerUnit,
    },
};
use eredu_runtime::{
    ArchitectureBoundary, LayerwisePartitionRuntime, LayerwisePartitionRuntimeError,
};
use serde_json::Value;

type Unit = TransformerUnit<NumericBackend>;
type State = OwnedTransformerRuntimeState<NumericBackend, CausalCache>;

type Runtime = LayerwisePartitionRuntime<
    TransformerArchitecture<NumericBackend>,
    NumericBackend,
    TransformerStateMarker<NumericBackend>,
    StagePolicy,
>;

struct Stage {
    runtime: Runtime,
    state: State,
    blocks: std::ops::Range<usize>,
}

// The shared provider supplies actual resident/bounded units. This wrapper
// injects one late acquisition failure to exercise stage transaction rollback.
struct StagePolicy {
    inner: NumericReplicatedPolicy<Unit>,
    fail_acquire: Option<usize>,
    aborts: usize,
}
impl LayerwisePolicy<NumericBackend, Unit> for StagePolicy {
    type Lease = NumericReplicatedLease<Unit>;
    type Error = Error;
    fn begin(&mut self, input: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        self.inner.begin(input, context)
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<Unit, E>,
    {
        if self.fail_acquire == Some(ordinal) {
            return Err(LayerwiseAcquireError::Policy(Error::backend(
                "injected stage acquisition failure",
            )));
        }
        self.inner.acquire(ordinal, address, build, context)
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a NumericTensor,
        state: S,
        retained: C,
        context: &NumericContext,
    ) -> Result<(), Error>
    where
        NumericTensor: 'a,
        S: Iterator<Item = &'a NumericTensor>,
        C: Iterator<Item = &'a NumericTensor>,
    {
        self.inner
            .complete(ordinal, address, lease, output, state, retained, context)
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        self.inner.finish(output, context)
    }
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &NumericContext,
    ) {
        self.aborts += 1;
        self.inner.abort(active, context);
    }
}

fn fixture_weights(
    parameters: &Value,
    layout: &LocalModelLayout,
    duplicate_kv: bool,
) -> BTreeMap<String, NumericTensor> {
    parameters
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let mut value = tensor(value);
            if duplicate_kv
                && (name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight"))
            {
                value = NumericTensor::concatenate(
                    &[value.clone(), value],
                    0,
                    &NumericContext::default(),
                )
                .unwrap();
            }
            (
                name.clone(),
                select_parameter(&value, layout.tensor(name).unwrap().placement()).unwrap(),
            )
        })
        .collect()
}

fn stages(
    config: &TransformerConfig,
    layout: &LocalModelLayout,
    weights: &BTreeMap<String, NumericTensor>,
    tensor_parallel: bool,
    resident: bool,
    cuts: &[(std::ops::Range<usize>, bool)],
    context: &NumericContext,
) -> Vec<Stage> {
    let description = tts_transformer_parameter_description(config).unwrap();
    let complete = if tensor_parallel {
        TransformerArchitecture::<NumericBackend>::new_parallel(config.clone(), layout).unwrap()
    } else {
        TransformerArchitecture::<NumericBackend>::new(config.clone()).unwrap()
    };
    let full_state = complete.state_layout().unwrap();
    let plan =
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(0, 0..2)]);
    cuts.iter()
        .enumerate()
        .map(|(stage, (blocks, norm))| {
            let mut groups = Vec::new();
            if !blocks.is_empty() {
                groups.push((description.graph().groups()[0].id(), blocks.clone()));
            }
            if *norm {
                groups.push((description.graph().groups()[1].id(), 0..1));
            }
            let partition = ArchitecturePartition::from_description(
                &description,
                groups.clone(),
                PartitionOwnership::new(stage == 0, *norm, std::iter::empty::<String>()).unwrap(),
                &full_state,
                &plan,
                (),
                TransformerBoundarySchema::new(config),
            )
            .unwrap();
            let architecture = TransformerArchitecture::from_partition(
                config.clone(),
                tensor_parallel.then_some(layout),
                &partition,
            )
            .unwrap();
            let mut inner = if resident {
                let mut units = Vec::new();
                for owned in partition.groups() {
                    for index in owned.global_units() {
                        let mut unit = architecture
                            .build_unit(owned.group_index(), index, context)
                            .unwrap();
                        replace_numeric_parameters(&mut unit, weights);
                        units.push(unit);
                    }
                }
                NumericReplicatedPolicy::resident(units)
            } else {
                NumericReplicatedPolicy::bounded()
            };
            inner.parameter_overrides = weights.clone();
            let policy = StagePolicy {
                inner,
                fail_acquire: None,
                aborts: 0,
            };
            let runtime =
                Runtime::new(architecture, policy, &partition, |group| group == 0).unwrap();
            let cache = if config.geometry().causal {
                blocks
                    .clone()
                    .map(|index| CausalCache(NumericCache::new((index == 0).then_some(3))))
                    .collect()
            } else {
                vec![]
            };
            let state = State::for_partition(runtime.architecture(), cache).unwrap();
            Stage {
                runtime,
                state,
                blocks: blocks.clone(),
            }
        })
        .collect()
}

fn run(
    stages: &mut [Stage],
    embedded: &NumericTensor,
    mask: Option<&NumericTensor>,
    parallel: Option<&NumericParallelContext>,
    context: &NumericContext,
) -> NumericTensor {
    let mut boundary: Option<(NumericTensor, TransformerBoundary<NumericTensor>)> = None;
    let count = stages.len();
    for (index, stage) in stages.iter_mut().enumerate() {
        let input = match boundary.take() {
            Some((hidden, auxiliary)) => {
                let auxiliary = if index == 1 && parallel.is_none() {
                    reject_changed_boundary(stage, &hidden, auxiliary, context)
                } else {
                    auxiliary
                };
                LayeredPartitionInput::Hidden { hidden, auxiliary }
            }
            None => LayeredPartitionInput::Tokens(embedded),
        };
        let output = stage
            .runtime
            .forward_with_preparation(
                &mut stage.state,
                parallel,
                context,
                |architecture, state, expected| {
                    architecture.begin_prepared_partition(
                        input,
                        (index == 0).then_some(mask).flatten(),
                        state,
                        expected,
                        parallel,
                        context,
                    )
                },
            )
            .unwrap_or_else(|error| {
                panic!(
                    "stage {:?}, frontier {:?}, sequence {}: {error:?}",
                    stage.blocks,
                    stage.state.causal_position(),
                    embedded.dim(1)
                )
            });
        if stage.runtime.architecture().config().geometry().causal {
            let position = stage
                .runtime
                .architecture()
                .validate_causal_state(&stage.state)
                .unwrap();
            assert!(position >= embedded.dim(1));
            let copy = stage
                .state
                .try_map_caches(|caches| Ok::<_, std::convert::Infallible>(caches.to_vec()))
                .unwrap();
            assert_eq!(copy.causal_position().unwrap(), position);
            if stage.state.caches().is_empty() {
                // A freshly reconstructed empty stage belongs to the initial
                // frontier; copying an advanced state retains its logical time.
                let cold = State::for_partition(stage.runtime.architecture(), vec![]).unwrap();
                assert_eq!(cold.causal_position().unwrap(), 0);
                assert!(position > cold.causal_position().unwrap());
            }
        }
        match output {
            LayeredPartitionOutput::Boundary { hidden, auxiliary } => {
                assert!(index + 1 < count, "the final normalization owns output");
                // Round-trip through the architecture's actual wire representation;
                // later stages receive neither the original mask nor host position.
                let schema = TransformerBoundarySchema::new(stage.runtime.architecture().config());
                let values = schema.encode(auxiliary).unwrap();
                let auxiliary = schema
                    .decode(
                        values
                            .into_iter()
                            .map(|value| value.into_parts().1)
                            .collect(),
                    )
                    .unwrap();
                boundary = Some((hidden, auxiliary));
            }
            LayeredPartitionOutput::Final { output, .. } => {
                assert_eq!(index + 1, count, "only the final norm owner publishes");
                return output;
            }
        }
    }
    panic!("pipeline did not publish its final normalized hidden state")
}

// A received frame cannot reuse another cache frontier or silently change key
// visibility. Rejection occurs before any learned unit or cache append.
fn reject_changed_boundary(
    stage: &mut Stage,
    hidden: &NumericTensor,
    auxiliary: TransformerBoundary<NumericTensor>,
    context: &NumericContext,
) -> TransformerBoundary<NumericTensor> {
    let schema = TransformerBoundarySchema::new(stage.runtime.architecture().config());
    let values = schema.encode(auxiliary).unwrap();
    let original = values
        .iter()
        .map(|value| value.tensor().clone())
        .collect::<Vec<_>>();
    let mut changed = original.clone();
    let causal = stage.runtime.architecture().config().geometry().causal;
    let role = if causal { "position" } else { "visibility" };
    let slot = values
        .iter()
        .position(|value| value.role() == role)
        .unwrap();
    if causal {
        changed[slot] =
            NumericTensor::from_i32_slice(&[original[slot].data[0] as i32 + 1, 1], &[2], context)
                .unwrap();
    } else {
        let end = changed[slot].dim(1) as usize;
        assert!(end > 1);
        changed[slot] = changed[slot].axis_slice(1, 0, end - 1);
    }
    let changed = schema.decode(changed).unwrap();
    let before = stage.state.caches().to_vec();
    {
        let result = stage.runtime.forward_with_preparation(
            &mut stage.state,
            None,
            context,
            |architecture, state, expected| {
                architecture.begin_prepared_partition(
                    LayeredPartitionInput::Hidden {
                        hidden: hidden.clone(),
                        auxiliary: changed,
                    },
                    None,
                    state,
                    expected,
                    None,
                    context,
                )
            },
        );
        assert!(matches!(result,
            Err(LayerwisePartitionRuntimeError::Begin(
                eredu_runtime::LayeredPartitionBeginError::Architecture(
                    eredu_architectures::voicechat::tts_transformer_layered::TransformerError::Configuration(_)
                )
            ))
        ));
    }
    for (before, after) in before.iter().zip(stage.state.caches()) {
        assert_eq!(before.offset(), after.offset());
        for (before, after) in [
            (&before.0.keys, &after.0.keys),
            (&before.0.values, &after.0.values),
        ] {
            match (before, after) {
                (None, None) => {}
                (Some(before), Some(after)) => {
                    assert_tensor_exact(before, after, "rejected pipeline boundary cache")
                }
                _ => panic!("rejected boundary changed cache allocation"),
            }
        }
    }
    schema.decode(original).unwrap()
}

fn pipeline_fixture(causal: bool) {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let data = &fixture[if causal { "gemma3" } else { "t5gemma" }];
    for ranks in [1, 2] {
        let mut dimensions = geometry(
            if causal {
                &data["config"]
            } else {
                &data["config"]["encoder"]
            },
            causal,
        );
        // Two identical KV heads preserve the oracle equation while giving the
        // ordinary KV-group planner one complete head per tensor rank.
        if ranks == 2 {
            dimensions.num_key_value_heads = 2;
        }
        let config = TransformerConfig::new(&dimensions).unwrap();
        let description = tts_transformer_parameter_description(&config).unwrap();
        let groups = description
            .groups()
            .iter()
            .map(|owned| owned.group().clone())
            .collect::<Vec<_>>();
        for cuts in [
            vec![(0..1, false), (1..2, false), (2..2, true)],
            vec![(0..1, false), (1..2, true)],
            vec![(0..2, false), (2..2, true)],
        ] {
            for resident in [true, false] {
                let collective = NumericParallelGroup::new(ranks);
                std::thread::scope(|scope| {
                    let handles = (0..ranks)
                        .map(|rank| {
                            let cuts = &cuts;
                            let config = config.clone();
                            let layout = numeric_local_layout(&groups, ranks, rank).unwrap();
                            let collective = Arc::clone(&collective);
                            let fixture = &fixture;
                            scope.spawn(move || {
                                let context = NumericContext::with_local_layout(layout.clone());
                                let parallel = NumericParallelContext::new(rank, collective);
                                let parallel = (ranks == 2).then_some(&parallel);
                                let weights =
                                    fixture_weights(&data["parameters"], &layout, ranks == 2);
                                let mut pipeline = stages(
                                    &config,
                                    &layout,
                                    &weights,
                                    ranks == 2,
                                    resident,
                                    cuts,
                                    &context,
                                );
                                if causal {
                                    for step in data["chunks"].as_array().unwrap() {
                                        let output = run(
                                            &mut pipeline,
                                            &tensor(&step["input"]),
                                            None,
                                            parallel,
                                            &context,
                                        );
                                        compare(&output, &step["output"], fixture);
                                        for stage in &pipeline {
                                            let expected = Value::Array(
                                                step["cache"].as_array().unwrap()
                                                    [stage.blocks.clone()]
                                                .to_vec(),
                                            );
                                            compare_cache(stage.state.caches(), &expected, fixture);
                                        }
                                    }
                                } else {
                                    let mask = tensor(&data["attention_mask"])
                                        .equal_i32(1, &context)
                                        .unwrap();
                                    for batch in data["batches"].as_array().unwrap() {
                                        let output = run(
                                            &mut pipeline,
                                            &tensor(&batch["input"]),
                                            Some(&mask),
                                            parallel,
                                            &context,
                                        );
                                        compare(&output, &batch["output"], fixture);
                                    }
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        }
    }
}

#[test]
fn voicechat_transformer_pipeline_cuts_and_tp2_cached_causal_match_pinned_hf() {
    pipeline_fixture(true);
}

#[test]
fn voicechat_transformer_pipeline_cuts_and_tp2_character_masks_match_pinned_hf() {
    pipeline_fixture(false);
}

#[test]
fn voicechat_transformer_compound_stage_failure_discards_branch_and_retries() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let data = &fixture["gemma3"];
    let config = TransformerConfig::new(&geometry(&data["config"], true)).unwrap();
    let groups = tts_transformer_parameter_description(&config)
        .unwrap()
        .groups()
        .iter()
        .map(|owned| owned.group().clone())
        .collect::<Vec<_>>();
    let layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let context = NumericContext::with_local_layout(layout.clone());
    let weights = fixture_weights(&data["parameters"], &layout, false);
    for resident in [true, false] {
        let mut pipeline = stages(
            &config,
            &layout,
            &weights,
            false,
            resident,
            &[(0..2, true)],
            &context,
        );
        let first = &data["chunks"][0];
        let output = run(
            &mut pipeline,
            &tensor(&first["input"]),
            None,
            None,
            &context,
        );
        compare(&output, &first["output"], &fixture);
        let stage = &mut pipeline[0];
        let parent = stage.state.caches().to_vec();
        compare_cache(&parent, &first["cache"], &fixture);
        let next = &data["chunks"][1];
        let input = tensor(&next["input"]);
        stage.runtime.policy_mut().fail_acquire = Some(2);
        let mut state = stage
            .state
            .try_map_caches(|caches| Ok::<_, std::convert::Infallible>(caches.to_vec()))
            .unwrap();
        let failed = stage.runtime.forward_with_preparation(
            &mut state,
            None,
            &context,
            |architecture, state, expected| {
                architecture.begin_prepared_partition(
                    LayeredPartitionInput::Tokens(&input),
                    None,
                    state,
                    expected,
                    None,
                    &context,
                )
            },
        );
        assert!(matches!(
            failed,
            Err(LayerwisePartitionRuntimeError::Policy(_))
        ));
        assert_eq!(stage.runtime.policy().aborts, 1);
        assert!(state
            .caches()
            .iter()
            .zip(&parent)
            .all(|(branch, parent)| branch.offset() == parent.offset() + input.dim(1)));
        compare_cache(&parent, &first["cache"], &fixture);
        compare_cache(stage.state.caches(), &first["cache"], &fixture);
        stage.runtime.policy_mut().fail_acquire = None;
        let output = run(&mut pipeline, &input, None, None, &context);
        compare(&output, &next["output"], &fixture);
        compare_cache(pipeline[0].state.caches(), &next["cache"], &fixture);
        assert_eq!(pipeline[0].runtime.policy().aborts, 1);
    }
}
