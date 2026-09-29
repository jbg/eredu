//! Shared merger equations preserve DeepStack geometry and TP capture order.
use super::*;
use eredu_architectures::qwen::vision::{VisionAttentionPolicy, VisionConfig, VisionLayerPolicy};
use std::collections::BTreeMap;

fn merger_config() -> VisionConfig {
    let (_directory, _target, ingress) = setup();
    let mut config = ingress.vision().config().clone();
    config.layer_schedule = eredu_core::attention::LayerSchedule::new(
        2,
        vec![
            VisionLayerPolicy {
                attention: VisionAttentionPolicy::Full,
                deepstack_merger: Some(0),
            },
            VisionLayerPolicy {
                attention: VisionAttentionPolicy::Full,
                deepstack_merger: Some(1),
            },
        ],
    )
    .unwrap();
    config
}

struct Nonzero(BTreeMap<String, NumericTensor>);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Nonzero {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = meta.id.as_str().to_owned();
        let seed = name.bytes().map(usize::from).sum::<usize>() % 31;
        let count = value.shape().iter().map(|n| *n as usize).product();
        let data = (0..count)
            .map(|index| {
                let v = ((index * 7 + seed) % 43) as f32 / 256. - 21. / 256.;
                v + if name.contains("norm") && name.ends_with(".weight") {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        *value = NumericTensor::new(value.shape(), data);
        assert!(self.0.insert(name, value.clone()).is_none());
    }
}

fn full_tower(
    config: &VisionConfig,
    context: &NumericContext,
) -> (VisionTower<NumericBackend>, BTreeMap<String, NumericTensor>) {
    let mut tower = VisionTower::new_with_root(config.clone(), "model.visual", context).unwrap();
    let mut weights = Nonzero(BTreeMap::new());
    tower.visit_parameters_mut(&mut weights);
    (tower, weights.0)
}

// The rank fixture splits independent full tensors along the declared parameter
// axes; it does not generate a different random tensor for the smaller shape.
struct RankWeights<'a> {
    weights: &'a BTreeMap<String, NumericTensor>,
    rank: usize,
    context: &'a NumericContext,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for RankWeights<'_> {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = meta.id.as_str();
        let full = &self.weights[name];
        let selected = if name.contains(".attn.qkv.") {
            let hidden = full.dim(0) as usize / 3;
            let width = hidden / 2;
            let pieces = (0..3)
                .map(|part| {
                    full.axis_slice(
                        0,
                        part * hidden + self.rank * width,
                        part * hidden + (self.rank + 1) * width,
                    )
                })
                .collect::<Vec<_>>();
            NumericTensor::concatenate(&pieces, 0, self.context).unwrap()
        } else if name.ends_with(".attn.proj.weight") || name.ends_with(".linear_fc2.weight") {
            let width = full.dim(1) as usize / 2;
            full.axis_slice(1, self.rank * width, (self.rank + 1) * width)
        } else if name.contains(".linear_fc1.") {
            let width = full.dim(0) as usize / 2;
            full.axis_slice(0, self.rank * width, (self.rank + 1) * width)
        } else {
            full.clone()
        };
        assert_eq!(selected.shape(), value.shape(), "{name}");
        *value = selected;
    }
}

#[test]
fn deepstack_and_final_mergers_match_tp2_and_preserve_capture_order() {
    let config = merger_config();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let (mut full, weights) = full_tower(&config, &context);
    assert_eq!(
        weights["model.visual.deepstack_merger_list.0.norm.weight"].shape(),
        &[32]
    );
    assert_eq!(weights["model.visual.merger.norm.weight"].shape(), &[8]);
    let pixels = pixels(12);
    let grid = [(1, 2, 2), (2, 2, 2)];
    let expected = full
        .forward(
            VisionInput {
                pixels: &pixels,
                grid: &grid,
            },
            &context,
        )
        .unwrap();
    assert!(expected
        .embeddings
        .data
        .iter()
        .any(|value| value.abs() > 1e-6));
    assert_eq!(expected.embeddings.shape(), &[1, 3, 32]);
    assert_eq!(expected.deepstack_features.len(), 2);
    let group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|rank| {
                let group = group.clone();
                let (config, weights, pixels, grid) = (&config, &weights, &pixels, &grid);
                scope.spawn(move || {
                    let context = NumericContext {
                        bind_checkpoint_values: true,
                        ..Default::default()
                    };
                    let parallel = NumericParallelContext::new(rank, group);
                    let mut modules = VisionStatic::<NumericBackend>::new_parallel_with_root(
                        config.clone(),
                        "model.visual",
                        &[16, 16, 16],
                        &context,
                    )
                    .unwrap();
                    modules.visit_parameters_mut(&mut RankWeights {
                        weights,
                        rank,
                        context: &context,
                    });
                    let (mut hidden, mut state) = modules
                        .begin(VisionInput { pixels, grid }, &context)
                        .unwrap();
                    for index in 0..2 {
                        let mut block = VisionBlock::<NumericBackend>::new_parallel_with_root(
                            config,
                            "model.visual",
                            index,
                            config.num_heads / 2,
                            config.intermediate_size / 2,
                            &context,
                        )
                        .unwrap();
                        block.visit_parameters_mut(&mut RankWeights {
                            weights,
                            rank,
                            context: &context,
                        });
                        hidden = modules
                            .forward_block_parallel(
                                &mut block, index, &hidden, &mut state, &parallel, &context,
                            )
                            .unwrap();
                        assert_eq!(state.deepstack_features().len(), index + 1);
                    }
                    let output = modules
                        .finish_parallel(&hidden, &mut state, &parallel, &context)
                        .unwrap();
                    assert!(state.deepstack_features().is_empty());
                    assert_eq!(
                        parallel.trace().len(),
                        7,
                        "two reductions per block and one per merger"
                    );
                    output
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for output in outputs {
        assert_tensor_close(&output.embeddings, &expected.embeddings, "TP2 final merger");
        assert_eq!(output.deepstack_features.len(), 2);
        for (actual, expected) in output
            .deepstack_features
            .iter()
            .zip(&expected.deepstack_features)
        {
            assert_tensor_close(actual, expected, "TP2 DeepStack merger");
        }
    }
}
