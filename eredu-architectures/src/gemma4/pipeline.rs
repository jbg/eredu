//! Shared-attention transport and its explicitly accounted receiver caches.

use std::{collections::HashMap, ops::Range};

use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
use eredu_runtime::{StateError, StateLayout};

use super::ModelArgs;

/// Parameter ownership follows the encoder ingress and output phases.
pub(crate) fn media_transport(
    args: &super::FamilyConfig,
    group: usize,
) -> eredu_runtime::ArchitectureGroupTransport {
    use eredu_runtime::*;
    let (kind, first, last) = match group {
        0 => (
            ArchitectureGroupKind::VisionEncoder,
            args.vision
                .as_ref()
                .map_or_else(Vec::new, |_| vec!["vision".into()]),
            args.vision.as_ref().map_or_else(Vec::new, |vision| {
                let mut roles = Vec::new();
                if vision.standardize {
                    roles.push("vision".into());
                }
                roles.push("vision_projection".into());
                roles
            }),
        ),
        1 => (
            ArchitectureGroupKind::AudioEncoder,
            args.audio
                .as_ref()
                .map_or_else(Vec::new, |_| vec!["audio".into()]),
            args.audio.as_ref().map_or_else(Vec::new, |_| {
                vec!["audio".into(), "audio_projection".into()]
            }),
        ),
        _ => unreachable!("Gemma media group"),
    };
    ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind,
        first_owner_static_roles: first,
        last_owner_static_roles: last,
        merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: Some(ArchitectureParallelSubgroup::TensorSharded),
        request_optional: true,
    }
}

/// Exact tensor-local cache geometry before pipeline receiver replicas.
pub(crate) fn tensor_state_layout(
    args: &ModelArgs,
    rank: eredu_core::ParallelRankTopology,
) -> Result<StateLayout, String> {
    let mut local = args.clone();
    let policies = args
        .layer_schedule
        .iter()
        .copied()
        .map(|mut policy| {
            let range = eredu_core::balanced_contiguous_range(
                policy.num_key_value_heads.get() as usize,
                rank.tensor_parallel_size(),
                rank.tensor_parallel_rank(),
                false,
            )
            .map_err(|error| error.to_string())?;
            policy.num_key_value_heads = std::num::NonZeroU32::new(range.len() as u32)
                .ok_or_else(|| "Gemma local KV head count is zero".to_owned())?;
            Ok(policy)
        })
        .collect::<Result<Vec<_>, String>>()?;
    local.layer_schedule =
        LayerSchedule::new(policies.len(), policies).map_err(|error| error.to_string())?;
    super::state_layout(&local).map_err(|error| error.to_string())
}

/// One receiver cache per remote publisher, attached to its first local consumer.
/// Local consumers of the same publication reuse that cache during the pass.
pub(crate) fn receiver_caches(args: &ModelArgs, units: Range<usize>) -> Vec<(usize, usize)> {
    let mut publishers = HashMap::new();
    let mut receivers = HashMap::<AttentionPolicy, usize>::new();
    let mut result = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.key_value.publishes_state() {
            publishers.insert(policy.attention, layer);
        } else if policy.key_value == eredu_nn::AttentionStateSource::Shared
            && units.contains(&layer)
        {
            let publisher = publishers[&policy.attention];
            if publisher < units.start && receivers.insert(policy.attention, layer).is_none() {
                result.push((layer, publisher));
            }
        }
    }
    result
}

/// Finds the retained history used by a local attention invocation. Shared
/// consumers borrow a local publisher or the first local remote-state receiver.
pub(crate) fn attention_cache_owner(
    args: &ModelArgs,
    units: Range<usize>,
    layer: usize,
) -> Option<usize> {
    if !units.contains(&layer) {
        return None;
    }
    let policy = args.layer_policy(layer)?;
    if policy.key_value != eredu_nn::AttentionStateSource::Shared {
        return Some(layer);
    }
    let publisher = (0..layer).rev().find(|&candidate| {
        let source = args.layer_policy(candidate).expect("preceding layer");
        source.attention == policy.attention && source.key_value.publishes_state()
    })?;
    if units.contains(&publisher) {
        return Some(publisher);
    }
    (units.start..=layer).find(|&candidate| {
        let consumer = args.layer_policy(candidate).expect("local layer");
        consumer.attention == policy.attention
            && consumer.key_value == eredu_nn::AttentionStateSource::Shared
    })
}

/// Adds only the histories required by this pipeline partition's remote reads.
/// The base layout already carries exact tensor-parallel head geometry.
pub(crate) fn partition_state_layout(
    args: &ModelArgs,
    base: &StateLayout,
    units: Range<usize>,
) -> Result<StateLayout, StateError> {
    let mut layers = base.layers().iter().cloned().collect::<Vec<_>>();
    for (receiver, publisher) in receiver_caches(args, units) {
        let source = base.layer(publisher).ok_or_else(|| {
            StateError::InvalidResidency("Gemma shared-state publisher has no layout".into())
        })?;
        let (attention, num_key_value_heads, head_dim) = match source {
            LayerCachePolicy::KeyValue {
                attention,
                num_key_value_heads,
                head_dim,
            }
            | LayerCachePolicy::KeyValueWithFixedState {
                attention,
                num_key_value_heads,
                head_dim,
                ..
            } => (*attention, *num_key_value_heads, *head_dim),
            LayerCachePolicy::KeyOnly {
                attention,
                num_key_heads,
                head_dim,
            }
            | LayerCachePolicy::KeyOnlyWithFixedState {
                attention,
                num_key_heads,
                head_dim,
                ..
            } => (*attention, *num_key_heads, *head_dim),
            _ => {
                return Err(StateError::InvalidResidency(
                    "Gemma publisher is not attention state".into(),
                ))
            }
        };
        // Published values are normalized independently of rotary keys even
        // when their projections share a weight. Retain both exact payloads.
        layers[receiver] = LayerCachePolicy::KeyValue {
            attention,
            num_key_value_heads,
            head_dim,
        };
    }
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cut_declares_only_first_remote_consumer_histories() {
        let args = ModelArgs::from_hf_json(br#"{
            "model_type":"gemma4_text", "hidden_size":16, "num_hidden_layers":8,
            "intermediate_size":32, "num_attention_heads":4, "num_key_value_heads":2,
            "head_dim":4, "rms_norm_eps":0.000001, "vocab_size":64,
            "max_position_embeddings":128, "sliding_window":4, "num_kv_shared_layers":4,
            "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention",
                "sliding_attention","full_attention","sliding_attention","full_attention"]
        }"#).unwrap();
        assert_eq!(
            args.pipeline_layer_ranges(4).unwrap(),
            [0..2, 2..4, 4..6, 6..8]
        );
        let base = super::super::state_layout(&args).unwrap();
        for start in 0..8 {
            let receivers = receiver_caches(&args, start..8);
            let expected = match start {
                0..=2 => vec![],
                3 => vec![(4, 2)],
                4 => vec![(4, 2), (5, 3)],
                5 => vec![(5, 3), (6, 2)],
                6 => vec![(6, 2), (7, 3)],
                7 => vec![(7, 3)],
                _ => unreachable!(),
            };
            assert_eq!(receivers, expected);
            let layout = partition_state_layout(&args, &base, start..8).unwrap();
            for layer in start..8 {
                let publisher = if layer < 4 { layer } else { 2 + layer % 2 };
                let owner = if publisher >= start {
                    publisher
                } else {
                    expected
                        .iter()
                        .find(|(_, source)| *source == publisher)
                        .unwrap()
                        .0
                };
                assert_eq!(attention_cache_owner(&args, start..8, layer), Some(owner));
                assert!(matches!(
                    layout.layer(owner),
                    Some(
                        LayerCachePolicy::KeyValue { .. }
                            | LayerCachePolicy::KeyValueWithFixedState { .. }
                    )
                ));
            }
            assert_eq!(attention_cache_owner(&args, start..8, 8), None);
            for layer in 0..8 {
                if expected.iter().any(|(receiver, _)| *receiver == layer) {
                    assert!(matches!(
                        layout.layer(layer),
                        Some(LayerCachePolicy::KeyValue { .. })
                    ));
                } else {
                    assert_eq!(layout.layer(layer), base.layer(layer));
                }
            }
        }
    }
}
