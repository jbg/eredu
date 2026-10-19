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
            | LayerCachePolicy::KeyValueWithState {
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
            | LayerCachePolicy::KeyOnlyWithState {
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
