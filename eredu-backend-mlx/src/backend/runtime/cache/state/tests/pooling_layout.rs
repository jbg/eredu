use std::num::NonZeroU32;

use eredu_core::{
    cache::{MutableStateResidency, StateResidencyClass, StateTensorDtype, StateTensorPolicy},
    AttentionPolicy, LayerSchedule,
};

use super::*;

fn pooling_stream(stream: u32, ratio: u32, overlapping: bool) -> Vec<StateTensorPolicy> {
    let ratio = NonZeroU32::new(ratio).unwrap();
    let role = |component| StateTensorRole::Pooling { stream, component };
    let pending = |component| {
        StateTensorPolicy::new(
            role(component),
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensRem(ratio),
                StateTensorDimension::fixed(8).unwrap(),
            ],
            StateTensorDtype::Floating,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .unwrap()
        .when_prefix_remainder_nonzero(ratio)
    };
    let mut policies = vec![
        pending(PoolingStateComponent::PendingValues),
        pending(PoolingStateComponent::PendingGates),
        StateTensorPolicy::new_with_residency(
            role(PoolingStateComponent::Pooled),
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(ratio),
                StateTensorDimension::fixed(8).unwrap(),
            ],
            StateTensorDtype::Floating,
            StateResidencyClass::SealablePaged,
        )
        .unwrap()
        .when_prefix_at_least(ratio),
    ];
    if overlapping {
        for component in [
            PoolingStateComponent::OverlapValues,
            PoolingStateComponent::OverlapGates,
        ] {
            policies.push(
                StateTensorPolicy::new(
                    role(component),
                    vec![
                        StateTensorDimension::Batch,
                        StateTensorDimension::Fixed(ratio),
                        StateTensorDimension::fixed(8).unwrap(),
                    ],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .unwrap()
                .when_prefix_at_least(ratio),
            );
        }
    }
    policies
}

#[test]
fn materializes_window_and_pooling_streams_from_layer_policy() {
    let mut tensors = pooling_stream(0, 4, true);
    tensors.extend(pooling_stream(1, 6, true));
    let policy = LayerCachePolicy::key_only_with_fixed_state(
        AttentionPolicy::sliding(37).unwrap(),
        1,
        8,
        tensors,
    )
    .unwrap();

    let geometry = pooling_attention_geometry(7, &policy).unwrap();
    assert_eq!(
        geometry,
        PoolingAttentionGeometry {
            sliding_window: 37,
            stream_ratios: vec![4, 6],
        }
    );
    let state = MlxPoolingAttentionCache::resident_from_policy(7, &policy).unwrap();
    let MlxPoolingAttentionCache::Sparse {
        pool, index_pool, ..
    } = state
    else {
        panic!("two declared streams must create sparse pooling state")
    };
    assert_eq!(pool.ratio(), 4);
    assert_eq!(index_pool.ratio(), 6);

    let layout = StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
    let state = MlxPoolingAttentionStateFactory::device(layout.clone()).unwrap();
    assert_eq!(state.layout(), &layout);
}

#[test]
fn materializes_local_only_policy_without_family_arguments() {
    let policy = LayerCachePolicy::key_only(AttentionPolicy::sliding(23).unwrap(), 1, 8).unwrap();

    assert_eq!(
        pooling_attention_geometry(2, &policy).unwrap(),
        PoolingAttentionGeometry {
            sliding_window: 23,
            stream_ratios: Vec::new(),
        }
    );
    assert!(matches!(
        MlxPoolingAttentionCache::resident_from_policy(2, &policy).unwrap(),
        MlxPoolingAttentionCache::Local(_)
    ));
}
