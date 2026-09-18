use super::*;
use eredu_core::{HostMetadataAccount, ModelConfigurationResolver};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
#[derive(Debug)]
struct Account {
    calls: Arc<AtomicUsize>,
    stop: usize,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(call <= self.stop, "producer after refusal");
        if call == self.stop {
            Err(HostMetadataFundingError::Unavailable)
        } else {
            Ok(())
        }
    }
}
fn account(stop: usize) -> (HostMetadataFunding, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let funding = HostMetadataFunding::new(Account {
        calls: calls.clone(),
        stop: stop + 1,
    })
    .unwrap();
    (funding, calls)
}
fn fixture() -> (
    ArchitectureDescriptor,
    ComponentGroup,
    LocalModelLayout,
    eredu_core::ParallelRankTopology,
) {
    let descriptor = crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&serde_json::json!({
            "model_type":"llama","hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":2,"vocab_size":16,
            "rms_norm_eps":0.00001,"max_position_embeddings":64
        }))
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let component = descriptor
        .components
        .iter()
        .find(|component| component.count == 12)
        .unwrap()
        .clone();
    let mut layout = LocalModelLayout::default();
    layout.insert(
        component.write_weight.clone(),
        LocalTensorLayout::new(
            "ffn",
            eredu_runtime::ParameterRole::FeedForwardIntermediate,
            vec![8, 12],
            vec![8, 4],
            TensorPlacement::Indices {
                axis: 1,
                indices: vec![9, 2, 7, 0],
            },
            None,
            None,
            false,
        ),
    );
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        0,
    )
    .unwrap();
    (descriptor, component, layout, topology)
}
#[test]
fn component_source_preserves_permutation_and_refuses_every_destination() {
    let (descriptor, component, layout, topology) = fixture();
    let components = std::slice::from_ref(&component);
    for local in [true, false] {
        let expected = ComponentPartitionLayout::from_components(
            &descriptor,
            components,
            &layout,
            topology,
            &|_| Ok(local),
        )
        .unwrap();
        let (funding, calls) = account(usize::MAX - 1);
        let actual = ComponentPartitionLayout::from_components_worker(
            &descriptor,
            components,
            &layout,
            topology,
            &|_| Ok(local),
            Destination(Some(&funding)),
        )
        .unwrap();
        assert_eq!(actual, expected);
        if local {
            let coordinates = actual.group(&component.id).unwrap().coordinates().unwrap();
            assert_eq!(
                (0..4)
                    .map(|i| coordinates.local_to_global(i).unwrap())
                    .collect::<Vec<_>>(),
                [9, 2, 7, 0]
            );
            assert_eq!(coordinates.global_to_local(2), Some(1));
        } else {
            assert!(actual.group(&component.id).unwrap().coordinates().is_none());
        }
        let count = calls.load(Ordering::SeqCst) - 1;
        assert!(count > 30);
        drop(actual);
        for stop in 0..count {
            let (funding, calls) = account(stop);
            let result = ComponentPartitionLayout::from_components_worker(
                &descriptor,
                components,
                &layout,
                topology,
                &|_| Ok(local),
                Destination(Some(&funding)),
            );
            assert!(
                matches!(
                    result,
                    Err(ComponentPartitionError::MetadataFunding(
                        HostMetadataFundingError::Unavailable
                    ))
                ),
                "cut {stop}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), stop + 2);
        }
    }
}
#[test]
fn component_diagnostic_keeps_original_meaning_and_fixed_funding_leaf() {
    let (descriptor, component, _, topology) = fixture();
    let layout = LocalModelLayout::default();
    let components = std::slice::from_ref(&component);
    let expected = ComponentPartitionLayout::from_components(
        &descriptor,
        components,
        &layout,
        topology,
        &|_| Ok(true),
    )
    .unwrap_err();
    let (funding, calls) = account(usize::MAX - 1);
    let actual = ComponentPartitionLayout::from_components_worker(
        &descriptor,
        components,
        &layout,
        topology,
        &|_| Ok(true),
        Destination(Some(&funding)),
    )
    .unwrap_err();
    assert_eq!(actual, expected);
    assert!(matches!(actual, ComponentPartitionError::MissingWeight(_)));
    let count = calls.load(Ordering::SeqCst) - 1;
    for stop in 0..count {
        let (funding, _) = account(stop);
        let error = ComponentPartitionLayout::from_components_worker(
            &descriptor,
            components,
            &layout,
            topology,
            &|_| Ok(true),
            Destination(Some(&funding)),
        )
        .unwrap_err();
        assert!(
            std::error::Error::source(&error)
                .unwrap()
                .is::<HostMetadataFundingError>()
        );
    }
}

/// Exercise the caller's original worker at every reached prospective destination.
pub(in crate::component_partition) fn verify<T: std::fmt::Debug + PartialEq>(
    build: impl Fn(Destination<'_>) -> Result<T, ComponentPartitionError>,
) {
    let expected = build(Destination(None));
    let (funding, calls) = account(usize::MAX - 1);
    assert_eq!(build(Destination(Some(&funding))), expected);
    let count = calls.load(Ordering::SeqCst) - 1;
    assert!(count > 0);
    for stop in 0..count {
        let (funding, calls) = account(stop);
        let error = build(Destination(Some(&funding))).unwrap_err();
        assert!(
            matches!(
                error,
                ComponentPartitionError::MetadataFunding(HostMetadataFundingError::Unavailable)
            ),
            "cut {stop}: {error}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), stop + 2);
    }
}

pub(in crate::component_partition) fn empty_source(
    topology: eredu_core::ParallelTopology,
    allocation: Destination<'_>,
) -> Result<ComponentPartitionLayouts, ComponentPartitionError> {
    allocation.controls::<(
        eredu_core::ParallelTopology,
        Vec<ComponentPartitionLayout>,
        ComponentPartitionLayout,
        usize,
    )>()?;
    let mut rows = allocation.vector(topology.world_size())?;
    for rank in 0..topology.world_size() {
        rows.push(ComponentPartitionLayout {
            topology: eredu_core::ParallelRankTopology::new(topology, rank).unwrap(),
            groups: SourceMap::new(),
            paths: SourceMap::new(),
            observations: SourceMap::new(),
            routed: SourceMap::new(),
        });
    }
    ComponentPartitionLayouts::new_worker(topology, rows, allocation)
}
