use super::*;
use crate::backend::nn::workspace::ExistingArrayProjection;
use eredu_core::{AttentionPolicy, cache::LayerCachePolicy};
use eredu_nn::{Tensor, workspace::*};
use std::{convert::Infallible, num::NonZeroU32};

// This fixture compares source/append geometry and custody, not an inference
// admission quote. Absent numerical facts must remain explicitly absent.
#[derive(Debug)]
struct AppendFacts;
impl WorkspaceMechanisms for AppendFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
impl WorkspaceFactMechanisms for AppendFacts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("absent operation facts")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("absent host facts")
    }
}

#[test]
#[ignore = "requires native CPU cache producers"]
fn native_paged_append_matches_projected_blocks_and_retains_failed_source_custody() {
    use crate::backend::runtime::cache::residency::CacheResidencyError;
    use eredu_runtime::CacheLifecycleError;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(3, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut cache = PagedKeyValueCache::new_with_layout(manager.clone(), 3, None, 0, None).unwrap();
    let first = Array::from_slice(&[0.5f32, 1.5, 2.5, 3.5, 4.5], &[1, 1, 5, 1]);
    let first_values = Array::from_slice(&[-0.5f32, -1.5, -2.5, -3.5, -4.5], &[1, 1, 5, 1]);
    cache.append(first, first_values, &stream).unwrap();
    settle(&cache);
    let capacity = 1 << 23;
    let pool = crate::memory_fixture::ledger(capacity, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(capacity),
        )
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(AppendFacts, funding.clone()).unwrap();
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap();
    let source = cold(|| cache.project_device_workspace(&context)).unwrap();
    let id = source.geometry().blocks[0].id.clone();
    let mut projected =
        cold(|| source.into_append_workspace(&policy, NonZeroU32::new(1).unwrap(), &context))
            .unwrap();
    let extra = Array::from_slice(
        &[5.5f32, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
        &[1, 1, 8, 1],
    );
    let extra_values = Array::from_slice(
        &[-5.5f32, -6.5, -7.5, -8.5, -9.5, -10.5, -11.5, -12.5],
        &[1, 1, 8, 1],
    );
    let input = cold(|| {
        let mut projection = ExistingArrayProjection::new(&context);
        [
            projection.project_prepared(&extra).unwrap(),
            projection.project_prepared(&extra_values).unwrap(),
        ]
    });
    cold(|| {
        projected
            .state_mut()
            .append_normalized(input, false, &context)
    })
    .unwrap();
    cache.append(extra, extra_values, &stream).unwrap();
    settle(&cache);
    let actual =
        cold(|| cache.with_workspace_source(&context, |source| source.prepare_geometry(&context)))
            .unwrap();
    assert_eq!(projected.state().geometry().offset, actual.offset);
    assert_eq!(projected.state().geometry().tail_start, actual.tail_start);
    assert_eq!(
        projected
            .state()
            .blocks()
            .iter()
            .map(|block| block.range())
            .collect::<Vec<_>>(),
        actual
            .blocks
            .iter()
            .map(|block| block.id.start..block.id.end)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        projected.state().tail().unwrap()[0].shape(),
        actual.tail.unwrap()[0].shape
    );
    assert_eq!(
        cache
            .tail_keys
            .as_ref()
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        [12.5]
    );
    assert_eq!(
        cache
            .tail_values
            .as_ref()
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        [-12.5]
    );
    for block in &actual.blocks {
        let lease = manager.lease_block(&block.id, &stream).unwrap();
        let crate::backend::runtime::cache::residency::CacheBlockArrays::KeyValue { keys, values } =
            lease.arrays()
        else {
            panic!("actual key/value source")
        };
        let expected = (block.id.start..block.id.end)
            .map(|position| position as f32 + 0.5)
            .collect::<Vec<_>>();
        assert_eq!(
            keys.evaluated().unwrap().as_slice::<f32>(),
            expected.as_slice()
        );
        let negative = expected.into_iter().map(|value| -value).collect::<Vec<_>>();
        assert_eq!(
            values.evaluated().unwrap().as_slice::<f32>(),
            negative.as_slice()
        );
    }
    let refused_source = cold(|| cache.project_device_workspace(&context)).unwrap();
    let wrong = LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 1).unwrap();
    let failure = match cold(|| {
        refused_source.into_append_workspace(&wrong, NonZeroU32::new(1).unwrap(), &context)
    }) {
        Ok(_) => panic!("declared heads cannot replace actual source geometry"),
        Err(failure) => failure,
    };
    drop((cache, actual, context, funding));
    assert!(pool.fixture_host_charge().unwrap() > 0);
    assert!(matches!(
        manager.remove_block(&id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    drop(projected);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    assert!(matches!(
        manager.remove_block(&id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    drop(failure);
    manager.remove_block(&id).unwrap();
    drop(manager);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
