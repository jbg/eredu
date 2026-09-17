use super::*;
use eredu_runtime::working_memory::WorkingMemoryPool;

/// Models a cache that can change state before reporting a restore failure.
#[derive(Debug)]
struct TestCache {
    value: i32,
    fail_restore: bool,
    panic_clone_from: bool,
}

impl Clone for TestCache {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            fail_restore: self.fail_restore,
            panic_clone_from: self.panic_clone_from,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.value = source.value;
        assert!(
            !source.panic_clone_from,
            "injected failure after payload mutation"
        );
        self.fail_restore = source.fail_restore;
    }
}

impl TestCache {
    fn restore_value(&mut self, checkpoint: &Self) -> Result<(), Error> {
        self.value = checkpoint.value;
        if self.fail_restore {
            Err(Error::backend("injected failure after payload restore"))
        } else {
            Ok(())
        }
    }
}

impl RuntimeLayerState<MlxNeuralBackend> for TestCache {
    type RetainedValues<'a> = std::iter::Empty<&'a MlxTensor>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
}

impl CompressedAttentionCache<MlxTensor> for TestCache {
    type Checkpoint = Self;
    fn offset(&self) -> i32 {
        self.value
    }
    fn is_paged(&self) -> bool {
        false
    }
    fn append(
        &mut self,
        state: CompressedAttentionState<MlxTensor>,
        _: &Stream,
    ) -> Result<CompressedAttentionView<MlxTensor>, Error> {
        Ok(CompressedAttentionView::Resident(state))
    }
    fn visit_blocks<F>(
        &mut self,
        _: i32,
        _: &Stream,
        _: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<MlxTensor>) -> Result<u64, Error>,
    {
        Ok(CompressedAttentionScan::default())
    }
    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }
    fn restore(&mut self, checkpoint: &Self, _: &Stream) -> Result<(), Error> {
        self.restore_value(checkpoint)
    }
    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn clear(&mut self) -> Result<(), Error> {
        self.value = 0;
        Ok(())
    }
}

impl PoolingAttentionCache<MlxTensor> for TestCache {
    type Checkpoint = Self;
    fn offset(&self) -> i32 {
        self.value
    }
    fn pooling_ratio(&self, _: u32) -> Option<i32> {
        None
    }
    fn append_local(&mut self, keys: MlxTensor, _: &Stream) -> Result<MlxTensor, Error> {
        Ok(keys)
    }
    fn local_mask(&self, _: i32, _: i32, _: &Stream) -> Result<MlxTensor, Error> {
        Err(Error::backend("fixture has no local mask"))
    }
    fn accumulate_pooling_windows(
        &mut self,
        _: u32,
        values: MlxTensor,
        gates: MlxTensor,
        absolute_offset: i32,
        _: &Stream,
    ) -> Result<PoolingWindows<MlxTensor>, Error> {
        Ok(PoolingWindows {
            values,
            gates,
            base_position: absolute_offset,
        })
    }
    fn replace_pooling_overlap(
        &mut self,
        _: u32,
        values: MlxTensor,
        gates: MlxTensor,
    ) -> Result<PoolingOverlap<MlxTensor>, Error> {
        Ok(PoolingOverlap {
            values: Some(values),
            gates: Some(gates),
        })
    }
    fn append_pooled(&mut self, _: u32, values: MlxTensor, _: &Stream) -> Result<MlxTensor, Error> {
        Ok(values)
    }
    fn pooling_mask(&self, _: u32, _: i32, _: i32, _: &Stream) -> Result<Option<MlxTensor>, Error> {
        Ok(None)
    }
    fn checkpoint(&self) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn restore(&mut self, checkpoint: &Self, _: &Stream) -> Result<(), Error> {
        self.restore_value(checkpoint)
    }
    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn clear(&mut self) -> Result<(), Error> {
        self.value = 0;
        Ok(())
    }
}

fn owned(pool: &WorkingMemoryPool, value: i32) -> OwnedPredictionCache<TestCache> {
    let owner = NativeMemoryOwner::acquire(pool).unwrap();
    OwnedPredictionCache::new(
        TestCache {
            value,
            fail_restore: false,
            panic_clone_from: false,
        },
        NativeMemoryRetention::from_owner(&owner),
    )
}

#[test]
fn clone_from_keeps_installed_and_incoming_owners_when_payload_mutation_panics() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let mut installed = owned(&pool, 3);
    let mut incoming = owned(&pool, 7);
    incoming.value.panic_clone_from = true;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        installed.clone_from(&incoming);
    }));
    assert!(result.is_err());
    assert_eq!(installed.inner().value, 7);
    drop(incoming);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(installed);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn failing_restore_and_subsequent_clear_keep_both_checkpoint_owners() {
    type Cache = OwnedPredictionCache<TestCache>;
    type Restore = fn(&mut Cache, &Cache, &Stream) -> Result<(), Error>;
    let restore: [Restore; 2] = [
        <Cache as CompressedAttentionCache<MlxTensor>>::restore,
        <Cache as PoolingAttentionCache<MlxTensor>>::restore,
    ];
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for restore in restore {
        let pool = WorkingMemoryPool::new(4096, 0).unwrap();
        let mut installed = owned(&pool, 11);
        installed.value.fail_restore = true;
        let incoming = owned(&pool, 17);
        let checkpoint = <Cache as CompressedAttentionCache<MlxTensor>>::checkpoint(&incoming);
        let pooling_checkpoint =
            <Cache as PoolingAttentionCache<MlxTensor>>::checkpoint(&incoming).unwrap();
        drop(incoming);
        assert!(restore(&mut installed, &checkpoint, &stream).is_err());
        assert_eq!(installed.inner().value, 17);
        drop(checkpoint);
        drop(pooling_checkpoint);
        <Cache as CompressedAttentionCache<MlxTensor>>::clear(&mut installed).unwrap();
        <Cache as PoolingAttentionCache<MlxTensor>>::clear(&mut installed).unwrap();
        assert_eq!(installed.inner().value, 0);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
        drop(installed);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn independent_copy_retention_preserves_source_and_new_authority_with_priced_metadata() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let source = owned(&pool, 19);
    let new_owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let estimate = source.copy_ownership_metadata_bytes().unwrap();
    let copied =
        OwnedPredictionCache::new(source.inner().clone(), source.memory_for_copy(&new_owner));
    assert!(copied.ownership_metadata_bytes().unwrap() <= estimate);
    drop(source);
    drop(new_owner);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    let checkpoint = copied.clone();
    drop(copied);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(checkpoint);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}
