//! Exact immutable memory bytes in the same encoded destination worker.
use super::*;
mod plan;
pub use plan::{
    MemoryEncodedReadBuildError, MemoryEncodedReadPlan, MemoryEncodedReadPlanError,
    MemoryEncodedReadRouteError, PreparedMemoryEncodedRead,
};
#[derive(Clone)]
pub(super) struct ReadMemory {
    pub(super) tensor: storage::SourceHandle<MemoryTensor>,
    pub(super) spans: Vec<ReadSpan>,
}
impl ReadMemory {
    pub(super) fn matches(&self, other: &Self) -> bool {
        self.tensor.same(&other.tensor)
            && self.spans.len() == other.spans.len()
            && self
                .spans
                .iter()
                .zip(&other.spans)
                .all(|(a, b)| a.source == b.source && a.destination == b.destination)
    }
    pub(super) fn validate(&self, bytes: usize) -> Option<()> {
        for span in &self.spans {
            let start = usize::try_from(span.source.start).ok()?;
            let end = usize::try_from(span.source.end).ok()?;
            if end < start
                || end > self.tensor.bytes.len()
                || span.destination.end < span.destination.start
                || span.destination.end > bytes
                || end - start != span.destination.len()
            {
                return None;
            }
        }
        Some(())
    }
    pub(super) fn copy_into(&self, destination: &mut [u8]) -> Result<(), EncodedReadFailure> {
        for span in &self.spans {
            let start = usize::try_from(span.source.start).ok();
            let end = usize::try_from(span.source.end).ok();
            let source = start
                .zip(end)
                .and_then(|(start, end)| self.tensor.bytes.get(start..end));
            let output = destination.get_mut(span.destination.clone());
            let (Some(source), Some(output)) = (source, output) else {
                return Err(EncodedReadFailure {
                    batch: None,
                    shard: None,
                    completed_shards: 0,
                    cause: EncodedReadFailureCause::Changed,
                });
            };
            if source.len() != output.len() {
                return Err(EncodedReadFailure {
                    batch: None,
                    shard: None,
                    completed_shards: 0,
                    cause: EncodedReadFailureCause::Changed,
                });
            }
            output.copy_from_slice(source);
        }
        Ok(())
    }
}
pub(in crate::store) fn prepare(
    store: &MemoryWeightStore,
    keys: &[String],
) -> Result<EncodedReadBatch, StoreError> {
    let plan =
        MemoryEncodedReadPlan::new(store, keys).map_err(|error| error.into_ordinary(keys))?;
    let prepared = plan.construct(()).map_err(|error| {
        StoreError::Internal(format!(
            "memory encoded read metadata reserve failed: {error}"
        ))
    })?;
    Ok(prepared.batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::{DerivedWeightRecipe, EncodedRecipeRead};
    fn store() -> MemoryWeightStore {
        MemoryWeightStore::from_safetensors([(
            "packed".into(),
            Dtype::U32,
            vec![3, 2],
            [0x76543210u32, 0xfedcba98, 11, 23, 37, 41]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )])
        .unwrap()
    }
    #[test]
    fn memory_encoded_projection_detaches_exact_owner_without_file_or_staging() {
        let store = store();
        let alive = store.tensors.get("packed").unwrap().ordinary_weak();
        let selection = TensorSelection::Indices {
            axis: 0,
            indices: vec![2, 0, 2],
        };
        let lease = store
            .acquire_lease(TensorReadRequest {
                key: "packed".into(),
                selection: selection.clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let expected = lease.encoded_bytes().unwrap().to_vec();
        drop(lease);
        let recipe = DerivedWeightRecipe::source("packed", selection);
        let read = recipe.prepare_encoded_read(&store).unwrap().unwrap();
        assert!(read.admitted_batch().shards.is_empty());
        assert!(read.admitted_batch().telemetry.is_none());
        let custody = Arc::new(());
        let funded = Arc::downgrade(&custody);
        let plan = EncodedRecipeRead::prepare_detached(std::iter::once(&read)).unwrap();
        let quoted = plan.read_layout::<Arc<()>>().unwrap().required_bytes();
        assert!(plan.required_bytes::<Arc<()>>().unwrap() > 0);
        let detached = plan.construct(custody).unwrap();
        assert!(quoted >= detached.read_layout().unwrap().required_bytes());
        assert!(detached.matches_read(0, &read));
        let other = MemoryWeightStore::from_safetensors([(
            "packed".into(),
            Dtype::U32,
            vec![3, 2],
            [0x76543210u32, 0xfedcba98, 11, 23, 37, 41]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )])
        .unwrap();
        let other_read = recipe.prepare_encoded_read(&other).unwrap().unwrap();
        assert!(!detached.matches_read(0, &other_read));
        let mut ordinary = vec![0; expected.len()];
        EncodedRecipeRead::read_many_borrowed_into(std::iter::once(&read), &mut [&mut ordinary])
            .unwrap();
        assert_eq!(ordinary, expected);
        drop((read, store, other, other_read));
        assert!(alive.upgrade().is_some());
        let mut output = vec![0; expected.len()];
        detached.read_many_into(&mut [&mut output]).unwrap();
        assert_eq!(output, expected);
        assert_eq!(detached.physical_read_bytes(0), None);
        detached.visit_payload_paths(|_, _| panic!("memory source has no file"));
        let mut short = vec![99; expected.len() - 1];
        let error = detached.read_many_into(&mut [&mut short]).unwrap_err();
        assert!(matches!(
            error.cause().cause,
            EncodedReadFailureCause::DestinationLengths
        ));
        assert!(short.iter().all(|byte| *byte == 99));
        drop(detached);
        assert!(alive.upgrade().is_none());
        assert!(funded.upgrade().is_some());
        drop(error);
        assert!(funded.upgrade().is_none());
    }
}
