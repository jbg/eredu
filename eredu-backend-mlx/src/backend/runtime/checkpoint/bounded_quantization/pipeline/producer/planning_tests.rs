use super::*;
use eredu_checkpoint::{
    AffineQuantization,
    recipe::RecipeInferenceCache,
    store::{EncodedReadBatch, EncodedTensorLease},
};
use safemlx::{Device, DeviceType};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Source {
    inner: MemoryWeightStore,
    producing: Arc<AtomicBool>,
    cache_calls: AtomicUsize,
}
impl CheckpointSource for Source {
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        assert!(
            self.producing.load(Ordering::SeqCst),
            "planning must not consult persistent inference"
        );
        self.cache_calls.fetch_add(1, Ordering::SeqCst);
        CheckpointSource::recipe_cache(&self.inner)
    }
    fn source_keys(&self) -> Vec<String> {
        self.inner.source_keys()
    }
    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        self.inner.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.inner.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.inner.source_diagnostics()
    }
    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        CheckpointSource::prepare_encoded_read(&self.inner, keys)
    }
}

struct Producer {
    inner: OrdinaryTileProducer,
    producing: Arc<AtomicBool>,
}
impl TileProducer for Producer {
    type Completion = WeightMaterialization;
    type Error = Error;
    fn submit(
        &mut self,
        source: &eredu_checkpoint::store::RetainedCheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<(WeightMaterialization, u64), Error> {
        self.producing.store(true, Ordering::SeqCst);
        let result = self
            .inner
            .submit(source, recipe, target, quantization, slot);
        self.producing.store(false, Ordering::SeqCst);
        result
    }
}

#[test]
fn shared_planning_avoids_source_cache_for_complete_leading_and_row_tiles() {
    let device = Device::new(DeviceType::Cpu, 0);
    let streams = [
        Stream::new_with_device(&device),
        Stream::new_with_device(&device),
    ];
    for (shape, budget, expected_tiles, expected_slots) in [
        (vec![8, 64], 320, 8, 1),
        (vec![8, 64], 640, 8, 2),
        (vec![4, 8, 64], 10_000, 2, 2),
        (vec![8, 64], 100_000, 1, 1),
    ] {
        let producing = Arc::new(AtomicBool::new(false));
        let elements = shape.iter().product::<usize>();
        let source = Arc::new(Source {
            inner: MemoryWeightStore::from_safetensors([(
                "weight".into(),
                SafeDtype::F32,
                shape,
                (0..elements)
                    .flat_map(|n| ((n % 16) as f32).to_le_bytes())
                    .collect(),
            )])
            .unwrap(),
            producing: producing.clone(),
            cache_calls: AtomicUsize::new(0),
        });
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::new(32, 4).unwrap(),
            budget,
            [
                BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
                    .unwrap()
                    .with_affine_companion_dtype(RecipeDtype::F16)
                    .unwrap(),
            ],
        )
        .unwrap();
        let source_dyn: Arc<dyn CheckpointSource> = source.clone();
        let prepared = ColdQuantization::prepare(source_dyn.into(), plan)
            .unwrap()
            .allocate_ordinary(&streams[0])
            .unwrap();
        assert_eq!(source.cache_calls.load(Ordering::SeqCst), 0);
        let mut producer = Producer {
            inner: OrdinaryTileProducer(
                streams
                    .each_ref()
                    .map(|s| MlxParameterMaterializationContext::new(s, s)),
            ),
            producing,
        };
        let (result, _) = prepared
            .materialize_with_producer(DeviceType::Cpu, &mut producer)
            .unwrap();
        assert!(source.cache_calls.load(Ordering::SeqCst) > 0);
        assert_eq!(result.report().source_tiles, expected_tiles);
        assert_eq!(result.report().peak_in_flight_tiles, expected_slots);
        assert_eq!(result.report().source_bytes_read, (elements * 4) as u64);
        let words = (0..elements / 8)
            .flat_map(|n| {
                (if n % 2 == 0 {
                    0x89abcdef_u32
                } else {
                    0x01234567_u32
                })
                .to_le_bytes()
            })
            .collect::<Vec<_>>();
        let scales = (0..elements / 32)
            .flat_map(|_| half::f16::from_f32(-1.0).to_le_bytes())
            .collect::<Vec<_>>();
        let biases = (0..elements / 32)
            .flat_map(|_| half::f16::from_f32(15.0).to_le_bytes())
            .collect::<Vec<_>>();
        for (key, expected) in [("weight", words), ("scales", scales), ("biases", biases)] {
            let lease = result.source()
                .acquire_lease(TensorReadRequest {
                    key: key.into(),
                    selection: TensorSelection::Full,
                    policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
                })
                .unwrap();
            assert_eq!(lease.encoded_bytes().unwrap(), expected);
        }
    }
}
