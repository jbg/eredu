use super::*;
use eredu_checkpoint::store::EncodedTensorLease;
use std::sync::Mutex;

/// Refuses an oversized physical selection before the backing store acquires it.
struct LimitedReads {
    source: MemoryWeightStore,
    max_source_bytes: u64,
    requests: Mutex<Vec<u64>>,
    refuse_read: Option<usize>,
}

impl CheckpointSource for LimitedReads {
    fn source_keys(&self) -> Vec<String> {
        self.source.source_keys()
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        self.source.source_metadata(key)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        assert_eq!(request.policy, WeightReadPolicy::RequireBounded);
        let bytes = DerivedWeightRecipe::source(&request.key, request.selection.clone())
            .infer(self as &dyn CheckpointSource)
            .unwrap()
            .byte_len();
        let mut requests = self.requests.lock().unwrap();
        requests.push(bytes);
        if bytes > self.max_source_bytes || self.refuse_read == Some(requests.len()) {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: request.key,
                message: "physical selection exceeds the test source's read allowance".into(),
            });
        }
        self.source.acquire_lease(request)
    }
}

fn source(shape: &[usize], max_source_bytes: u64, refuse_read: Option<usize>) -> Arc<LimitedReads> {
    let values = (0..shape.iter().product())
        .map(|index: usize| (index % 64) as f32 / 4.0 - (index / 64) as f32 * 3.0)
        .collect::<Vec<_>>();
    Arc::new(LimitedReads {
        source: MemoryWeightStore::from_safetensors([(
            "model.proj.weight".into(),
            SafeDtype::F32,
            shape.to_vec(),
            float_bytes(&values),
        )])
        .unwrap(),
        max_source_bytes,
        requests: Mutex::new(Vec::new()),
        refuse_read,
    })
}

#[test]
fn rejected_row_and_leading_candidates_never_acquire_payloads() {
    let context = cpu_context();
    // Per row: 256 source bytes and 40 packed output bytes. Leading tiles
    // contain two rows; the final case admits the whole two-row target.
    for (shape, budget, max_source_bytes, tiles, slots) in [
        (vec![8, 64], 320, 256, 8, 1),
        (vec![8, 64], 640, 256, 8, 2),
        (vec![8, 2, 64], 1184, 512, 8, 2),
        (vec![2, 64], 1184, 512, 1, 1),
    ] {
        let source = source(&shape, max_source_bytes, None);
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            budget,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();
        assert_eq!(transformed.report().source_tiles, tiles);
        assert_eq!(transformed.report().peak_in_flight_tiles, slots);
        assert!(transformed.report().peak_planned_working_set_bytes <= budget);
        // One bounded proof and one materialization read per selected tile.
        // No rejected candidate or eager whole-plan probe reads the source.
        assert_eq!(
            *source.requests.lock().unwrap(),
            vec![max_source_bytes; tiles * 2]
        );
        let lease = transformed
            .acquire_lease(TensorReadRequest {
                key: "model.proj.weight".into(),
                selection: TensorSelection::Full,
                policy: WeightReadPolicy::RequireBounded,
            })
            .unwrap();
        assert!(lease
            .encoded_bytes()
            .unwrap()
            .iter()
            .any(|&value| value != 0));
    }
}

#[test]
fn bounded_read_refusal_survives_a_pending_conversion() {
    use crate::backend::runtime::checkpoint::recipe::WeightRecipeError;
    use eredu_checkpoint::recipe::RecipeError;
    // Reads 1 and 2 belong to the first tile. Read 3 verifies the second tile
    // and read 4 materializes it while the first submission is still pending.
    let context = cpu_context();
    for refused in [3, 4] {
        let source = source(&[8, 64], 256, Some(refused));
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            640,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let error = BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream())
            .unwrap_err();
        assert_eq!(*source.requests.lock().unwrap(), vec![256; refused]);
        let cause = match &error {
            Error::WeightRecipe(WeightRecipeError::Neutral(RecipeError::Store(cause)))
            | Error::WeightRecipe(WeightRecipeError::CheckpointStore(cause)) => cause,
            other => panic!("bounded read refusal must preserve its typed cause: {other:?}"),
        };
        assert!(matches!(
            cause,
            StoreError::BoundedSelectionUnavailable { key, .. } if key == "model.proj.weight"
        ));
    }
}
