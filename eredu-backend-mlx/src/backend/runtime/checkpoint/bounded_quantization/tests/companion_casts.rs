use super::*;
use eredu_checkpoint::store::EncodedTensorLease;
use std::{cell::Cell, sync::Mutex};

struct Source {
    store: MemoryWeightStore,
    reads: Mutex<Vec<u64>>,
}

impl CheckpointSource for Source {
    fn source_keys(&self) -> Vec<String> {
        self.store.source_keys()
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        self.store.source_metadata(key)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.store.source_diagnostics()
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        assert_eq!(request.policy, WeightReadPolicy::RequireBounded);
        let bytes = DerivedWeightRecipe::source(&request.key, request.selection.clone())
            .infer(self as &dyn CheckpointSource)
            .unwrap()
            .byte_len();
        self.reads.lock().unwrap().push(bytes);
        self.store.acquire_lease(request)
    }
}

fn fixture(shape: &[usize], dtype: SafeDtype) -> Arc<Source> {
    let values = (0..shape.iter().product()).map(|i: usize| (i % 16 + i / 64) as f32);
    let bytes = match dtype {
        SafeDtype::F32 => values.flat_map(f32::to_le_bytes).collect(),
        SafeDtype::F16 => values
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        SafeDtype::BF16 => values
            .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    };
    Arc::new(Source {
        store: MemoryWeightStore::from_safetensors([(
            "model.proj.weight".into(),
            dtype,
            shape.to_vec(),
            bytes,
        )])
        .unwrap(),
        reads: Mutex::new(Vec::new()),
    })
}

fn target(dtype: RecipeDtype) -> BoundedQuantizationTarget {
    direct_test_target("model.proj.weight")
        .with_affine_companion_dtype(dtype)
        .unwrap()
}

// One group of 64 values: 32 packed bytes, two final companions, plus
// both native source-precision companions whenever a cast is needed. F16
// and BF16 casts require separate storage even though their widths match.
const CASES: [(SafeDtype, RecipeDtype, u64); 9] = [
    (SafeDtype::F16, RecipeDtype::F16, 164),
    (SafeDtype::F16, RecipeDtype::BF16, 168),
    (SafeDtype::F16, RecipeDtype::F32, 172),
    (SafeDtype::BF16, RecipeDtype::F16, 168),
    (SafeDtype::BF16, RecipeDtype::BF16, 164),
    (SafeDtype::BF16, RecipeDtype::F32, 172),
    (SafeDtype::F32, RecipeDtype::F16, 300),
    (SafeDtype::F32, RecipeDtype::BF16, 300),
    (SafeDtype::F32, RecipeDtype::F32, 296),
];

#[test]
fn companion_cast_minimum_precedes_output_allocation_and_source_reads() {
    for (source_dtype, output_dtype, required) in CASES {
        let source = fixture(&[1, 64], source_dtype);
        for budget in [required - 1, required] {
            let plan = BoundedQuantizationPlan::new(
                AffineQuantization::default(),
                budget,
                [target(output_dtype.clone())],
            )
            .unwrap();
            let allocations = Cell::new(0);
            let result = ColdQuantization::prepare(source.clone().into(), plan).and_then(|cold| {
                cold.allocate(cpu_context().stream(), |layout| {
                    allocations.set(allocations.get() + 1);
                    eredu_checkpoint::store::MemoryTensorBuffer::allocate(
                        &layout.name,
                        layout.dtype,
                        &layout.shape,
                        layout.byte_len,
                    )
                    .map_err(|cause| Error::Other(Box::new(cause)))
                })
            });
            if budget < required {
                let Err(error) = result else {
                    panic!("missing cast allowance was accepted")
                };
                assert!(error
                    .to_string()
                    .contains(&format!("requires at least {required} working-set bytes")));
                assert_eq!(allocations.get(), 0);
            } else {
                let prepared = result.unwrap();
                assert_eq!(allocations.get(), 3);
                assert_eq!(prepared.output_shards[0].geometry.complete_peak, required);
                assert_eq!(prepared.output_shards[0].geometry.tile_buffers, 1);
            }
            assert!(source.reads.lock().unwrap().is_empty());
        }
    }
}

fn check_outputs(store: &BoundedQuantizedWeightStore, rows: usize, dtype: &RecipeDtype) {
    for (name, expected) in [
        (
            "model.proj.weight",
            [0x89abcdef_u32, 0x01234567]
                .repeat(rows * 4)
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            "model.proj.scales",
            encode(std::iter::repeat_n(-1.0, rows), dtype),
        ),
        (
            "model.proj.biases",
            encode((0..rows).map(|row| row as f32 + 15.0), dtype),
        ),
    ] {
        let lease = store
            .acquire_lease(TensorReadRequest {
                key: name.into(),
                selection: TensorSelection::Full,
                policy: WeightReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(lease.encoded_bytes().unwrap(), expected, "{name}");
    }
}

fn encode(values: impl Iterator<Item = f32>, dtype: &RecipeDtype) -> Vec<u8> {
    match dtype {
        RecipeDtype::F32 => values.flat_map(f32::to_le_bytes).collect(),
        RecipeDtype::F16 => values
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        RecipeDtype::BF16 => values
            .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

#[test]
fn companion_cast_tiles_preserve_precision_and_count_live_original_outputs() {
    let context = cpu_context();
    for (source_dtype, output_dtype, one_row_peak) in CASES {
        let source = fixture(&[8, 64], source_dtype);
        let source_row = (64 * source_dtype.bitsize() / 8) as u64;
        let target = target(output_dtype.clone());
        let final_row = 32 + 2 * target.affine_companion_bytes();
        let plan =
            BoundedQuantizationPlan::new(AffineQuantization::default(), one_row_peak * 2, [target])
                .unwrap();
        let result =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();
        assert_eq!(result.report().source_tiles, 8);
        assert_eq!(result.report().peak_in_flight_tiles, 2);
        assert_eq!(
            result.report().peak_planned_working_set_bytes,
            one_row_peak * 2
        );
        assert_eq!(result.report().output_bytes, 8 * final_row);
        assert_eq!(result.report().largest_output_tile_bytes, final_row);
        assert_eq!(*source.reads.lock().unwrap(), vec![source_row; 16]);
        check_outputs(&result, 8, &output_dtype);
    }
}

#[test]
fn companion_cast_sizing_covers_complete_leading_and_row_candidates() {
    let context = cpu_context();
    // F32 -> BF16: 256 source + 36 final + 8 original companion bytes per row.
    // Tight budgets distinguish the live casts from the encoded payload size.
    for (shape, budget, tiles, slots, read_bytes, peak) in [
        (vec![8, 64], 584, 8, 1, 256, 300),
        (vec![8, 64], 1168, 8, 2, 256, 600),
        (vec![8, 2, 64], 2336, 8, 2, 512, 1200),
        (vec![2, 2, 64], 1168, 4, 2, 256, 600),
        (vec![2, 64], 1168, 2, 2, 256, 600),
        (vec![2, 64], 1200, 1, 1, 512, 600),
    ] {
        let source = fixture(&shape, SafeDtype::F32);
        let rows = shape[..shape.len() - 1].iter().product();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            budget,
            [target(RecipeDtype::BF16)],
        )
        .unwrap();
        let result =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();
        assert_eq!(result.report().source_tiles, tiles, "{shape:?}, {budget}");
        assert_eq!(result.report().peak_in_flight_tiles, slots);
        assert_eq!(result.report().peak_planned_working_set_bytes, peak);
        assert_eq!(result.report().output_bytes, rows as u64 * 36);
        assert_eq!(*source.reads.lock().unwrap(), vec![read_bytes; tiles * 2]);
        check_outputs(&result, rows, &RecipeDtype::BF16);
    }
}
