use super::*;
use eredu_checkpoint::store::{EncodedTensorLease, MemoryTensorBuffer};
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Source {
    store: MemoryWeightStore,
    reads: AtomicUsize,
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
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.store.acquire_lease(request)
    }
}

fn fixture(dtype: SafeDtype, shape: &[usize]) -> Arc<Source> {
    let codebook = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let values = codebook.into_iter().cycle().take(shape.iter().product());
    let bytes = match dtype {
        SafeDtype::F16 => values
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        SafeDtype::BF16 => values
            .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
            .collect(),
        SafeDtype::F32 => values.flat_map(f32::to_le_bytes).collect(),
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
        reads: AtomicUsize::new(0),
    })
}

fn plan(bytes: u64) -> BoundedQuantizationPlan {
    BoundedQuantizationPlan::new(
        WeightQuantization::MxFp4,
        bytes,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap()
}

#[test]
fn cpu_mxfp4_minimum_is_checked_before_output_allocation_or_source_reads() {
    let context = cpu_context();
    for (dtype, minimum) in [
        (SafeDtype::F16, 5372),
        (SafeDtype::BF16, 5372),
        (SafeDtype::F32, 9988),
    ] {
        let source = fixture(dtype, &[8, 64]);
        for budget in [minimum - 1, minimum] {
            let allocations = Cell::new(0);
            let result = ColdQuantization::prepare(source.clone().into(), plan(budget))
                .and_then(|cold| {
                    cold.allocate(context.stream(), |layout| {
                        allocations.set(allocations.get() + 1);
                        MemoryTensorBuffer::allocate(
                            &layout.name,
                            layout.dtype,
                            &layout.shape,
                            layout.byte_len,
                        )
                        .map_err(|cause| Error::Other(Box::new(cause)))
                    })
                });
            if budget < minimum {
                let Err(error) = result else {
                    panic!("insufficient CPU workspace accepted")
                };
                assert!(error
                    .to_string()
                    .contains(&format!("requires at least {minimum} working-set bytes")));
                assert_eq!(allocations.get(), 0);
            } else {
                assert!(result.is_ok());
                assert_eq!(allocations.get(), 2);
            }
            assert_eq!(source.reads.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn cpu_mxfp4_public_allocators_qualify_before_original_pool_admission() {
    use eredu_runtime::working_memory::{DependencyMemoryPolicy, WorkingMemoryError, WorkingMemoryPool};
    let context = cpu_context();
    let policy = DependencyMemoryPolicy {
        fixed_bytes: 1024,
        bytes_per_input_byte: 8,
    };
    let output_quote = [("model.proj.weight", [8, 8], 256), ("model.proj.scales", [8, 2], 16)]
        .into_iter()
        .map(|(name, shape, bytes)| {
            WorkingMemoryPool::memory_tensor_buffer_quote(name, &shape, bytes, policy)
                .map(|quote| quote.total_bytes())
        })
        .collect::<Result<Vec<_>, _>>();
    let output_quote = match output_quote {
        Ok(quotes) => quotes.into_iter().sum(),
        Err(WorkingMemoryError::UnknownBound)
            if std::env::var_os("EREDU_REQUIRE_QUALIFIED_MEMORY_TENSOR_SOURCE").is_none() =>
        {
            return;
        }
        Err(cause) => panic!("original output quote: {cause}"),
    };
    for (dtype, minimum) in [
        (SafeDtype::F16, 5372),
        (SafeDtype::BF16, 5372),
        (SafeDtype::F32, 9988),
    ] {
        let source = fixture(dtype, &[8, 64]);
        for original in [false, true] {
            for budget in [minimum - 1, minimum] {
                // A destination constructor would fail admission at zero. The
                // workspace refusal must win before any such constructor runs.
                let pool = WorkingMemoryPool::new(
                    if budget < minimum { 0 } else { output_quote },
                    0,
                )
                .unwrap();
                let cold = ColdQuantization::prepare(source.clone().into(), plan(budget)).unwrap();
                let result = if original {
                    cold.allocate_original(&pool, policy, context.stream())
                } else {
                    cold.allocate_ordinary(context.stream())
                };
                if budget < minimum {
                    let Err(error) = result else {
                        panic!("insufficient CPU workspace accepted")
                    };
                    assert!(error
                        .to_string()
                        .contains(&format!("requires at least {minimum} working-set bytes")));
                } else {
                    let prepared = result.unwrap();
                    assert_eq!(pool.used_bytes().unwrap(), if original { output_quote } else { 0 });
                    drop(prepared);
                }
                assert_eq!(pool.used_bytes().unwrap(), 0);
                assert_eq!(source.reads.load(Ordering::SeqCst), 0);
            }
        }
    }
}

#[test]
fn later_cpu_mxfp4_target_is_qualified_before_any_destination_allocation() {
    let context = cpu_context();
    let source = Arc::new(Source {
        store: MemoryWeightStore::from_safetensors([
            ("a.weight".into(), SafeDtype::F32, vec![1, 32], vec![0; 128]),
            ("b.weight".into(), SafeDtype::F32, vec![1, 64], vec![0; 256]),
        ])
        .unwrap(),
        reads: AtomicUsize::new(0),
    });
    let plan = BoundedQuantizationPlan::new(
        WeightQuantization::MxFp4,
        5070,
        [direct_test_target("a.weight"), direct_test_target("b.weight")],
    )
    .unwrap();
    let allocations = Cell::new(0);
    let result = ColdQuantization::prepare(source.clone().into(), plan)
        .unwrap()
        .allocate(context.stream(), |layout| {
            allocations.set(allocations.get() + 1);
            MemoryTensorBuffer::allocate(&layout.name, layout.dtype, &layout.shape, layout.byte_len)
                .map_err(|cause| Error::Other(Box::new(cause)))
        });
    let Err(error) = result else {
        panic!("second target must exceed the CPU workspace")
    };
    assert!(error.to_string().contains("b.weight"));
    assert!(error.to_string().contains("requires at least 9988 working-set bytes"));
    assert_eq!(allocations.get(), 0);
    assert_eq!(source.reads.load(Ordering::SeqCst), 0);
}

#[test]
fn cpu_mxfp4_tiled_payloads_preserve_the_independent_codebook() {
    let context = cpu_context();
    for (dtype, one_row, two_rows) in [
        (SafeDtype::F16, 5372, 10568),
        (SafeDtype::BF16, 5372, 10568),
        (SafeDtype::F32, 9988, 19824),
    ] {
        for (shape, minimum, rows) in [(vec![8, 64], one_row, 8), (vec![8, 2, 64], two_rows, 16)] {
            let source = fixture(dtype, &shape);
            let converted = BoundedQuantizedWeightStore::create(
                source.clone(),
                plan(2 * minimum),
                context.stream(),
            )
            .unwrap();
            assert_eq!(converted.report().source_tiles, 8);
            assert_eq!(converted.report().peak_in_flight_tiles, 2);
            assert_eq!(
                converted.report().peak_planned_working_set_bytes,
                2 * minimum
            );
            assert_eq!(converted.report().output_bytes, rows as u64 * 34);
            assert_eq!(source.reads.load(Ordering::SeqCst), 16);
            // Each group has maximum magnitude 6, giving exponent zero/E8M0 127.
            // ArgMin's first match encodes both signed zeroes with code zero.
            for (key, expected) in [
                (
                    "model.proj.weight",
                    [0x76543210_u32, 0xfedcba90]
                        .repeat(rows * 4)
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
                ("model.proj.scales", vec![127; rows * 2]),
            ] {
                let lease = converted
                    .acquire_lease(TensorReadRequest {
                        key: key.into(),
                        selection: TensorSelection::Full,
                        policy: WeightReadPolicy::RequireBounded,
                    })
                    .unwrap();
                assert_eq!(lease.encoded_bytes().unwrap(), expected);
            }
        }
    }
}

#[test]
fn stream_profile_cannot_borrow_direct_quantizer_geometry_for_cpu_fallback() {
    use super::super::workspace::QuantizerWorkspace;
    let context = cpu_context();
    let source = fixture(SafeDtype::F32, &[8, 64]);
    let mut unqualified = ColdQuantization::prepare(source.clone().into(), plan(9988))
        .unwrap()
        .allocate_ordinary(context.stream())
        .unwrap();
    // Emulate a destination prepared for a different native quantizer profile.
    unqualified.workspace = QuantizerWorkspace::Direct;
    let error = unqualified.materialize(context.stream()).unwrap_err();
    assert!(error
        .to_string()
        .contains("preparation for the selected stream"));
    assert_eq!(source.reads.load(Ordering::SeqCst), 0);
    let direct = QuantizerWorkspace::selected(WeightQuantization::MxFp4, DeviceType::Gpu);
    let payload = direct.payload(&RecipeDtype::F32, 64).unwrap();
    assert_eq!((payload.row_bytes, payload.fixed_bytes), (0, 0));
    assert_eq!(direct.maximum_submission_elements(), i32::MAX as usize);
    let cpu = QuantizerWorkspace::selected(WeightQuantization::MxFp4, DeviceType::Cpu);
    assert_eq!(cpu.maximum_submission_elements(), i32::MAX as usize / 16);
    assert!(cpu.payload(&RecipeDtype::F32, 0).is_err());
    assert!(cpu.payload(&RecipeDtype::F32, usize::MAX - 31).is_err());
}
