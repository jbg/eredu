use super::super::preparation::ColdQuantization;
use super::*;
use eredu_checkpoint::store::{EncodedTensorLease, MemoryTensorBuffer};
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn later_target_working_set_failure_precedes_every_output_allocation() {
    let source = Arc::new(
        MemoryWeightStore::from_safetensors([
            ("a.weight".into(), SafeDtype::F32, vec![1, 64], vec![0; 256]),
            (
                "b.weight".into(),
                SafeDtype::F32,
                vec![1, 128],
                vec![0; 512],
            ),
        ])
        .unwrap(),
    );
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        296,
        [
            direct_test_target("a.weight"),
            direct_test_target("b.weight"),
        ],
    )
    .unwrap();
    let allocations = Cell::new(0);
    let result = ColdQuantization::prepare(source.into(), plan).and_then(|plan| {
        plan.allocate(cpu_context().stream(), |layout| {
            allocations.set(allocations.get() + 1);
            MemoryTensorBuffer::allocate(&layout.name, layout.dtype, &layout.shape, layout.byte_len)
                .map_err(|cause| Error::Other(Box::new(cause)))
        })
    });
    let Err(error) = result else {
        panic!("second target must exceed the working set")
    };
    assert!(error.to_string().contains("b.weight"));
    assert!(error
        .to_string()
        .contains("requires at least 592 working-set bytes"));
    assert_eq!(allocations.get(), 0);
}

#[test]
fn conversion_fills_the_original_destinations_and_readers_retain_their_custody() {
    let (_directory, source, _) = direct_fixture();
    let reads_before_preparation = source.source_diagnostics().unwrap().physical_reads;
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        320,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let cold = ColdQuantization::prepare(source.clone().into(), plan).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let custody = Arc::new(Custody(drops.clone()));
    let allocations = Cell::new(0);
    let original = Cell::new(std::ptr::null());
    let prepared = cold
        .allocate(cpu_context().stream(), |layout| {
            allocations.set(allocations.get() + 1);
            let mut buffer = MemoryTensorBuffer::allocate_with_custody(
                &layout.name,
                layout.dtype,
                &layout.shape,
                layout.byte_len,
                custody.clone(),
            )
            .map_err(|cause| Error::Other(Box::new(cause)))?;
            if layout.name == "model.proj.weight" {
                original.set(buffer.bytes_mut().as_ptr());
            }
            Ok(buffer)
        })
        .unwrap();
    drop(custody);
    assert_eq!(allocations.get(), 3);
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads_before_preparation
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    let context = cpu_context();
    let transformed = prepared.materialize(context.stream()).unwrap();
    assert_eq!(transformed.report().transformed_weights, 1);
    assert_eq!(transformed.report().output_bytes, 320);
    let lease = transformed.source()
        .acquire_lease(TensorReadRequest {
            key: "model.proj.weight".into(),
            selection: TensorSelection::Full,
            policy: WeightReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_eq!(lease.encoded_bytes().unwrap().as_ptr(), original.get());
    assert_eq!(lease.encoded_bytes().unwrap().len(), 256);
    assert!(lease.encoded_bytes().unwrap().iter().any(|&byte| byte != 0));
    drop(transformed);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(lease);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn output_allocation_failure_retires_prefix_and_retains_failed_constructor_custody() {
    let (_directory, source, _) = direct_fixture();
    let reads_before_preparation = source.source_diagnostics().unwrap().physical_reads;
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        320,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let cold = ColdQuantization::prepare(source.clone().into(), plan).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let custody = Arc::new(Custody(drops.clone()));
    let allocations = Cell::new(0);
    let result = cold.allocate(cpu_context().stream(), |layout| {
        let ordinal = allocations.get();
        allocations.set(ordinal + 1);
        let bytes = layout.byte_len - u64::from(ordinal == 2);
        MemoryTensorBuffer::allocate_with_custody(
            &layout.name,
            layout.dtype,
            &layout.shape,
            bytes,
            custody.clone(),
        )
        .map_err(|cause| Error::Other(Box::new(cause)))
    });
    let Err(error) = result else {
        panic!("third output has invalid byte geometry")
    };
    drop(custody);
    assert_eq!(allocations.get(), 3);
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads_before_preparation
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(error.to_string().contains("payload contradicts metadata"));
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

const ORIGINAL_METADATA_POLICY: eredu_runtime::working_memory::DependencyMemoryPolicy =
    eredu_runtime::working_memory::DependencyMemoryPolicy {
        fixed_bytes: 1024,
        bytes_per_input_byte: 8,
    };

fn original_output_quotes() -> Option<[u64; 3]> {
    use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};
    let mut quotes = [0; 3];
    for (index, (name, shape, bytes)) in [
        ("model.proj.weight", [8, 8], 256),
        ("model.proj.scales", [8, 1], 32),
        ("model.proj.biases", [8, 1], 32),
    ]
    .into_iter()
    .enumerate()
    {
        match WorkingMemoryPool::memory_tensor_buffer_quote(
            name,
            &shape,
            bytes,
            ORIGINAL_METADATA_POLICY,
        ) {
            Ok(quote) => quotes[index] = quote.total_bytes(),
            Err(WorkingMemoryError::UnknownBound)
                if std::env::var_os("EREDU_REQUIRE_QUALIFIED_MEMORY_TENSOR_SOURCE").is_none() =>
            {
                return None;
            }
            Err(cause) => panic!("original output quote: {cause}"),
        }
    }
    Some(quotes)
}

#[test]
fn conversion_preserves_the_pools_original_payload_inventory() {
    use eredu_runtime::working_memory::WorkingMemoryPool;
    let Some(quotes) = original_output_quotes() else {
        return;
    };
    let (_directory, source, _) = direct_fixture();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let pool = WorkingMemoryPool::new(quotes.iter().sum(), 0).unwrap();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        320,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let prepared = ColdQuantization::prepare(source.clone().into(), plan)
        .unwrap()
        .allocate_original(&pool, ORIGINAL_METADATA_POLICY, cpu_context().stream())
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), quotes.iter().sum::<u64>());
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    let owner = pool.acquire_unquoted().unwrap();
    let context = cpu_context();
    let transformed = prepared.materialize(context.stream()).unwrap();
    let mut inventories = Vec::new();
    assert!(transformed.source()
        .visit_source_storage(&mut |row| {
            inventories.push((row.identity(), row.bytes()));
        })
        .unwrap());
    assert_eq!(inventories.len(), 3);
    let mut capacities = inventories
        .iter()
        .map(|(_, bytes)| *bytes)
        .collect::<Vec<_>>();
    capacities.sort_unstable();
    assert_eq!(capacities, [32, 32, 256]);
    for (identity, bytes) in &inventories {
        pool.validate_original_source_inventory(identity, *bytes)
            .unwrap();
    }
    let lease = transformed.source()
        .acquire_lease(TensorReadRequest {
            key: "model.proj.weight".into(),
            selection: TensorSelection::Full,
            policy: WeightReadPolicy::RequireBounded,
        })
        .unwrap();
    assert!(lease.encoded_bytes().unwrap().iter().any(|&byte| byte != 0));
    drop((inventories, transformed, context, owner));
    assert_eq!(pool.used_bytes().unwrap(), quotes[0]);
    drop(lease);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_output_admission_failure_retires_the_allocated_prefix_without_reads() {
    use eredu_runtime::working_memory::{
        OriginalMemoryTensorError, WorkingMemoryError, WorkingMemoryPool,
    };
    let Some(quotes) = original_output_quotes() else {
        return;
    };
    let (_directory, source, _) = direct_fixture();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let pool = WorkingMemoryPool::new(quotes.iter().sum::<u64>() - 1, 0).unwrap();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        320,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let result = ColdQuantization::prepare(source.clone().into(), plan)
        .unwrap()
        .allocate_original(&pool, ORIGINAL_METADATA_POLICY, cpu_context().stream());
    let Err(Error::Other(error)) = result else {
        panic!("third output must fail original admission");
    };
    let error = error.downcast_ref::<OriginalMemoryTensorError>().unwrap();
    assert!(matches!(error.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
        if *required_bytes == quotes[2] && *available_bytes == quotes[2] - 1));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
}
