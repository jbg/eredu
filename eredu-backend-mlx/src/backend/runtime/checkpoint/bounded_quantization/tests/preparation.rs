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
        plan.allocate(|layout| {
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
        .allocate(|layout| {
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

    // Conversion context creation follows complete destination construction.
    let context = cpu_context();
    let transformed = prepared.materialize(context.stream()).unwrap();
    assert_eq!(transformed.report().transformed_weights, 1);
    assert_eq!(transformed.report().output_bytes, 320);
    let lease = transformed
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
    let result = cold.allocate(|layout| {
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
