use super::*;
use eredu_nn::Tensor;

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16_384 },
        sdpa_blocks: None,
    }
}
fn operation(dtype: WorkspaceDtype) -> WorkspaceOperation {
    WorkspaceOperation {
        kind: WorkspaceOperationKind::ParameterPlaceholder,
        inputs: vec![],
        outputs: vec![WorkspaceLayout::new(&[], dtype).unwrap()],
    }
}

#[test]
fn unloaded_seed_fact_is_dtype_sized_and_has_no_disjoint_host_staging() {
    let mechanisms = mechanisms();
    for dtype in [
        WorkspaceDtype::Float32,
        WorkspaceDtype::Int32,
        WorkspaceDtype::Uint32,
        WorkspaceDtype::Uint8,
        WorkspaceDtype::Bool,
    ] {
        let op = operation(dtype);
        let bound = mechanisms.operation_bound(&op).unwrap().unwrap();
        let [WorkspaceOutputStorage::Allocate(bytes)] = bound.outputs.as_slice() else {
            panic!("each constructor owns a fresh scalar seed")
        };
        assert_eq!(
            *bytes,
            mechanisms
                .allocation
                .buffer_capacity(dtype.bytes())
                .unwrap()
        );
        assert_eq!(bound.scratch_bytes, 0);
        assert_eq!(
            mechanisms.host_workspace_bound(&op).unwrap().unwrap().bytes,
            0
        );
    }
    let context = WorkspaceContext::new(mechanisms);
    let huge = WorkspaceTensor::unloaded_f32(&[65_536, 65_536], &context).unwrap();
    let empty = WorkspaceTensor::unloaded_i32(&[0, 65_536], &context).unwrap();
    let _clone = huge.clone();
    assert_eq!(huge.shape(), [65_536, 65_536]);
    assert_eq!(empty.shape(), [0, 65_536]);
    let report = context.report(&[]).unwrap();
    assert_eq!(report.operations.len(), 2);
    assert_eq!(
        report.total_bytes,
        Some(2 * mechanisms.allocation.buffer_capacity(4).unwrap())
    );
    assert_eq!(report.host_workspace_bytes, Some(0));
}

#[test]
fn invalid_placeholder_descriptors_cannot_price_logical_weight_materialization() {
    let mechanisms = mechanisms();
    let mut invalid = Vec::new();
    let mut op = operation(WorkspaceDtype::Float32);
    op.outputs[0] = WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap();
    invalid.push(op);
    let mut op = operation(WorkspaceDtype::Float32);
    op.inputs.push(op.outputs[0].clone());
    invalid.push(op);
    let mut op = operation(WorkspaceDtype::Float32);
    op.outputs.clear();
    invalid.push(op);
    let mut op = operation(WorkspaceDtype::Float32);
    op.outputs.push(op.outputs[0].clone());
    invalid.push(op);
    for op in invalid {
        assert!(mechanisms.operation_bound(&op).is_err());
        assert!(mechanisms.host_workspace_bound(&op).is_err());
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_unloaded_parameter_construction_fits_scalar_bound_without_evaluating_weights() {
    use crate::backend::nn::module::PhysicalParam;
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for (dtype, metadata) in [
        (Dtype::Float32, WorkspaceDtype::Float32),
        (Dtype::Int32, WorkspaceDtype::Int32),
        (Dtype::Uint32, WorkspaceDtype::Uint32),
        (Dtype::Uint8, WorkspaceDtype::Uint8),
    ] {
        for shape in [&[8_192, 8_192][..], &[0, 8_192][..]] {
            let before = safemlx::memory::active_memory().unwrap();
            safemlx::memory::reset_peak_memory().unwrap();
            let a = PhysicalParam::<Array>::unloaded(shape, dtype, &stream).unwrap();
            let b = PhysicalParam::<Array>::unloaded(shape, dtype, &stream).unwrap();
            let observed = safemlx::memory::peak_memory()
                .unwrap()
                .saturating_sub(before) as u64;
            let allowed = 2 * selected
                .allocation
                .buffer_capacity(metadata.bytes())
                .unwrap();
            assert_eq!(a.as_ref().shape(), shape);
            assert_eq!(b.as_ref().dtype(), dtype);
            assert!(
                observed > 0,
                "constructor must include eager scalar allocation"
            );
            assert!(
                observed <= allowed,
                "{dtype:?} {shape:?}: observed {observed}, bound {allowed}"
            );
            eprintln!(
                "parameter-placeholder dtype={dtype:?} shape={shape:?} observed={observed} bound={allowed}"
            );
            // Neither placeholder is evaluated: strict parameter binding replaces
            // the lazy full-weight result. Destruction releases both seed graphs.
            drop((a, b));
            assert_eq!(safemlx::memory::active_memory().unwrap(), before);
        }
    }
}
