use super::*;
use crate::workspace::*;
use crate::Tensor;
use std::cell::Cell;
use std::error::Error as _;

thread_local! {
    static ALLOCATION: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
    static BUFFER_CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
}
pub(super) fn allocated(capacity: usize, pointer: *const f32) {
    ALLOCATION.set(Some((capacity, pointer as usize)));
    BUFFER_CONSTRUCTIONS.set(BUFFER_CONSTRUCTIONS.get() + 1);
}
fn reset() {
    ALLOCATION.set(None);
    BUFFER_CONSTRUCTIONS.set(0);
}

#[test]
fn fixed_worker_uses_one_exact_buffer_and_one_scalar_evaluation_per_element() {
    for (shape, count) in [(&[2, 3][..], 6), (&[][..], 1), (&[7, 0, 3][..], 0)] {
        reset();
        let calls = Cell::new(0);
        let plan = F32InitializationPlan::new(shape).unwrap();
        let result = initialize(
            shape,
            |index| {
                assert_eq!(index, calls.get());
                calls.set(index + 1);
                (index as f32 + 1.25) * -2.0
            },
            |values| {
                assert_eq!(values.len(), count);
                assert_eq!(ALLOCATION.get(), Some((count, values.as_ptr() as usize)));
                assert_eq!(plan.host_buffer_bytes(), (count * 4) as u64);
                assert_eq!(plan.elements(), count);
                assert_eq!(plan.shape(), shape);
                Ok(values.iter().copied().sum::<f32>())
            },
        )
        .unwrap();
        assert_eq!(calls.get(), count);
        assert_eq!(BUFFER_CONSTRUCTIONS.get(), 1);
        assert_eq!(
            result,
            (0..count)
                .map(|index| (index as f32 + 1.25) * -2.0)
                .sum::<f32>()
        );
    }
}

#[test]
fn invalid_geometry_rejects_before_allocation_scalar_or_realization() {
    for shape in [
        &[2, -1][..],
        &[0, -1][..],
        &[i32::MAX, i32::MAX, 8][..],
        &[i32::MAX, i32::MAX][..],
    ] {
        reset();
        let error = initialize::<()>(
            shape,
            |_| panic!("scalar ran"),
            |_| panic!("realization ran"),
        )
        .unwrap_err();
        assert!(error.source().unwrap().is::<F32InitializationError>());
        assert_eq!(BUFFER_CONSTRUCTIONS.get(), 0);
    }
    // Zero storage does not overflow because unrelated large extents precede it.
    assert_eq!(
        F32InitializationPlan::new(&[i32::MAX, i32::MAX, i32::MAX, 0])
            .unwrap()
            .host_buffer_bytes(),
        0
    );
}

#[derive(Debug, thiserror::Error)]
#[error("original realization failure")]
struct RealizationFailure;

#[test]
fn realization_failure_preserves_original_typed_cause() {
    reset();
    let error = initialize::<()>(
        &[17],
        |i| i as f32 + 0.5,
        |values| {
            assert_eq!(values.len(), 17);
            assert_eq!(ALLOCATION.get(), Some((17, values.as_ptr() as usize)));
            Err(Error::backend_retained_source(RealizationFailure))
        },
    )
    .unwrap_err();
    assert!(error.source().unwrap().is::<RealizationFailure>());
    assert_eq!(BUFFER_CONSTRUCTIONS.get(), 1);
}

#[derive(Debug)]
struct FixedTestMechanism {
    host: bool,
}
impl WorkspaceMechanisms for FixedTestMechanism {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        assert!(matches!(
            op.kind,
            WorkspaceOperationKind::GeneratedF32Initialization
        ));
        assert!(op.inputs.is_empty());
        assert_eq!(op.outputs.len(), 1);
        assert_eq!(op.outputs[0].dtype(), WorkspaceDtype::Float32);
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![WorkspaceOutputStorage::Allocate(op.outputs[0].bytes()?)],
            scratch_bytes: 0,
            assumptions: "test realization copies fixed buffer into exact independent F32 storage"
                .into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.then(|| WorkspaceHostBound {
            bytes: F32InitializationPlan::new(op.outputs[0].shape())
                .unwrap()
                .host_buffer_bytes(),
            assumptions: "test implementation uses the actual shared fixed-buffer worker".into(),
        }))
    }
}

#[test]
fn metadata_skips_values_and_sums_each_constructor_in_the_completed_span() {
    reset();
    let context = WorkspaceContext::new(FixedTestMechanism { host: true });
    let first =
        WorkspaceTensor::from_f32_fn(&[3], |_| panic!("cold scalar ran"), &context).unwrap();
    let second =
        WorkspaceTensor::from_f32_fn(&[3], |_| panic!("cold scalar ran"), &context).unwrap();
    let empty =
        WorkspaceTensor::from_f32_fn(&[0], |_| panic!("empty scalar ran"), &context).unwrap();
    let report = context
        .report(&[first.clone(), first, second, empty])
        .unwrap();
    assert_eq!(report.host_workspace_bytes, Some(24));
    assert_eq!(report.retained_bytes, Some(24));
    assert_eq!(report.total_bytes, Some(48));
    assert_eq!(report.operations.len(), 3);
    assert_eq!(BUFFER_CONSTRUCTIONS.get(), 0);
    // Dropping constructor outputs cannot reset possible construction overlap.
    assert_eq!(context.report(&[]).unwrap().host_workspace_bytes, Some(24));
    context.begin_span();
    assert_eq!(context.report(&[]).unwrap().host_workspace_bytes, Some(0));
}

#[test]
fn metadata_geometry_errors_and_missing_host_facts_do_not_become_complete() {
    let context = WorkspaceContext::new(FixedTestMechanism { host: false });
    let error =
        WorkspaceTensor::from_f32_fn(&[-1], |_| panic!("cold scalar ran"), &context).unwrap_err();
    assert!(error.source().unwrap().is::<F32InitializationError>());
    assert!(context.report(&[]).unwrap().operations.is_empty());
    let output =
        WorkspaceTensor::from_f32_fn(&[4], |_| panic!("cold scalar ran"), &context).unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(report.tensor_buffers.total_bytes, Some(16));
    assert_eq!(report.host_workspace_bytes, None);
    assert_eq!(report.total_bytes, None);
}
