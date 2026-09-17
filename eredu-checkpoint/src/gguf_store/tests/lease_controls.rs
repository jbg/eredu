use super::*;
use crate::store::{CheckpointLease, LeaseProvider};

#[test]
fn actual_gguf_lease_layout_uses_physical_selection_and_retains_lazy_store() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("packed.gguf");
    write_tensor(&path, "matrix.weight", &[32, 2], GgmlType::Q4_0, &[0; 36]);
    let store = test_store(&path);
    let source = store.source_lease_controls("matrix.weight").unwrap();
    assert_eq!(source.provider(), LeaseProvider::Gguf);
    let selection = TensorSelection::Indices {
        axis: 1,
        indices: vec![0, 1, 2, 3],
    };
    let lease = store
        .acquire_lease(TensorReadRequest {
            key: "matrix.weight".into(),
            selection,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let layout = lease.clone_control_layout().unwrap();
    let gguf = layout.gguf.unwrap();
    let CheckpointLease::Gguf(actual) = &lease else {
        panic!("GGUF lease")
    };
    let Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Indices { indices, .. })) =
        actual.identity().physical_selection()
    else {
        panic!("physical indices")
    };
    assert_eq!(indices.len(), 32); // four packed logical words expand to 32 native values
    assert_eq!(
        gguf.physical_selection.unwrap().size(),
        indices.len() * std::mem::size_of::<usize>()
    );
    assert_eq!(
        layout.selection.elements.unwrap().size(),
        4 * std::mem::size_of::<usize>()
    );
    assert_eq!(
        gguf.descriptor_dimensions.size(),
        2 * std::mem::size_of::<u64>()
    );
    assert_eq!(gguf.identity_name.size(), "matrix.weight".len());
    assert_eq!(gguf.physical_name.size(), "matrix.weight".len());
    assert!(layout.boxed_lease.is_some());
    assert!(!std::ptr::eq(lease.metadata(), source.metadata()));
    let weak = store.inner.ordinary_weak();
    let clone = lease.clone();
    assert_eq!(clone.clone_control_layout(), Some(layout));
    assert_ne!(clone.output_shape().as_ptr(), lease.output_shape().as_ptr());
    assert_eq!(store.diagnostics().unwrap().physical_reads, 0);
    std::fs::remove_file(path).unwrap(); // neither layout nor clone reads the deferred payload
    drop(store);
    drop(lease);
    assert!(weak.upgrade().is_some());
    drop(clone);
    assert!(weak.upgrade().is_none());
}

#[test]
fn actual_gguf_full_range_and_dense_span_layouts_match_nonzero_lazy_leases() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense.gguf");
    let data = [1.25_f32, -2.5, 3.75, 5.0, -6.25, 7.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    write_tensor(&path, "matrix.weight", &[3, 2], GgmlType::F32, &data);
    let store = test_store(&path);
    for selection in [
        TensorSelection::Full,
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        },
        TensorSelection::Contiguous {
            offset_elements: 1,
            shape: vec![2, 2],
        },
    ] {
        let lease = store
            .acquire_lease(TensorReadRequest {
                key: "matrix.weight".into(),
                selection,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let layout = lease.clone_control_layout().unwrap();
        let CheckpointLease::Gguf(actual) = &lease else {
            panic!("GGUF lease")
        };
        match actual.identity().physical_selection() {
            Some(GgufPhysicalSelection::DenseSpan(span)) => {
                assert_eq!(span.shape(), &[2, 2]);
                assert_eq!(
                    layout.gguf.unwrap().physical_selection.unwrap().size(),
                    2 * std::mem::size_of::<u64>()
                );
            }
            _ => assert!(layout.gguf.unwrap().physical_selection.is_none()),
        }
        assert_eq!(lease.clone().clone_control_layout(), Some(layout));
    }
    assert_eq!(store.diagnostics().unwrap().physical_reads, 0);
}
