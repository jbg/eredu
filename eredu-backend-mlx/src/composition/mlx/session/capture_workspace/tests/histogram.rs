#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn complete_histogram_cold_worker_quotes_only_the_actual_producer() {
    use super::*;
    use crate::backend::nn::workspace::{
        MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms,
    };
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let source = source(
        &[2, 3],
        CaptureTransform::Histogram {
            edges: vec![-1.0, 0.25, 2.0],
        },
        vec![],
    );
    let mut usage = None;
    for local in [true, false] {
        let context = WorkspaceContext::new(cpu);
        let value = WorkspaceTensor::existing(
            context
                .layout(&[2, 3], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            &context,
        )
        .unwrap();
        let transfers = Cell::new(CaptureNativePopulation::default());
        let scalar = [Cell::new(None)];
        let (mut observer, _host) =
            CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
        // These are the same internal source loans selected by the partition
        // binding; this test targets the complete producer/remote worker split.
        observer.transfers = Some(&transfers);
        observer.scalar_source = Some(&scalar);
        // Advance the same unselected prefill as the actual observer; a fresh
        // owner cannot skip prediction zero and begin at decode one.
        begin(&mut observer, &context);
        context.begin_span();
        observer
            .observe_value_with_origin("block.output", &value, local)
            .unwrap();
        assert_eq!(scalar[0].get(), Some(WorkspaceFloatingType::Float32));
        assert_eq!(observer.roots.is_empty(), !local);
        if local {
            let report = context.report(&observer.roots).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(!report.operations.is_empty());
            usage = Some(observer.ledger.total());
        } else {
            assert_eq!(observer.ledger.total(), usage.unwrap());
            let transfer = transfers.get();
            assert_eq!(
                (
                    transfer.publications,
                    transfer.completions,
                    transfer.controls,
                    transfer.retained_roots
                ),
                (0, 0, 0, 0)
            );
            assert!(context.report(&[]).unwrap().operations.is_empty());
        }
    }
}
