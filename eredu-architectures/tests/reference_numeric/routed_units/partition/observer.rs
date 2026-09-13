//! Actual prepared executors use the shared capture transaction and receipt transport.
use super::*;
use eredu_core::{capture::*, *};
use eredu_runtime::capture::{partition::*, CaptureSession};
#[path = "observer/collector.rs"]
mod collector;
#[path = "observer/transport.rs"]
mod transport;
pub(super) use transport::{Transport, World};

pub(super) fn run(
    executable: &mut NumericPartitionExecutable,
    layouts: &eredu_architectures::component_partition::ComponentPartitionLayouts,
    descriptor: &ArchitectureDescriptor,
    transport: &Transport,
    inputs: &[NumericTensor],
    expected: Option<&[(NumericTensor, Values, Values)]>,
    fault: Option<&'static str>,
) -> CaptureUsage {
    executable.reset().unwrap();
    let mut session = CaptureSession::new(placement::plan(descriptor));
    session
        .configure_partition_capture(
            PartitionCaptureIdentity::new(
                "prepared-numeric-fixture".into(),
                "retained-routed-selection".into(),
                "capture-conformance".into(),
                None,
                layouts.topology().world_size(),
            )
            .unwrap(),
        )
        .unwrap();
    let mut last = CaptureUsage::default();
    for (prediction, tokens) in inputs.iter().enumerate() {
        let before = executable.positions().unwrap();
        let calls = Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)));
        let backend = collector::Collector {
            fault,
            calls: calls.clone(),
        };
        let mut observer = PartitionCaptureObserver::for_step(
            &mut session,
            backend,
            transport,
            layouts,
            prediction as u64,
            PartitionCaptureReceiptLimits {
                max_record_bytes: 16384,
                ..placement::limits()
            },
            |_: &[u64],
             _: &CaptureSelection,
             _: &ResolvedCaptureSlice|
             -> Result<PartitionCaptureNativeEstimate, CaptureError> {
                panic!("dense estimate")
            },
            |error: PartitionCaptureObserverError<Error>| Error::backend_source(error),
        );
        let output = executable.forward_observed(tokens, prediction == 0, &mut observer);
        drop(observer);
        assert_eq!(
            output.is_ok(),
            fault.is_none(),
            "actual shared prepared observer {fault:?}: {output:?}"
        );
        let step=session.take_step().unwrap_or_else(||panic!("shared transaction did not settle capture {fault:?}: {output:?}, source/collector calls {:?}", calls.iter().map(|n|n.load(Ordering::SeqCst)).collect::<Vec<_>>()));
        assert!(step.cumulative_usage.host_bytes > last.host_bytes);
        last = step.cumulative_usage;
        if fault.is_some() {
            let error = output.as_ref().unwrap_err();
            if calls[2].load(Ordering::SeqCst) > 0 {
                assert_unit_source(error);
            }
            if matches!(fault, Some("source-active" | "source-idle")) {
                assert_eq!(
                    calls[1].load(Ordering::SeqCst),
                    0,
                    "failed source must not enter collection"
                );
            }
            assert!(
                !error.to_string().contains("deadline")
                    && !error.to_string().contains("missing capture participant"),
                "injected source failure must settle all prepaid votes: {error:?}"
            );
            assert_eq!(
                executable.positions().unwrap(),
                before,
                "capture failure rolls back prepared model state"
            );
            assert!(step.records.iter().all(|r| r.payload.is_none()));
            assert!(step.partitions.is_empty());
            break;
        }
        if let Some(expected) = expected {
            let output = output.as_ref().unwrap();
            let reference = &expected[prediction].0;
            assert_eq!(output.shape, reference.shape);
            assert!(
                output
                    .data
                    .iter()
                    .zip(&reference.data)
                    .all(|(actual, expected)| (actual - expected).abs() < 2e-5),
                "capture preserves ordinary logits within 2e-5"
            );
        }
        assert_eq!(step.records.len(), descriptor.routed_components.len() * 2);
        assert_eq!(step.partitions.len(), step.records.len());
        for (index, record) in step.records.iter().enumerate() {
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            let Some(CapturePayload::RoutedUnits(payload)) = &record.payload else {
                panic!("sparse payload")
            };
            assert_eq!(
                payload.rows.len(),
                tokens.data.len() * payload.geometry.routes_per_token as usize
            );
            let component = &descriptor.routed_components[index / 2];
            for row in &payload.rows {
                assert_eq!(
                    row.source_peer,
                    (layouts.topology().expert() > 1).then_some(0)
                );
                let TensorObservationData::F32(data) = row.values.data() else {
                    panic!("float values")
                };
                assert_eq!(row.unit_start, 1);
                assert_eq!(row.unit_stride, 2);
                assert_eq!(data.len(), component.units_per_expert / 2);
                if let Some(expected) = expected {
                    let reference = if index % 2 == 0 {
                        &expected[prediction].1
                    } else {
                        &expected[prediction].2
                    };
                    for (offset, value) in data.iter().enumerate() {
                        let key = (
                            component.routing.clone(),
                            None,
                            row.token as usize,
                            row.slot as usize,
                            row.expert as usize,
                            1 + 2 * offset,
                        );
                        assert!(
                            (value - reference[&key]).abs() < 5e-6,
                            "selected record {key:?}: {value}/{}",
                            reference[&key]
                        );
                    }
                }
            }
        }
    }
    last
}
