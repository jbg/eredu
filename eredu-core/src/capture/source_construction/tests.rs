use super::*;
use crate::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
struct Account {
    calls: Arc<AtomicUsize>,
    retired: Arc<AtomicBool>,
    stop: usize,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            n <= self.stop,
            "source construction continued after refusal"
        );
        if n == self.stop {
            Err(HostMetadataFundingError::Unavailable)
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
fn funding(stop: usize) -> (HostMetadataFunding, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        calls: calls.clone(),
        retired: retired.clone(),
        stop,
    })
    .unwrap();
    (funding, calls, retired)
}
fn fixture() -> CaptureDiscovery {
    CaptureDiscovery {
        artifact_identity: "exact-retained-identity".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Partial(vec!["omission".into()]),
            points: vec![ObservationPoint {
                path: "model.values".into(),
                node_id: "model".into(),
                meaning: "actual values".into(),
                value_type: ObservationValueType::Tensor,
                dtype: ObservationDtype::Floating,
                axes: Some(vec![TensorAxis {
                    name: "sequence".into(),
                    dimension: SymbolicDimension::Sequence,
                }]),
                prefill: true,
                decode: false,
                requirements: vec![ObservationRequirement::ActivationHooks],
                position: ObservationPosition::BeforeIntervention,
                retained_bytes: Some(16),
                host_bytes: Some(16),
            }],
        },
        support: ObservationSupportReport {
            schema_version: 1,
            points: vec![ObservationSupport {
                path: "model.values".into(),
                prefill: ObservationSupportStatus::Conditional("input".into()),
                decode: ObservationSupportStatus::Unsupported("phase".into()),
                floating_to_f32: true,
            }],
            capture: CaptureCapabilities {
                transformations: vec![CaptureTransformKind::Preview, CaptureTransformKind::Summary],
                max_histogram_bins: 32,
                conditions: vec!["selected worker".into()],
            },
        },
    }
}
fn copy(
    source: &CaptureDiscovery,
    construction: CaptureSourceConstruction<'_>,
) -> Result<CaptureDiscovery, CaptureError> {
    Ok(CaptureDiscovery {
        artifact_identity: construction.text(&source.artifact_identity)?,
        catalog: construction.catalog(&source.catalog)?,
        support: construction.support(&source.support)?,
    })
}
#[test]
fn capture_source_fields_match_and_every_original_destination_refuses() {
    let expected = fixture();
    assert_eq!(
        copy(&expected, CaptureSourceConstruction::new(None)).unwrap(),
        expected
    );
    let (account, calls, retired) = funding(usize::MAX);
    // This concrete enclosure keeps the actual policy after caller retirement.
    let output = (
        copy(&expected, CaptureSourceConstruction::new(Some(&account))).unwrap(),
        account.clone(),
    );
    assert_eq!(output.0, expected);
    let count = calls.load(Ordering::SeqCst);
    assert!(count > 15);
    drop(account);
    assert!(!retired.load(Ordering::SeqCst));
    drop(output);
    assert!(retired.load(Ordering::SeqCst));
    for stop in 1..count {
        let (account, calls, retired) = funding(stop);
        let error = copy(&expected, CaptureSourceConstruction::new(Some(&account))).unwrap_err();
        assert!(matches!(
            error,
            CaptureError::AdmissionStorage(CaptureAdmissionStorageError::Funding(
                HostMetadataFundingError::Unavailable
            ))
        ));
        let mut source: &(dyn std::error::Error + 'static) = &error;
        while !source.is::<HostMetadataFundingError>() {
            source = source
                .source()
                .expect("original fixed refusal remains in the source chain");
        }
        assert_eq!(calls.load(Ordering::SeqCst), stop + 1);
        drop(error);
        drop(account);
        assert!(retired.load(Ordering::SeqCst));
    }
}
