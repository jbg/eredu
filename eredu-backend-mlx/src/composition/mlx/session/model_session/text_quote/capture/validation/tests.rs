use super::*;
use eredu_core::{capture::*, *};
use std::{sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}}};

fn fixture() -> (AdmittedCapturePlan, ObservationCatalog, ObservationSupportReport) {
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![ObservationPoint {
            path: MODEL_LOGITS_OBSERVATION_PATH.into(), node_id: "decoder.output".into(),
            meaning: "raw output".into(), value_type: ObservationValueType::Tensor,
            dtype: ObservationDtype::Floating,
            axes: Some(vec![TensorAxis { name: "width".into(), dimension: SymbolicDimension::Known(4) }]),
            prefill: true, decode: true, requirements: vec![ObservationRequirement::ActivationHooks],
            position: ObservationPosition::BeforeIntervention, retained_bytes: None, host_bytes: None,
        }],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        points: vec![ObservationSupport {
            path: MODEL_LOGITS_OBSERVATION_PATH.into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported, floating_to_f32: true,
        }],
        capture: CaptureCapabilities {
            transformations: vec![CaptureTransformKind::Preview], max_histogram_bins: 0,
            physical_native_limit: false, conditions: vec![],
        },
    };
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "output".into(), path: MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(), slices: vec![],
        transform: CaptureTransform::Preview { max_elements: 2 },
    });
    let unlimited = CaptureUsage {
        captures: u64::MAX, retained_bytes: u64::MAX, host_bytes: u64::MAX, encoded_bytes: u64::MAX,
    };
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    let admitted = raw.admit(&catalog, &support, &support.capture,
        CaptureRequestShape { batch: 1, prompt_tokens: 2, max_predictions: 3 }).unwrap();
    (admitted, catalog, support)
}

#[derive(Debug)]
struct State { calls: AtomicUsize, refuse: AtomicUsize, retired: AtomicBool }
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
        let current = self.0.calls.fetch_add(1, Ordering::SeqCst);
        if current >= self.0.refuse.load(Ordering::SeqCst) {
            Err(HostMetadataFundingError::Unavailable)
        } else { Ok(()) }
    }
}
impl Drop for Account {
    fn drop(&mut self) { self.0.retired.store(true, Ordering::SeqCst); }
}
fn account(refuse: usize) -> (HostMetadataFunding, Arc<State>) {
    let state = Arc::new(State { calls: AtomicUsize::new(0), refuse: AtomicUsize::new(usize::MAX), retired: AtomicBool::new(false) });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    state.calls.store(0, Ordering::SeqCst);
    state.refuse.store(refuse, Ordering::SeqCst);
    (funding, state)
}
fn leaf<T: std::error::Error + 'static>(error: &eredu_nn::Error) -> bool {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(error) = cause {
        if error.downcast_ref::<T>().is_some() { return true; }
        cause = error.source();
    }
    false
}

#[test]
fn borrowed_revalidation_matches_existing_catalog_and_phase_decisions() {
    let (admitted, catalog, support) = fixture();
    for mutation in 0..7 {
        let mut catalog = catalog.clone();
        let mut support = support.clone();
        match mutation {
            1 => catalog.schema_version += 1,
            2 => catalog.points[0].meaning.push_str(" changed"),
            3 => support.capture.transformations.clear(),
            4 => support.points[0].decode = ObservationSupportStatus::Unsupported("unavailable".into()),
            5 => support.points.clear(),
            6 => support.points[0].decode = ObservationSupportStatus::Conditional("input dependent".into()),
            _ => {},
        }
        let ordinary = admitted.revalidate_parts(&catalog, &support).is_ok();
        let (funding, state) = account(usize::MAX);
        let result = validate_parts(&admitted, &catalog, &support, WorkspaceReportMetadata::with_funding(&funding));
        assert_eq!(result.is_ok(), ordinary, "mutation {mutation}");
        if let Err(error) = &result { assert!(leaf::<CaptureRevalidationError>(error)); }
        drop(result);
        drop(funding);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}

#[test]
fn cold_rejection_funds_every_reached_destination_and_retains_original_account() {
    let (admitted, catalog, mut support) = fixture();
    support.points.clear();
    let (funding, state) = account(usize::MAX);
    let error = validate_parts(&admitted, &catalog, &support, WorkspaceReportMetadata::with_funding(&funding)).unwrap_err();
    let requests = state.calls.load(Ordering::SeqCst);
    assert!(requests >= 2, "comparison controls and typed error destination");
    assert!(leaf::<CaptureRevalidationError>(&error));
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);
    assert!(state.retired.load(Ordering::SeqCst));
    for refuse in 0..requests {
        let (funding, state) = account(refuse);
        let error = validate_parts(&admitted, &catalog, &support, WorkspaceReportMetadata::with_funding(&funding)).unwrap_err();
        assert!(state.calls.load(Ordering::SeqCst) > refuse);
        assert!(leaf::<HostMetadataFundingError>(&error), "refusal {refuse}: {error}");
        assert!(!leaf::<CaptureRevalidationError>(&error), "unfunded rejection must not escape");
        drop(error);
        drop(funding);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
