//! Borrowed validation uses actual prepared selected facts without content hashing.
use super::*;
use eredu_core::capture::*;
use eredu_core::{ObservationMechanisms, ObservationSupportStatus};

fn fixture() -> (
    tempfile::TempDir,
    PreparedModelSources,
    PreparedModelDiscovery,
    SharedCapturePlan,
) {
    let (root, inspection) = crate::preparation_selection::tests::inspected_llama();
    let selected = crate::select_preparation(
        &inspection,
        &eredu_runtime::NormalizedLoadRequest::default(),
        &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
    )
    .unwrap();
    let plan =
        ModelPreparationPlan::from_retained_admission(inspection, selected.admission()).unwrap();
    let sources = prepare_model_sources(plan, selected).unwrap();
    let discovery = sources.prepare_discovery(
        ObservationMechanisms {
            activation_tensors: true,
            floating_to_f32: true,
            ..Default::default()
        },
        CaptureCapabilities {
            transformations: vec![CaptureTransformKind::FullTensor],
            ..Default::default()
        },
    );
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule {
            prefill: false,
            ..Default::default()
        },
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    let admission = raw
        .admit_with_text_origin(
            &discovery.descriptor.observations,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: 7,
            },
        )
        .unwrap();
    (root, sources, discovery, SharedCapturePlan::new(admission))
}

#[test]
fn actual_selected_discovery_revalidates_without_hashing_or_payload_reads() {
    let (_root, sources, discovery, source) = fixture();
    assert!(!discovery.identity.is_resolved());
    let before = sources.primary().source_diagnostics().unwrap();
    let pointers = (
        source.admission().points().as_ptr(),
        source.admission().plan().selections.as_ptr(),
        source.admission().identity().as_ptr(),
    );
    let catalog = discovery.descriptor.observations.points.as_ptr();
    let support = discovery.support.points.as_ptr();
    let capacity = source.capacity_bytes();
    let alias = source.clone();
    for _ in 0..3 {
        discovery
            .validate_capture_admission(source.admission())
            .unwrap();
        assert!(
            !discovery.identity.is_resolved(),
            "borrowed validation must never demand content identity"
        );
    }
    assert!(source.same_storage(&alias));
    assert_eq!(source.capacity_bytes(), capacity);
    assert_eq!(
        (
            source.admission().points().as_ptr(),
            source.admission().plan().selections.as_ptr(),
            source.admission().identity().as_ptr()
        ),
        pointers
    );
    assert_eq!(discovery.descriptor.observations.points.as_ptr(), catalog);
    assert_eq!(discovery.support.points.as_ptr(), support);
    assert_eq!(
        source.admission().text_origin(),
        Some(CaptureTextOrigin {
            cached_positions: 7
        })
    );
    assert_eq!(
        source
            .admission()
            .geometry_at(CapturePhase::Decode, 2, None)
            .unwrap()
            .context,
        Some(12)
    );
    let after = sources.primary().source_diagnostics().unwrap();
    assert_eq!(after.physical_reads, before.physical_reads);
    assert_eq!(after.physical_read_bytes, before.physical_read_bytes);
}

#[test]
fn retained_discovery_rejects_changed_selected_semantics_and_current_support() {
    let (_root, _sources, mut discovery, source) = fixture();
    let index = discovery
        .descriptor
        .observations
        .points
        .iter()
        .position(|p| p.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        .unwrap();
    discovery.descriptor.observations.points[index]
        .meaning
        .push_str(" changed");
    assert!(matches!(
        discovery.validate_capture_admission(source.admission()),
        Err(CaptureError::Invalid(_))
    ));
    discovery.descriptor.observations.points[index] = source.admission().points()[0].clone();
    let selected = discovery
        .support
        .points
        .iter_mut()
        .find(|p| p.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        .unwrap();
    selected.decode = ObservationSupportStatus::Unsupported("collector unavailable".into());
    assert!(matches!(
        discovery.validate_capture_admission(source.admission()),
        Err(CaptureError::Unsupported(_))
    ));
    discovery
        .support
        .points
        .iter_mut()
        .find(|p| p.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        .unwrap()
        .decode = ObservationSupportStatus::Supported;
    discovery
        .validate_capture_admission(source.admission())
        .unwrap();
    discovery.support.capture.transformations.clear();
    assert!(matches!(
        discovery.validate_capture_admission(source.admission()),
        Err(CaptureError::Unsupported(_))
    ));
    assert!(!discovery.identity.is_resolved());
}
