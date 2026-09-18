use super::*;
use crate::working_memory::InferenceExecutionIdentity;
use eredu_core::{capture::*, *};

fn declarations() -> (CapturePlan, ObservationCatalog, ObservationSupportReport) {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "actual output rows".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(3),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let usage = CaptureUsage {
        captures: 5,
        retained_bytes: 4096,
        host_bytes: 4096,
        encoded_bytes: 16384,
    };
    let plan = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "rows".into(),
            path: point.path.clone(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Summary,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let support = ObservationSupportReport {
        schema_version: DISCOVERY_SCHEMA_VERSION,
        capture: CaptureCapabilities {
            transformations: vec![CaptureTransformKind::Summary],
            ..Default::default()
        },
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: DISCOVERY_SCHEMA_VERSION,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    (plan, catalog, support)
}

#[test]
fn raw_capture_source_matches_admission_and_aliases_keep_original_charge() {
    let pool = WorkingMemoryPool::new(64 << 20, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 64 << 20)
        .unwrap();
    let (plan, catalog, support) = declarations();
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 4,
        max_predictions: 3,
    };
    let origin = CaptureTextOrigin {
        cached_positions: 7,
    };
    let expected = plan
        .clone()
        .admit_with_text_origin(&catalog, &support, &support.capture, request, origin)
        .unwrap();
    let source = pool
        .compile_capture_declaration(&plan, &catalog, &support, request, origin, &funding)
        .unwrap();
    assert_eq!(source.plan().admission().identity(), expected.identity());
    assert_eq!(source.plan().admission().points(), expected.points());
    source.validate_pool(&pool).unwrap();
    let other = WorkingMemoryPool::new(64 << 20, 0).unwrap();
    assert!(source.validate_pool(&other).is_err());
    let alias = source.plan().clone();
    drop((source, funding));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(alias.admission().text_origin(), Some(origin));
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn failed_raw_capture_retains_its_admission_payer_until_diagnostic_retirement() {
    let pool = WorkingMemoryPool::new(64 << 20, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 64 << 20)
        .unwrap();
    let (mut plan, catalog, support) = declarations();
    plan.selections[0].path = "missing".into();
    let error = pool
        .compile_capture_declaration(
            &plan,
            &catalog,
            &support,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 4,
                max_predictions: 3,
            },
            CaptureTextOrigin::default(),
            &funding,
        )
        .unwrap_err();
    assert!(
        matches!(&error.cause, Cause::Admission(CaptureError::MissingPath(path)) if path == "missing")
    );
    drop(funding);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
