use super::*;
use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
use eredu_core::{
    DescriptionCompleteness, ObservationCatalog, ObservationSupportReport,
    ObservationSupportStatus, SymbolicDimension, TensorAxis,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, OriginalInterventionSource, WorkingMemoryPool,
};
fn sources(pool: &WorkingMemoryPool) -> (OriginalInterventionSource, AdmittedCapturePlan) {
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 5,
        max_context: None,
        max_predictions: 4,
    };
    let point = InterventionPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        stage: InterventionStage::Activation,
        axes: vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ],
        dtypes: vec![InterventionDtype::Float32],
        operations: vec![InterventionKind::Scale],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routing: None,
        routed_units: None,
    };
    let declaration = InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "model".into(),
        session_identity: Some("session".into()),
        points: vec![point],
    };
    let plan = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: [
            InterventionEvidence::Summary,
            InterventionEvidence::Preview { max_elements: 3 },
        ]
        .into_iter()
        .enumerate()
        .map(|(i, evidence)| InterventionOperation {
            id: i.to_string(),
            target: "block.output".into(),
            schedule: Default::default(),
            slices: if i == 1 {
                vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: 3,
                    end: 5,
                    stride: 1,
                }]
            } else {
                vec![]
            },
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 0.5,
            },
            evidence,
        })
        .collect(),
    }
    .admit_invocations(&declaration, bounds, "session")
    .unwrap();
    let source = pool
        .compile_intervention_source(
            eredu_core::intervention::PreparedInterventionPlanCopy::inspect(&plan).unwrap(),
        )
        .unwrap();
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let capture = CapturePlan {
        schema_version: 1,
        selections: vec![],
        limits: CaptureLimits {
            per_step: unlimited,
            cumulative: unlimited,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit_invocations(
        &ObservationCatalog {
            schema_version: 1,
            points: vec![],
            completeness: DescriptionCompleteness::Complete,
        },
        &ObservationSupportReport {
            schema_version: 1,
            capture: Default::default(),
            points: vec![],
        },
        &CaptureCapabilities {
            transformations: vec![],
            max_histogram_bins: 0,
            physical_native_limit: false,
            conditions: vec![],
        },
        bounds,
    )
    .unwrap();
    (source, capture)
}
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn model_companion_trace_counts_lazy_summary_frontiers_and_metadata_only_window_sides() {
    let capacity = 1 << 27;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let (source, capture) = sources(&pool);
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), capacity)
        .unwrap();
    let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(facts, funding.clone()).unwrap();
    let invocation = CaptureInvocationShape {
        batch: 1,
        sequence: 3,
        context: None,
    };
    let skips = [
        [None, None],
        [
            Some(CaptureSkipReason::Limit {
                budget: CaptureBudget::Captures,
                cumulative: true,
            }),
            None,
        ],
    ];
    let mut model = PreparedModelInterventions::prepare_with_evidence(
        &source,
        &[true, true],
        CapturePhase::Prefill,
        0,
        invocation,
        Some(CaptureInvocationWindow {
            logical_sequence: 5,
            start: 0,
        }),
        Some(&skips),
        &context,
    )
    .unwrap();
    let input = WorkspaceTensor::existing(
        context
            .layout(&[3, 2], eredu_nn::workspace::WorkspaceDtype::Float32)
            .unwrap(),
        &context,
    )
    .unwrap();
    context.begin_state_span([&input]).unwrap();
    let mut ledger = CaptureLedger::new(&capture);
    model
        .begin(CapturePhase::Prefill, 0, &mut ledger, &context)
        .unwrap();
    let initial = ledger.total();
    let mut roots = Vec::new();
    let (output, population) = model
        .trace("block.output", &input, &context, &mut ledger, &mut roots)
        .unwrap();
    assert_eq!(output.as_ref().unwrap().shape(), &[3, 2]);
    assert_eq!(ledger.total().captures - initial.captures, 2);
    assert_eq!(population.publications, 0);
    assert!(population.completions > 2);
    assert!(population.retained_roots >= roots.len());
    assert!(population.controls > 0);
    let Row::Ready(first) = &model.rows[0] else {
        panic!("first row")
    };
    assert_eq!(first.evidence, [State::Traced; 2]);
    let Row::Ready(second) = &model.rows[1] else {
        panic!("second row")
    };
    assert!(!second.overlap);
    assert_eq!(second.evidence, [State::Inactive, State::Empty]);
    model.finish(&context).unwrap();
    let report = context.finish_report(&roots).unwrap();
    assert!(!report.operations.is_empty());
    drop((
        report, roots, output, input, context, funding, source, capture,
    ));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(model);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
