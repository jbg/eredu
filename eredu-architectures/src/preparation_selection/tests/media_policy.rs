use super::*;
use eredu_runtime::{MediaExecutionPolicy, MediaLoadRequest, ProcessorSelectionRequest};
use std::sync::atomic::Ordering;

fn required_media() -> MediaLoadRequest {
    MediaLoadRequest::Required(
        MediaExecutionPolicy::new(ProcessorSelectionRequest::new([
            eredu_core::InputModality::Image,
        ]))
        .unwrap(),
    )
}

#[test]
fn explicit_media_is_rejected_before_backend_facts_with_distinct_cached_intent() {
    let (root, inspection) = inspected_llama();
    std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
    let validation = inspection
        .architecture_plan()
        .validation(inspection.admission_token());
    let default = NormalizedLoadRequest::default();
    let required = default.clone().with_media_execution(required_media());
    let disabled = default
        .clone()
        .with_media_execution(MediaLoadRequest::Disabled);
    for _ in 0..2 {
        assert!(matches!(
            select_preparation(&inspection, &required, &UnreachableMechanisms),
            Err(PreparationSelectionError::MediaPreparationRequired)
        ));
        assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 1);
    }
    let mechanisms = BoundedIndependentAdapter::default();
    for (index, request) in [&default, &disabled, &default, &disabled]
        .into_iter()
        .enumerate()
    {
        let selected = select_preparation(&inspection, request, &mechanisms).unwrap();
        selected
            .execution()
            .clone()
            .dispatch(SemanticExecutionProbe {
                partitioned: false,
                routed: false,
                processor: false,
            })
            .unwrap();
        assert_eq!(
            validation.selection_runs.load(Ordering::Relaxed),
            (index + 2).min(3)
        );
    }
    assert!(matches!(
        select_preparation(&inspection, &required, &UnreachableMechanisms),
        Err(PreparationSelectionError::MediaPreparationRequired)
    ));
    assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 3);
    let MediaLoadRequest::Required(policy) = required.media_execution() else {
        unreachable!();
    };
    for (index, changed) in [
        MediaExecutionPolicy::new(policy.processor().clone().with_projected_embeddings(true))
            .unwrap(),
        MediaExecutionPolicy::new(policy.processor().clone().with_raw_media(true)).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let request = default
            .clone()
            .with_media_execution(MediaLoadRequest::Required(changed));
        assert!(matches!(
            select_preparation(&inspection, &request, &UnreachableMechanisms),
            Err(PreparationSelectionError::MediaPreparationRequired)
        ));
        assert_eq!(validation.selection_runs.load(Ordering::Relaxed), index + 4);
    }
    mechanisms.assert_cold_only();
}

#[test]
fn disabled_media_keeps_routed_text_but_rejects_composite_processor_retention() {
    for (config, composite) in [(routed_config(), false), (composite_config(), true)] {
        let (root, inspection) = inspected_config(config);
        std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
        let validation = inspection
            .architecture_plan()
            .validation(inspection.admission_token());
        let mechanisms = BoundedIndependentAdapter::default();
        let disabled =
            NormalizedLoadRequest::default().with_media_execution(MediaLoadRequest::Disabled);
        for _ in 0..2 {
            let result = select_preparation(&inspection, &disabled, &mechanisms);
            if composite {
                assert!(matches!(
                    result,
                    Err(PreparationSelectionError::MediaDisabledForComposite)
                ));
            } else {
                result
                    .unwrap()
                    .execution()
                    .clone()
                    .dispatch(SemanticExecutionProbe {
                        partitioned: false,
                        routed: true,
                        processor: false,
                    })
                    .unwrap();
            }
            assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 1);
        }
        assert_eq!(mechanisms.counters.processor_queries.get(), 0);
        if composite {
            assert_eq!(mechanisms.counters.text_queries.get(), 0);
        }
        select_preparation(&inspection, &NormalizedLoadRequest::default(), &mechanisms)
            .unwrap()
            .execution()
            .clone()
            .dispatch(SemanticExecutionProbe {
                partitioned: false,
                routed: !composite,
                processor: composite,
            })
            .unwrap();
        assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 2);
        mechanisms.assert_cold_only();
    }
}
