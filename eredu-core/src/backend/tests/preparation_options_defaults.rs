use super::*;
use crate::capture::*;

fn empty_source() -> crate::capture::SharedCapturePlan {
    let catalog = crate::ObservationCatalog {
        schema_version: 1,
        points: vec![],
        completeness: crate::DescriptionCompleteness::Complete,
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    let support = crate::ObservationSupportReport {
        schema_version: 1,
        capture: capabilities.clone(),
        points: vec![],
    };
    crate::capture::SharedCapturePlan::new(
        CapturePlan::none()
            .admit(
                &catalog,
                &support,
                &capabilities,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 2,
                    max_predictions: 3,
                },
            )
            .unwrap(),
    )
}
fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

#[test]
fn default_options_admission_rejects_even_empty_source_and_delegates_none() {
    let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
    let source = empty_source();
    assert!(source.admission().is_empty());
    assert!(source.capacity_bytes().unwrap() > 0);
    let error = match TextGeneration::new_with_options(
        &mut runtime,
        vec![1, 2],
        config(),
        TextPreparationOptions {
            interventions: None,
            capture: Some(source),
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("empty shared source is not proof of zero storage"),
    };
    assert_eq!(error.kind(), BackendFailureKind::Unsupported);
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<CaptureError>());
    assert!(runtime.session().tokens.is_empty());
    let actual = TextGeneration::from_prompt_with_options(
        &mut runtime,
        vec![1, 2],
        config(),
        TextPreparationOptions::default(),
    )
    .unwrap()
    .collect::<Result<Vec<_>, _>>()
    .unwrap();
    let mut reference = ModelRuntime::prepare(Mock, 10).unwrap();
    let expected = TextGeneration::from_prompt(&mut reference, vec![1, 2], config())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn default_install_cannot_silently_accept_a_shared_source() {
    let runtime = ModelRuntime::prepare(Mock, 10).unwrap();
    let source = empty_source();
    let mut state = (4, 7);
    let error = Mock::install_text_capture_admitted(
        &runtime,
        &mut state,
        &(),
        &source,
        &TextStepContext::new(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), BackendFailureKind::Unsupported);
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<CaptureError>());
    assert_eq!(state, (4, 7));
    assert!(runtime.session().tokens.is_empty());
}
