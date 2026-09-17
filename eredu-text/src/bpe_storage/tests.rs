use super::*;
#[test]
fn component_owns_exact_canonical_model_after_input_retires() {
    let input = String::from(
        r#"{"vocab":{"h":1,"i":4,"hi":90,"\u0068":2,"😃":4294967295},"merges":[["h","i"]]}"#,
    );
    let plan = BpeModelPlan::prepare_model_json(input.as_bytes()).unwrap();
    let facts = plan.requirements();
    let model = plan.compile().unwrap();
    drop(input);
    assert_eq!(model.token_count(), 4);
    assert_eq!(model.token_id("h"), Some(2));
    assert_eq!(model.spelling(1), None);
    assert_eq!(model.spelling(90), Some("hi"));
    assert_eq!(model.spelling(u32::MAX), Some("😃"));
    let mut ids: Vec<_> = model.ids().collect();
    ids.sort_unstable();
    assert_eq!(ids, [2, 4, 90, u32::MAX]);
    assert_eq!(
        facts.required_bytes(),
        facts.buffer_bytes() + facts.control_bytes()
    );
}
#[test]
fn component_preserves_fixed_profile_and_actual_late_failure_causes() {
    let error =
        BpeModelPlan::prepare_model_json(br#"{"vocab":{},"merges":[],"dropout":0.5}"#).unwrap_err();
    assert_eq!(error.kind(), BpeModelSourceErrorKind::DropoutProfile);
    let input = String::from(r#"{"vocab":{"a":1,"b":1},"merges":[]}"#);
    let plan = BpeModelPlan::prepare_model_json(input.as_bytes()).unwrap();
    let bytes = plan.requirements().buffer_bytes();
    let error = plan.compile().unwrap_err();
    drop(input);
    assert_eq!(
        error.source_error().unwrap().kind(),
        BpeModelSourceErrorKind::AmbiguousId
    );
    assert_eq!(error.allocated_bytes(), Some(bytes));
    assert!(error.allocation_error().is_none());
}
