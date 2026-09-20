use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Retired(Arc<AtomicBool>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
fn funding(fail_at: usize) -> (PreparationFunding, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    let source = crate::runtime::chat::preparation_memory::test_funding({
        let calls = calls.clone();
        let guard = Retired(retired.clone());
        move |bytes| {
            let _ = &guard;
            if calls.fetch_add(1, Ordering::SeqCst) == fail_at {
                Err(HostMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: 0,
                })
            } else {
                Ok(())
            }
        }
    })
    .unwrap();
    (source, calls, retired)
}
fn rows(schema: &serde_json::Value) -> [super::super::ToolDefinition<'_>; 3] {
    ["measure", "count", "read"].map(|name| super::super::ToolDefinition {
        name,
        parameters: schema,
    })
}

#[test]
fn compiled_schema_declaration_keeps_row_funding_through_escaping_source_aliases() {
    let schema = serde_json::json!({"type":"object", "properties":{"count":{"type":"integer","minimum":17}}});
    let (funding, _, retired) = funding(usize::MAX);
    let pending = PendingSchemas::compile(
        &rows(&schema),
        &funding,
        &HostPreparationAuthority::unmanaged(),
        false,
    )
    .unwrap();
    let recipe = ConstraintRecipe::new(
        None,
        &llguidance::api::TopLevelGrammar::from_regex("a"),
        &[],
        &[],
        &[],
        &[],
        &[],
        None,
    )
    .unwrap();
    let source = pending.bind(&recipe);
    let alias = source.source.clone();
    drop((source, funding, schema));
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(alias.declaration::<Data>().unwrap().schemas.rows.len(), 3);
    drop(alias);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn facade_schema_row_reservations_refuse_before_growth_and_preserve_the_cause() {
    let schema = serde_json::json!({"type":"object", "properties":{"count":{"type":"integer","minimum":17}}});
    let (accepted, calls, _) = funding(usize::MAX);
    let pending = PendingSchemas::compile(
        &rows(&schema),
        &accepted,
        &HostPreparationAuthority::unmanaged(),
        false,
    )
    .unwrap();
    let count = calls.load(Ordering::SeqCst);
    assert!(count >= 5);
    drop((accepted, pending));
    for cut in 1..count {
        let (funding, calls, retired) = funding(cut);
        let failure = match PendingSchemas::compile(
            &rows(&schema),
            &funding,
            &HostPreparationAuthority::unmanaged(),
        false,
        ) {
            Ok(pending) => {
                // Equal schemas can reach slightly different allocation counts.
                // A cut beyond this concrete trace refused nothing. Every
                // reached refusal must still stop at its first callback below.
                assert!(calls.load(Ordering::SeqCst) <= cut,
                    "reached reservation {cut}/{count} was ignored");
                assert_eq!(pending.schemas.rows.len(), 3);
                drop(funding);
                assert!(!retired.load(Ordering::SeqCst));
                drop(pending);
                assert!(retired.load(Ordering::SeqCst));
                continue;
            }
            Err(error) => error,
        };
        assert_eq!(calls.load(Ordering::SeqCst), cut + 1);
        let mut cause: &(dyn std::error::Error + 'static) = &failure;
        loop {
            if matches!(
                cause.downcast_ref::<HostMetadataFundingError>(),
                Some(HostMetadataFundingError::Capacity { available: 0, .. })
            ) {
                break;
            }
            cause = cause
                .source()
                .expect("actual funding refusal must remain in the error chain");
        }
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn tagged_parameter_rooted_references_defer_to_the_complete_source_validator() {
    #[derive(Debug)]
    struct Invocation;
    impl eredu_core::HostMetadataAccount for Invocation {
        fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> { Ok(()) }
    }
    let schema = serde_json::json!({
        "type":"object", "$defs":{"count":{"type":"integer","minimum":2}},
        "properties":{"nested":{"type":"object","properties":{"count":{"$ref":"#/$defs/count"}},
            "required":["count"]}}, "required":["nested"]
    });
    let (funding, _, _) = funding(usize::MAX);
    let tools = [super::super::ToolDefinition { name:"write", parameters:&schema }];
    let source = PendingSchemas::compile(&tools, &funding, &HostPreparationAuthority::unmanaged(), true).unwrap();
    let (_, complete, tagged) = &source.schemas.rows[0];
    let tagged = tagged.as_ref().unwrap();
    let invocation = HostMetadataFunding::new(Invocation).unwrap();
    for count in [1, 2] {
        let raw = format!("{{\"count\":{count}}}");
        let value = tagged.parse("nested", None, &raw, &invocation).unwrap();
        assert_eq!(value, serde_json::json!({"count":count}));
        let arguments = format!("{{\"nested\":{raw}}}");
        assert_eq!(complete.validate(&arguments, &invocation).is_ok(), count == 2);
    }
}
