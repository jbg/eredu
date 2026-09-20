use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

fn tools() -> Vec<Value> {
    (0..37).map(|n| serde_json::json!({
        "type": "function", "function": {
            "name": format!("measure_{n}"), "description": "é🙂".repeat(64),
            "parameters": {"type":"object", "properties":{"count":{"type":"integer", "minimum":17}}}
        }
    })).collect()
}

#[test]
fn declarations_borrow_caller_schemas_and_names_in_request_order() {
    let input = tools();
    let declarations =
        ToolDeclarations::prepare(&input, &PreparationFunding::unmanaged()).unwrap();
    assert_eq!(declarations.as_slice().len(), input.len());
    for (row, input) in declarations.as_slice().iter().zip(&input) {
        assert!(std::ptr::eq(
            row.parameters,
            &input["function"]["parameters"]
        ));
        assert!(std::ptr::eq(
            row.name,
            input["function"]["name"].as_str().unwrap()
        ));
        assert_eq!(row.parameters["properties"]["count"]["minimum"], 17);
    }
}

#[test]
fn every_prospective_reservation_refusal_retains_the_actual_payer() {
    struct Retired(Arc<AtomicBool>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let make = |fail_at| {
        let calls = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicBool::new(false));
        let funding = crate::runtime::chat::preparation_memory::test_funding({
            let calls = calls.clone();
            let guard = Retired(retired.clone());
            move |bytes| {
                let _ = &guard;
                if calls.fetch_add(1, Ordering::SeqCst) == fail_at {
                    Err(eredu_core::HostMetadataFundingError::Capacity {
                        required: bytes as u64,
                        available: 0,
                    })
                } else {
                    Ok(())
                }
            }
        })
        .unwrap();
        (funding, calls, retired)
    };
    let input = tools();
    let (funding, calls, retired) = make(usize::MAX);
    let accepted = ToolDeclarations::prepare(&input, &funding).unwrap();
    let total = calls.load(Ordering::SeqCst);
    assert!(total >= 3);
    drop(funding);
    assert!(!retired.load(Ordering::SeqCst));
    drop(accepted);
    assert!(retired.load(Ordering::SeqCst));
    // Callback construction's first reservation precedes the declaration pass.
    for fail_at in 1..total {
        let (funding, calls, retired) = make(fail_at);
        let failure = ToolDeclarations::prepare(&input, &funding).unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), fail_at + 1);
        let mut cause: &(dyn std::error::Error + 'static) = &failure;
        loop {
            if matches!(
                cause.downcast_ref::<eredu_core::HostMetadataFundingError>(),
                Some(eredu_core::HostMetadataFundingError::Capacity { available: 0, .. })
            ) {
                break;
            }
            cause = cause
                .source()
                .expect("the original typed funding cause is retained");
        }
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn malformed_declarations_fail_before_any_schema_compiler_is_needed() {
    let funding = PreparationFunding::unmanaged();
    let input = tools();
    let mut duplicate = input.clone();
    duplicate[29]["function"]["name"] = duplicate[3]["function"]["name"].clone();
    assert!(matches!(
        ToolDeclarations::prepare(&duplicate, &funding)
            .unwrap_err()
            .cause,
        Cause::Declaration(InvalidDeclaration::Duplicate { index: 29 })
    ));
    let mut malformed = input;
    malformed[17]["function"]["name"] = Value::String("bad.name".into());
    assert!(matches!(
        ToolDeclarations::prepare(&malformed, &funding)
            .unwrap_err()
            .cause,
        Cause::Declaration(InvalidDeclaration::NameCharacters { index: 17 })
    ));
}
