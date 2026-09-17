use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Retired(Arc<AtomicUsize>);

impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn tools() -> Vec<Value> {
    vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "measure",
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "integer", "minimum": 3}},
                "required": ["count"],
                "additionalProperties": false
            }
        }
    })]
}

#[test]
fn standalone_schema_constructor_keeps_validation_behavior() {
    let schemas = ToolSchemas::new(&tools()).unwrap();
    schemas.validate("measure", r#"{"count":41}"#).unwrap();
    assert!(schemas.validate("measure", r#"{"count":0}"#).is_err());
    assert!(schemas.validate("absent", r#"{"count":41}"#).is_err());
}

#[test]
fn compiled_validator_alias_retains_authority_after_source_and_root_retire() {
    let retired = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(Retired(retired.clone()));
    let source = tools();
    let schemas = ToolSchemas::new_under_authority(&source, &authority).unwrap();
    let alias = schemas.clone();
    assert!(Arc::ptr_eq(&schemas.0, &alias.0));
    assert!(std::ptr::eq(
        schemas.0.schemas.get("measure").unwrap(),
        alias.0.schemas.get("measure").unwrap(),
    ));
    let backing = Arc::downgrade(&schemas.0);
    drop((source, schemas, authority));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    alias.validate("measure", r#"{"count":17}"#).unwrap();
    assert!(alias.validate("measure", r#"{"count":2}"#).is_err());
    assert!(alias
        .validate("measure", r#"{"count":17,"extra":1}"#)
        .is_err());
    drop(alias);
    assert!(backing.upgrade().is_none());
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_later_schema_releases_partial_validators_under_callers_live_authority() {
    let retired = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(Retired(retired.clone()));
    let mut source = tools();
    source.push(serde_json::json!({
        "type": "function",
        "function": {"name": "broken", "parameters": {"type": 17}}
    }));
    let error = ToolSchemas::new_under_authority(&source, &authority).unwrap_err();
    assert!(error.contains("tools[1].function.parameters"));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    // The failed construction left no shared validator owner behind.
    drop((error, source, authority));
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}


#[test]
fn original_borrowed_schema_input_matches_full_engine_and_retains_first_failure() {
    use super::original::Source;
    use eredu_nn::workspace::{WorkspaceMetadataAccount, WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
    use std::sync::atomic::AtomicBool;
    #[derive(Debug)]
    struct Payer { refused: Arc<AtomicBool>, retired: Arc<AtomicBool>, calls: AtomicUsize }
    impl WorkspaceMetadataAccount for Payer {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.refused.load(Ordering::SeqCst) {
                Err(WorkspaceMetadataFundingError::Capacity { required: bytes as u64, available: 0 })
            } else { Ok(()) }
        }
    }
    impl Drop for Payer { fn drop(&mut self) { self.retired.store(true, Ordering::SeqCst); } }
    let make = || {
        let refused = Arc::new(AtomicBool::new(false));
        let retired = Arc::new(AtomicBool::new(false));
        let funding = WorkspaceMetadataFunding::new(Payer { refused: refused.clone(), retired: retired.clone(), calls: AtomicUsize::new(0) }).unwrap();
        (funding, refused, retired)
    };
    // Typed source compilation is ordinary, before every paid input/validation loan.
    let schema = serde_json::json!({"type":"object", "properties": {
        "count":{"type":"integer"}, "name":{"type":"string","minLength":2},
        "items":{"type":"array","items":{"type":"boolean"}}
    }, "required":["count","name"], "additionalProperties":false});
    let ordinary = compile(&schema).unwrap();
    let source_retired = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(Retired(source_retired.clone()));
    let source = Source::compile(&schema, &authority).unwrap();
    let retained = source.capacity_bytes().unwrap();
    assert!(retained > std::mem::size_of_val(&source));
    assert_eq!(source.clone().capacity_bytes().unwrap(), retained);
    let recursive_schema = serde_json::json!({"$id":"https://example.invalid/paid-node","$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}}}},"$ref":"#/$defs/node"});
    let recursive = Source::compile(&recursive_schema, &authority).unwrap();
    assert!(recursive.capacity_bytes().unwrap() > 0);
    let recursive_ordinary = compile(&recursive_schema).unwrap();
    let (recursive_funding, _, _) = make();
    for raw in [r#"{"next":{"next":{}}}"#, r#"{"next":3}"#] {
        let value: Value = serde_json::from_str(raw).unwrap();
        let actual = recursive.validate(raw, &recursive_funding);
        assert_eq!(actual.is_ok(), recursive_ordinary.is_valid(&value), "{raw}: {actual:?}");
    }
    drop((recursive, recursive_ordinary, recursive_funding));
    let (funding, refused, retired) = make();
    for raw in [r#"{"count":18446744073709551615,"name":"é🙂","items":[true,false]}"#,
        r#"{"count":0,"\u0063ount":3.5,"name":"valid"}"#,
        r#"{"count":-0.0,"name":"ok"}"#, r#"{"count":1,"name":"é"}"#,
        r#"{"count":2,"name":"ok","extra":null}"#] {
        let value: Value = serde_json::from_str(raw).unwrap();
        let expected = ordinary.is_valid(&value);
        let result = source.validate(raw, &funding);
        assert_eq!(result.is_ok(), expected, "{raw}: {result:?}");
        if let Err(error) = result { assert!(error.to_string().contains("do not match")); }
    }
    let syntax = source.validate(r#"{"name":"kept","count":"#, &funding).unwrap_err();
    assert!(syntax.to_string().contains("EOF"));
    refused.store(true, Ordering::SeqCst);
    let refusal = source.validate(r#"{"count":1,"name":"ok"}"#, &funding).unwrap_err();
    let mut cause: &(dyn std::error::Error + 'static) = &refusal;
    let mut typed = false;
    loop {
        if matches!(cause.downcast_ref::<WorkspaceMetadataFundingError>(), Some(WorkspaceMetadataFundingError::Capacity { available: 0, .. })) { typed = true; break; }
        let Some(next) = cause.source() else { break; };
        cause = next;
    }
    assert!(typed, "actual funding cause remains in the source chain");
    drop((funding, authority, source));
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(source_retired.load(Ordering::SeqCst), 0);
    drop(syntax);
    assert!(!retired.load(Ordering::SeqCst));
    drop(refusal);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(source_retired.load(Ordering::SeqCst), 1);

    // These scalar bodies borrow the input directly; no equals_value/to_value
    // fallback or numeric helper is implicitly admitted by this qualification.
    let schema = serde_json::json!({"type":"object", "properties": {
        "mode":{"enum":["read","write"]}, "active":{"const":true},
        "empty":{"const":null}, "name":{"const":"é🙂"}
    }, "required":["mode","active","empty","name"], "additionalProperties":false});
    let ordinary = compile(&schema).unwrap();
    let scalar = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
    assert!(scalar.capacity_bytes().unwrap() > 0);
    let (funding, _, _) = make();
    for raw in [r#"{"mode":"read","active":true,"empty":null,"name":"é🙂"}"#,
        r#"{"mode":"write","active":false,"empty":null,"name":"é🙂"}"#,
        r#"{"mode":"other","active":true,"empty":null,"name":"é🙂"}"#,
        r#"{"mode":"read","active":true,"empty":0,"name":"é🙂"}"#,
        r#"{"mode":"read","active":true,"empty":null,"name":"é"}"#] {
        let value: Value = serde_json::from_str(raw).unwrap();
        let actual = scalar.validate(raw, &funding);
        assert_eq!(actual.is_ok(), ordinary.is_valid(&value), "{raw}: {actual:?}");
        if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
    }
    drop((scalar, funding));

    // Cross the actual property-table and string-enum thresholds. Both ordinary
    // and original validation now inspect the same pinned completed hash maps.
    let properties: serde_json::Map<String, Value> = (0..19)
        .map(|i| (format!("p{i}"), serde_json::json!({"type":"boolean"}))).collect();
    let alternatives: Vec<String> = (0..17).map(|i| format!("choice{i}")).collect();
    let mut properties = properties;
    properties.insert("mode".into(), serde_json::json!({"enum":alternatives}));
    for extra in [Value::Bool(false), serde_json::json!({"type":"boolean"}), Value::Bool(true)] {
        let schema = serde_json::json!({"type":"object", "properties":properties, "additionalProperties":extra});
        let ordinary = compile(&schema).unwrap();
        let source = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
        assert!(source.capacity_bytes().unwrap() > 0);
        let (funding, _, _) = make();
        for raw in [r#"{"p0":true,"p18":false,"mode":"choice16"}"#,
            r#"{"p0":true,"mode":"absent"}"#, r#"{"p0":false,"extra":true}"#,
            r#"{"p18":7}"#] {
            let value: Value = serde_json::from_str(raw).unwrap();
            let actual = source.validate(raw, &funding);
            assert_eq!(actual.is_ok(), ordinary.is_valid(&value), "{raw}: {actual:?}");
            if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
        }
    }

    // The selected lower-level numeric helper certifies its primitive path.
    // Wide integers must retain their exact comparison, never an f64 rewrite.
    let schema = serde_json::json!({"type":"object", "properties": {
        "exact":{"const":18446744073709551615u64},
        "range":{"minimum":-2,"exclusiveMaximum":2.5},
        "upper":{"maximum":18446744073709551615u64},
        "lower":{"exclusiveMinimum":-9223372036854775808i64}
    }, "additionalProperties":false});
    let ordinary = compile(&schema).unwrap();
    let source = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
    assert!(source.capacity_bytes().unwrap() > 0);
    let (funding, _, _) = make();
    for raw in [r#"{"exact":18446744073709551615,"range":-2,"upper":18446744073709551615,"lower":-1}"#,
        r#"{"exact":18446744073709551614}"#, r#"{"exact":18446744073709551616.0}"#,
        r#"{"range":2.5}"#, r#"{"range":-0.0}"#, r#"{"upper":18446744073709551616.0}"#,
        r#"{"lower":-9223372036854775808}"#, r#"{"range":"not a number"}"#] {
        let value: Value = serde_json::from_str(raw).unwrap();
        let actual = source.validate(raw, &funding);
        assert_eq!(actual.is_ok(), ordinary.is_valid(&value), "{raw}: {actual:?}");
        if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
    }
    drop((source, funding));

    let schema = serde_json::json!({"type":"object", "properties":{
        "array":{"const":[[1,"é🙂"],true,null]},
        "mixed":{"enum":[null,false,17,["x",0]]},
        "single":{"enum":[[1,2]]}
    }, "additionalProperties":false});
    let ordinary = compile(&schema).unwrap();
    let source = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
    assert!(source.capacity_bytes().unwrap() > 0);
    let (funding, _, _) = make();
    for raw in [r#"{"array":[[1.0,"é🙂"],true,null],"mixed":["x",-0.0],"single":[1,2]}"#,
        r#"{"array":[[1,"é"],true,null]}"#, r#"{"array":[[1,"é🙂"],false,null]}"#,
        r#"{"mixed":17.0}"#, r#"{"mixed":18}"#, r#"{"single":[1,3]}"#] {
        let value: Value = serde_json::from_str(raw).unwrap();
        let actual = source.validate(raw, &funding);
        assert_eq!(actual.is_ok(), ordinary.is_valid(&value), "{raw}: {actual:?}");
        if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
    }
    drop(funding);
    #[derive(Debug)]
    struct Cut { calls: Arc<AtomicUsize>, limit: Arc<AtomicUsize> }
    impl WorkspaceMetadataAccount for Cut {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) >= self.limit.load(Ordering::SeqCst) {
                Err(WorkspaceMetadataFundingError::Capacity { required: bytes as u64, available: 0 })
            } else { Ok(()) }
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let limit = Arc::new(AtomicUsize::new(usize::MAX));
    let funding = WorkspaceMetadataFunding::new(Cut { calls: calls.clone(), limit: limit.clone() }).unwrap();
    let raw = r#"{"array":[[1,"é🙂"],true,null],"mixed":["x",0],"single":[1,2]}"#;
    source.validate(raw, &funding).unwrap();
    let count = calls.swap(0, Ordering::SeqCst);
    assert!(count > 3);
    limit.store(count - 2, Ordering::SeqCst);
    let failed = source.validate(raw, &funding).unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), count - 1, "first late funding refusal prevents further spending");
    let mut cause: &(dyn std::error::Error + 'static) = &failed;
    loop {
        if cause.downcast_ref::<WorkspaceMetadataFundingError>().is_some() { break; }
        cause = cause.source().expect("reached recursive equality keeps its real funding cause");
    }
    drop((source, funding, failed));

    // The same compiled literal owner holds nested object/array declarations.
    // All known object backing is inspected; no opaque serde map is retained.
    let schema = serde_json::json!({"type":"object","properties":{
        "object":{"const":{"a":[1,{"z":"é🙂"}],"b":true}},
        "array":{"const":[{"first":1},{"second":[2]}]},
        "mixed":{"enum":[null,{"a":1},{"b":[2]}]},
        "single":{"enum":[{"a":1}]}
    }});
    let ordinary = compile(&schema).unwrap();
    let source_retired = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(Retired(source_retired.clone()));
    let source = Source::compile(&schema, &authority).unwrap();
    assert!(source.capacity_bytes().unwrap() > 0);
    let (funding, _, retired) = make();
    for (raw, expected) in [
        (r#"{"object":{"a":[1.0,{"z":"é🙂"}],"b":true},"array":[{"first":1},{"second":[2]}],"mixed":{"b":[2]},"single":{"a":1}}"#, true),
        // The pinned preexisting equality zips member order; preserve that
        // selected ordinary behavior instead of normalizing only paid input.
        (r#"{"object":{"b":true,"a":[1.0,{"z":"é🙂"}]}}"#, !serde_json::bounded_events::preserves_object_order()),
        (r#"{"object":{"a":[],"b":true,"\u0061":[1.0,{"z":"é🙂"}]}}"#, true),
        (r#"{"mixed":{"a":1.0}}"#, true),
        (r#"{"object":{"a":[1,{"z":"é"}],"b":true}}"#, false),
        (r#"{"object":{"a":[1,{"z":"é🙂"}],"b":true,"extra":0}}"#, false),
        (r#"{"array":[{"second":[2]},{"first":1}]}"#, false),
        (r#"{"mixed":{"a":2}}"#, false),
        (r#"{"single":{"a":1,"b":2}}"#, false),
    ] {
        let value: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(ordinary.is_valid(&value), expected, "ordinary: {raw}");
        let actual = source.validate(raw, &funding);
        assert_eq!(actual.is_ok(), expected, "{raw}: {actual:?}");
        if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
    }
    let failed = source.validate(r#"{"object":{"a":[]}}"#, &funding).unwrap_err();
    drop((source, authority, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(source_retired.load(Ordering::SeqCst), 0);
    drop(failed);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(source_retired.load(Ordering::SeqCst), 1);

    let constant = serde_json::json!({"a":[1,{"z":"é🙂"}],"b":true});
    let diagnostic_schema = serde_json::json!({"const":constant});
    let ordinary = compile(&diagnostic_schema).unwrap();
    let mismatch = serde_json::json!({});
    let error = ordinary.validate(&mismatch).unwrap_err();
    match error.kind() {
        jsonschema::error::ValidationErrorKind::Constant { expected_value } => assert_eq!(expected_value, &constant),
        other => panic!("unchanged ordinary constant diagnostic: {other:?}"),
    }

    // Exact integer modulo must not round wide integral instances through f64.
    let schema = serde_json::json!({"type":"object","properties":{
        "three":{"multipleOf":3}, "two":{"multipleOf":2}, "wide":{"multipleOf":9007199254740994.0}
    }});
    let ordinary = compile(&schema).unwrap();
    let source = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
    assert!(source.capacity_bytes().unwrap() > 0);
    let (funding, _, _) = make();
    for raw in [r#"{"three":18446744073709551615,"two":-9223372036854775808}"#,
        r#"{"three":18446744073709551614}"#, r#"{"two":-0.0}"#,
        r#"{"two":1.5}"#, r#"{"wide":9007199254740994.0}"#, r#"{"two":"skip"}"#] {
        let value: Value = serde_json::from_str(raw).unwrap();
        let actual = source.validate(raw, &funding);
        assert_eq!(actual.is_ok(), ordinary.is_valid(&value), "{raw}: {actual:?}");
        if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
    }
    drop((source, funding));

    // These are the existing compiler-selected literal workers. They share
    // exactly the ordinary body and do not reach a general regex matcher.
    for (pattern, texts) in [
        (r"^e", vec!["e🙂", "e", "xé"]),
        (r"^\$ref$", vec!["$ref", "$refs", "$ref\n"]),
        (r"^(get|put|post)$", vec!["get", "post", "GET", "patch"]),
        (r"^\S*$", vec!["", "é🙂", "a b", "a\u{a0}b", "a\u{feff}b", "a\u{85}b"]),
    ] {
        let schema = serde_json::json!({"type":"object","properties":{"name":{"pattern":pattern}}});
        let ordinary = compile(&schema).unwrap();
        let source = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
        assert!(source.capacity_bytes().unwrap() > 0);
        let (funding, _, _) = make();
        for name in texts {
            let value = serde_json::json!({"name":name});
            let raw = serde_json::to_string(&value).unwrap();
            let actual = source.validate(&raw, &funding);
            assert_eq!(actual.is_ok(), ordinary.is_valid(&value), "{pattern}: {raw}: {actual:?}");
            if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
        }
        let raw = r#"{"name":7}"#;
        assert_eq!(source.validate(raw, &funding).is_ok(), ordinary.is_valid(&serde_json::from_str(raw).unwrap()));
    }


    // Property names use the actual borrowed string wrapper once per key,
    // while dependencies retain their compiled prepared keys and child graph.
    for schema in [
        serde_json::json!({"type":"object","propertyNames":{"minLength":2,"maxLength":3}}),
        serde_json::json!({"type":"object","propertyNames":false}),
        serde_json::json!({"type":"object","propertyNames":{"pattern":"^e"}}),
        serde_json::json!({"type":"object","dependentRequired":{"ab":["cd","é🙂"]}}),
        serde_json::json!({"type":"object","dependentSchemas":{"ab":{"required":["cd"],"properties":{"cd":{"type":"boolean"}}}}}),
        serde_json::json!({"$schema":"http://json-schema.org/draft-07/schema#","type":"object","dependencies":{"ab":["cd"],"cd":{"required":["é🙂"]}}}),
    ] {
        let ordinary = compile(&schema).unwrap();
        let source = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
        assert!(source.capacity_bytes().unwrap() > 0);
        let (funding, _, _) = make();
        for raw in [r#"{}"#, r#"{"ab":1,"cd":true,"é🙂":2}"#, r#"{"ab":1}"#,
            r#"{"ab":1,"cd":2,"é🙂":2}"#, r#"{"é🙂":1}"#, r#"{"é":1}"#, r#"{"e🙂":1}"#,
            r#"{"abcd":1,"é🙂":2}"#, r#"{"ab":0,"\u0061b":1,"cd":true}"#] {
            let value: Value = serde_json::from_str(raw).unwrap();
            let actual = source.validate(raw, &funding);
            assert_eq!(actual.is_ok(), ordinary.is_valid(&value), "{schema}: {raw}: {actual:?}");
            if let Err(error) = actual { assert!(error.to_string().contains("do not match")); }
        }
    }


    // Both sides cross the actual >15-item hash threshold; nested structural
    // equality, integer/floating equality and input key order stay shared.
    let schema = serde_json::json!({"type":"object","properties":{"items":{"type":"array","uniqueItems":true}}});
    let ordinary = compile(&schema).unwrap();
    let source = Source::compile(&schema, &HostPreparationAuthority::unmanaged()).unwrap();
    assert!(source.capacity_bytes().unwrap()>0);
    let (funding, _, _) = make();
    for count in [0,1,2,3,15,16,31] {
        for duplicate in [false,true] {
            let mut items: Vec<Value> = (0..count).map(|i| serde_json::json!({"n":i,"nested":[true,null,"é🙂"]})).collect();
            if duplicate && count>1 { items[count-1] = serde_json::json!({"n":0.0,"nested":[true,null,"é🙂"]}); }
            let value = serde_json::json!({"items":items});
            let raw = serde_json::to_string(&value).unwrap();
            let result = source.validate(&raw,&funding);
            assert_eq!(result.is_ok(),ordinary.is_valid(&value),"{raw}: {result:?}");
            assert_eq!(result.is_ok(),!(duplicate && count>1));
        }
    }
    for raw in [r#"{"items":[1,1.0]}"#,r#"{"items":[-0.0,0.0]}"#,
        r#"{"items":[{"a":1,"b":2},{"b":2,"a":1}]}"#,
        r#"{"items":[18446744073709551615,18446744073709551614]}"#,
        r#"{"items":[[1,2],[2,1]]}"#] {
        let value: Value = serde_json::from_str(raw).unwrap();
        let result = source.validate(raw,&funding);
        assert_eq!(result.is_ok(),ordinary.is_valid(&value),"{raw}: {result:?}");
    }


    // Refuse after entering the actual large-array worker. Partial table/handle
    // scratch drops, but the escaped failure still retains input and payer.
    #[derive(Debug)]
    struct UniqueCut { calls: Arc<AtomicUsize>, fail_at: usize, retired: Arc<AtomicBool> }
    impl WorkspaceMetadataAccount for UniqueCut {
        fn reserve_metadata(&self, bytes: usize) -> Result<(),WorkspaceMetadataFundingError> {
            if self.calls.fetch_add(1,Ordering::SeqCst)==self.fail_at {
                Err(WorkspaceMetadataFundingError::Capacity {required:bytes as u64,available:0})
            } else { Ok(()) }
        }
    }
    impl Drop for UniqueCut {
        fn drop(&mut self) { self.retired.store(true,Ordering::SeqCst); }
    }
    let cut_funding = |fail_at| {
        let calls=Arc::new(AtomicUsize::new(0));
        let retired=Arc::new(AtomicBool::new(false));
        let funding=WorkspaceMetadataFunding::new(UniqueCut {calls:calls.clone(), fail_at, retired:retired.clone()}).unwrap();
        (funding,calls,retired)
    };
    let value=serde_json::json!({"items":(0..31).map(|n| serde_json::json!({"n":n,"v":[true,"é🙂"]})).collect::<Vec<_>>()});
    let raw=serde_json::to_string(&value).unwrap();
    let (all,calls,_)=cut_funding(usize::MAX);
    source.validate(&raw,&all).unwrap();
    let total=calls.load(Ordering::SeqCst);
    drop(all);
    assert!(total>4);
    for cut in [total/2,total-2,total-1] {
        let (funding,calls,retired)=cut_funding(cut);
        let failure=source.validate(&raw,&funding).unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst),cut+1,"first failure must stop spending");
        let mut cause: &(dyn std::error::Error+'static)=&failure;
        let mut typed=false;
        loop {
            if matches!(cause.downcast_ref::<WorkspaceMetadataFundingError>(),Some(WorkspaceMetadataFundingError::Capacity{available:0,..})) {typed=true;break;}
            let Some(next)=cause.source() else {break;};cause=next;
        }
        assert!(typed);
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }

    // Modern-draft format/content are ordinary annotation-only declarations.
    // They do not call an email validator, base64 decoder or JSON content parser.
    let schema=serde_json::json!({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object",
        "properties":{"email":{"type":"string","format":"email"},"blob":{"type":"string","contentEncoding":"base64","contentMediaType":"application/json"}}});
    let ordinary=compile(&schema).unwrap();
    let source=Source::compile(&schema,&HostPreparationAuthority::unmanaged()).unwrap();
    assert!(source.capacity_bytes().unwrap()>0);
    let (funding,_,_)=make();
    for raw in [r#"{"email":"not an email","blob":"not base64 or json"}"#,
        r#"{"email":"a@example.invalid","blob":"e30="}"#,r#"{"email":7}"#,r#"{"blob":false}"#] {
        let value:Value=serde_json::from_str(raw).unwrap();
        let actual=source.validate(raw,&funding);
        assert_eq!(actual.is_ok(),ordinary.is_valid(&value),"{raw}: {actual:?}");
        if let Err(error)=actual { assert!(error.to_string().contains("do not match")); }
    }
    for bounds in [serde_json::json!({}),serde_json::json!({"minContains":2}),
        serde_json::json!({"maxContains":2}),serde_json::json!({"minContains":1,"maxContains":2}),
        serde_json::json!({"minContains":0,"maxContains":0})] {
        let mut items=serde_json::json!({"type":"array","contains":{"const":1}});
        items.as_object_mut().unwrap().extend(bounds.as_object().unwrap().clone());
        let schema=serde_json::json!({"type":"object","properties":{"items":items}});
        let ordinary=compile(&schema).unwrap();
        let source=Source::compile(&schema,&HostPreparationAuthority::unmanaged()).unwrap();
        assert!(source.capacity_bytes().unwrap()>0);
        let (funding,_,_)=make();
        for raw in [r#"{"items":[]}"#,r#"{"items":[0]}"#,r#"{"items":[1.0]}"#,
            r#"{"items":[1,0,1.0]}"#,r#"{"items":[1,1.0,1]}"#] {
            let value:Value=serde_json::from_str(raw).unwrap();
            let actual=source.validate(raw,&funding);
            assert_eq!(actual.is_ok(),ordinary.is_valid(&value),"{schema}: {raw}: {actual:?}");
            if let Err(error)=actual { assert!(error.to_string().contains("do not match")); }
        }
    }

    // The pinned analyzer deliberately routes a non-ASCII literal prefix and
    // a lookahead to the general regex engine. Neither acquires its source or
    // body authority merely because these inputs happen to match ordinarily.
    for pattern in ["^é", "^(?!bad)"] {
        let schema=serde_json::json!({"type":"object", "properties": {"name":{"pattern":pattern}}});
        let ordinary=compile(&schema).unwrap();
        let raw=r#"{"name":"é🙂"}"#;
        assert!(ordinary.is_valid(&serde_json::from_str(raw).unwrap()));
        let source=Source::compile(&schema,&HostPreparationAuthority::unmanaged()).unwrap();
        assert!(matches!(source.capacity_bytes(),Err(jsonschema::OriginalValidationError::Unqualified(_))));
        let (funding, _, retired) = make();
        let failure = source.validate(raw, &funding).unwrap_err();
        assert!(failure.to_string().contains("not yet qualified"));
        drop((source, funding));
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }
}
