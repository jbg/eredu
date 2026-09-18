use super::*;
use std::{
    error::Error,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

#[derive(Debug)]
struct Refused(usize);
impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "schema frame refusal {}", self.0)
    }
}
impl Error for Refused {}
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
struct Account {
    funding: ParserAllocationFunding,
    calls: Arc<Mutex<Vec<usize>>>,
    cut: Arc<AtomicUsize>,
    retired: Arc<AtomicUsize>,
}
impl Account {
    fn new() -> Self {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let cut = Arc::new(AtomicUsize::new(usize::MAX));
        let retired = Arc::new(AtomicUsize::new(0));
        let (events, limit, owner) = (calls.clone(), cut.clone(), Retired(retired.clone()));
        let funding = ParserAllocationFunding::prepare(move |bytes| {
            let _owner = &owner;
            let mut events = events.lock().unwrap();
            let ordinal = events.len();
            events.push(bytes);
            if ordinal == limit.load(Ordering::SeqCst) {
                Err(Refused(ordinal))
            } else {
                Ok(())
            }
        })
        .unwrap();
        Self {
            funding,
            calls,
            cut,
            retired,
        }
    }
    fn reset(&self) {
        self.calls.lock().unwrap().clear();
    }
    fn requests(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }
}
fn alternatives(depth: usize, width: usize) -> Schema {
    if depth == 0 {
        Schema::Null
    } else {
        Schema::OneOf((0..width).map(|_| alternatives(depth - 1, width)).collect())
    }
}
fn nested(depth: usize, value: Option<bool>) -> Schema {
    if depth == 0 {
        return Schema::Boolean(value);
    }
    let mut properties = IndexMap::new();
    properties.insert(
        "item".to_owned(),
        Schema::Array(ArraySchema {
            min_items: 1,
            max_items: None,
            prefix_items: vec![nested(depth - 1, value)],
            items: Some(Box::new(Schema::Any)),
        }),
    );
    let mut required = IndexSet::new();
    required.insert("item".to_owned());
    Schema::Object(ObjectSchema {
        properties,
        pattern_properties: IndexMap::new(),
        required,
        additional_properties: Some(Box::new(Schema::Boolean(value))),
        min_properties: 1,
        max_properties: None,
    })
}
fn refusal(error: &ParserError) -> usize {
    let mut cause: &(dyn Error + 'static) = error;
    loop {
        if let Some(error) = cause.downcast_ref::<Refused>() {
            return error.0;
        }
        cause = cause.source().expect("original schema funding refusal");
    }
}

#[test]
fn recursive_disjoint_frames_pay_live_depth_and_reuse_after_return() {
    let account = Account::new();
    let allocation = CompilerAllocation(&account.funding);
    let pre = PreContext::new(serde_json::json!({}), None, &allocation).unwrap();
    let ctx = Context::new(&pre).unwrap();
    account.reset();
    assert!(Schema::Null
        .is_verifiably_disjoint_from(&Schema::Boolean(None), &ctx)
        .unwrap());
    let one = account.requests();
    assert_eq!(one.len(), 1);
    assert!(one[0] > 0);
    account.reset();
    assert!(alternatives(4, 1)
        .is_verifiably_disjoint_from(&Schema::Boolean(None), &ctx)
        .unwrap());
    assert_eq!(
        account.requests(),
        vec![one[0]; 4],
        "each recursive parent remains live"
    );
    account.reset();
    assert!(alternatives(4, 3)
        .is_verifiably_disjoint_from(&Schema::Boolean(None), &ctx)
        .unwrap());
    assert!(
        account.requests().is_empty(),
        "returned sibling frames reuse only the paid peak"
    );
    assert!(!alternatives(4, 3)
        .is_verifiably_disjoint_from(&Schema::Null, &ctx)
        .unwrap());
    assert!(account.requests().is_empty());
}

#[test]
fn array_object_intersection_refusals_stop_at_first_callback_and_retain_payer() {
    let reference = {
        let funding = ParserAllocationFunding::unenforced();
        let allocation = CompilerAllocation(&funding);
        let pre = PreContext::new(serde_json::json!({}), None, &allocation).unwrap();
        let ctx = Context::new(&pre).unwrap();
        format!(
            "{:?}",
            nested(3, None)
                .intersect(nested(3, Some(true)), &ctx, 0)
                .unwrap()
        )
    };
    let successful = {
        let account = Account::new();
        let allocation = CompilerAllocation(&account.funding);
        let pre = PreContext::new(serde_json::json!({}), None, &allocation).unwrap();
        let ctx = Context::new(&pre).unwrap();
        account.reset();
        let schema = nested(3, None)
            .intersect(nested(3, Some(true)), &ctx, 0)
            .unwrap();
        assert_eq!(format!("{schema:?}"), reference);
        account.requests().len()
    };
    assert!(successful > 3);
    for cut in 0..successful {
        let account = Account::new();
        let error = {
            let allocation = CompilerAllocation(&account.funding);
            let pre = PreContext::new(serde_json::json!({}), None, &allocation).unwrap();
            let ctx = Context::new(&pre).unwrap();
            account.reset();
            account.cut.store(cut, Ordering::SeqCst);
            let error = nested(3, None)
                .intersect(nested(3, Some(true)), &ctx, 0)
                .unwrap_err();
            assert_eq!(refusal(&error), cut);
            assert_eq!(
                account.requests().len(),
                cut + 1,
                "no callback after the first refusal"
            );
            // Even previously paid capacity cannot hide the account's refusal.
            assert_eq!(
                refusal(&Schema::Null.intersect(Schema::Null, &ctx, 0).unwrap_err()),
                cut
            );
            assert_eq!(account.requests().len(), cut + 1);
            error
        };
        let retired = account.retired.clone();
        drop(account);
        assert_eq!(
            retired.load(Ordering::SeqCst),
            0,
            "the original error retains its payer"
        );
        drop(error);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn public_schema_compile_preserves_nested_intersection_semantics() {
    let schema = serde_json::json!({"allOf": [
        {"type":"object", "required":["items"], "properties": {"items": {
            "type":"array", "items": {"type":"object", "properties": {"enabled":{"type":"boolean"}}}}}},
        {"type":"object", "properties": {"items": {
            "type":"array", "minItems":1, "items": {"type":"object", "properties": {"enabled":{"const":true}}}}}}
    ]});
    let ordinary = build_schema(
        schema.clone(),
        &JsonCompileOptions::default(),
        &ParserAllocationFunding::unenforced(),
    )
    .unwrap();
    let account = Account::new();
    let original = build_schema(schema, &JsonCompileOptions::default(), &account.funding).unwrap();
    assert_eq!(
        format!("{:?}", original.schema),
        format!("{:?}", ordinary.schema)
    );
    assert!(original.warnings.is_empty());
}

fn copy_fixture() -> Schema {
    let mut source = nested(3, Some(true));
    let Schema::Object(root) = &mut source else {
        unreachable!()
    };
    root.pattern_properties.insert(
        "^extra".to_owned(),
        Schema::String(StringSchema {
            min_length: 1,
            max_length: Some(9),
            regex: Some(RegexAst::And(vec![
                RegexAst::Literal("alpha".to_owned()),
                RegexAst::Literal("beta".to_owned()),
            ])),
        }),
    );
    root.properties.insert(
        "alternatives".to_owned(),
        Schema::AnyOf(vec![
            Schema::Null,
            Schema::Ref("#/definitions/alias".to_owned()),
            Schema::OneOf(vec![Schema::Boolean(None), Schema::Any]),
        ]),
    );
    source
}

#[test]
fn recursive_schema_copy_preserves_independent_destinations_and_every_refusal() {
    let source = copy_fixture();
    let expected = format!("{source:?}");
    let account = Account::new();
    account.reset();
    let destination = source.copy_with_funding(&account.funding).unwrap();
    let requests = account.requests().len();
    assert!(requests > 3);
    let (Schema::Object(source_root), Schema::Object(destination_root)) = (&source, &destination)
    else {
        unreachable!()
    };
    for ((name, _), (copy, _)) in source_root
        .properties
        .iter()
        .zip(&destination_root.properties)
    {
        assert_eq!(name, copy);
        assert_ne!(name.as_ptr(), copy.as_ptr());
    }
    drop(source);
    assert_eq!(format!("{destination:?}"), expected);
    drop(destination);
    let retired = account.retired.clone();
    drop(account);
    assert_eq!(retired.load(Ordering::SeqCst), 1);

    for cut in 0..requests {
        let source = copy_fixture();
        let account = Account::new();
        account.reset();
        account.cut.store(cut, Ordering::SeqCst);
        let error = source.copy_with_funding(&account.funding).unwrap_err();
        assert_eq!(refusal(&error), cut);
        assert_eq!(account.requests().len(), cut + 1);
        assert_eq!(
            format!("{source:?}"),
            expected,
            "refusal leaves the borrowed source intact"
        );
        let retired = account.retired.clone();
        drop((source, account));
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(error);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
