use super::*;
use crate::api::ParserLimits;
use std::{error::Error, fmt, sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}}};

#[derive(Debug)]
struct Refused(usize);
impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "emitter refusal {}", self.0) }
}
impl Error for Refused {}
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }
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
        let (events, stop, owner) = (calls.clone(), cut.clone(), Retired(retired.clone()));
        let funding = ParserAllocationFunding::prepare(move |bytes| {
            let _owner = &owner;
            let mut events = events.lock().unwrap();
            let ordinal = events.len(); events.push(bytes);
            if ordinal == stop.load(Ordering::SeqCst) { Err(Refused(ordinal)) } else { Ok(()) }
        }).unwrap();
        Self { funding, calls, cut, retired }
    }
    fn reset(&self) { self.calls.lock().unwrap().clear(); }
    fn requests(&self) -> Vec<usize> { self.calls.lock().unwrap().clone() }
}
fn refusal(error: &ParserError) -> usize {
    let mut cause: &dyn Error = error;
    loop {
        if let Some(cause) = cause.downcast_ref::<Refused>() { return cause.0; }
        cause = cause.source().expect("original emitter refusal");
    }
}
fn ast(depth: usize, width: usize) -> RegexAst {
    if depth == 0 { RegexAst::Literal("x".to_owned()) }
    else { RegexAst::Concat((0..width).map(|_| ast(depth-1, width)).collect()) }
}

#[test]
fn emitter_inspection_pays_live_depth_reuses_siblings_and_preserves_first_failure() {
    let account = Account::new();
    let scope = Scope::new(&account.funding).unwrap();
    account.reset();
    assert!(always_non_empty(&ast(0, 1), &scope, &account.funding).unwrap());
    let first = account.requests(); assert_eq!(first.len(), 1);
    account.reset();
    assert!(always_non_empty(&ast(5, 1), &scope, &account.funding).unwrap());
    assert_eq!(account.requests(), vec![first[0]; 5]);
    account.reset();
    assert!(always_non_empty(&ast(5, 3), &scope, &account.funding).unwrap());
    assert!(account.requests().is_empty());
    // A heap refusal is sticky even when the next recursive frame fits the peak.
    account.cut.store(0, Ordering::SeqCst);
    let original = account.funding.reserve(17).unwrap_err();
    let error = always_non_empty(&ast(0, 1), &scope, &account.funding).unwrap_err();
    assert_eq!(refusal(&error), 0); assert_eq!(account.requests(), vec![17]);
    let retired = account.retired.clone();
    drop(scope); drop(account); drop(original);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(error); assert_eq!(retired.load(Ordering::SeqCst), 1);
}

fn builder(funding: &ParserAllocationFunding) -> GrammarBuilder<'static> {
    GrammarBuilder::new(None, ParserLimits::default(), funding.clone()).unwrap()
}

#[test]
fn field_sequence_emission_refuses_before_growth_and_retains_escaping_cause() {
    let run = |cut| {
        let account = Account::new();
        let scope = Scope::new(&account.funding).unwrap();
        let mut compiler = Compiler::new(JsonCompileOptions::default(), builder(&account.funding), &scope).unwrap();
        compiler.builder.add_grammar(LLGuidanceOptions::default(), RegexAst::NoMatch).unwrap();
        let node = compiler.builder.string("x").unwrap();
        let items = vec![(node, false); 48];
        let mut cache = HashMap::default();
        account.reset(); account.cut.store(cut, Ordering::SeqCst);
        let result = compiler.ordered_sequence(&items, false, &mut cache);
        let requests = account.requests();
        let retired = account.retired.clone();
        drop(compiler); drop(scope); drop(cache); drop(items); drop(account);
        match result {
            Ok(_) => { assert_eq!(cut, usize::MAX); assert_eq!(retired.load(Ordering::SeqCst), 1); }
            Err(error) => {
                assert_eq!(refusal(&error), cut);
                assert_eq!(requests.len(), cut+1);
                assert_eq!(retired.load(Ordering::SeqCst), 0);
                drop(error); assert_eq!(retired.load(Ordering::SeqCst), 1);
            }
        }
        requests
    };
    let reached = run(usize::MAX);
    assert!(reached.len() > 48, "field-count recursion is reached independently of schema depth");
    for cut in [0, 1, reached.len()/2, reached.len()-1] { run(cut); }
}

fn public_schema() -> Value {
    serde_json::json!({"type":"object", "properties": {
        "rows": {"type":"array", "minItems":1, "maxItems":2, "items": {
            "anyOf": [{"const":17}, {"type":"object", "properties":{"enabled":{"const":true}},
                "required":["enabled"], "additionalProperties":false}]}},
        "label":{"enum":["ok","ready"]}}, "required":["rows"], "additionalProperties":false})
}
fn emitted(entry: usize, schema: Value, funding: &ParserAllocationFunding) -> Result<GrammarResult<'static>> {
    let options = JsonCompileOptions::default();
    let builder = builder(funding);
    match entry {
        0 => options.json_to_llg(builder, schema),
        1 => options.json_to_llg_no_validate(builder, schema),
        _ => options.json_to_llg_with_overrides(builder, schema),
    }
}
fn accepts(grammar: &crate::earley::SharedGrammar, text: &str) -> bool {
    let mut parser = crate::earley::Parser::new(toktrie::ApproximateTokEnv::single_byte_env(),
        grammar.clone(), ParserLimits::default(), Arc::new(crate::earley::perf::ParserPerfCounters::new())).unwrap();
    for byte in text.bytes() {
        if parser.apply_token(&[byte], u32::from(byte)).is_err() { return false; }
    }
    parser.is_accepting()
}

#[test]
fn public_json_emitter_entries_preserve_nested_arrays_objects_unions_and_custody() {
    let examples = [(r#"{"rows":[17]}"#, true),
        (r#"{"rows":[{"enabled":true},17],"label":"ready"}"#, true),
        (r#"{"rows":[]}"#, false), (r#"{"rows":[18]}"#, false),
        (r#"{"rows":[{"enabled":false}]}"#, false),
        (r#"{"rows":[17],"label":"bad"}"#, false)];
    for entry in 0..3 {
        let account = Account::new();
        let mut schema = public_schema();
        if entry == 2 { schema["x-guidance"] = serde_json::json!({"whitespace_flexible":true}); }
        let grammar = {
            let result = emitted(entry, schema.clone(), &account.funding).unwrap();
            let ordinary = emitted(entry, schema, &ParserAllocationFunding::unenforced()).unwrap();
            assert_eq!(result.builder.grammar.to_string(Some(&result.builder.regex.spec)),
                ordinary.builder.grammar.to_string(Some(&ordinary.builder.regex.spec)));
            let GrammarResult { builder, .. } = result;
            let GrammarBuilder { grammar, regex, funding, .. } = builder;
            crate::earley::SharedGrammar::new(grammar.compile(
                regex.spec, &ParserLimits::default(), funding).unwrap()).unwrap()
        };
        let retired = account.retired.clone(); drop(account);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        for (text, expected) in examples { assert_eq!(accepts(&grammar, text), expected, "entry={entry}: {text}"); }
        drop(grammar); assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn each_public_json_emitter_entry_refuses_before_its_first_unpaid_control() {
    for entry in 0..3 {
        let account = Account::new();
        let options = JsonCompileOptions::default();
        let builder = builder(&account.funding);
        let mut schema = public_schema();
        if entry == 2 { schema["x-guidance"] = serde_json::json!({"whitespace_flexible":true}); }
        account.reset(); account.cut.store(0, Ordering::SeqCst);
        let result = match entry {
            0 => options.json_to_llg(builder, schema),
            1 => options.json_to_llg_no_validate(builder, schema),
            _ => options.json_to_llg_with_overrides(builder, schema),
        };
        let error = result.err().expect("first reached control must be paid");
        assert_eq!(refusal(&error), 0); assert_eq!(account.requests().len(), 1);
        let retired = account.retired.clone(); drop(account);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(error); assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
