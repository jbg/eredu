use super::*;
use eredu_core::SharedStorageDomain;
use serde_json::json;
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

fn grammar() -> TopLevelGrammar {
    TopLevelGrammar::from_json_schema(json!({"type":"object", "maxProperties":0}))
}

fn recipe(
    tools: &[Value],
    spellings: &[String],
    ids: &[u32],
    stops: &[String],
) -> ConstraintRecipe {
    ConstraintRecipe::new(
        Some(
            br#"{"model":{"type":"BPE","vocab":{"a":7,"z":9001}},"decoder":{"type":"ByteLevel"}}"#,
        ),
        &grammar(),
        tools,
        &[9001, u32::MAX],
        spellings,
        ids,
        stops,
        Some("工具:"),
    )
    .unwrap()
}

fn assert_borrowed_from_source(recipe: &ConstraintRecipe, value: &[u8]) {
    let source = recipe.source().as_ref();
    let first = source.as_ptr() as usize;
    let last = first + source.len();
    let borrowed = value.as_ptr() as usize;
    assert!(borrowed >= first && borrowed <= last);
    assert!(value.len() <= last - borrowed);
}

#[test]
fn packed_recipe_round_trips_original_tools_separately_from_chosen_fallback_grammar() {
    let tools = vec![json!({
        "type":"function", "function": {
            "name":"測定", "parameters": {
                "type":"object", "properties":{"length":{"minimum":17}},
                "required":["length"]
            }
        }
    })];
    let spellings = vec!["<|tool|>".to_owned(), "終わり".to_owned(), String::new()];
    let ids = [7, 9001, u32::MAX];
    let stops = vec!["\n終".to_owned(), "\0done".to_owned(), String::new()];
    let recipe = recipe(&tools, &spellings, &ids, &stops);
    assert_eq!(recipe.tools().unwrap(), tools);
    assert_eq!(
        serde_json::to_value(recipe.grammar().unwrap()).unwrap(),
        serde_json::to_value(grammar()).unwrap()
    );
    assert_eq!(recipe.eos_token_ids().collect::<Vec<_>>(), [9001, u32::MAX]);
    assert_eq!(recipe.structural_tokens().len(), 3);
    assert_eq!(
        recipe.structural_tokens().collect::<Vec<_>>(),
        [(7, "<|tool|>"), (9001, "終わり"), (u32::MAX, "")]
    );
    assert_eq!(
        recipe.stop_sequences().collect::<Vec<_>>(),
        ["\n終", "\0done", ""]
    );
    assert_eq!(recipe.trigger(), Some("工具:"));
    let tokenizer: Value = serde_json::from_slice(recipe.tokenizer_json().unwrap()).unwrap();
    assert_eq!(tokenizer["model"]["vocab"]["z"], json!(9001));
    assert_eq!(tokenizer["decoder"]["type"], "ByteLevel");
    assert_borrowed_from_source(&recipe, recipe.tokenizer_json().unwrap());
    for (_, spelling) in recipe.structural_tokens() {
        assert_borrowed_from_source(&recipe, spelling.as_bytes());
    }
    for stop in recipe.stop_sequences() {
        assert_borrowed_from_source(&recipe, stop.as_bytes());
    }
    assert_borrowed_from_source(&recipe, recipe.trigger().unwrap().as_bytes());
    assert!(recipe.source().capacity_bytes().unwrap() >= recipe.source().as_ref().len() as u64);
}

#[test]
fn clone_retains_one_source_and_decoded_values_are_independent_after_inputs_retire() {
    let recipe = {
        let tools = vec![json!({"name":"original","schema":{"minimum":23}})];
        recipe(&tools, &["λ".into()], &[311], &["stop".into()])
    };
    let copy = recipe.clone();
    assert!(recipe.source().same_storage(copy.source()));
    assert_eq!(
        recipe.trigger().unwrap().as_ptr(),
        copy.trigger().unwrap().as_ptr()
    );
    let mut decoded = recipe.tools().unwrap();
    decoded[0]["name"] = json!("changed");
    assert_eq!(copy.tools().unwrap()[0]["name"], "original");
    drop(recipe);
    assert_eq!(copy.structural_tokens().collect::<Vec<_>>(), [(311, "λ")]);
    assert_eq!(copy.stop_sequences().collect::<Vec<_>>(), ["stop"]);
    assert_eq!(copy.tools().unwrap()[0]["schema"]["minimum"], 23);
}

#[test]
fn semantic_equality_matches_values_including_key_order_and_signed_zero() {
    let first: Value = serde_json::from_str(r#"{"z":17,"a":{"limit":-0.0,"word":"é"}}"#).unwrap();
    let second: Value = serde_json::from_str(r#"{"a":{"word":"é","limit":0.0},"z":17}"#).unwrap();
    let left = recipe(
        std::slice::from_ref(&first),
        &["x".into()],
        &[41],
        &["stop".into()],
    );
    let right = ConstraintRecipe::new(
        Some(br#"{"model":{"type":"WordLevel","vocab":{"x":2}}}"#),
        &TopLevelGrammar::from_regex("different grammar"),
        std::slice::from_ref(&second),
        &[3],
        &["x".into()],
        &[41],
        &["stop".into()],
        None,
    )
    .unwrap();
    // This also respects serde_json's alternate arbitrary_precision equality
    // if another consumer enables that dependency feature.
    assert_eq!(left.semantic_eq(&right), first == second);
    assert_eq!(left.tools().unwrap(), [first]);
    assert_eq!(right.tools().unwrap(), [second]);

    let reordered: Value =
        serde_json::from_str(r#"{"a":{"word":"é","limit":-0.0},"z":17}"#).unwrap();
    let same = recipe(&[reordered], &["x".into()], &[41], &["stop".into()]);
    assert!(left.semantic_eq(&same));
    for changed in [
        recipe(
            &left.tools().unwrap(),
            &["x".into()],
            &[43],
            &["stop".into()],
        ),
        recipe(
            &left.tools().unwrap(),
            &["y".into()],
            &[41],
            &["stop".into()],
        ),
        recipe(
            &left.tools().unwrap(),
            &["x".into()],
            &[41],
            &["other".into()],
        ),
        recipe(&[json!({"z":19})], &["x".into()], &[41], &["stop".into()]),
    ] {
        assert!(!left.semantic_eq(&changed));
    }
}

#[test]
fn absent_and_empty_optional_sections_remain_distinct_with_exact_empty_iterators() {
    let absent = ConstraintRecipe::new(None, &grammar(), &[], &[], &[], &[], &[], None).unwrap();
    let empty =
        ConstraintRecipe::new(Some(&[]), &grammar(), &[], &[], &[], &[], &[], Some("")).unwrap();
    assert_eq!(absent.tokenizer_json(), None);
    assert_eq!(empty.tokenizer_json(), Some(&[][..]));
    assert_eq!(absent.trigger(), None);
    assert_eq!(empty.trigger(), Some(""));
    for recipe in [&absent, &empty] {
        assert_eq!(recipe.eos_token_ids().len(), 0);
        assert_eq!(recipe.structural_tokens().len(), 0);
        assert_eq!(recipe.stop_sequences().len(), 0);
        assert_eq!(recipe.tools().unwrap(), Vec::<Value>::new());
    }
    assert!(absent.semantic_eq(&empty));
}

#[test]
fn mismatched_token_metadata_and_checked_geometry_reject_without_partial_cursor_changes() {
    let mismatch = ConstraintRecipe::new(None, &grammar(), &[], &[], &["x".into()], &[], &[], None);
    assert!(mismatch.unwrap_err().contains("differ in length"));
    let mut cursor = usize::MAX;
    assert!(take_span(&mut cursor, 1).is_err());
    assert_eq!(cursor, usize::MAX);
    cursor = isize::MAX as usize;
    assert!(take_span(&mut cursor, 1).is_err());
    assert_eq!(cursor, isize::MAX as usize);
    assert!(token_bytes(usize::MAX).is_err());
    let mut cursor = HEADER_BYTES;
    let span = take_span(&mut cursor, 17).unwrap();
    assert_eq!(
        (span.start, span.end, cursor),
        (HEADER_BYTES, HEADER_BYTES + 17, HEADER_BYTES + 17)
    );
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn preexisting_recipe_and_source_aliases_share_attachment_until_the_final_source_retires() {
    let recipe = recipe(
        &[json!({"minimum":29})],
        &["x".into()],
        &[101],
        &["end".into()],
    );
    let alias = recipe.clone();
    let source = recipe.source().clone();
    let domain = SharedStorageDomain::default();
    let drops = Arc::new(AtomicUsize::new(0));
    assert!(recipe
        .source()
        .try_attach(&domain, || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Retired(Arc::clone(&drops))))
        })
        .unwrap());
    assert!(!alias
        .source()
        .try_attach(&domain, || -> Result<Box<dyn Send + Sync>, Infallible> {
            panic!("a preexisting alias must share the attached domain");
        })
        .unwrap());
    drop(recipe);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(!source.as_ref().is_empty());
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
