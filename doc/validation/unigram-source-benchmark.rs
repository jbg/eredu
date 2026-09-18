use std::{hint::black_box, time::Instant};
fn main() {
    let mut vocab = vec![("<unk>".to_owned(), -9.0), ("a".into(), -1.0), ("b".into(), -1.0), ("ab".into(), -1.0), (" ".into(), -0.1), ("é".into(), -0.2), ("中文".into(), -0.3)];
    for n in 0..100_000 { vocab.push((format!("v{n:05x}"), -5.0)); }
    let json = serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"Unigram","vocab":vocab,"unk_id":0,"byte_fallback":false}}).to_string();
    let now=Instant::now();let old=reference::Tokenizer::from_bytes(json.as_bytes()).unwrap();let old_build=now.elapsed();
    let now=Instant::now();let new=local::Tokenizer::from_bytes(json.as_bytes()).unwrap();let new_build=now.elapsed();
    let plan=local::TokenizerCompilePlan::prepare_json(json.as_bytes()).unwrap();let quote=plan.requirements();
    let now=Instant::now();let source=plan.compile().unwrap();let source_build=now.elapsed();
    let text="ab é 中文 🙂 v01000 ".repeat(512);
    let expected=old.encode(text.as_str(),false).unwrap();assert_eq!(source.encode(text.as_str(),false).unwrap().get_ids(),expected.get_ids());
    let now=Instant::now();for _ in 0..50 {black_box(old.encode(text.as_str(),false).unwrap());}let old_time=now.elapsed();
    let now=Instant::now();for _ in 0..50 {black_box(new.encode(text.as_str(),false).unwrap());}let new_time=now.elapsed();
    let now=Instant::now();for _ in 0..50 {black_box(local::EncodeIdsPlan::prepare(&source,&text,false).unwrap().encode().unwrap());}let ids_time=now.elapsed();
    println!("100007 scored pieces: ordinary build pristine={old_build:?} local={new_build:?}; planned original build={source_build:?}; source quote={quote:?}");
    println!("50 x {} bytes: pristine={old_time:?} local={new_time:?} ratio={:.3} original IDs={ids_time:?}",text.len(),new_time.as_secs_f64()/old_time.as_secs_f64());
}
