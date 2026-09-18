//! Run with local/onig,reference/onig for the default regex feature profile.
//! Both aliases use pinned tokenizers0.23.2; local selects this repository fork.
use local::PreTokenizer as _;
use reference::PreTokenizer as _;
use serde_json::json;
fn main() {
    let patterns:Vec<String>=serde_json::from_str(r###"["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", "\\w+|[^\\w\\s]+", "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\x{200C}|\\x{200D})+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"]"###).unwrap();
    for (index, pattern) in patterns.iter().enumerate() {
        let json=json!({"version":"1.0","truncation":null,"padding":null,"normalizer":null,
   "pre_tokenizer":{"type":"Split","pattern":{"Regex":pattern},"behavior":"Isolated","invert":false},
   "post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],
   "model":{"type":"BPE","vocab":{"a":0},"merges":[]}}).to_string();
        let actual = local::tokenizer::TokenizerCompilePlan::prepare_json(json.as_bytes())
            .unwrap()
            .compile()
            .unwrap();
        let expected = reference::Tokenizer::from_bytes(json.as_bytes()).unwrap();
        let ordinary = local::Tokenizer::from_bytes(json.as_bytes()).unwrap();
        assert!(
            actual.matches_compiled_configuration(&ordinary),
            "source spelling and engine selection {index}"
        );
        let mut input = String::new();
        let mut samples = 0;
        let mut differences = 0;
        for scalar in (0..=0x10ffff).filter_map(char::from_u32) {
            use std::fmt::Write as _;
            write!(input, "a{scalar}B1 '{scalar}s {scalar}\n").unwrap();
            samples += 1;
            if samples % 256 == 0 || scalar == '\u{10ffff}' {
                let mut a = local::PreTokenizedString::from(input.as_str());
                actual
                    .get_pre_tokenizer()
                    .unwrap()
                    .pre_tokenize(&mut a)
                    .unwrap();
                let mut b = reference::PreTokenizedString::from(input.as_str());
                expected
                    .get_pre_tokenizer()
                    .unwrap()
                    .pre_tokenize(&mut b)
                    .unwrap();
                let a: Vec<_> = a
                    .get_splits(local::OffsetReferential::Original, local::OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect();
                let b: Vec<_> = b
                    .get_splits(
                        reference::OffsetReferential::Original,
                        reference::OffsetType::Byte,
                    )
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect();
                if a != b {
                    differences += 1;
                    if differences <= 5 {
                        println!("pattern={index} firstdifferent chunk end U+{:X}; first split diff={:?}",scalar as u32,a.iter().zip(&b).find(|(a,b)|a!=b));
                    }
                }
                input.clear();
            }
        }
        println!("pattern={index} Unicode scalars={samples} differing chunks={differences}");
        assert_eq!(
            differences, 0,
            "selected feature semantics for expression {index}"
        );
    }
}
