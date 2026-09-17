use super::*;
use crate::pre_tokenizers::{
    byte_level::ByteLevel,
    sequence::Sequence,
    split::{Split, SplitPattern},
    PreTokenizerWrapper,
};
use crate::{OffsetReferential, OffsetType};

fn splits(pre: &impl PreTokenizer, text: &str) -> Vec<(String, Offsets)> {
    let mut input = PreTokenizedString::from(text);
    pre.pre_tokenize(&mut input).unwrap();
    input
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, offsets, _)| (s.to_owned(), offsets))
        .collect()
}
fn ordinary(pattern: &str) -> Split {
    Split::new(
        SplitPattern::Regex(pattern.to_owned()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .unwrap()
}
#[test]
fn all_actual_sources_preserve_nonzero_split_spans_and_bytelevel_offsets() {
    for pattern in construction::patterns() {
        let compiled =
            CompiledRegexSplit::compile(construction::Plan::new(pattern).unwrap()).unwrap();
        let ordinary = ordinary(pattern);
        for text in [
            "Hello WORLD's 1234567!\r\n αβé e\u{301} 中文🙂 ",
            "can't I'M\t ",
            "a\u{200c}b\u{200d}c",
            "",
            "  /\r\n",
        ] {
            assert_eq!(
                splits(&compiled, text),
                splits(&ordinary, text),
                "{pattern}: {text:?}"
            );
        }
        let bounded = Sequence::new(vec![
            PreTokenizerWrapper::CompiledRegexSplit(compiled),
            ByteLevel::new(false, true, false).into(),
        ]);
        let legacy = Sequence::new(vec![
            ordinary.into(),
            ByteLevel::new(false, true, false).into(),
        ]);
        for text in [
            "é🙂 ABC's 123456 ",
            "\0\u{7f}\u{80}\u{ad}\n",
            "e\u{301}𐐀\u{200d}",
            "",
        ] {
            assert_eq!(splits(&bounded, text), splits(&legacy, text));
        }
    }
}
#[test]
fn closed_clone_keeps_actual_source_and_serializes_to_ordinary_split_shape() {
    let pattern = construction::patterns().nth(2).unwrap();
    let compiled = CompiledRegexSplit::compile(construction::Plan::new(pattern).unwrap()).unwrap();
    let clone = compiled.clone();
    assert_eq!(Arc::strong_count(compiled.owner.0.as_ref().unwrap()), 2);
    assert!(std::ptr::eq(compiled.owner.source(), clone.owner.source()));
    let expected = splits(&ordinary(pattern), "ABC's élève 12345 ");
    let json = serde_json::to_string(&compiled).unwrap();
    assert_eq!(json, serde_json::to_string(&ordinary(pattern)).unwrap());
    let restored: PreTokenizerWrapper = serde_json::from_str(&json).unwrap();
    assert!(matches!(restored, PreTokenizerWrapper::Split(_)));
    drop(compiled);
    assert_eq!(Arc::strong_count(clone.owner.0.as_ref().unwrap()), 1);
    assert_eq!(splits(&clone, "ABC's élève 12345 "), expected);
    assert_eq!(splits(&restored, "ABC's élève 12345 "), expected);
}
#[test]
fn repeated_coverage_reuses_one_workspace_and_keeps_full_mapped_splits() {
    let pattern = construction::patterns().nth(2).unwrap();
    let source = CompiledRegexSplit::compile(construction::Plan::new(pattern).unwrap()).unwrap();
    let mut workspace = source.workspace_plan().unwrap().prepare().unwrap();
    let before = workspace.capacities();
    let text = "ABC's 123456 élève\r\n🙂";
    let expected = splits(&ordinary(pattern), text);
    for _ in 0..100 {
        let mut actual = Vec::new();
        visit_spans::<fancy_regex::Error>(&mut workspace, text, |(start, end)| {
            if start != end {
                actual.push((text[start..end].to_owned(), (start, end)));
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(workspace.capacities(), before);
    }
}
