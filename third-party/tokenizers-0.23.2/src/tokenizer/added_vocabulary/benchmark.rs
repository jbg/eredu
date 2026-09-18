//! Opt-in reproducible CPU matcher measurement with independent literal answers.
use super::*;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn expected(patterns: &[String], input: &str) -> Vec<(u32, Offsets)> {
    let mut answer = Vec::new();
    let mut pos = 0;
    while pos < input.len() {
        let next = patterns
            .iter()
            .enumerate()
            .filter_map(|(id, pattern)| {
                input[pos..]
                    .find(pattern)
                    .map(|start| (id as u32, (pos + start, pos + start + pattern.len())))
            })
            .min_by_key(|(id, (start, end))| (*start, std::cmp::Reverse(end - start), *id));
        let Some(found) = next else { break };
        pos = found.1 .1;
        answer.push(found);
    }
    answer
}
fn run(name: &str, patterns: Vec<String>, input: String) {
    let mut added = AddedVocabulary::new();
    let construction = Instant::now();
    added
        .add_tokens(
            patterns.iter().map(|s| AddedToken::from(s.as_str(), true)),
            &crate::models::bpe::BPE::default(),
            None::<&crate::NormalizerWrapper>,
        )
        .unwrap();
    let construction = construction.elapsed();
    let answer = expected(&patterns, &input);
    assert_eq!(added.raw_matches(&input, false).collect::<Vec<_>>(), answer);
    let start = Instant::now();
    let mut iterations = 0;
    while start.elapsed() < Duration::from_millis(350) {
        assert_eq!(
            black_box(added.raw_matches(black_box(&input), false).count()),
            answer.len()
        );
        iterations += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "{name}: patterns={} input_bytes={} matches={} build_us={} MiB/s={:.2}",
        patterns.len(),
        input.len(),
        answer.len(),
        construction.as_micros(),
        input.len() as f64 * iterations as f64 / elapsed / 1048576.0
    );
}
#[test]
#[ignore = "release throughput measurement; run with --release --ignored --nocapture"]
fn added_matcher_throughput() {
    let chat = (0..256)
        .map(|i| format!("<|special_{i}|>"))
        .collect::<Vec<_>>();
    run(
        "chat_sparse",
        chat.clone(),
        "The weather in Madrid is sunny. café 東京 🦀. <|special_42|>\n".repeat(512),
    );
    run(
        "chat_absent",
        chat,
        "Ordinary prose without delimiters, including café 東京.\n".repeat(512),
    );
    run(
        "shared_prefix_absent",
        (32..64).map(|n| format!("{}b", "a".repeat(n))).collect(),
        "a".repeat(32768),
    );
    run(
        "shared_prefix_matches",
        (1..64).map(|n| "a".repeat(n)).collect(),
        "a".repeat(32768),
    );
    run(
        "delayed_short_match",
        vec!["a".into(), format!("{}b", "a".repeat(255))],
        "a".repeat(32768),
    );
    run(
        "unicode",
        vec!["é".into(), "é🦀".into(), "東京".into(), "京".into()],
        "é🦀 東京 é!".repeat(2048),
    );
}

#[test]
fn literal_matches_equal_independent_leftmost_longest_search() {
    let mut seed = 19u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };
    let alphabet = ["a", "b", "é", "🦀"];
    for _ in 0..250 {
        let mut patterns = Vec::new();
        for _ in 0..12 {
            let n = (next() % 7 + 1) as usize;
            let pattern = (0..n)
                .map(|_| alphabet[(next() >> 16) as usize % alphabet.len()])
                .collect::<String>();
            if !patterns.contains(&pattern) {
                patterns.push(pattern)
            }
        }
        let input = (0..128)
            .map(|_| alphabet[(next() >> 16) as usize % alphabet.len()])
            .collect::<String>();
        let mut added = AddedVocabulary::new();
        added
            .add_tokens(
                patterns.iter().map(|s| AddedToken::from(s.as_str(), true)),
                &crate::models::bpe::BPE::default(),
                None::<&crate::NormalizerWrapper>,
            )
            .unwrap();
        assert_eq!(
            added.raw_matches(&input, false).collect::<Vec<_>>(),
            expected(&patterns, &input),
            "patterns={patterns:?}, input={input:?}"
        );
    }
}
