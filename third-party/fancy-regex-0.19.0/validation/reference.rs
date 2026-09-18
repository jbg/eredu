use fancy_regex as current;
use fancy_regex_reference as reference;

fn compare(pattern: &str, texts: &[String]) -> usize {
    let actual = current::RegexOptionsBuilder::new()
        .build_with_allocations(pattern, &current::allocation::Unenforced);
    let expected = reference::Regex::new(pattern);
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            let mut scoped = actual
                .search_workspace_with_allocations(&current::allocation::Unenforced)
                .unwrap();
            for text in texts {
                let actual_spans: Result<Vec<_>, _> = actual
                    .find_iter(text)
                    .map(|m| m.map(|m| (m.start(), m.end())))
                    .collect();
                let expected_spans: Result<Vec<_>, _> = expected
                    .find_iter(text)
                    .map(|m| m.map(|m| (m.start(), m.end())))
                    .collect();
                assert_eq!(
                    actual_spans.map_err(|e| e.to_string()),
                    expected_spans.map_err(|e| e.to_string()),
                    "{pattern:?} / {text:?}"
                );
                assert_eq!(
                    scoped
                        .find(text, &current::allocation::Unenforced)
                        .map(|m| m.map(|m| (m.start(), m.end())))
                        .map_err(|e| e.to_string()),
                    expected
                        .find(text)
                        .map(|m| m.map(|m| (m.start(), m.end())))
                        .map_err(|e| e.to_string()),
                    "scoped find {pattern:?} / {text:?}",
                );
                assert_eq!(
                    scoped
                        .is_match(text, &current::allocation::Unenforced)
                        .map_err(|e| e.to_string()),
                    expected.is_match(text).map_err(|e| e.to_string()),
                    "scoped boolean {pattern:?} / {text:?}",
                );
                let a = actual.captures(text).map(|row| {
                    row.map(|row| {
                        row.iter()
                            .map(|m| m.map(|m| (m.start(), m.end())))
                            .collect::<Vec<_>>()
                    })
                });
                let b = expected.captures(text).map(|row| {
                    row.map(|row| {
                        row.iter()
                            .map(|m| m.map(|m| (m.start(), m.end())))
                            .collect::<Vec<_>>()
                    })
                });
                assert_eq!(
                    a.map_err(|e| e.to_string()),
                    b.map_err(|e| e.to_string()),
                    "captures {pattern:?} / {text:?}"
                );
            }
            texts.len() * 4
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(
                actual.to_string(),
                expected.to_string(),
                "diagnostic {pattern:?}"
            );
            1
        }
        (a, b) => panic!("acceptance differs: {pattern:?}: current={a:?}, pristine={b:?}"),
    }
}

fn main() {
    let texts: Vec<String> = [
        "",
        "a",
        "aaa",
        "abab",
        "abcc",
        "bca",
        "a\r\nb",
        "élève",
        "éÉ",
        "kKK",
        "ſSs",
        "水🙂",
        "a水a",
        "123",
        " a \n",
        "a\u{301}",
        "\0",
        "\u{10ffff}",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let atoms = [
        "", "a", "ab", "a|b", "[a-z]", "[^a]", r"\d", r"\w", r"\s", r"\R", r"\N", "水", "(?=a)",
        "(?!b)", "(?<=a)", "(?<!b)", "(a)", r"(a)\1", "(?i:k)", "(?i:ſ)", "(?>a|ab)", r"\Ga",
        "(*FAIL)", "(?~a)", "(", "[", "a{3,1}",
    ];
    let mut comparisons = 0;
    let mut patterns = 0;
    for atom in atoms {
        for suffix in ["", "?", "+", "*", "{0,2}", "|z", "$", "(?!z)"] {
            for prefix in ["", "^"] {
                comparisons += compare(&format!("{prefix}{atom}{suffix}"), &texts);
                patterns += 1;
            }
        }
    }
    let mut inputs = texts;
    let alphabet = [
        'a', 'Z', 'é', '東', '🙂', '\u{301}', '\u{200c}', '0', '९', ' ', '\r', '\n', '\t', '\'',
        '/', '!', '\0',
    ];
    let mut seed = 0x7f123abcu32;
    for _ in 0..512 {
        let mut text = String::new();
        for _ in 0..32 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            text.push(alphabet[seed as usize % alphabet.len()]);
        }
        inputs.push(text);
    }
    for pattern in current::workspace::construction::patterns() {
        let source = current::workspace::construction::Plan::new(pattern)
            .unwrap()
            .prepare()
            .unwrap();
        let mut workspace = source.plan().unwrap().prepare().unwrap();
        let original = reference::Regex::new(pattern).unwrap();
        for text in &inputs {
            let actual: Vec<_> = workspace
                .find_iter(text)
                .map(|m| {
                    let m = m.unwrap();
                    (m.start(), m.end())
                })
                .collect();
            let expected: Vec<_> = original
                .find_iter(text)
                .map(|m| {
                    let m = m.unwrap();
                    (m.start(), m.end())
                })
                .collect();
            assert_eq!(actual, expected, "closed source {pattern:?} / {text:?}");
            comparisons += 1;
        }
        patterns += 1;
    }
    println!("{patterns} patterns: {comparisons} independent matching/capture/diagnostic comparisons passed");
}
