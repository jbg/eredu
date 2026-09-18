use jsonschema_regex as current;
use jsonschema_regex_reference as reference;

fn compare(pattern: &str) {
    let actual = current::to_rust_regex(pattern).map(|value| (matches!(value, std::borrow::Cow::Borrowed(_)), value.into_owned()));
    let expected = reference::to_rust_regex(pattern).map(|value| (matches!(value, std::borrow::Cow::Borrowed(_)), value.into_owned()));
    assert_eq!(actual, expected, "translation: {pattern:?}");
    assert_eq!(current::is_valid_ecma_regex(pattern), reference::is_valid_ecma_regex(pattern), "syntax: {pattern:?}");
    assert_eq!(format!("{:?}", current::analyze_pattern(pattern)), format!("{:?}", reference::analyze_pattern(pattern)), "analysis: {pattern:?}");
    assert_eq!(current::pattern_witness(pattern), reference::pattern_witness(pattern), "witness: {pattern:?}");
    assert_eq!(current::pattern_as_prefix(pattern), reference::pattern_as_prefix(pattern), "prefix: {pattern:?}");
}

fn main() {
    let atoms = ["", "a", "abc", "水", "a-b", "é", r"\d", r"\D", r"\w", r"\W", r"\s", r"\S", r"\cA", r"\cz", r"\c0", r"\a", r"\.", r"\/", r"\$", r"[\d\w]", r"[^\s]", "[a-z]", "[水🦀]", r"\p{Letter}", "(?=a)", "(?<!x)", "(?<name>a)", r"\k<name>", r"\1", "[", "(", "a{3,2}"];
    let suffixes = ["", "*", "+", "?", "{2}", "{0,3}", "$", "|z"];
    let mut patterns = 0;
    for atom in atoms {
        for suffix in suffixes {
            for prefix in ["", "^", "a", r"\cB"] {
                compare(&format!("{prefix}{atom}{suffix}"));
                compare(&format!("^({atom}{suffix}|water)$"));
                patterns += 2;
            }
        }
    }
    for first in [r"\d", r"\w", r"\s", r"\cA", r"\cB", "water"] {
        for second in [r"\d", r"\w", r"\s", r"\cA", r"\cB", "水"] {
            compare(&format!("{first}{second}"));
            compare(&format!("[{first}{second}]"));
            patterns += 2;
        }
    }
    for source in [r"(?<\u0061>a)\k<a>", r"(?<a>x)|(?<a>y)", r"(?<a>x)(?<a>y)", r"(?ims-i:a)", r"(?s:a)(?-s:b)", r"^(x|a\/b|x)$", r"^\S*$", r"^\$ref$", r"(?<a>(?<b>x)|(?<b>y))"] {
        compare(source); patterns += 1;
    }
    println!("{patterns} patterns: {} independent exact comparisons passed", patterns * 5);
}
