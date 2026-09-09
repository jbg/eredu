use eredu_text::tokenizer::Tokenizer;
use serde::Deserialize;
use serde_json::{json, Value};

fn tokenizer() -> Tokenizer {
    Tokenizer::from_tokenizer(tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::default(),
    ))
}

#[test]
fn tojson_matches_python_json_dumps() {
    #[derive(Deserialize)]
    struct Case {
        name: String,
        value: Value,
        template: String,
        expected: String,
    }
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/tojson/python-json-dumps.json")).unwrap();
    let mut tokenizer = tokenizer();
    for case in cases {
        let rendered = tokenizer
            .apply_chat_template_json(
                case.template.as_str(),
                [Vec::new()],
                Some(&[case.value]),
                &case.name,
                false,
                None,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(rendered, [case.expected], "{}", case.name);
    }
}

#[test]
fn tojson_rejects_invalid_options() {
    for arguments in [
        "unknown=true",
        "indent=1.5",
        "indent=[]",
        "separators=(\",\",)",
        "separators=(\",\", \":\", \"!\")",
        "separators=(1, 2)",
        "false, ensure_ascii=true",
        "false, none, none, false, true",
    ] {
        let template = format!("{{{{ tools[0] | tojson({arguments}) }}}}");
        assert!(
            tokenizer()
                .apply_chat_template_json(
                    template.as_str(),
                    [Vec::new()],
                    Some(&[json!({"a": 1})]),
                    "invalid-json-options",
                    false,
                    None,
                )
                .is_err(),
            "accepted {arguments}"
        );
    }
}
