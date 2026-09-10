use eredu_text::tokenizer::Tokenizer;
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
struct Case {
    name: String,
    template: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
    kwargs: Map<String, Value>,
    template_defaults: Map<String, Value>,
    add_generation_prompt: bool,
    rendered: String,
    token_ids: Vec<u32>,
}

fn cases() -> Vec<Case> {
    serde_json::from_str(include_str!("fixtures/k2_horizon/reference.json")).unwrap()
}

fn template(name: &str) -> &'static str {
    match name {
        "dense-safetensors" => include_str!("fixtures/k2_horizon/dense-safetensors.jinja"),
        "dense-gguf" => include_str!("fixtures/k2_horizon/dense-gguf.jinja"),
        "mova-safetensors" => include_str!("fixtures/k2_horizon/mova-safetensors.jinja"),
        "mova-gguf" => include_str!("fixtures/k2_horizon/mova-gguf.jinja"),
        _ => panic!("unknown template {name}"),
    }
}

#[test]
fn publisher_templates_match_python_jinja_exactly() {
    let mut tokenizer = Tokenizer::from_tokenizer(tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::default(),
    ));
    for case in cases() {
        tokenizer.set_template_kwargs(case.template_defaults);
        let rendered = tokenizer
            .apply_chat_template_json(
                template(&case.template),
                [case.messages],
                Some(&case.tools),
                &case.template,
                case.add_generation_prompt,
                Some(&case.kwargs),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(rendered, [case.rendered], "{}", case.name);
    }
}

#[test]
#[ignore = "requires verified pinned artifacts; set EREDU_K2_ARTIFACT_ROOT"]
fn released_tokenizers_match_publisher_token_ids() {
    let root = std::path::PathBuf::from(std::env::var("EREDU_K2_ARTIFACT_ROOT").unwrap());
    for (family, filename) in [
        ("dense", "K2-Horizon-1B-BF16.gguf"),
        ("mova", "K2-Horizon-36B-BF16.gguf"),
    ] {
        let tokenizer = Tokenizer::from_file(root.join(family).join("tokenizer.json")).unwrap();
        let reader =
            eredu_gguf::Reader::open(root.join(format!("{family}-gguf")).join(filename)).unwrap();
        let gguf =
            eredu_text::gguf::from_metadata(&reader.metadata().clone().into_iter().collect())
                .unwrap()
                .unwrap();
        for case in cases()
            .into_iter()
            .filter(|case| case.template.starts_with(family))
        {
            let tokenizer = if case.template.ends_with("safetensors") {
                &*tokenizer
            } else {
                &gguf.tokenizer
            };
            assert_eq!(
                tokenizer.encode(case.rendered, false).unwrap().get_ids(),
                case.token_ids,
                "{}",
                case.name
            );
        }
    }
}
