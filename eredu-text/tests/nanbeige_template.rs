use eredu_text::tokenizer::Tokenizer;
use serde_json::json;

#[test]
fn released_nanbeige_template_renders_messages_and_thinking_toggle() {
    let mut tokenizer = Tokenizer::from_tokenizer(tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::default(),
    ));
    let template = include_str!("fixtures/nanbeige/chat_template.jinja");
    for thinking in [true, false] {
        let kwargs = serde_json::Map::from_iter([("enable_thinking".into(), thinking.into())]);
        let rendered = tokenizer
            .apply_chat_template_json(
                template,
                [vec![
                    json!({"role":"system", "content":"Be concise."}),
                    json!({"role":"user", "content":"Hello"}),
                ]],
                None,
                "Nanbeige/Nanbeige4.2-3B",
                true,
                Some(&kwargs),
            )
            .unwrap();
        assert!(rendered[0].contains("Be concise."));
        assert!(rendered[0].contains("Hello<|im_end|>"));
        assert!(rendered[0].contains("<|im_start|>assistant"));
        assert!(rendered[0].ends_with(if thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        }));
    }
}
