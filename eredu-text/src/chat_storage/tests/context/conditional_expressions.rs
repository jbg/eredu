use super::*;
#[test]
fn conditional_expressions_keep_value_stack_and_skip_unselected_calls(){
    let source="{% set prefix = 'é' if flag else '界' %}{{ ('<' if flag else '[') + (prefix if text else '🦀') + ('>' if flag else ']') }}|{{ missing_call() if false else ('NUL\u{0}' if flag else 'other') }}|{{ 'quiet' if false }}|{{ ('yes' if true else 'no') if flag else ('左' if false else '右') }}";
    for flag in [true,false]{
        let caller=json!({"flag":flag,"text":true});
        let output=compare(source,&[],&serde_json::Map::new(),caller.as_object().unwrap());
        assert_eq!(output.prompt(false),if flag{"<é>|NUL\u{0}||yes"}else{"[界]|other||右"});
    }
}
