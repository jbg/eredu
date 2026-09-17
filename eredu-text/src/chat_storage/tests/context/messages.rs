use super::*;

#[test]
fn rich_message_objects_use_the_same_borrowed_json_workers() {
    let messages = json!([
        {"role":"assistant","content":null,"reasoning_content":"é\u{0}界",
         "tool_calls":[{"id":"call-1","function":{"name":"lookup","arguments":{"city":"Madrid","limit":2}}}]},
        {"role":"tool","content":[{"type":"text","text":"answer"},{"type":"image","image":"host-owned"}],"tool_call_id":"call-1"},
        {"role":"assistant","content":"done"}
    ]);
    let template = "{% for message in messages %}{{ message.role }}:{{ message|length }}:{% if message.reasoning_content is string %}{{ message.reasoning_content }}{% endif %}{% for call in message.tool_calls|default([]) %}{{ call.id }}={{ call.function.name }}({% for key,value in call.function.arguments.items() %}{{ key }}={{ value }};{% endfor %}){% endfor %}{% if message.content is string %}{{ message.content }}{% elif message.content is sequence %}{% for part in message.content %}{{ part.type }}={{ part.text|default('') }};{% endfor %}{% else %}null{% endif %}|{% endfor %}{{ messages[0].tool_calls[0].id }}:{{ messages[-1].content }}";
    let rendered = compare(
        template,
        messages.as_array().unwrap(),
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    drop(messages);
    assert!(rendered.prompt(false).contains("é\u{0}界"));
    assert!(rendered.prompt(false).contains("call-1=lookup("));
    assert!(rendered.prompt(false).ends_with("call-1:done"));
}

#[test]
fn rich_message_aliases_and_overrides_preserve_input_and_error_custody() {
    let template = "{% set saved = messages[0] %}{% set borrowed = saved.reasoning_content %}{% for message in messages %}{% set chosen = message.get('tool_calls', []) %}{% for call in chosen %}{{ call.function.name }};{% endfor %}{% endfor %}{{ saved.content|default('missing') }}:{{ borrowed }}:{% for key,value in saved.items() %}{{ key }};{% endfor %}";
    let base = json!([{"role":"user","content":"discarded"}]);
    let caller = json!({"messages":[{"role":"assistant","reasoning_content":"kept 🦀","tool_calls":[{"function":{"name":"first"}},{"function":{"name":"second"}}]}]});
    let rendered = compare(
        template,
        base.as_array().unwrap(),
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    let source = ChatTemplatePlan::prepare_utf8(template, "rich-message")
        .unwrap()
        .compile()
        .unwrap();
    let context =
        ChatRenderContext::from_json(base.as_array().unwrap(), None, caller.as_object()).unwrap();
    let failure = source
        .render_plan_with_context(context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::Borrowed)
        .render()
        .unwrap_err();
    drop((source, base, caller));
    assert!(rendered
        .prompt(false)
        .starts_with("first;second;missing:kept 🦀:"));
    assert!(failure.retained_buffer_bytes() > 0);
}
