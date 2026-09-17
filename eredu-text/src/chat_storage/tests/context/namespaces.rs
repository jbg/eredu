use super::*;

#[test]
fn namespace_released_deepseek_scan_preserves_mutation_and_unicode_accumulation() {
    // Exact released 7e28c67d system scan, followed only by observing its result.
    // Tool JSON formatting is a separate still-unqualified operation family.
    let template=format!("{}{{{{ ns.system_prompt }}}}|{{{{ ns.is_first_sp }}}}",
        include_str!("namespace_deepseek_v3.jinja"));
    let messages=json!([
        {"role":"system","content":"é\u{0}界"},
        {"role":"user","content":"excluded"},
        {"role":"system","content":"🦀第二"},
        {"role":"assistant","content":"excluded too"}
    ]);
    let rendered=compare(&template,messages.as_array().unwrap(),&serde_json::Map::new(),&serde_json::Map::new());
    assert!(rendered.prompt(false).contains("é\u{0}界\n\n🦀第二"));
    assert!(rendered.prompt(false).ends_with("|False"));
    assert!(!rendered.prompt(false).contains("excluded"));
}

#[test]
fn namespace_aliases_keep_fresh_loop_instances_and_outer_mutation() {
    let template="{% set root = namespace(text='',count=0) %}{% set root_alias = root %}{% for row in rows %}{% set local = namespace(value=row) %}{% set previous = local %}{% set local = namespace(value='fresh') %}{% set previous.value = previous.value + '!' %}{% for child in children %}{% set inner = namespace(value=child) %}{% set root_alias.text = root_alias.text + previous.value + inner.value %}{% endfor %}{{ local.value }}:{{ previous.value }};{% set root.count = root.count + 1 %}{% endfor %}|{{ root.text }}:{{ root_alias.count }}|{{ local|default('gone') }}";
    let caller=json!({"rows":["é","界"],"children":["🦀","λ"]});
    let rendered=compare(template,&[],&serde_json::Map::new(),caller.as_object().unwrap());
    drop(caller);
    assert_eq!(rendered.prompt(false),"fresh:é!;fresh:界!;|é!🦀é!λ界!🦀界!λ:2|gone");
}

#[test]
fn namespace_fields_retain_input_loans_beyond_operand_reuse_and_reassignment() {
    let template="{% set ns = namespace(record=entry, value=entry.text, missing=none) %}{% set alias = ns %}{% for row in rows %}{% set ns.record = row %}{% set ns.value = row.text %}{% set discard = other %}{{ alias.record.name }}={{ alias.value }};{% endfor %}{% set alias.new = '追加' %}|{{ ns.new }}:{{ ns['new'] }}:{{ ns.missing is none }}:{{ ns.absent|default('absent') }}:{{ ns|length }}:{{ ns is mapping }}";
    let caller=json!({"entry":{"name":"old","text":"old"},"rows":[{"name":"é","text":"界\u{0}"},{"name":"B","text":"🦀"}],"other":{"text":"wrong"}});
    let rendered=compare(template,&[],&serde_json::Map::new(),caller.as_object().unwrap());
    let source=ChatTemplatePlan::prepare_utf8(template,"namespace-custody").unwrap().compile().unwrap();
    for buffer in [ChatRenderBuffer::Namespaces,ChatRenderBuffer::NamespaceFields,ChatRenderBuffer::Borrowed] {
        let context=ChatRenderContext::from_json(&[],None,Some(caller.as_object().unwrap())).unwrap();
        let failed=source.render_plan_with_context(context).unwrap().fail_reservation(buffer).render().unwrap_err();
        assert!(failed.retained_buffer_bytes()>0);
    }
    let context=ChatRenderContext::from_json(&[],None,Some(caller.as_object().unwrap())).unwrap();
    let failed=source.render_plan_with_context(context).unwrap().fail_reservation(ChatRenderBuffer::NamespaceFields).render().unwrap_err();
    drop((source,caller));
    assert!(failed.retained_buffer_bytes()>0);
    assert_eq!(rendered.prompt(false),"é=界\u{0};B=🦀;|追加:追加:True:absent:4:True");
}

#[test]
fn namespace_invalid_targets_and_unqualified_object_graphs_refuse_without_alias_reuse() {
    for template in ["{% set missing.field = 1 %}","{% set ns = namespace() %}{% set ns.child = namespace() %}","{% set a = namespace() %}{% set b = namespace(child=a) %}","{% set namespace = false %}{% set n = namespace() %}"] {
        let source=ChatTemplatePlan::prepare_utf8(template,"namespace-refusal").unwrap().compile().unwrap();
        assert!(source.render_plan_with_context(ChatRenderContext::from_json(&[],None,None).unwrap()).is_err(),"{template}");
    }
    for template in ["{% set n = namespace(mapping) %}","{% set n = namespace(**mapping) %}"] {
        assert!(match ChatTemplatePlan::prepare_utf8(template,"namespace-unqualified") {Ok(plan)=>plan.compile().is_err(),Err(_)=>true});
    }
    // Empty objects and later fields follow actual Namespace object truth/length.
    compare("{% set n = namespace() %}{{ n|length }}:{% if n %}yes{% else %}no{% endif %}{% set n.field = 2 %}:{{ n|length }}:{% if n %}yes{% endif %}",&[],&serde_json::Map::new(),&serde_json::Map::new());
}
