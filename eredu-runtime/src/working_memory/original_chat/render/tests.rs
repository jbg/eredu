use super::super::tests::{consumer, messages, source_plan, tokenizer};
use super::*;
use eredu_text::chat_storage::ChatMessages;
use eredu_text::chat_storage::ChatRenderBuffer;

#[test]
fn semantic_consumer_tools_use_exact_render_account_and_source_custody() {
    use eredu_text::chat_storage::{ChatInputArray, ChatRenderContext, ChatTemplatePlan};
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{{ tools|tojson }}{% if add_generation_prompt %}assistant{% endif %}",
                "semantic",
            )
            .unwrap(),
        )
        .unwrap();
    let tools = vec![
        serde_json::json!({"function":{"name":"reading", "parameters":{
            "type":"object", "properties":{"value":{"type":"integer", "minimum":17}}
        }}}),
    ];
    let context = ChatRenderContext::from_json(&[], None, None)
        .unwrap()
        .with_tools(ChatInputArray::Json(&tools));
    let base = pool.payload_used_bytes().unwrap();
    let expected = pool.chat_render_required_bytes(&j, &c, context).unwrap();
    let render = pool.render_original_chat(&j, &c, context).unwrap();
    assert_eq!(render.original_bytes(), expected);
    assert_eq!(pool.payload_used_bytes().unwrap(), base + expected);
    assert!(render.has_sources(&j, &c));
    let alias = render.clone();
    drop((render, tools, j, c));
    assert_eq!(pool.payload_used_bytes().unwrap(), base + expected);
    let prompt = alias.prompt(true);
    assert!(prompt.contains("reading") && prompt.contains("17"));
    assert!(prompt.ends_with("assistant"));
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn h_unwind_and_poison_keep_completed_or_partial_render_and_original_sources() {
    for partial in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
        let c = tokenizer(&pool);
        let j = pool.compile_chat_template(source_plan()).unwrap();
        let input = messages();
        let base = c.original_bytes() + j.original_bytes();
        let plan = j
            .payload()
            .source
            .render_plan(ChatMessages::from_text(&input))
            .unwrap();
        let h = required(&plan).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = pool.render_original_chat_plan(&j, &c, plan, || {
                    panic!("unwind after real H admission")
                });
            }))
            .is_err()
        );
        assert_eq!(pool.payload_used_bytes().unwrap(), base);
        let plan = j
            .payload()
            .source
            .render_plan(ChatMessages::from_text(&input))
            .unwrap();
        let plan = if partial {
            plan.fail_reservation(ChatRenderBuffer::WithPrompt)
        } else {
            plan
        };
        let error = pool
            .render_original_chat_plan(&j, &c, plan, || {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _usage = pool.0.usage.lock().unwrap();
                        panic!("poison actual H account");
                    }))
                    .is_err()
                );
            })
            .unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(error.retained_bytes(), h);
        if partial {
            assert!(error.render_failure().unwrap().retained_buffer_bytes() > 0);
            assert!(error.completed.is_none());
        } else {
            let rendered = error.completed.as_ref().unwrap();
            assert!(rendered.prompt(false).contains("Nonempty policy"));
            assert_eq!(rendered.generation_suffix(), "<|im_start|>assistant\n");
            assert!(error.render_failure().is_none());
        }
        assert!(error.template.as_ref().unwrap().same_source(&j));
        assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
        drop((j, c));
        drop(error);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.reservations, 1);
        assert_eq!(
            usage.reserved,
            base + h,
            "poison does not fabricate a successful refund"
        );
    }
}

#[test]
fn borrowed_context_render_and_partial_failure_retain_only_original_owned_storage() {
    use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{{ prefix + suffix }}{% if add_generation_prompt %}!{% endif %}",
                "chat",
            )
            .unwrap(),
        )
        .unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let defaults = serde_json::json!({"prefix":"default", "suffix":"default"});
    let caller = serde_json::json!({"prefix":"borrowed 界", "suffix":" tail"});
    let context =
        ChatRenderContext::from_json(&[], defaults.as_object(), caller.as_object()).unwrap();
    let expected = pool.chat_render_required_bytes(&j, &c, context).unwrap();
    let render = pool.render_original_chat(&j, &c, context).unwrap();
    assert_eq!(render.original_bytes(), expected);
    assert_eq!(pool.payload_used_bytes().unwrap(), base + expected);
    let alias = render.clone();
    drop((defaults, caller, render));
    assert_eq!(alias.prompt(true), "borrowed 界 tail!");
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), base);
    for buffer in [
        ChatRenderBuffer::WithoutPrompt,
        ChatRenderBuffer::WithPrompt,
    ] {
        let caller = serde_json::json!({"prefix":"actual retained", "suffix":" prefix"});
        let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
        let plan = pool.chat_render_plan(&j, &c, context).unwrap();
        let expected = required(&plan).unwrap();
        let error = pool
            .render_original_chat_plan(&j, &c, plan.fail_reservation(buffer), || {})
            .unwrap_err();
        drop(caller);
        assert_eq!(error.retained_bytes(), expected);
        assert_eq!(
            error.render_failure().unwrap().retained_buffer_bytes() > 0,
            buffer == ChatRenderBuffer::WithPrompt
        );
        assert_eq!(pool.payload_used_bytes().unwrap(), base + expected);
        drop(error);
        assert_eq!(pool.payload_used_bytes().unwrap(), base);
    }
    drop((j, c));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn nested_borrowed_context_retains_new_text_destination_and_original_h_until_retirement() {
    use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8("{{ cfg.prefix + cfg.suffix }}", "chat").unwrap(),
        )
        .unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let caller = serde_json::json!({"cfg":{"prefix":"actual nested 界", "suffix":" copy"}});
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let bytes = pool.chat_render_required_bytes(&j, &c, context).unwrap();
    let render = pool.render_original_chat(&j, &c, context).unwrap();
    let plan = pool
        .chat_render_plan(&j, &c, context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt);
    let failure = pool
        .render_original_chat_plan(&j, &c, plan, || {})
        .unwrap_err();
    assert_eq!(failure.retained_bytes(), bytes);
    assert!(failure.render_failure().unwrap().retained_buffer_bytes() > 0);
    drop((caller, j, c));
    assert_eq!(pool.payload_used_bytes().unwrap(), base + 2 * bytes);
    assert_eq!(render.prompt(true), "actual nested 界 copy");
    drop(failure);
    assert_eq!(pool.payload_used_bytes().unwrap(), base + bytes);
    drop(render);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn typed_short_circuit_chat_keeps_actual_render_and_failure_custody_after_context_retirement() {
    use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(r#"{% if cfg.reasoning is false or cfg.payload is mapping %}{{ (cfg.preferred or cfg.fallback).text + cfg.suffix }}{% endif %}{% if add_generation_prompt is true %}{{ cfg.suffix }}{% endif %}"#, "chat").unwrap(),
        )
        .unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let caller = serde_json::json!({"cfg":{"preferred":{},"fallback":{"text":"actual nested 界"},"suffix":" copy","reasoning":true,"payload":{"type":"text"}}});
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let bytes = pool.chat_render_required_bytes(&j, &c, context).unwrap();
    let render = pool.render_original_chat(&j, &c, context).unwrap();
    let plan = pool
        .chat_render_plan(&j, &c, context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt);
    let failure = pool
        .render_original_chat_plan(&j, &c, plan, || {})
        .unwrap_err();
    assert_eq!(failure.retained_bytes(), bytes);
    assert!(failure.render_failure().unwrap().retained_buffer_bytes() > 0);
    drop((caller, j, c));
    assert_eq!(pool.payload_used_bytes().unwrap(), base + 2 * bytes);
    assert_eq!(render.prompt(false), "actual nested 界 copy");
    assert_eq!(render.prompt(true), "actual nested 界 copy copy");
    drop(failure);
    assert_eq!(pool.payload_used_bytes().unwrap(), base + bytes);
    drop(render);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn render_headroom_is_admitted_before_nested_dispatch() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let template = "{% for call in calls %}{% for key,value in call.items() %}{{ key + ':' + value }};{% endfor %}{% endfor %}";
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(ChatTemplatePlan::prepare_utf8(template, "nested").unwrap())
        .unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let caller = serde_json::json!({"calls":[{"é":"one"},{"界":"two"}]});
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let measured = pool.chat_render_required_bytes(&j, &c, context).unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        base,
        "planning does not consume operation admission"
    );
    let rendered = pool.render_original_chat(&j, &c, context).unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), base + measured);
    drop((caller, j, c));
    assert_eq!(rendered.prompt(false), "é:one;界:two;");
    drop(rendered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);

    // Keep source owners and one byte less than the selected render reservation.
    let pool = crate::working_memory::memory_fixture::host_ledger(base + measured - 1, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(ChatTemplatePlan::prepare_utf8(template, "nested").unwrap())
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), base);
    let caller = serde_json::json!({"calls":[{"é":"one"},{"界":"two"}]});
    let error = pool
        .render_original_chat(
            &j,
            &c,
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap_err();
    assert!(error.accounting_failure().is_some());
    assert_eq!(
        error.retained_bytes(),
        0,
        "no scratch was constructed before refusal"
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), base);
}

#[test]
fn failed_loop_execution_retains_sources_and_render_admission() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{% for a,b in rows %}{{ a }}{{ b }}{% endfor %}",
                "bad-unpack",
            )
            .unwrap(),
        )
        .unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let caller = serde_json::json!({"rows":[["only one"]]});
    let prefix = pool
        .chat_render_required_bytes(
            &j,
            &c,
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap();
    let error = pool
        .render_original_chat(
            &j,
            &c,
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap_err();
    assert_eq!(error.retained_bytes(), prefix);
    assert!(matches!(error.cause, Cause::Render(_)));
    drop((caller, j, c));
    assert_eq!(pool.payload_used_bytes().unwrap(), base + prefix);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn json_outputs_and_output_reserve_errors_retain_sources_and_admission() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{{ tools|tojson(sort_keys=true,ensure_ascii=true) }}",
                "original-json",
            )
            .unwrap(),
        )
        .unwrap();
    let input = serde_json::json!({"tools":[{"z":7,"a":[{"z":-11,"a":{"z":23,"a":"É🙂"}}]}]});
    let context = ChatRenderContext::from_json(&[], None, input.as_object()).unwrap();
    let baseline = pool.payload_used_bytes().unwrap();
    let plan = pool.chat_render_plan(&j, &c, context).unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline,
        "planning consumes no operation admission"
    );
    let rendered = pool.render_original_chat_plan(&j, &c, plan, || {}).unwrap();
    let plan = pool
        .chat_render_plan(&j, &c, context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt);
    let error = pool
        .render_original_chat_plan(&j, &c, plan, || {})
        .unwrap_err();
    assert!(error.render_failure().unwrap().retained_buffer_bytes() > 0);
    assert!(error.template.as_ref().unwrap().same_source(&j));
    assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline + rendered.original_bytes() + error.retained_bytes()
    );
    drop((input, j, c));
    assert!(rendered.prompt(false).contains("23"));
    assert!(rendered.prompt(false).contains("\\ud83d\\ude42"));
    drop(error);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline + rendered.original_bytes()
    );
    drop(rendered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn generated_slices_and_output_reserve_errors_retain_sources() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{% set out=payload|tojson %}{{ out[:-1] }}",
                "generated-slice",
            )
            .unwrap(),
        )
        .unwrap();
    let input = serde_json::json!({"payload":{"label":"É🙂","value":23}});
    let context = ChatRenderContext::from_json(&[], None, input.as_object()).unwrap();
    let baseline = pool.payload_used_bytes().unwrap();
    let plan = pool.chat_render_plan(&j, &c, context).unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline,
        "planning consumes no operation admission"
    );
    let rendered = pool.render_original_chat_plan(&j, &c, plan, || {}).unwrap();
    let plan = pool
        .chat_render_plan(&j, &c, context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt);
    let error = pool
        .render_original_chat_plan(&j, &c, plan, || {})
        .unwrap_err();
    assert!(error.render_failure().unwrap().retained_buffer_bytes() > 0);
    assert!(error.template.as_ref().unwrap().same_source(&j));
    assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline + rendered.original_bytes() + error.retained_bytes()
    );
    drop((input, j, c));
    assert!(rendered.prompt(false).contains("23"));
    assert!(!rendered.prompt(false).ends_with('}'));
    drop(error);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline + rendered.original_bytes()
    );
    drop(rendered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn generated_lists_and_output_reserve_errors_retain_sources() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{% set values=[payload.label,payload.value] %}{{ values[1] }}{{ values[0] }}",
                "generated-values",
            )
            .unwrap(),
        )
        .unwrap();
    let input = serde_json::json!({"payload":{"label":"É🙂","value":23}});
    let context = ChatRenderContext::from_json(&[], None, input.as_object()).unwrap();
    let baseline = pool.payload_used_bytes().unwrap();
    let plan = pool.chat_render_plan(&j, &c, context).unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline,
        "planning consumes no operation admission"
    );
    let rendered = pool.render_original_chat_plan(&j, &c, plan, || {}).unwrap();
    let plan = pool
        .chat_render_plan(&j, &c, context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt);
    let error = pool
        .render_original_chat_plan(&j, &c, plan, || {})
        .unwrap_err();
    assert!(error.render_failure().unwrap().retained_buffer_bytes() > 0);
    assert!(error.template.as_ref().unwrap().same_source(&j));
    assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline + rendered.original_bytes() + error.retained_bytes()
    );
    drop((input, j, c));
    assert!(rendered.prompt(false).contains("23"));
    assert_eq!(rendered.prompt(false), "23É🙂");
    drop(error);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        baseline + rendered.original_bytes()
    );
    drop(rendered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn generated_loop_output_preserves_unicode_and_enforces_admission() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let template = "{% for row in rows %}{{ (row + 'é')|upper }};{% endfor %}";
    let input = serde_json::json!({"rows":(0..33).map(|i|format!("row-{i}")).collect::<Vec<_>>()});
    let context = ChatRenderContext::from_json(&[], None, input.as_object()).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(ChatTemplatePlan::prepare_utf8(template, "growth").unwrap())
        .unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let admission = pool.chat_render_required_bytes(&j, &c, context).unwrap();
    let rendered = pool.render_original_chat(&j, &c, context).unwrap();
    assert_eq!(
        rendered.prompt(false),
        (0..33).map(|i| format!("ROW-{i}É;")).collect::<String>()
    );
    drop((rendered, j, c));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    for short in [true, false] {
        let pool = crate::working_memory::memory_fixture::host_ledger(
            base + admission - u64::from(short),
            0,
        )
        .unwrap();
        let c = tokenizer(&pool);
        let j = pool
            .compile_chat_template(ChatTemplatePlan::prepare_utf8(template, "growth").unwrap())
            .unwrap();
        let result = pool.render_original_chat(&j, &c, context);
        if short {
            let error = result.unwrap_err();
            assert!(
                matches!(error.accounting_failure(), Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))
                if *required_bytes == admission && (limit_bytes - existing_bytes) == admission - 1)
            );
            assert_eq!(error.retained_bytes(), 0);
        } else {
            let output = result.unwrap();
            assert_eq!(
                output.prompt(false),
                (0..33).map(|i| format!("ROW-{i}É;")).collect::<String>()
            );
            assert_eq!(pool.payload_used_bytes().unwrap(), base + admission);
            drop(output);
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), base);
        drop((j, c));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn one_render_has_independently_funded_exact_consumers_without_prompt_copies() {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let tokenizer = tokenizer(&pool);
    let template = pool.compile_chat_template(source_plan()).unwrap();
    let input = messages();
    let render = pool
        .render_original_chat(
            &template,
            &tokenizer,
            ChatRenderContext::from_messages(ChatMessages::from_text(&input)),
        )
        .unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let ordinary = GenerationSequenceConsumerLayout::for_driver_with_ordinary_output::<
        [u64; 4],
        (),
        std::io::Error,
    >()
    .unwrap();
    let terminal = consumer();
    let ordinary_bytes = OriginalChatConsumer::required_bytes(&ordinary).unwrap();
    let terminal_bytes = OriginalChatConsumer::required_bytes(&terminal).unwrap();
    let first = render.bind_consumer(ordinary).unwrap();
    let second = render.bind_consumer(terminal).unwrap();
    assert!(first.render().same_render(second.render()));
    assert_eq!(
        first.render().prompt(false).as_ptr(),
        second.render().prompt(false).as_ptr()
    );
    assert!(first.accepts_consumer(&ordinary));
    assert!(!first.accepts_consumer(&terminal));
    assert!(second.accepts_consumer(&terminal));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        base + ordinary_bytes + terminal_bytes
    );
    let other = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    assert!(matches!(
        first.validate_pool(&other),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let alias = first.clone();
    drop((first, render, tokenizer, template));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        base + ordinary_bytes + terminal_bytes
    );
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), base + ordinary_bytes);
    assert!(alias.render().prompt(false).contains("Héllo 世界"));
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn exact_consumer_capacity_refuses_before_association_and_retains_render_on_failure() {
    let probe = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let token_source = tokenizer(&probe);
    let template = probe.compile_chat_template(source_plan()).unwrap();
    let input = messages();
    let source_bytes = probe.payload_used_bytes().unwrap();
    let plan = template
        .payload()
        .source
        .render_plan(ChatMessages::from_text(&input))
        .unwrap();
    let render_bytes = required(&plan).unwrap();
    let consumer_bytes = OriginalChatConsumer::required_bytes(&consumer()).unwrap();
    drop(plan);
    drop((token_source, template, probe));
    for spare in [consumer_bytes - 1, consumer_bytes] {
        let pool = crate::working_memory::memory_fixture::host_ledger(
            source_bytes + render_bytes + spare,
            0,
        )
        .unwrap();
        let token_source = tokenizer(&pool);
        let template = pool.compile_chat_template(source_plan()).unwrap();
        let plan = template
            .payload()
            .source
            .render_plan(ChatMessages::from_text(&input))
            .unwrap();
        let render = pool
            .render_original_chat_plan(&template, &token_source, plan, || {})
            .unwrap();
        if spare < consumer_bytes {
            let error = render.bind_consumer(consumer()).unwrap_err();
            assert!(
                matches!(error.accounting_failure(), WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })
                if *required_bytes == consumer_bytes && (limit_bytes - existing_bytes) == spare)
            );
            assert_eq!(error.retained_bytes(), 0);
            drop((render, token_source, template));
            assert_eq!(
                pool.payload_used_bytes().unwrap(),
                source_bytes + render_bytes
            );
            drop(error);
        } else {
            let association = render.bind_consumer(consumer()).unwrap();
            let refusal = render.bind_consumer(consumer()).unwrap_err();
            assert_eq!(refusal.retained_bytes(), 0);
            let alias = association.clone();
            drop((association, refusal, render, token_source, template));
            assert_eq!(
                pool.payload_used_bytes().unwrap(),
                source_bytes + render_bytes + consumer_bytes
            );
            drop(alias);
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
