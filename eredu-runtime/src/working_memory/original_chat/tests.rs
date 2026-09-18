use super::*;
use crate::working_memory::OriginalTokenizer;
use eredu_core::GenerationSequenceConsumerLayout;
use eredu_text::chat_storage::{ChatMessages, ChatRenderBuffer, ChatSourceBuffer, TextMessage};
use eredu_text::tokenizer_storage::TokenizerPlan;
use std::error::Error as _;
const SOURCE: &str = include_str!("tests/template.jinja");
const TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"h":0,"i":1},"merges":[]}}"#;
pub(super) fn source_plan() -> ChatTemplatePlan<'static> {
    ChatTemplatePlan::prepare_utf8(SOURCE, "chat").unwrap()
}
pub(super) fn tokenizer(pool: &WorkingMemoryPool) -> OriginalTokenizer {
    pool.compile_tokenizer(
        TokenizerPlan::prepare_json(TOKENIZER.as_bytes())
            .unwrap()
            .with_generation_domain()
            .unwrap(),
    )
    .unwrap()
}
pub(super) fn consumer() -> GenerationSequenceConsumerLayout {
    GenerationSequenceConsumerLayout::for_driver_with_terminal_text::<(), (), std::io::Error>()
        .unwrap()
}
pub(super) fn messages() -> [TextMessage<'static>; 2] {
    [
        TextMessage {
            role: "system",
            content: "Nonempty policy",
        },
        TextMessage {
            role: "user",
            content: "Héllo 世界\0\n",
        },
    ]
}

#[test]
fn original_j_exact_short_active_idle_and_all_strong_retirement() {
    let j = WorkingMemoryPool::chat_template_required_bytes(&source_plan()).unwrap();
    let short = WorkingMemoryPool::new(j - 1, 0).unwrap();
    let error = short
        .compile_chat_template_with(source_plan(), || panic!("short J entered constructor"))
        .unwrap_err();
    assert!(
        matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes==j && *available_bytes==j-1)
    );
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(j, 0).unwrap();
    let source = pool
        .compile_chat_template_with(source_plan(), || {
            assert_eq!(pool.used_bytes().unwrap(), j);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        })
        .unwrap();
    assert_eq!(source.name(), "chat");
    assert_eq!(source.original_bytes(), j);
    drop(pool.acquire_unquoted().unwrap());
    let alias = source.clone();
    assert!(alias.same_source(&source));
    let foreign = WorkingMemoryPool::new(j, 0).unwrap();
    assert!(matches!(
        alias.validate_pool(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let witness = pool.clone();
    drop((source, pool));
    assert_eq!(witness.used_bytes().unwrap(), j);
    let peer = alias.clone();
    std::thread::scope(|threads| {
        threads.spawn(move || drop(alias));
        threads.spawn(move || drop(peer));
    });
    assert_eq!(witness.used_bytes().unwrap(), 0);
}

#[test]
fn original_j_real_three_reserve_errors_and_unwind_keep_original_accounting() {
    for (index, buffer) in [
        ChatSourceBuffer::Instructions,
        ChatSourceBuffer::Bytes,
        ChatSourceBuffer::Locations,
    ]
    .into_iter()
    .enumerate()
    {
        let plan = source_plan().fail_reservation(buffer);
        let j = WorkingMemoryPool::chat_template_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(j, 0).unwrap();
        let error = pool.compile_chat_template(plan).unwrap_err();
        assert_eq!(error.retained_bytes(), j);
        assert_eq!(pool.used_bytes().unwrap(), j);
        let failure = error.compiler_failure().unwrap();
        assert_eq!(failure.retained_buffer_bytes() > 0, index != 0);
        assert!(failure.source().is_some());
        drop(pool.acquire_unquoted().unwrap());
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    let j = WorkingMemoryPool::chat_template_required_bytes(&source_plan()).unwrap();
    let pool = WorkingMemoryPool::new(j, 0).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.compile_chat_template_with(source_plan(), || panic!("actual installed J scope unwind"))
    }));
    assert!(panic.is_err());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn original_h_exact_short_foreign_sources_and_retained_render_aliases() {
    let input = messages();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool.compile_chat_template(source_plan()).unwrap();
    let h = pool
        .chat_render_required_bytes(&j, &c, eredu_text::chat_storage::ChatRenderContext::from_messages(ChatMessages::from_text(&input)))
        .unwrap();
    let source_bytes = c.original_bytes() + j.original_bytes();
    drop((j, c));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let short = WorkingMemoryPool::new(source_bytes + h - 1, 0).unwrap();
    let c = tokenizer(&short);
    let j = short.compile_chat_template(source_plan()).unwrap();
    let plan = j
        .payload()
        .source
        .render_plan(ChatMessages::from_text(&input))
        .unwrap();
    let error = short
        .render_original_chat_plan(&j, &c, plan, || {
            panic!("short H entered constructor")
        })
        .unwrap_err();
    assert_eq!(error.retained_bytes(), 0);
    assert!(
        matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes==h && *available_bytes==h-1)
    );
    assert_eq!(short.used_bytes().unwrap(), source_bytes);
    drop((error, j, c));
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(source_bytes + h, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool.compile_chat_template(source_plan()).unwrap();
    let plan = j
        .payload()
        .source
        .render_plan(ChatMessages::from_text(&input))
        .unwrap();
    let render = pool
        .render_original_chat_plan(&j, &c, plan, || {
            assert_eq!(pool.used_bytes().unwrap(), source_bytes + h);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        })
        .unwrap();
    assert!(render.has_sources(&j, &c));
    assert!(render.prompt(false).contains("Héllo 世界\0\n"));
    assert_eq!(render.generation_suffix(), "<|im_start|>assistant\n");
    drop(pool.acquire_unquoted().unwrap());
    let other = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c2 = tokenizer(&other);
    let j2 = other.compile_chat_template(source_plan()).unwrap();
    assert!(!render.has_sources(&j2, &c2));
    let foreign = other
        .render_original_chat(&j, &c2, eredu_text::chat_storage::ChatRenderContext::from_messages(ChatMessages::from_text(&input)))
        .unwrap_err();
    assert!(matches!(
        foreign.accounting_failure(),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(foreign.retained_bytes(), 0);
    drop((j, c));
    assert_eq!(pool.used_bytes().unwrap(), source_bytes + h);
    let alias = render.clone();
    drop(render);
    assert!(alias.prompt(true).ends_with("<|im_start|>assistant\n"));
    let peer = alias.clone();
    std::thread::scope(|threads| {
        threads.spawn(move || drop(alias));
        threads.spawn(move || drop(peer));
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_h_six_actual_reserve_frontiers_retain_j_c_and_full_h() {
    let input = messages();
    for (index, buffer) in [
        ChatRenderBuffer::Operands,
        ChatRenderBuffer::Frames,
        ChatRenderBuffer::Locals,
        ChatRenderBuffer::Concat,
        ChatRenderBuffer::WithoutPrompt,
        ChatRenderBuffer::WithPrompt,
    ]
    .into_iter()
    .enumerate()
    {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let c = tokenizer(&pool);
        let j = pool.compile_chat_template(source_plan()).unwrap();
        let base = c.original_bytes() + j.original_bytes();
        let h = pool
            .chat_render_required_bytes(&j, &c, eredu_text::chat_storage::ChatRenderContext::from_messages(ChatMessages::from_text(&input)))
            .unwrap();
        let plan = j
            .payload()
            .source
            .render_plan(ChatMessages::from_text(&input))
            .unwrap()
            .fail_reservation(buffer);
        let error = pool
            .render_original_chat_plan(&j, &c, plan, || {})
            .unwrap_err();
        assert_eq!(error.retained_bytes(), h);
        assert_eq!(pool.used_bytes().unwrap(), base + h);
        assert_eq!(
            error.render_failure().unwrap().retained_buffer_bytes() > 0,
            index != 0
        );
        assert!(error.render_failure().unwrap().source().is_some());
        drop((j, c));
        drop(pool.acquire_unquoted().unwrap());
        assert_eq!(pool.used_bytes().unwrap(), base + h);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn concurrent_original_h_admissions_share_c_j_without_recredit_or_blocked_failure_cleanup() {
    let input = messages();
    let probe = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c = tokenizer(&probe);
    let j = probe.compile_chat_template(source_plan()).unwrap();
    let h = probe
        .chat_render_required_bytes(&j, &c, eredu_text::chat_storage::ChatRenderContext::from_messages(ChatMessages::from_text(&input)))
        .unwrap();
    let base = c.original_bytes() + j.original_bytes();
    drop((c, j));
    assert_eq!(probe.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(base + 3 * h - 1, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool.compile_chat_template(source_plan()).unwrap();
    let (ready_send, ready_recv) = std::sync::mpsc::channel();
    std::thread::scope(|threads| {
        let mut releases = Vec::new();
        let mut workers = Vec::new();
        for _ in 0..2 {
            let (release_send, release_recv) = std::sync::mpsc::channel::<()>();
            releases.push(release_send);
            let ready = ready_send.clone();
            let pool = &pool;
            let c = &c;
            let j = &j;
            let input = &input;
            workers.push(threads.spawn(move || {
                let mut notified = false;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let plan = j
                        .payload()
                        .source
                        .render_plan(ChatMessages::from_text(input))
                        .unwrap();
                    pool.render_original_chat_plan(j, c, plan, || {
                        notified = true;
                        let _ = ready.send(true);
                        let _ = release_recv.recv();
                    })
                }));
                if !notified {
                    let _ = ready.send(false);
                }
                result
            }));
        }
        drop(ready_send);
        let admissions = [ready_recv.recv(), ready_recv.recv()];
        // Gather observations without assertions while either worker is held.
        // Every path releases and joins both workers before interpreting them.
        let held = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (
                pool.used_bytes(),
                matches!(
                    pool.acquire_unquoted(),
                    Err(WorkingMemoryError::ReservedWorkActive)
                ),
                pool.render_original_chat(&j, &c, eredu_text::chat_storage::ChatRenderContext::from_messages(ChatMessages::from_text(&input))),
            )
        }));
        drop(releases);
        let completed: Vec<_> = workers.into_iter().map(|worker| worker.join()).collect();
        assert!(admissions
            .into_iter()
            .all(|value| matches!(value, Ok(true))));
        let (used, busy, third) = held.unwrap();
        let error = third.unwrap_err();
        assert!(
            matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == h && *available_bytes == h - 1)
        );
        assert_eq!(error.retained_bytes(), 0);
        drop(error);
        assert_eq!(used.unwrap(), base + 2 * h);
        assert!(busy);
        let mut outputs: Vec<_> = completed
            .into_iter()
            .map(|join| join.unwrap().unwrap().unwrap())
            .collect();
        assert_eq!(pool.used_bytes().unwrap(), base + 2 * h);
        assert_eq!(outputs[0].prompt(true), outputs[1].prompt(true));
        assert!(outputs.iter().all(|out| out.has_sources(&j, &c)));
        drop(pool.acquire_unquoted().unwrap());
        drop(outputs.pop());
        assert_eq!(pool.used_bytes().unwrap(), base + h);
        drop(outputs);
        assert_eq!(pool.used_bytes().unwrap(), base);
    });
    drop((j, c));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn j_poisoned_settlement_retains_actual_completed_or_partial_source_and_charge() {
    for partial in [false, true] {
        let plan = if partial {
            source_plan().fail_reservation(ChatSourceBuffer::Locations)
        } else {
            source_plan()
        };
        let j = WorkingMemoryPool::chat_template_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(j, 0).unwrap();
        let error = pool
            .compile_chat_template_with(plan, || {
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _usage = pool.0.usage.lock().unwrap();
                    panic!("poison the actual source account");
                }))
                .is_err());
            })
            .unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(error.retained_bytes(), j);
        if partial {
            assert!(error.compiler_failure().unwrap().retained_buffer_bytes() > 0);
            assert!(error.completed.is_none());
        } else {
            assert_eq!(error.completed.as_ref().unwrap().name(), "chat");
            assert!(error.compiler_failure().is_none());
        }
        drop(error);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.reservations, 1);
        assert_eq!(
            usage.reserved, j,
            "poisoned settlement never refunds as if successful"
        );
    }
}

#[test]
fn generic_chat_compiler_j_precedes_decode_and_retains_failed_prefix() {
    const TEXT: &str = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}> {% endif %}";
    let plan = || ChatTemplatePlan::prepare_utf8(TEXT, "general").unwrap();
    let j = WorkingMemoryPool::chat_template_required_bytes(&plan()).unwrap();
    let short = WorkingMemoryPool::new(j - 1, 0).unwrap();
    let refused = short
        .compile_chat_template_with(plan(), || panic!("short J decoded source"))
        .unwrap_err();
    assert!(
        matches!(refused.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == j && *available_bytes == j - 1)
    );
    assert_eq!(refused.retained_bytes(), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(j, 0).unwrap();
    let failed = pool
        .compile_chat_template_with(plan().fail_reservation(ChatSourceBuffer::Locations), || {
            assert_eq!(pool.used_bytes().unwrap(), j)
        })
        .unwrap_err();
    assert!(failed.compiler_failure().unwrap().retained_buffer_bytes() >= TEXT.len());
    assert_eq!(failed.retained_bytes(), j);
    drop(pool.acquire_unquoted().unwrap());
    assert_eq!(pool.used_bytes().unwrap(), j);
    drop(failed);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    let source = pool
        .compile_chat_template_with(plan(), || assert_eq!(pool.used_bytes().unwrap(), j))
        .unwrap();
    let peer = source.clone();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), j);
    assert_eq!(peer.name(), "general");
    drop(peer);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    // Invalid source constructs its parser storage under J and returns an
    // owning fixed failure; no borrowed parser storage escapes the account.
    let text = String::from("{% if %}");
    let plan = ChatTemplatePlan::prepare_utf8(&text, "general").unwrap();
    let j = WorkingMemoryPool::chat_template_required_bytes(&plan).unwrap();
    let pool = WorkingMemoryPool::new(j, 0).unwrap();
    let failed = pool.compile_chat_template(plan).unwrap_err();
    drop(text);
    assert!(failed.compiler_failure().unwrap().retained_buffer_bytes() > 0);
    assert_eq!(pool.used_bytes().unwrap(), j);
    drop(failed);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_numeric_chat_parser_and_error_remain_under_source_custody() {
    let source = serde_json::to_string(SOURCE).unwrap();
    for number in ["1.25", "1e999"] {
        let config = format!("{{\"unused\":{number},\"chat_template\":{source}}}");
        let plan = || ChatTemplatePlan::prepare_config(config.as_bytes(), "numeric", false).unwrap();
        let j = WorkingMemoryPool::chat_template_required_bytes(&plan()).unwrap();
        let short = WorkingMemoryPool::new(j - 1, 0).unwrap();
        let error = short
            .compile_chat_template_with(plan(), || {
                panic!("short source grant entered scalar parser")
            })
            .unwrap_err();
        assert!(error.accounting_failure().is_some());
        assert!(error.compiler_failure().is_none());
        assert_eq!(short.used_bytes().unwrap(), 0);
        let pool = WorkingMemoryPool::new(j, 0).unwrap();
        let result =
            pool.compile_chat_template_with(plan(), || assert_eq!(pool.used_bytes().unwrap(), j));
        match result {
            Ok(template) => {
                assert_eq!(number, "1.25");
                assert_eq!(pool.used_bytes().unwrap(), j);
                let alias = template.clone();
                drop(template);
                assert_eq!(pool.used_bytes().unwrap(), j);
                drop(alias);
            }
            Err(error) => {
                assert_eq!(number, "1e999");
                assert!(error.compiler_failure().unwrap().numeric_error().is_some());
                assert_eq!(pool.used_bytes().unwrap(), j);
                drop(pool.acquire_unquoted().unwrap());
                drop(error);
            }
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
