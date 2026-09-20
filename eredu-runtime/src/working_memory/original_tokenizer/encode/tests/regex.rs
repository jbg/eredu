use super::*;
pub(super) fn json(implicit: bool) -> String {
    if implicit {
        return JSON.replace("\"pre_tokenizer\":null",concat!("\"pre_tokenizer\":",r###"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"###))
            .replace("\" \":3","\"Ġ\":3")
            .replace("\"?\":4","\"?\":4,\"1\":5,\"2\":6,\"12\":7").replace("\"id\":5","\"id\":8")
            .replace("[[\"h\",\"i\"]]","[[\"h\",\"i\"],[\"1\",\"2\"]]");
    }
    JSON.replace("\"pre_tokenizer\":null", concat!("\"pre_tokenizer\":",r###"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"###)).replace("\" \":3","\"Ġ\":3")
}
fn sizes(implicit: bool) -> (u64, u64) {
    let text = if implicit { "hi12<S>hi ?" } else { TEXT };
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = source(&pool, &json(implicit));
    (
        source.original_bytes(),
        WorkingMemoryPool::tokenizer_encode_required_bytes(&source, text, false).unwrap(),
    )
}
#[test]
fn regex_c_and_e_exact_one_short_and_closed_result_lifetimes() {
    profile_case_0(false);
}
#[test]
fn implicit_regex_c_and_e_exact_one_short_and_closed_result_lifetimes() {
    profile_case_0(true);
}
fn profile_case_0(implicit: bool) {
    let text = if implicit { "hi12<S>hi ?" } else { TEXT };
    let (c, e) = sizes(implicit);
    let json = json(implicit);
    let short = WorkingMemoryPool::new(c - 1, 0).unwrap();
    assert!(
        short
            .compile_tokenizer(TokenizerPlan::prepare_json(json.as_bytes()).unwrap())
            .is_err()
    );
    assert_eq!(short.used_bytes().unwrap(), 0);
    for one_short in [true, false] {
        let pool = WorkingMemoryPool::new(c + e - u64::from(one_short), 0).unwrap();
        let source = source(&pool, &json);
        let result = pool.encode_tokenizer_ids_with(
            &source,
            text,
            false,
            |p| p,
            || {
                assert!(!one_short);
                assert_eq!(pool.used_bytes().unwrap(), c + e);
                assert!(matches!(
                    pool.acquire_unquoted(),
                    Err(WorkingMemoryError::ReservedWorkActive)
                ));
            },
            || {},
        );
        if one_short {
            let error = result.unwrap_err();
            assert_eq!(error.retained_bytes(), 0);
            assert!(
                matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes}) if *required_bytes==e && *available_bytes==e-1)
            );
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            drop(error);
        } else {
            let output = result.unwrap();
            assert_eq!(
                output.ids(),
                if implicit {
                    &[2, 5, 6, 8, 2, 3, 4][..]
                } else {
                    &[2, 5, 2, 3, 4][..]
                }
            );
            assert!(output.matches_source(&source));
            drop(pool.acquire_unquoted().unwrap());
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), c + e);
            drop(output);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn same_regex_source_has_independent_concurrent_original_workspaces_and_no_foreign_custody() {
    profile_case_2(false);
}
#[test]
fn implicit_same_regex_source_has_independent_concurrent_original_workspaces_and_no_foreign_custody()
 {
    profile_case_2(true);
}
fn profile_case_2(implicit: bool) {
    let text = if implicit { "hi12<S>hi ?" } else { TEXT };
    let (c, e) = sizes(implicit);
    let pool = WorkingMemoryPool::new(c + 2 * e, 0).unwrap();
    let source = source(&pool, &json(implicit));
    let foreign = WorkingMemoryPool::new(c + e, 0).unwrap();
    let rejects: Vec<_> = (0..16)
        .map(|_| {
            foreign
                .encode_tokenizer_ids(&source, text, false)
                .unwrap_err()
        })
        .collect();
    assert!(rejects.iter().all(|e| e.retained_bytes() == 0
        && matches!(
            e.accounting_failure(),
            Some(WorkingMemoryError::IdentityMismatch)
        )));
    let (a, b, arrivals, observed) = std::thread::scope(|scope| {
        let (notify, rx) = std::sync::mpsc::channel();
        let (release_a, wait_a) = std::sync::mpsc::channel::<()>();
        let (release_b, wait_b) = std::sync::mpsc::channel::<()>();
        let make = |notify: std::sync::mpsc::Sender<bool>, wait: std::sync::mpsc::Receiver<()>| {
            let mut entered = false;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.encode_tokenizer_ids_with(
                    &source,
                    text,
                    false,
                    |p| p,
                    || {
                        entered = true;
                        let _ = notify.send(true);
                        let _ = wait.recv();
                    },
                    || {},
                )
            }));
            if !entered {
                let _ = notify.send(false);
            }
            result
        };
        let peer = notify.clone();
        let a = scope.spawn(move || make(peer, wait_a));
        let b = scope.spawn(move || make(notify, wait_b));
        let arrivals = [rx.recv(), rx.recv()];
        let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (
                pool.used_bytes(),
                pool.encode_tokenizer_ids(&source, text, false),
            )
        }));
        drop((release_a, release_b));
        (a.join(), b.join(), arrivals, observed)
    });
    assert!(arrivals.iter().all(|a| matches!(a, Ok(true))));
    let (used, rejected) = observed.unwrap();
    assert_eq!(used.unwrap(), c + 2 * e);
    assert_eq!(rejected.unwrap_err().retained_bytes(), 0);
    let a = a.unwrap().unwrap().unwrap();
    let b = b.unwrap().unwrap().unwrap();
    assert_eq!(a.ids(), b.ids());
    assert_ne!(a.ids().as_ptr(), b.ids().as_ptr());
    drop(source);
    drop(a);
    assert_eq!(pool.used_bytes().unwrap(), c + e);
    drop(b);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    drop(rejects);
}

#[test]
fn malformed_regex_compilation_retains_admission_and_its_upstream_error() {
    let mut value: serde_json::Value = serde_json::from_str(&json(false)).unwrap();
    value["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "[".into();
    let input = value.to_string();
    let plan = TokenizerPlan::prepare_json(input.as_bytes()).unwrap();
    let c = WorkingMemoryPool::tokenizer_required_bytes(&plan).unwrap();
    let pool = WorkingMemoryPool::new(c, 0).unwrap();
    let error = pool.compile_tokenizer(plan).unwrap_err();
    assert_eq!(error.retained_bytes(), c);
    assert!(error.compiler_failure().unwrap().root_failure().is_some());
    assert!(error.source().is_some());
    drop(input);
    drop(pool.acquire_unquoted().unwrap());
    assert_eq!(pool.used_bytes().unwrap(), c);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
