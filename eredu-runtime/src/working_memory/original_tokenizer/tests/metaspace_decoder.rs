use super::*;

#[test]
fn metaspace_decoder_source_cold_refusal_and_failed_destinations_retain_custody() {
    for sequence in [false, true] {
        let decoder = serde_json::json!({"type":"Metaspace", "replacement":"▁", "prepend_scheme":"first", "split":false});
        let decoder = if sequence {
            serde_json::json!({"type":"Sequence", "decoders":[decoder]})
        } else {
            decoder
        };
        let input = serde_json::json!({"version":"1.0", "truncation":null, "padding":null,
            "normalizer":null, "pre_tokenizer":{"type":"Metaspace", "replacement":"▁", "prepend_scheme":"first", "split":true},
            "post_processor":null, "decoder":decoder, "added_tokens":[],
            "model":{"type":"WordLevel", "vocab":{"[UNK]":0,"▁hello":2,"▁world":7}, "unk_token":"[UNK]"}}).to_string();
        let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(&input)).unwrap();
        let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
        let error = short
            .compile_tokenizer_with(plan(&input), || {
                panic!("refused before source construction")
            })
            .unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(short.used_bytes().unwrap(), 0);
        for stage in 0..3 {
            let plan = plan(&input).fail_decode_reservation(stage);
            let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
            let error = pool.compile_tokenizer(plan).unwrap_err();
            assert_eq!(error.retained_bytes(), bytes);
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            drop(error);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let source = pool.compile_tokenizer(plan(&input)).unwrap();
        let declared = eredu_text::tokenizer::Tokenizer::from_bytes(input.as_bytes()).unwrap();
        assert!(source.matches_configuration(&declared));
        let mut changed: serde_json::Value = serde_json::from_str(&input).unwrap();
        let decoder = if sequence {
            &mut changed["decoder"]["decoders"][0]
        } else {
            &mut changed["decoder"]
        };
        decoder["prepend_scheme"] = "never".into();
        let changed =
            eredu_text::tokenizer::Tokenizer::from_bytes(changed.to_string().as_bytes()).unwrap();
        assert!(!source.matches_configuration(&changed));
        let alias = source.clone();
        drop((source, declared, input));
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        assert_eq!(alias.token_id("▁world"), Some(7));
        drop(alias);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
