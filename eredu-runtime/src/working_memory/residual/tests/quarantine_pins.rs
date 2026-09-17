use super::*;
use crate::working_memory::InferenceRequest;
use eredu_core::{ResolvedGenerationConfig, TextGenerationConfig};

#[test]
fn residual_prompt_additional_sources_survive_either_native_scope_quarantining_first() {
    for prompt_first in [false, true] {
        let pool = WorkingMemoryPool::new(1000, 0).unwrap();
        let source_a = pool.register_storage([(1u32, 64)]).unwrap();
        let source_b = pool.register_storage([(2u32, 48), (3, 0)]).unwrap();
        let g = geometry();
        let quote = replacement_quote(&pool, g, 0);
        let execution = InferenceExecutionIdentity::default();
        let (_, reservation, accepted) = plan(&pool, &execution, &quote, request(g), 1000).unwrap();
        assert_eq!(reservation.bytes(), 96);
        let (metadata, run) = reservation.into_funding().unwrap();
        let config = TextGenerationConfig::new(ResolvedGenerationConfig {
            do_sample: false,
            temperature: 0.0,
            top_k: 17,
            top_p: 0.83,
            min_p: 0.07,
            repetition_penalty: 1.13,
            repeat_last_n: 23,
            frequency_penalty: 0.17,
            presence_penalty: 0.29,
            max_new_tokens: Some(2),
        });
        let preparation = InferenceRequest::from(metadata)
            .prepare_text(&execution, g, config)
            .unwrap();
        let ordinary = run.scope().unwrap();
        let complete = pool
            .pin_registered_storage([(1u32, 64), (2, 48), (3, 0)])
            .unwrap();
        let (completion, prompt) = preparation
            .claim_prompt()
            .unwrap()
            .construct_without_decoder(&run, complete)
            .unwrap();
        // Cancelling the original Prompt does not settle its independent scope.
        drop(completion);
        assert!(matches!(
            preparation.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        drop((source_a, source_b, quote, accepted, preparation, run));
        assert_eq!(used(&pool), (208, 208));
        if prompt_first {
            drop(prompt);
            drop(ordinary);
        } else {
            drop(ordinary);
            drop(prompt);
        }
        assert_eq!(used(&pool), (208, 208));
        // The additional B and zero-byte identity remain available as registered
        // sources even after their original owners and the fresh request retire.
        let retained = pool
            .pin_registered_storage([(1u32, 64), (2, 48), (3, 0)])
            .unwrap();
        assert_eq!(retained.bytes(), 112);
        drop(retained);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}
