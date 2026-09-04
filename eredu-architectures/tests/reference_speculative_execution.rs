//! Mechanisms-only reference backend proof through production architecture construction.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use eredu_core::{
    generation::SpeculativeSchedulerOptions, Completion, FinishReason, GenerationCancellationToken,
    GenerationSequence, PreparedSpeculativeLane, SamplingPlacement, SpeculativeConfig,
    SpeculativeConstraint, SpeculativeDraftRandomPosition, SpeculativeExecutionTopology,
    SpeculativeOutputError, SpeculativeOutputRuntime, SpeculativePublisher, SpeculativeRandomness,
    SpeculativeSampling, TokenFilter,
};
use eredu_nn::Tensor as _;
use eredu_nn::{
    AttentionCache, AttentionMask, AttentionRequest, EmbeddingLookupPolicy, EmbeddingOperator,
    EmbeddingSpec, Error, GroupSelection, GroupSelectionOperator, GroupedGatedProductOperator,
    GroupedGatedProductSpec, GroupedNeuralBackend, GroupedRelu2Operator, GroupedRelu2Spec, Index,
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, PadMode, ParameterMetadata, ParameterVisitor,
    ParameterVisitorMut, Parameterized, RotaryOperator, RotaryPosition, RotarySpec,
    TensorParallelGroupedGatedProductOperator, TensorParallelGroupedOutput,
    TensorParallelGroupedRelu2Operator, TopKGroupSelectorSpec, VocabularyParallelRange,
};
use eredu_runtime::SpeculativeScheduler;
use eredu_runtime::{
    ParameterBackend, PenaltyConfig, ResettableRuntimeLayerState, RuntimeLayerState,
    RuntimeStateComponents, Sampler, SamplingBackend, StateError, SubmissionBackend, TokenDomain,
};

include!("support/reference_backend.rs");
include!("support/reference_composite.rs");

#[derive(Default)]
struct Constraint;

impl SpeculativeConstraint for Constraint {
    fn fork(&self) -> Result<Self, SpeculativeOutputError> {
        Ok(Self)
    }

    fn push_token(&mut self, _: u32) -> Result<bool, SpeculativeOutputError> {
        Ok(false)
    }

    fn finish(&mut self, _: FinishReason) -> Result<(), SpeculativeOutputError> {
        Ok(())
    }
}

#[derive(Default)]
struct Publisher {
    publications: Rc<Cell<usize>>,
}

impl SpeculativePublisher<Constraint> for Publisher {
    fn publish_committed(
        &mut self,
        _: &mut Constraint,
        _: &[u32],
        _: &GenerationCancellationToken,
        _: bool,
    ) -> Result<bool, SpeculativeOutputError> {
        self.publications.set(self.publications.get() + 1);
        Ok(false)
    }

    fn publish_cancelled(&mut self, _: &mut Constraint) -> Result<(), SpeculativeOutputError> {
        Ok(())
    }
}

fn production_gemma_target_config() -> serde_json::Value {
    serde_json::json!({
        "architectures":["Gemma4ForConditionalGeneration"], "model_type":"gemma4",
        "tie_word_embeddings":false,
        "text_config": {"model_type":"gemma4_text","hidden_size":32,
            "num_hidden_layers":2,"intermediate_size":64,"num_attention_heads":4,
            "rms_norm_eps":0.00001,"vocab_size":32,"pad_token_id":0,
            "num_key_value_heads":2,"max_position_embeddings":128,"head_dim":8,
            "attention_k_eq_v":false,"num_kv_shared_layers":1,
            "layer_types":["full_attention","full_attention"],"tie_word_embeddings":false}
    })
}

fn production_muse_target_config() -> serde_json::Value {
    serde_json::json!({
        "architectures":["MuseGlimmerForConditionalGeneration"],"model_type":"muse_glimmer",
        "image_token_id":201817,"video_token_id":201816,"out_hidden_size":32,
        "projector_hidden_size":16,
        "text_config":{"model_type":"muse_glimmer_text","hidden_size":6656,
            "num_hidden_layers":50,"intermediate_size":19968,"num_attention_heads":32,
            "num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.000001,
            "post_norm_eps":0.000001,"vocab_size":201819,"max_position_embeddings":131072,
            "rope_theta":500000.0,"layer_types":vec!["sliding_attention";50],
            "layer_rope_theta":vec![500000.0;50],"sliding_window":2048,
            "tie_word_embeddings":false,"hidden_act":"silu","attention_dropout":0.0,
            "qk_scale_factor":1.0,"output_multiplier":1.0,"final_logit_softcapping":30.0},
        "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,
            "intermediate_size":12,"num_attention_heads":2,"num_hidden_layers":1,
            "patch_size":2,"patch_temporal":1,"merge_size":2,"pos_emb_height":2,
            "pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
            "hidden_act":"gelu","layer_types":["full_attention"],
            "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
    })
}

fn production_deepseek_v3_prediction_config() -> serde_json::Value {
    serde_json::json!({
        "architectures":["DeepseekV3ForCausalLM"],"model_type":"deepseek_v3",
        "hidden_size":16,"intermediate_size":32,"moe_intermediate_size":8,
        "num_hidden_layers":2,"num_attention_heads":2,"vocab_size":32,
        "max_position_embeddings":128,"q_lora_rank":4,"kv_lora_rank":4,
        "qk_nope_head_dim":6,"qk_rope_head_dim":2,"v_head_dim":8,
        "first_k_dense_replace":1,"moe_layer_freq":1,"n_routed_experts":2,
        "n_shared_experts":1,"num_experts_per_tok":1,"n_group":1,"topk_group":1,
        "topk_method":"noaux_tc","scoring_func":"sigmoid","norm_topk_prob":true,
        "routed_scaling_factor":1.0,"tie_word_embeddings":false,"attention_dropout":0.0,
        "hidden_act":"silu","num_nextn_predict_layers":1
    })
}

fn production_deepseek_v4_dspark_config() -> serde_json::Value {
    serde_json::json!({
        "architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4",
        "hidden_size":16,"moe_intermediate_size":8,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "qk_rope_head_dim":2,"q_lora_rank":2,"o_lora_rank":2,"o_groups":2,
        "vocab_size":32,"max_position_embeddings":128,"sliding_window":8,
        "compress_ratios":[0,0,0],"index_n_heads":2,"index_head_dim":4,"index_topk":1,
        "hc_mult":2,"hc_sinkhorn_iters":2,"n_routed_experts":2,"n_shared_experts":1,
        "num_experts_per_tok":1,"num_hash_layers":0,"scoring_func":"sqrtsoftplus",
        "topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.0,
        "swiglu_limit":4.0,"num_nextn_predict_layers":1,"dspark_block_size":2,
        "dspark_noise_token_id":0,"dspark_target_layer_ids":[0,1],"dspark_markov_rank":2
    })
}

fn assert_construction_stages(outcome: &ReferenceProductionOutcome) {
    assert_eq!(outcome.tokens, [2, 2, 2]);
    assert_eq!(outcome.publications, 2);
    assert_eq!(
        outcome.construction_stages,
        [
            eredu_core::SpeculativeLifecycleStage::Admission,
            eredu_core::SpeculativeLifecycleStage::Compatibility,
            eredu_core::SpeculativeLifecycleStage::Input,
            eredu_core::SpeculativeLifecycleStage::Execution,
        ]
    );
}

#[test]
fn sequential_embedded_runs_the_inspected_materialized_scheduler_path() {
    std::thread::Builder::new()
        .name("reference-embedded-sequential".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let outcome =
                run_reference_embedded_production(&production_deepseek_v3_prediction_config())
                    .expect("sequential embedded executor must reach the shared scheduler");
            assert_construction_stages(&outcome);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn fused_dspark_runs_the_inspected_materialized_scheduler_path() {
    std::thread::Builder::new()
        .name("reference-embedded-fused".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let outcome =
                run_reference_embedded_production(&production_deepseek_v4_dspark_config())
                    .expect("fused embedded executor must reach the shared scheduler");
            assert_construction_stages(&outcome);
            assert_eq!(outcome.accepted, [2]);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn gemma_runs_the_inspected_materialized_scheduler_path() {
    std::thread::Builder::new()
        .name("reference-gemma-production".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            clear_reference_trace();
            let _sampler = ReferenceSampler;
            let assistant = reference_gemma_assistant_artifact();
            let outcome = run_reference_external_production(
                &production_gemma_target_config(),
                assistant.path(),
            )
            .expect("Gemma executor must reach the shared scheduler");
            assert_construction_stages(&outcome);
            assert_eq!(outcome.accepted, [2]);
            assert_eq!(outcome.target_tokens, 6);
            assert!(!reference_trace().linear_outputs.is_empty());
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn released_muse_dflash_runs_the_inspected_materialized_scheduler_path() {
    std::thread::Builder::new()
        .name("reference-muse-production".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            clear_reference_trace();
            let assistant = reference_muse_assistant_artifact();
            let outcome = run_reference_external_production(
                &production_muse_target_config(),
                assistant.path(),
            )
            .expect("released DFlash executor must reach the shared scheduler");
            assert_construction_stages(&outcome);
            assert_eq!(outcome.accepted, [2]);
            assert!(!reference_trace().linear_outputs.is_empty());
        })
        .unwrap()
        .join()
        .unwrap();
}
