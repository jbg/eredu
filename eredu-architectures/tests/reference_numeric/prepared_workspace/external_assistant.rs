//! Actual ordinary proposal and cold projection share their mutation boundary.
use super::*;
use eredu_architectures::{
    external_assistant::invocation::ExternalAssistantOperation,
    gemma4::{
        Assistant, AssistantConfig, AssistantState,
        assistant::invocation::{DraftStep, DraftStepArguments},
    },
};

fn configuration(ordered: bool) -> AssistantConfig {
    let value = serde_json::json!({
        "model_type":"gemma4_assistant", "backbone_hidden_size":8,
        "use_ordered_embeddings":ordered, "num_centroids":4, "centroid_intermediate_top_k":2,
        "tie_word_embeddings":false, "block_size":4,
        "text_config":{"model_type":"gemma4_text","hidden_size":8,
            "num_hidden_layers":1,"intermediate_size":12,"num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":4,"rms_norm_eps":0.00001,
            "vocab_size":16,"max_position_embeddings":64,"tie_word_embeddings":false,
            "attention_k_eq_v":false,"layer_types":["full_attention"]}
    });
    AssistantConfig::from_json(&serde_json::to_vec(&value).unwrap()).unwrap()
}
fn tensor(shape: impl Into<Vec<i32>>, shift: f32) -> NumericTensor {
    let shape = shape.into();
    let count = shape.iter().product::<i32>();
    NumericTensor::new(
        shape,
        (0..count)
            .map(|i| (i as f32 * 0.17 + shift).sin() * 0.8)
            .collect(),
    )
}
fn state() -> AssistantState<NumericTensor> {
    AssistantState {
        evidence: None,
        shared_kv: std::collections::HashMap::from([(
            eredu_core::AttentionPolicy::Full,
            (tensor(vec![2, 1, 3, 4], 0.5), tensor(vec![2, 1, 3, 4], 0.9)),
        )]).into(),
        kv_offset: 3,
        hidden: tensor(vec![2, 1, 8], 0.2),
    }
}

#[test]
fn external_assistant_operation_preserves_nonzero_proposals_and_failed_state() {
    let native = NumericContext::default();
    let mut ordinary = Assistant::<NumericBackend>::new(configuration(false), &native).unwrap();
    // NumericBackend constructors leave unloaded weights at zero. Populate the
    // real parameter visitor before cloning the two equivalent equation paths.
    struct Populate(usize);
    impl<'a> eredu_nn::ParameterVisitorMut<'a, NumericTensor> for Populate {
        fn visit_mut(&mut self, _: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            self.0 += 1;
            for (index, scalar) in value.data.iter_mut().enumerate() {
                *scalar = 0.03 * ((index as f32 + self.0 as f32 * 3.0) * 0.31).sin();
            }
        }
    }
    let mut populate = Populate(0);
    ordinary.visit_parameters_mut(&mut populate);
    assert!(populate.0 > 0);
    let mut hooked = ordinary.clone();
    let mut left = state();
    let mut right = left.clone();
    for turn in 0..3 {
        let embedding = tensor(vec![2, 1, 8], 1.0 + turn as f32);
        let expected = ordinary
            .draft_step::<NumericCache>(&embedding, &mut left, &native)
            .unwrap();
        let actual = DraftStep::execute::<NumericBackend, NumericCache>(
            &mut hooked,
            DraftStepArguments {
                embedding: &embedding,
                state: &mut right,
            },
            &native,
        )
        .unwrap();
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.data, expected.data);
        assert!(actual.data.iter().any(|value| value.abs() > 1e-7));
        assert_eq!(right.hidden.data, left.hidden.data);
        assert_eq!(right.kv_offset, 4 + turn);
    }
    right.shared_kv = Default::default();
    let hidden = right.hidden.data.clone();
    let offset = right.kv_offset;
    let embedding = tensor(vec![2, 1, 8], 2.0);
    assert!(
        DraftStep::execute::<NumericBackend, NumericCache>(
            &mut hooked,
            DraftStepArguments {
                embedding: &embedding,
                state: &mut right
            },
            &native
        )
        .is_err()
    );
    assert_eq!(right.hidden.data, hidden);
    assert_eq!(right.kv_offset, offset);
}

#[test]
fn external_ordered_assistant_cold_operation_preserves_actual_inputs_and_frontier() {
    let context = WorkspaceContext::new(Facts::default());
    let mut module = DraftStep::workspace_module(&configuration(true), &context).unwrap();
    let mut current = state();
    let embedding = tensor(vec![2, 1, 8], 0.1);
    let original_hidden = current.hidden.data.clone();
    let mut visits = Vec::new();
    let arguments = DraftStepArguments {
        embedding: &embedding,
        state: &mut current,
    };
    let mut projected = DraftStep::project(
        &arguments,
        |value| {
            visits.push(value.shape.clone());
            WorkspaceTensor::initialized(&value.shape, WorkspaceDtype::Float32, &context)
        },
        &context,
    )
    .unwrap();
    assert_eq!(
        visits,
        [
            vec![2, 1, 8],
            vec![2, 1, 8],
            vec![2, 1, 3, 4],
            vec![2, 1, 3, 4]
        ]
    );
    let geometry = DraftStep::geometry(&projected, &context).unwrap();
    assert_eq!(
        (
            geometry.batch_size,
            geometry.cached_positions,
            geometry.input_positions
        ),
        (2, 3, 1)
    );
    let mut opening = Vec::new();
    DraftStep::visit_projected(&projected, &mut |value| opening.push(value.clone()));
    context.begin_state_span(&opening).unwrap();
    let output = DraftStep::trace(&mut module, &mut projected, &context).unwrap();
    assert_eq!(output.shape(), [2, 1, 16]);
    assert_eq!(
        DraftStep::geometry(&projected, &context)
            .unwrap()
            .cached_positions,
        4
    );
    let mut retained = Vec::new();
    DraftStep::visit_projected(&projected, &mut |value| retained.push(value.clone()));
    assert_eq!(retained.len(), 4);
    assert_eq!(retained[1].shape(), [2, 1, 8]);
    assert_eq!(
        current.kv_offset, 3,
        "cold tracing does not mutate actual native state"
    );
    assert_eq!(current.hidden.data, original_hidden);
}

#[test]
fn external_fused_raw_context_quote_preserves_order_and_lazy_encoding(){
    use eredu_architectures::muse_glimmer::{DFlashConfig,assistant::invocation::{RawContext,RawContextArguments}};
    let config=fused_configuration();
    let facts=Facts::default();let operations=facts.operations.clone();
    let context=WorkspaceContext::new(facts);
    let mut module=RawContext::workspace_module(&config,&context).unwrap();
    let native=NumericContext::default();
    let taps=(0..5).map(|i|NumericTensor::new(vec![1,2,6656],vec![(i+1) as f32;2*6656])).collect::<Vec<_>>();
    let previous=NumericTensor::new(vec![1,3,33280],vec![9.0;3*33280]);
    let expected=config.prepare_raw_context_span(Some(&previous),&taps,&native).unwrap();
    assert_eq!(expected.shape,[1,5,33280]);
    assert!(expected.data[..3*33280].iter().all(|v|*v==9.0));
    for (tap,chunk) in expected.data[3*33280..4*33280].chunks_exact(6656).enumerate(){
        assert!(chunk.iter().all(|v|*v==(tap+1) as f32));
    }
    let arguments=RawContextArguments{previous:Some(&previous),states:&taps};
    let mut projected=RawContext::project(&arguments,|value|WorkspaceTensor::initialized(&value.shape,WorkspaceDtype::Float32,&context),&context).unwrap();
    let geometry=RawContext::geometry(&projected,&context).unwrap();
    assert_eq!((geometry.cached_positions,geometry.input_positions),(0,2));
    let mut opening=Vec::new();RawContext::visit_projected(&projected,&mut |value|opening.push(value.clone()));
    assert_eq!(opening.len(),6);context.begin_state_span(&opening).unwrap();
    operations.lock().unwrap().clear();
    let output=RawContext::trace(&mut module,&mut projected,&context).unwrap();
    assert_eq!(output.shape(),expected.shape.as_slice());
    // This boundary contains the tap and temporal concatenations only. The
    // first encoder projection remains at the next UpdateContext operation.
    let traced=operations.lock().unwrap();
    assert_eq!(traced.len(),2);
    assert!(traced.iter().all(|operation|matches!(operation.kind,WorkspaceOperationKind::Concatenate)));
    drop(traced);
    let mut closing=0;RawContext::visit_projected(&projected,&mut |_|closing+=1);
    RawContext::visit_output(&output,&mut |_|closing+=1);assert_eq!(closing,7);
    let empty=RawContextArguments::<NumericTensor>{previous:None,states:&[]};
    let projected=RawContext::project(&empty,|value|WorkspaceTensor::initialized(&value.shape,WorkspaceDtype::Float32,&context),&context).unwrap();
    assert!(RawContext::geometry(&projected,&context).is_err());
}

fn fused_configuration()->eredu_architectures::muse_glimmer::DFlashConfig{
    eredu_architectures::muse_glimmer::DFlashConfig::from_hf_json(br#"{
      "model_type":"muse_glimmer_assistant","hidden_size":6656,
      "intermediate_size":19968,"num_hidden_layers":5,"num_attention_heads":32,
      "num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.000001,
      "max_position_embeddings":131072,"sliding_window":2048,"block_size":16,
      "mask_token_id":201818,"target_layer_ids":[1,13,25,37,49],
      "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
      "hidden_act":"silu","attention_dropout":0.0,"rope_parameters":{"rope_theta":500000.0}
    }"#).unwrap()
}

#[test]
fn external_fused_context_operation_preserves_append_frontier_and_proposal_roots(){
    use eredu_architectures::muse_glimmer::{DFlashContext,DFlashLayerContext,
        assistant::invocation::{UpdateContext,UpdateContextArguments,FusedProposal,FusedProposalArguments}};
    let config=fused_configuration();
    let context=WorkspaceContext::new(Facts::default());
    let mut module=UpdateContext::workspace_module(&config,&context).unwrap();
    let tensor=|shape:&[i32]|WorkspaceTensor::initialized(shape,WorkspaceDtype::Float32,&context).unwrap();
    let previous=DFlashContext{encoded:tensor(&[1,3,6656]),layers:(0..5).map(|_|DFlashLayerContext{
        keys:tensor(&[1,8,3,128]),values:tensor(&[1,8,3,128])}).collect(),start:0,end:3};
    let pending=tensor(&[1,2,33280]);
    let arguments=UpdateContextArguments{previous:Some(&previous),pending:&pending,absolute_end:5};
    let mut projected=UpdateContext::project(&arguments,|value|Ok(value.clone()),&context).unwrap();
    let geometry=UpdateContext::geometry(&projected,&context).unwrap();
    assert_eq!((geometry.input_positions,geometry.cached_positions),(2,0));
    let mut opening=Vec::new();UpdateContext::visit_projected(&projected,&mut |value|opening.push(value.clone()));
    assert_eq!(opening.len(),12);context.begin_state_span(&opening).unwrap();
    let committed=UpdateContext::trace(&mut module,&mut projected,&context).unwrap();
    assert_eq!((committed.start,committed.end),(0,5));assert_eq!(committed.encoded.shape(),[1,5,6656]);
    assert_eq!(committed.layers.len(),5);
    for layer in &committed.layers{assert_eq!(layer.keys.shape(),[1,8,5,128]);assert_eq!(layer.values.shape(),[1,8,5,128]);}
    let mut roots=0;UpdateContext::visit_output(&committed,&mut |_|roots+=1);assert_eq!(roots,11);
    assert_eq!(previous.encoded.shape(),[1,3,6656]);assert_eq!((previous.start,previous.end),(0,3));
    let embeddings=tensor(&[1,4,6656]);
    let arguments=FusedProposalArguments{embeddings:&embeddings,committed:&committed,absolute_end:5};
    let mut projected=FusedProposal::project(&arguments,|value|Ok(value.clone()),&context).unwrap();
    let geometry=FusedProposal::geometry(&projected,&context).unwrap();
    assert_eq!((geometry.input_positions,geometry.cached_positions),(4,5));
    let mut opening=Vec::new();FusedProposal::visit_projected(&projected,&mut |value|opening.push(value.clone()));
    assert_eq!(opening.len(),12);context.begin_state_span(&opening).unwrap();
    let output=FusedProposal::trace(&mut module,&mut projected,&context).unwrap();
    assert_eq!(output.shape(),[1,3,6656]);
    assert_eq!((committed.start,committed.end),(0,5),"fused proposals leave committed context immutable");
    let bad=UpdateContextArguments{previous:Some(&previous),pending:&pending,absolute_end:6};
    let mut projected=UpdateContext::project(&bad,|value|Ok(value.clone()),&context).unwrap();
    assert!(UpdateContext::trace(&mut module,&mut projected,&context).is_err());
    assert_eq!(previous.end,3,"failed append does not alter the actual source");
}
