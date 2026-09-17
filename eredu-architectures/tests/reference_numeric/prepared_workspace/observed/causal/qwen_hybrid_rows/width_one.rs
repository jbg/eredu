//! Shared history-free mixer: recurrence remains real and stateful.
use super::*;

#[test]
fn width_one_direct_next_and_35_dense_and_routed_preserve_all_rows_and_state() {
    for kind in ["qwen3_next", "qwen3_5_text"] {
        for routed in [false, true] {
            let mut config = configuration(kind, routed, routed);
            config["linear_conv_kernel_dim"] = 1.into();
            compare_equations(config);
        }
    }
}

#[test]
fn width_one_layout_keeps_recurrence_and_wider_mixers_still_require_history() {
    let mut args =
        hybrid::model_args_from_config_value(&configuration("qwen3_5_text", false, false))
            .unwrap()
            .text;
    args.linear_conv_kernel_dim = 0;
    assert!(matches!(
        hybrid::state_layout(&args),
        Err(hybrid::HybridConfigError::Invalid(_))
    ));
    args.linear_conv_kernel_dim = 1;
    let layout = hybrid::state_layout(&args).unwrap();
    assert_eq!(layout.layer(0).unwrap().fixed_state().len(), 1);
    assert_eq!(
        layout.layer(0).unwrap().fixed_state()[0].role,
        StateTensorRole::Recurrent
    );
    for width in [2, 4] {
        args.linear_conv_kernel_dim = width;
        let mut state = HybridState::create(hybrid::state_layout(&args).unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
        assert!(state
            .layer(0)
            .unwrap()
            .fixed
            .remove(&StateTensorRole::Convolution { slot: 0 })
            .is_some());
        let before = state.clone();
        let context = NumericContext::default();
        let mut model =
            ResidentRuntime::new(HybridModel::new(args.clone(), &context).unwrap(), &context)
                .unwrap();
        load_recurrent_parameters(&mut model, 4);
        let error = model
            .forward(
                hybrid::EmbeddedInput::target(&NumericTensor::token_ids(&[1, 2]), None),
                &mut state,
                &context,
            )
            .unwrap_err();
        assert!(error.to_string().contains("Convolution"), "{error}");
        assert_hybrid_state(&state, &before);
    }
}

type Conditional = hybrid::ConditionalLayeredModel<NumericBackend>;
enum Owner {
    Text(ResidentRuntime<HybridModel, NumericBackend, HybridState>),
    Conditional(ResidentRuntime<Conditional, NumericBackend, HybridState>),
}
impl Owner {
    fn target(
        &mut self,
        ids: &[usize],
        state: &mut HybridState,
        context: &NumericContext,
    ) -> (NumericTensor, NumericTensor) {
        let tokens = NumericTensor::token_ids(ids);
        match self {
            Self::Text(model) => {
                let (scores, forward) = model
                    .forward_with_context(
                        hybrid::EmbeddedInput::target(&tokens, None),
                        state,
                        context,
                    )
                    .unwrap();
                (scores, forward.target_hidden().unwrap().clone())
            }
            Self::Conditional(model) => {
                let parts = [eredu_architectures::qwen::vl::InputPart::Text(&tokens)];
                let (scores, forward) = model
                    .forward_with_context(
                        hybrid::ConditionalInput::Target {
                            parts: &parts,
                            pixels: None,
                            mask: None,
                        },
                        state,
                        context,
                    )
                    .unwrap();
                (scores, forward.target_hidden().unwrap().clone())
            }
        }
    }
    fn draft(
        &mut self,
        hidden: &NumericTensor,
        state: &mut HybridState,
        context: &NumericContext,
    ) -> NumericTensor {
        let tokens = NumericTensor::token_ids(&[2]);
        match self {
            Self::Text(model) => model
                .forward(
                    hybrid::EmbeddedInput::draft(&tokens, hidden, 0),
                    state,
                    context,
                )
                .unwrap(),
            Self::Conditional(model) => model
                .forward(
                    hybrid::ConditionalInput::Draft {
                        tokens: &tokens,
                        hidden,
                        depth: 0,
                    },
                    state,
                    context,
                )
                .unwrap(),
        }
    }
}

fn owner(
    kind: Option<&str>,
    routed: bool,
    mtp: bool,
    context: &NumericContext,
) -> (Owner, HybridState) {
    if let Some(kind) = kind {
        let mut value = configuration(kind, routed, false);
        value["linear_conv_kernel_dim"] = 1.into();
        value["mtp_num_hidden_layers"] = usize::from(mtp).into();
        let args = hybrid::model_args_from_config_value(&value).unwrap().text;
        let state = HybridState::create(hybrid::state_layout(&args).unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
        let architecture = HybridModel::new(args, context).unwrap();
        assert_target_hooks::<HybridModel>(&architecture, mtp);
        let mut model = ResidentRuntime::new(architecture, context).unwrap();
        load_recurrent_parameters(&mut model, 4);
        (Owner::Text(model), state)
    } else {
        let mut value = conditional_qwen_partition_config(routed);
        value["text_config"]["linear_conv_kernel_dim"] = 1.into();
        value["text_config"]["mtp_num_hidden_layers"] = usize::from(mtp).into();
        let parsed = hybrid::model_args_from_config_value(&value).unwrap();
        let state =
            HybridState::create(hybrid::state_layout(&parsed.text).unwrap(), |_, policy| {
                Ok::<_, Error>(NumericHybridLayerState::new(policy))
            })
            .unwrap();
        let architecture = Conditional::new(parsed, context).unwrap();
        assert_target_hooks::<Conditional>(&architecture, mtp);
        let mut model = ResidentRuntime::new(architecture, context).unwrap();
        load_recurrent_parameters(&mut model, 4);
        (Owner::Conditional(model), state)
    }
}

fn target_and_prediction(kind: Option<&str>, routed: bool, mtp: bool) {
    let context = NumericContext::default();
    let (mut model, mut full) = owner(kind, routed, mtp, &context);
    model.target(&[4, 2], &mut full, &context);
    let prefix = full.clone();
    let mut chunked = prefix.clone();
    let (full_scores, full_hidden) = model.target(&[1, 3, 5, 2, 6], &mut full, &context);
    let mut consumed = Vec::new();
    let mut last_hidden = None;
    let mut score_parts = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let (scores, hidden) = model.target(ids, &mut chunked, &context);
        consumed.extend_from_slice(ids);
        let mut expected = prefix.clone();
        model.target(&consumed, &mut expected, &context);
        assert_hybrid_state(&chunked, &expected);
        last_hidden = Some(hidden);
        score_parts.push(scores);
    }
    assert_tensor_close(
        &NumericTensor::concatenate(&score_parts, 1, &context).unwrap(),
        &full_scores,
        "width-one target",
    );
    assert_hybrid_state(&chunked, &full);
    for state in [&full, &chunked] {
        for layer in &state.as_ref()[..2] {
            assert_eq!(layer.position(), 7);
            if layer.attention.is_none() {
                assert_eq!(layer.fixed.len(), 1);
                assert!(!layer
                    .fixed
                    .contains_key(&StateTensorRole::Convolution { slot: 0 }));
                let recurrent = layer.fixed[&StateTensorRole::Recurrent].as_ref().unwrap();
                assert!(recurrent.data.iter().all(|v| v.is_finite()));
                assert!(recurrent.data.iter().any(|v| v.abs() > 1e-9));
            }
        }
    }
    if mtp {
        assert_eq!(full.as_ref().len(), 3);
        assert!(full.as_ref()[2].fixed.is_empty());
        assert_eq!(full.as_ref()[2].position(), 0);
        let full_prior = full_hidden.axis_slice(1, 4, 5);
        let chunk_prior = last_hidden.unwrap().axis_slice(1, 1, 2);
        assert_tensor_close(
            &full_prior,
            &chunk_prior,
            "same actual target hidden for MTP",
        );
        let before = full.clone();
        let a = model.draft(&full_prior, &mut full, &context);
        let b = model.draft(&chunk_prior, &mut chunked, &context);
        assert_tensor_close(&a, &b, "actual full-attention draft");
        assert_hybrid_state(&full, &chunked);
        assert_eq!(full.as_ref()[2].position(), 1);
        let mut expected_target = before;
        *expected_target.layer(2).unwrap() = full.as_ref()[2].clone();
        assert_hybrid_state(&full, &expected_target);
    }
    for id in [1, 3, 2] {
        let (a, _) = model.target(&[id], &mut full, &context);
        let (b, _) = model.target(&[id], &mut chunked, &context);
        assert_tensor_close(&a, &b, "target resumes after draft");
        assert_hybrid_state(&full, &chunked);
        if mtp {
            assert_eq!(full.as_ref()[2].position(), 1);
        }
    }
}

#[test]
fn width_one_conditional_dense_and_routed_targets_preserve_complete_state() {
    for routed in [false, true] {
        target_and_prediction(None, routed, false);
    }
}

#[test]
fn width_one_target_mixer_preserves_actual_full_attention_mtp_and_return() {
    for kind in [Some("qwen3_next"), Some("qwen3_5_text"), None] {
        for routed in [false, true] {
            target_and_prediction(kind, routed, true);
        }
    }
}

fn assert_target_hooks<A>(architecture: &A, mtp: bool)
where
    A: LayeredArchitecture<NumericBackend, HybridState, Error = Error>,
{
    if mtp {
        use eredu_runtime::inspection::ObservationHookSite;
        let hooks = architecture.observation_hooks();
        for site in [
            ObservationHookSite::Input,
            ObservationHookSite::Unit,
            ObservationHookSite::Readout,
        ] {
            assert!(!hooks.supports(site));
        }
    }
}

#[path = "width_one/partition.rs"]
mod partition;
