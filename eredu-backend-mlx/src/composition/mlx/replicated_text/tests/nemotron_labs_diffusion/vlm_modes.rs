//! Sampled mode handoffs retain the prepared released image context.
use super::*;
use eredu_architectures::decoder::Config as _;

#[test]
#[ignore = "requires the pinned released VLM checkpoint and prepared image reference"]
fn released_prepared_image_context_survives_all_six_sampled_handoffs() {
    use crate::composition::mlx::session::MlxModelInput;
    use eredu_core::{TextAdvancement, decoding::*, execution_control::GenerationControlHandle};
    use safemlx::ops::indexing::TryIndexOp;

    let root = std::path::PathBuf::from(
        std::env::var_os("EREDU_NEMOTRON_RELEASE_ROOT").expect("pinned VLM checkpoint"),
    );
    let config = eredu_architectures::nemotron_labs_diffusion::FamilyConfig::from_hf_json(
        &std::fs::read(root.join("config.json")).unwrap(),
    )
    .unwrap();
    let (stream, weights) = execution_streams();
    let reference_path = std::env::var_os("EREDU_NEMOTRON_IMAGE_REFERENCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            root.join(if config.tensor_fp8() {
                "reference-mixed-context.safetensors"
            } else {
                "reference-bf16-context.safetensors"
            })
        });
    let source = std::env::var_os("EREDU_NEMOTRON_MODEL_SOURCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.clone());
    let reference = Array::load_safetensors(reference_path, &weights).unwrap();
    let grids = reference["vision.grids"].clone().into_evaluated().unwrap();
    let grids = grids.as_slice::<i32>().to_vec();
    assert_eq!(grids.len(), 4);
    let prompt = |suffix: &[u32]| -> MlxModelInput {
        let mut parts = vec![input::token_ids_part(&Array::from_slice(&[1u32], &[1, 1])).unwrap()];
        let mut offset = 0;
        for image in 0..2 {
            let (height, width) = (grids[2 * image], grids[2 * image + 1]);
            let end = offset + height * width;
            let patches = reference["vision.patches"]
                .try_index_device((offset..end, ..), &stream)
                .unwrap();
            parts.push(
                input::input_part(
                    eredu_core::InputModality::Image,
                    input::InputPayload::Tensor(patches),
                    [(
                        eredu_core::InputMetadataKey::PatchGrid,
                        Array::from_slice(&[1i32, height, width], &[1, 3]),
                    )],
                    [eredu_core::InputExtent::PatchGrid {
                        time: 1,
                        height: height as usize,
                        width: width as usize,
                    }],
                )
                .unwrap(),
            );
            if image == 0 {
                parts.push(input::token_ids_part(&Array::from_slice(&[4u32, 5], &[1, 2])).unwrap());
            }
            offset = end;
        }
        let tokens = std::iter::once(7u32)
            .chain(suffix.iter().copied())
            .collect::<Vec<_>>();
        parts.push(
            input::token_ids_part(&Array::from_slice(&tokens, &[1, tokens.len() as i32])).unwrap(),
        );
        MlxModelInput::from(input::ModelInput::new(&parts))
    };
    #[derive(Clone)]
    struct VocabularyFilter {
        vocabulary: usize,
        mask: u32,
    }
    impl TokenFilterController for VocabularyFilter {
        type Error = std::convert::Infallible;
        fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
            Ok(eredu_core::TokenFilter::allowed(
                (0..self.vocabulary)
                    .map(|id| id != self.mask as usize)
                    .collect(),
            )
            .unwrap())
        }
        fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
            assert_ne!(token, self.mask);
            Ok(())
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    impl eredu_core::SpeculativeTokenFilterController for VocabularyFilter {
        fn filter_at(&self, _: &[u32]) -> Result<eredu_core::TokenFilter, Self::Error> {
            self.clone().current_filter()
        }
        fn prefix_is_complete(&self, _: &[u32]) -> Result<bool, Self::Error> {
            Ok(false)
        }
        fn control_snapshot_bytes(&self) -> Option<u64> {
            Some(std::mem::size_of::<Self>() as u64)
        }
    }
    let filter = || VocabularyFilter {
        vocabulary: config.vocabulary_size() as usize,
        mask: config.mask_token_id,
    };
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(80),
            temperature: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        },
    )
    .unwrap();
    let generation_config = eredu_core::TextGenerationConfig::new(sampling);
    let backend = crate::native::backend(&stream, &weights);
    let model = eredu_core::load_model(
        &backend,
        &source,
        crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::layerwise_host(Default::default()),
            ),
        ),
    )
    .unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let baseline = ControlledTextGeneration::from_prompt(
        &mut runtime,
        prompt(&[]),
        generation_config.clone(),
        filter(),
    )
    .unwrap()
    .take(8)
    .map(|token| token.unwrap().token_id())
    .collect::<Vec<_>>();
    runtime.reset().unwrap();
    // A one-invocation schedule unmasking the full admitted block exercises
    // commitment and cache handoff without repeating numerical-policy coverage.
    let refinement = eredu_core::refinement::RefinementPolicy::MaskedUnmasking {
        config: eredu_core::refinement::MaskedRefinementOptions {
            max_denoising_steps: 1,
            ..Default::default()
        }
        .try_into()
        .unwrap(),
        mask_token_id: config.mask_token_id,
    };
    let speculation = DecodingMode::SelfSpeculation(
        SpeculationOptionsRequest {
            max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
            refinement: Some(refinement),
            ..Default::default()
        }
        .try_into()
        .unwrap(),
    );
    let diffusion = DecodingMode::Diffusion {
        policy: Some(refinement),
    };
    let mut committed = if std::env::var_os("EREDU_NEMOTRON_IMAGE_REQUEST_TABLE").is_some() {
        use crate::composition::mlx::speculative::{
            MlxSpeculativeSampling, SpeculativeExecutionStreams,
        };
        use eredu_core::{
            GenerationRequestConfig, GenerationRequestTable, SpeculativeCallbackPublisher,
            SpeculativeOutputRuntime, SpeculativeRandomness, SpeculativeSemanticConstraint,
        };
        let DecodingMode::SelfSpeculation(options) = speculation else {
            unreachable!()
        };
        runtime.session_mut().with_model_operation(|model| {
            let context = SpeculativeExecutionStreams::single(&stream);
            let policy = model.erased().capability_estimate().shared_backbone_draft().unwrap();
            let draft = policy.prepare(options, 0.0).unwrap();
            let mut executor = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(model.erased_mut(), policy.proposal_capacity());
            let mut cache = executor.new_cache(context)?;
            let sampler = crate::composition::mlx::session::MlxTextSampler::from_config(&generation_config).unwrap();
            let output = SpeculativeOutputRuntime::new(
                MlxSpeculativeSampling::new(eredu_runtime::ConstrainedSampler::new(sampler, filter())),
                eredu_core::GenerationSequence::new(80, []), SpeculativeSemanticConstraint::plain(),
                SpeculativeCallbackPublisher::tokens(|_| Ok(())), Default::default(),
            );
            let mut table = GenerationRequestTable::new(Default::default(), context.topology()).unwrap();
            let id = table.submit(&mut executor, &mut cache, SharedBackboneInput::new(prompt(&[]), draft,
                crate::backend::random::RandomState::with_seed(17)?, crate::backend::random::RandomState::with_seed(29)?),
                GenerationRequestConfig { max_tokens: 80, ..Default::default() }, output,
                SpeculativeRandomness::new(None, None), false, context,
            ).map_err(|error| Error::Exception(safemlx::error::Exception::from_source(error)))?;
            for mode in [speculation, DecodingMode::Autoregressive, diffusion, speculation, diffusion, DecodingMode::Autoregressive] {
                table.request(id).unwrap().request_decoding_mode(mode).unwrap();
                while table.request(id).unwrap().decoding_state().pending.is_some() { table.step(&mut executor, false, context).unwrap(); }
                let request = table.request(id).unwrap();
                assert!(request.pending_committed_tokens().is_empty());
                assert_eq!(request.decoding_state().acknowledgement.unwrap().committed_position, request.sequence().tokens().len() as u64);
                println!("released image-context request-table handoff {mode:?}: committed_position={}", request.sequence().tokens().len());
                if mode == diffusion && request.sequence().tokens().len() <= baseline.len() {
                    assert_eq!(request.sequence().tokens(), &baseline[..request.sequence().tokens().len()]);
                }
                let position = request.sequence().tokens().len();
                while table.request(id).unwrap().sequence().tokens().len() == position { table.step(&mut executor, false, context).unwrap(); }
            }
            Ok(table.request(id).unwrap().sequence().tokens().to_vec())
        }).unwrap()
    } else {
        let handle = GenerationControlHandle::default();
        let mut generation = ControlledTextGeneration::from_prompt(
            &mut runtime,
            prompt(&[]),
            generation_config.clone(),
            filter(),
        )
        .unwrap();
        generation.bind_control_handle(&handle).unwrap();
        let mut committed = vec![generation.next().unwrap().unwrap().token_id()];
        let mut previous = DecodingMode::Autoregressive;
        for mode in [
            speculation,
            DecodingMode::Autoregressive,
            diffusion,
            speculation,
            diffusion,
            DecodingMode::Autoregressive,
        ] {
            handle.request_decoding_mode(mode).unwrap();
            loop {
                match generation
                    .advance_operation()
                    .unwrap()
                    .expect("handoff before output limit")
                {
                    TextAdvancement::ModeChanged(ack) => {
                        assert_eq!(ack.committed_position, committed.len() as u64);
                        assert_eq!(handle.decoding_state().unwrap().active, mode);
                        assert_eq!(handle.decoding_state().unwrap().pending, None);
                        println!(
                            "released image-context handoff {previous:?} -> {mode:?}: committed_position={}",
                            ack.committed_position
                        );
                        break;
                    }
                    TextAdvancement::Token(token) => committed.push(token.token_id()),
                    TextAdvancement::Progress(_) => {}
                }
            }
            if mode == diffusion && previous == DecodingMode::Autoregressive {
                assert_eq!(
                    committed,
                    baseline[..committed.len()],
                    "greedy verification before diffusion preserves image-conditioned AR"
                );
            }
            committed.push(generation.next().unwrap().unwrap().token_id());
            previous = mode;
        }
        drop(generation);
        committed
    };
    let resumed = committed.pop().unwrap();
    runtime.reset().unwrap();
    let expected = ControlledTextGeneration::from_prompt(
        &mut runtime,
        prompt(&committed),
        generation_config,
        filter(),
    )
    .unwrap()
    .next()
    .unwrap()
    .unwrap()
    .token_id();
    assert_eq!(
        resumed, expected,
        "image-conditioned causal context matches a fresh prefill after both diffusion blocks"
    );
    println!(
        "released image-context six handoffs complete; {} committed prefix tokens; resumed AR token={resumed}",
        committed.len()
    );
}
