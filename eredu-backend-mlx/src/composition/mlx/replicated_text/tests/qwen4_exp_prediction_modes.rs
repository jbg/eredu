//! Mode commands survive typed embedded prediction erasure.
use super::*;
use eredu_core::{
    ControlledTextGeneration, ModelRuntime, TextGenerationConfig, TokenFilterController,
};

struct AllowAll;
impl TokenFilterController for AllowAll {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
        Ok(eredu_core::TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

struct ModeProbe<'a> {
    stream: &'a Stream,
    config: TextGenerationConfig,
    baseline: &'a [u32],
}
impl MlxEmbeddedExecutorContinuation for ModeProbe<'_> {
    fn execute(
        &mut self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut eredu_core::speculative::DynGenerationExecutor<
            '_,
            MlxGenerationExecutorTypes,
        >,
    ) -> Result<eredu_core::SpeculativeGenerationBatchOutput, Error> {
        use crate::composition::mlx::speculative::MlxSpeculativeSampling;
        use eredu_core::{
            GenerationRequestConfig, GenerationRequestTable, SpeculativeCallbackPublisher,
            SpeculativeOutputRuntime, SpeculativeRandomness, SpeculativeSemanticConstraint,
            decoding::{DecodingMode, SpeculationOptions},
            generation::SpeculativeRequestStatus,
        };
        assert_eq!(
            selected.requirements().strategy().proposal_capacity().get(),
            1
        );
        let context = SpeculativeExecutionStreams::single(self.stream);
        let tokens = Array::from_slice(&PROMPT, &[1, PROMPT.len() as i32]);
        let parts = [input::token_ids_part(&tokens)?];
        let prepared = eredu_runtime::PreparedModelInput::new(parts.to_vec(), |tensor| {
            eredu_runtime::PreparedInputInspector::identity(&input::MlxInputInspector, tensor)
        })
        .unwrap();
        let identity = prepared
            .cache_identity("miniature-qwen4-mode-handoffs")
            .unwrap();
        let prompt = MlxModelInput::from(input::ModelInput::with_cache_identity(&parts, &identity));
        let sampler = crate::composition::mlx::session::MlxTextSampler::from_config(&self.config)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let output = SpeculativeOutputRuntime::new(
            MlxSpeculativeSampling::new(sampler),
            eredu_core::GenerationSequence::new(self.baseline.len(), []),
            SpeculativeSemanticConstraint::plain(),
            SpeculativeCallbackPublisher::tokens(|_| Ok(())),
            Default::default(),
        );
        let mut cache = executor.new_cache(context)?;
        let mut table = GenerationRequestTable::new(
            Default::default(),
            eredu_core::SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                executor,
                &mut cache,
                prompt,
                GenerationRequestConfig {
                    max_tokens: self.baseline.len(),
                    ..Default::default()
                },
                output,
                SpeculativeRandomness::new(None, None),
                false,
                context,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let options = SpeculationOptions::default()
            .with_proposal_capacity(std::num::NonZeroUsize::new(1).unwrap());
        table
            .request(id)
            .unwrap()
            .request_decoding_mode(DecodingMode::SelfSpeculation(options))
            .unwrap();
        table.step(executor, false, context).unwrap();
        let request = table.request(id).unwrap();
        assert!(request.pending_committed_tokens().is_empty());
        assert_eq!(
            request
                .decoding_state()
                .acknowledgement
                .unwrap()
                .committed_position,
            1
        );
        let settled = request
            .continuation_memory_observation(executor, 0)
            .unwrap()
            .unwrap()
            .target
            .current_positions;
        assert_eq!(settled, PROMPT.len() as u64 + 1);
        table.step(executor, false, context).unwrap();
        let request = table.request(id).unwrap();
        assert_eq!(request.sequence().tokens().len(), 2);
        assert_eq!(
            request
                .continuation_memory_observation(executor, 0)
                .unwrap()
                .unwrap()
                .target
                .current_positions,
            settled
        );
        table.step(executor, false, context).unwrap();
        assert_eq!(table.request(id).unwrap().stats().draft_tokens(), 1);
        table
            .request(id)
            .unwrap()
            .request_decoding_mode(DecodingMode::Autoregressive)
            .unwrap();
        while table
            .request(id)
            .unwrap()
            .decoding_state()
            .pending
            .is_some()
        {
            table.step(executor, false, context).unwrap();
        }
        let request = table.request(id).unwrap();
        assert_eq!(request.status(), SpeculativeRequestStatus::ReadyToDraft);
        assert_eq!(
            request
                .decoding_state()
                .acknowledgement
                .unwrap()
                .committed_position,
            request.sequence().tokens().len() as u64
        );
        assert_eq!(
            request
                .continuation_memory_observation(executor, 0)
                .unwrap()
                .unwrap()
                .target
                .current_positions,
            PROMPT.len() as u64 + request.sequence().tokens().len() as u64
        );
        table.run(executor, false, context).unwrap();
        assert_eq!(
            table.request(id).unwrap().sequence().tokens(),
            self.baseline
        );
        Ok(eredu_core::SpeculativeGenerationBatchOutput::new(
            vec![],
            Default::default(),
        ))
    }
}

#[test]
fn erased_embedded_prediction_mode_handoffs_match_ordinary_ar_across_residencies() {
    // Normal loading and nested typed prediction visitors exceed the test
    // harness's default debug stack. Bound the fixture's thread explicitly.
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(check_embedded_mode_handoffs)
        .unwrap()
        .join()
        .unwrap();
}

fn check_embedded_mode_handoffs() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(8),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let config = TextGenerationConfig::new(sampling);
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
        LayerWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let backend = crate::native::backend(&stream, &weights);
        let model = eredu_core::load_model(
            &backend,
            &fixtures.safetensors_path,
            crate::MlxLoadRequest::from_normalized(
                NormalizedLoadRequest::default()
                    .with_weight_residency(eredu_runtime::WeightResidency::with_layers(residency))
                    .with_drafting(eredu_runtime::DraftingLoadRequest::embedded(1).unwrap()),
            ),
        )
        .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let baseline =
            ControlledTextGeneration::new(&mut runtime, PROMPT.to_vec(), config.clone(), AllowAll)
                .unwrap()
                .map(|token| token.unwrap().token_id())
                .collect::<Vec<_>>();
        runtime.reset().unwrap();
        runtime
            .session_mut()
            .with_model_operation(|model| {
                model
                    .erased_mut()
                    .with_embedded_prediction(&mut ModeProbe {
                        stream: &stream,
                        config: config.clone(),
                        baseline: &baseline,
                    })
                    .expect("retained embedded executor")?;
                Ok(())
            })
            .unwrap();
    }
}
