//! Ordinary requests use the common executor without an assistant parameter owner.
use super::*;

#[test]
fn single_model_causal_requests_match_ar_and_restore_the_pending_suffix() {
    use crate::composition::mlx::speculative::{
        MlxSpeculativeSampling, SpeculativeExecutionStreams,
    };
    use eredu_core::{
        GenerationRequestConfig, GenerationRequestTable, SpeculativeCallbackPublisher,
        SpeculativeOutputRuntime, SpeculativeRandomness, SpeculativeSemanticConstraint,
    };
    use eredu_runtime::{ConstrainedSampler, speculative::autoregressive::CausalExecutor};
    let root = tiny_heterogeneous_artifact(config());
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let model = eredu_core::load_model(
            &backend,
            root.path(),
            crate::MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                    eredu_runtime::WeightResidency::with_layers(policy.layers()),
                ),
            ),
        )
        .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(8),
                temperature: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap();
        let source_sampler = crate::composition::mlx::session::MlxTextSampler::from_config(
            &eredu_core::TextGenerationConfig::new(sampling),
        )
        .unwrap();
        let baseline = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling),
            ExcludeMask,
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect::<Vec<_>>();
        runtime.reset().unwrap();
        let prompt = <MlxBackend<'_> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
            runtime.backend(),
            vec![1, 4, 2],
        )
        .unwrap();
        runtime
            .session_mut()
            .with_model_operation(|model| {
                let context = SpeculativeExecutionStreams::single(&stream);
                let mut executor =
                    CausalExecutor::<MlxAutoregressiveMechanisms, eredu_runtime::speculative::autoregressive::NoTargetRefinement>::new(model.erased_mut(), eredu_runtime::speculative::autoregressive::NoTargetRefinement);
                let mut cache = executor.new_cache(context)?;
                let mut delivered = Vec::new();
                let output = SpeculativeOutputRuntime::new(
                    MlxSpeculativeSampling::new(ConstrainedSampler::new(
                        source_sampler,
                        ExcludeMask,
                    )),
                    eredu_core::GenerationSequence::new(8, []),
                    SpeculativeSemanticConstraint::plain(),
                    SpeculativeCallbackPublisher::tokens(|tokens| {
                        delivered.extend_from_slice(tokens);
                        Ok(())
                    }),
                    Default::default(),
                );
                let mut table = GenerationRequestTable::new(
                    Default::default(),
                    eredu_core::SpeculativeExecutionTopology::Single,
                )
                .unwrap();
                let id = table
                    .submit(
                        &mut executor,
                        &mut cache,
                        eredu_runtime::speculative::autoregressive::AutoregressiveInput::new(prompt, ()),
                        GenerationRequestConfig {
                            max_tokens: 8,
                            ..Default::default()
                        },
                        output,
                        SpeculativeRandomness::new(None, None),
                        false,
                        context,
                    )
                    .map_err(|error| {
                        Error::Exception(safemlx::error::Exception::from_source(error))
                    })?;
                while table.request(id).unwrap().sequence().tokens().len() < 3 {
                    table.step(&mut executor, false, context).unwrap();
                }
                let request = table.request(id).unwrap();
                assert_eq!(request.pending_committed_tokens(), &baseline[2..3]);
                let memory = request
                    .continuation_memory_observation(&executor, 0)
                    .unwrap()
                    .unwrap();
                assert_eq!(memory.target.current_positions, 5);
                assert!(memory.draft.is_none() && memory.embedded.is_none());
                assert_eq!(memory.seed_bytes, Some(0));
                let budget = eredu_runtime::execution_control::SnapshotBudget::new(
                    eredu_core::execution_control::SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 0,
                        retained_bytes: 64 << 20,
                        cumulative_copy_bytes: 128 << 20,
                    },
                );
                let estimate = request.control_snapshot_estimate(&executor).unwrap();
                let _saved_charge = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Snapshot,
                        Some(estimate),
                    )
                    .unwrap();
                let saved = request.control_snapshot(&executor, context).unwrap();
                table.run(&mut executor, true, context).unwrap();
                assert_eq!(table.request(id).unwrap().sequence().tokens(), baseline);
                assert_eq!(table.request(id).unwrap().stats().draft_tokens(), 0);
                let _restore_charge = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Restore,
                        Some(estimate),
                    )
                    .unwrap();
                table
                    .request_mut(id)
                    .unwrap()
                    .restore_control_snapshot(&mut executor, &saved, context)
                    .unwrap();
                table.run(&mut executor, true, context).unwrap();
                assert_eq!(table.request(id).unwrap().sequence().tokens(), baseline);
                table.finish().unwrap();
                assert_eq!(&delivered[..8], baseline);
                assert_eq!(&delivered[8..], &baseline[3..]);
                Ok(())
            })
            .unwrap();
    }
}
