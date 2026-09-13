// Native phase hooks and public bounded capture, intervention and parameter APIs.
mod prediction_components {
    include!("prediction_components/bounded.rs");
    include!("prediction_components/partitioned.rs");
    include!("prediction_components/parameters.rs");
    include!("prediction_components/readout_bounds.rs");
    use super::*;
    use eredu_core::{
        speculative::SpeculativeActivationPhase as Phase, ModelConfigurationResolver,
    };
    use eredu_runtime::{inspection::SpeculativeActivationObserver, ActivationObserver};
    use safemlx::error::Exception;

    #[derive(Clone, Copy)]
    enum Mode {
        Observe,
        Mask,
        Fail(Phase),
    }
    #[derive(Default)]
    struct Trace {
        phases: Vec<(Phase, usize)>,
        finishes: Vec<bool>,
        values: Vec<(Phase, String, Vec<i32>, Vec<f32>)>,
        active: Option<(Phase, usize)>,
        remaining: usize,
    }
    struct Observer {
        trace: Arc<Mutex<Trace>>,
        paths: Vec<String>,
        channel: String,
        failure_path: String,
        mode: Mode,
        stream: Stream,
    }
    impl ActivationObserver<MlxTensor, Exception> for Observer {
        fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), Exception> {
            let mut trace = self.trace.lock().unwrap();
            let (phase, sequence) = trace.active.expect("internal hook requires an invocation");
            if self.paths.iter().any(|wanted| wanted == path) {
                let bytes = value.as_array().size().checked_mul(4).unwrap();
                trace.remaining = trace
                    .remaining
                    .checked_sub(bytes)
                    .ok_or_else(|| Exception::custom("native phase fixture exhausted allowance"))?;
                let shape = value.as_array().shape().to_vec();
                assert_eq!(
                    shape[1] as usize, sequence,
                    "physical geometry for {phase:?}/{path}"
                );
                let evaluated = value.as_array().evaluated()?;
                let values = evaluated
                    .try_to_vec::<f32>()
                    .map_err(|error| Exception::custom(error.to_string()))?;
                trace.values.push((phase, path.into(), shape, values));
            }
            if let Mode::Fail(failed) = self.mode {
                if failed == phase
                    && (path == self.failure_path
                        || (phase == Phase::TargetPrefill
                            && path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH))
                {
                    return Err(Exception::custom("injected native internal phase failure"));
                }
            }
            Ok(())
        }
        fn observe_generated(
            &mut self,
            path: &str,
            _: &MlxTensor,
            _: &eredu_core::capture::GeneratedCaptureSource,
            generate: &mut dyn FnMut() -> Result<MlxTensor, Exception>,
        ) -> Result<(), Exception> {
            // This fixture requests existing component values only. Unrequested
            // generated projections must not allocate native tensors.
            if self.paths.iter().any(|wanted| wanted == path) {
                self.observe(path, &generate()?)?;
            }
            Ok(())
        }
        fn intervene(
            &mut self,
            path: &str,
            value: &MlxTensor,
        ) -> Result<Option<MlxTensor>, Exception> {
            if !matches!(self.mode, Mode::Mask) || path != self.channel {
                return Ok(None);
            }
            let shape = value.as_array().shape();
            let width = *shape.last().unwrap() as usize;
            let mut values = vec![1.0_f32; value.as_array().size()];
            let index = values.len() - width;
            values[index] = 0.0;
            let mask = Array::from_slice(&values, shape);
            Ok(Some(MlxTensor::from_array(safemlx::ops::multiply(
                value.as_array(),
                &mask,
                &self.stream,
            )?)))
        }
    }
    impl SpeculativeActivationObserver<MlxTensor, Exception> for Observer {
        fn begin_activation_invocation(
            &mut self,
            phase: Phase,
            sequence: usize,
        ) -> Result<(), Exception> {
            let mut trace = self.trace.lock().unwrap();
            assert!(trace.active.replace((phase, sequence)).is_none());
            trace.phases.push((phase, sequence));
            Ok(())
        }
        fn complete_activation_invocation(&mut self) -> Result<(), Exception> {
            Ok(())
        }
        fn finish_activation_invocation(&mut self, success: bool) {
            let mut trace = self.trace.lock().unwrap();
            assert!(trace.active.take().is_some());
            trace.finishes.push(success);
        }
    }
    fn prove(device: DeviceType, fused: bool) {
        let checkpoint = tempfile::tempdir().unwrap();
        if fused {
            write_deepseek_v4_dspark_fixture(checkpoint.path(), false);
        } else {
            write_deepseek_fixture_with_values(checkpoint.path(), 2, 2, true);
        }
        let config =
            serde_json::from_slice(&std::fs::read(checkpoint.path().join("config.json")).unwrap())
                .unwrap();
        let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let scope = &graph.component_scopes[0];
        let channel = scope
            .components
            .iter()
            .find(|group| {
                matches!(
                    group.activation_equation,
                    eredu_core::component::ComponentActivation::Attention { .. }
                )
            })
            .unwrap();
        let mut paths = vec![
            channel.activation.clone(),
            channel.effective_activation.clone(),
            scope.readout.normalized.clone(),
            scope.readout.logits.clone(),
        ];
        if fused {
            paths.push("dspark.context.normalized".into());
        }
        let proposal_phase = if fused {
            Phase::FusedProposal
        } else {
            Phase::Proposal { depth: 0 }
        };
        let run = |mode: Option<Mode>| {
            let stream = Stream::new_with_device(&Device::new(device, 0));
            let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let backend = crate::native::backend(&stream, &weights);
            let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default()).unwrap();
            let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
            let trace = Arc::new(Mutex::new(Trace {
                remaining: 128 * 1024,
                ..Default::default()
            }));
            let outer = Arc::new(Mutex::new(ExternalObservationTrace::default()));
            let mut observers =
                eredu_architectures::speculative_execution::EmbeddedPredictionObservers::new(
                    eredu_runtime::NoopObserver,
                    EmbeddedLogitsObserver {
                        trace: Arc::clone(&outer),
                        intervention: ExternalTensorIntervention::None,
                    },
                );
            if let Some(mode) = mode {
                observers = observers.with_internal(Observer {
                    trace: Arc::clone(&trace),
                    paths: paths.clone(),
                    channel: channel.activation.clone(),
                    failure_path: if fused && matches!(mode, Mode::Fail(Phase::PredictionPrefill)) {
                        "dspark.context.normalized".into()
                    } else {
                        scope.readout.normalized.clone()
                    },
                    mode,
                    stream: stream.clone(),
                });
            }
            runtime
                .session_mut()
                .install_embedded_prediction_observer_set(observers)
                .unwrap();
            let tokens = [1_u32, 2, 3];
            let prompt = Array::from_slice(&tokens, &[1, 3]);
            let parts = [text_input_part(&prompt)];
            let (output, publications) = execute_neutral_embedded_mtp(
                &mut runtime,
                synthetic_prediction_input(&parts, &tokens),
                SpeculativeConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
            );
            (output, publications, trace, outer)
        };
        let (ordinary, _, disabled, ordinary_outer) = run(None);
        let ordinary = ordinary.unwrap();
        assert!(disabled.lock().unwrap().phases.is_empty());
        let (observed, _, trace, outer) = run(Some(Mode::Observe));
        let observed = observed.unwrap();
        assert_eq!(observed.token_ids(), ordinary.token_ids());
        assert_eq!(
            observed.stats().accept_lens(),
            ordinary.stats().accept_lens()
        );
        assert_eq!(
            outer.lock().unwrap().proposal_logits,
            ordinary_outer.lock().unwrap().proposal_logits
        );
        let trace = trace.lock().unwrap();
        assert_eq!(
            &trace.phases[..2],
            [
                (Phase::TargetPrefill, 3),
                (Phase::PredictionPrefill, if fused { 3 } else { 2 }),
            ]
        );
        assert!(trace
            .phases
            .iter()
            .any(|(phase, _)| *phase == proposal_phase));
        assert!(trace
            .phases
            .iter()
            .any(|(phase, _)| *phase == Phase::Verification));
        assert!(trace.finishes.iter().all(|success| *success));
        assert_eq!(trace.finishes.len(), trace.phases.len());
        assert!(trace.active.is_none());
        let original = trace
            .values
            .iter()
            .find(|(phase, path, _, _)| *phase == proposal_phase && path == &channel.activation)
            .unwrap();
        assert!(original.3.iter().any(|value| value.abs() > 1e-5));
        drop(trace);
        let (masked, _, trace, masked_outer) = run(Some(Mode::Mask));
        masked.unwrap();
        assert_ne!(
            masked_outer.lock().unwrap().proposal_logits[0],
            outer.lock().unwrap().proposal_logits[0],
            "channel intervention must reach actual draft head"
        );
        let trace = trace.lock().unwrap();
        let before = trace
            .values
            .iter()
            .filter(|(_, path, _, _)| path == &channel.activation)
            .collect::<Vec<_>>();
        let after = trace
            .values
            .iter()
            .filter(|(_, path, _, _)| path == &channel.effective_activation)
            .collect::<Vec<_>>();
        assert_eq!(before.len(), after.len());
        assert!(!before.is_empty());
        for (before, after) in before.into_iter().zip(after) {
            assert_eq!(before.0, after.0);
            assert_eq!(before.2, after.2);
            let index = before.3.len() - *before.2.last().unwrap() as usize;
            for i in 0..before.3.len() {
                assert_eq!(after.3[i], if i == index { 0.0 } else { before.3[i] });
            }
        }
        drop(trace);
        let mut failure_phases = vec![Phase::TargetPrefill, proposal_phase];
        if fused {
            failure_phases.push(Phase::PredictionPrefill);
        }
        for phase in failure_phases {
            let (failed, publications, trace, _) = run(Some(Mode::Fail(phase)));
            let error = match failed {
                Ok(_) => panic!("injected internal observer failure succeeded"),
                Err(error) => error,
            };
            assert!(
                error
                    .to_string()
                    .contains("injected native internal phase failure"),
                "unexpected failure: {error}"
            );
            if matches!(phase, Phase::TargetPrefill | Phase::PredictionPrefill) {
                assert_eq!(publications, 0);
            }
            let trace = trace.lock().unwrap();
            assert_eq!(trace.phases.last().unwrap().0, phase);
            assert_eq!(trace.finishes.last(), Some(&false));
            assert_eq!(trace.finishes.len(), trace.phases.len());
            assert!(trace.active.is_none());
        }
    }
    #[test]
    fn native_v3_internal_phases_reach_components_on_cpu() {
        prove(DeviceType::Cpu, false);
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a local MLX Metal device"]
    fn native_v3_internal_phases_reach_components_on_metal() {
        prove(DeviceType::Gpu, false);
    }
    #[test]
    fn native_dspark_internal_phases_reach_components_on_cpu() {
        prove(DeviceType::Cpu, true);
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a local MLX Metal device"]
    fn native_dspark_internal_phases_reach_components_on_metal() {
        prove(DeviceType::Gpu, true);
    }
}
