struct LowStep {
    state: State,
    hidden: NumericTensor,
    output: Option<NumericTensor>,
    diagnostics: Vec<NumericTensor>,
    projections: Vec<(String, Vec<i32>)>,
    values: Vec<(String, NumericTensor)>,
}
struct LowVisitor {
    demand: OutputDemand,
    diagnostics: bool,
    observed: bool,
    config: moshi::MoshiConfig,
    widths: Vec<i32>,
    token_card: [i32; 2],
}
struct LastObserver(Vec<(String, NumericTensor)>);
impl ActivationObserver<NumericTensor, Error> for LastObserver {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        assert!(value.data.iter().all(|v| v.is_finite()));
        assert!(value.data.iter().any(|v| v.abs() > 1e-9));
        self.0.push((path.to_owned(), value.clone()));
        Ok(())
    }
}
impl moshi::MoshiRealtimeArchitectureVisitor<NumericBackend, State> for LowVisitor {
    type Output = Vec<LowStep>;
    type Error = Error;
    fn visit<A>(
        self,
        mut prepared: moshi::PreparedMoshiRealtimeArchitecture<A>,
        store: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: moshi::MoshiRealtimeExecutionArchitecture<NumericBackend, State>
            + RealtimeArchitectureIdentity
            + 'static,
        A::Error: std::fmt::Display,
    {
        let context = NumericContext::default();
        let architecture = prepared.take_architecture();
        let contract = prepared.take_contract();
        let constructed = construct_realtime_model::<A, NumericBackend, _>(
            architecture,
            None,
            contract,
            Construction {
                store,
                bound: Rc::new(Cell::new(0)),
                residency: Rc::new(RefCell::new(residency::Evidence::default())),
            },
            &context,
        )
        .map_err(Error::backend)?;
        let (mut execution, mut state) = constructed.into_execution_and_state();
        let mut steps = Vec::new();
        let mut frontier = 0;
        for width in self.widths {
            let temporal = (0..self.config.frame_schedule().total_audio_codebooks() + 1)
                .map(|slot| {
                    NumericTensor::new(
                        [1, width],
                        (0..width)
                            .map(|i| {
                                ((i + frontier + slot as i32 + 1)
                                    % if slot == 0 {
                                        self.token_card[0]
                                    } else {
                                        self.token_card[1]
                                    }) as f32
                            })
                            .collect(),
                    )
                })
                .collect::<Vec<_>>();
            let forced = (0..self.config.frame_schedule().depth_audio_codebooks() + 1)
                .map(|slot| {
                    PredictionDirective::Force(NumericTensor::new(
                        [1, width],
                        (0..width)
                            .map(|i| {
                                ((i + slot as i32 + 2)
                                    % if slot == 0 {
                                        self.token_card[0]
                                    } else {
                                        self.token_card[1]
                                    }) as f32
                            })
                            .collect(),
                    ))
                })
                .collect::<Vec<_>>();
            // Execute all bodies for this low-level comparison; a public demand
            // never changes depth state when no tail-skip permission is given.
            let plan = SequentialDecisionPlan::new(forced, self.diagnostics, false).unwrap();
            let mut driver = SequentialDecisionDriver::<NumericBackend, _>::new(
                plan,
                vec![
                    StatefulNumericSampler {
                        calls: 0,
                        invalid: false
                    };
                    self.config.frame_schedule().depth_audio_codebooks() + 1
                ],
                vec![0.0; self.config.frame_schedule().depth_audio_codebooks() + 1],
                Some(17),
            )
            .unwrap();
            context.projections.lock().unwrap().clear();
            let mut observer = LastObserver(Vec::new());
            let (output, forward) = if self.observed {
                moshi::execute_detached_replicated_moshi_realtime_with_observer_and_readout(
                    &mut execution,
                    &mut state,
                    &temporal,
                    &mut driver,
                    &context,
                    &mut observer,
                    self.demand,
                )
            } else {
                moshi::execute_detached_replicated_moshi_realtime_with_readout(
                    &mut execution,
                    &mut state,
                    &temporal,
                    &mut driver,
                    &context,
                    self.demand,
                )
            }
            .map_err(Error::backend)?;
            driver.finish().unwrap();
            assert_eq!(driver.random_state(), Some(&17));
            let hidden = forward.temporal_output().unwrap().clone();
            assert_eq!(
                hidden.shape,
                [1, width, self.config.temporal().hidden_size()]
            );
            assert!(hidden.data.iter().any(|v| v.abs() > 1e-9));
            steps.push(LowStep {
                state: state.clone(),
                hidden,
                output,
                diagnostics: driver
                    .diagnostics()
                    .iter()
                    .map(|d| d.logits().clone())
                    .collect(),
                projections: vocabulary(&context.projections.lock().unwrap()),
                values: observer.0,
            });
            frontier += width;
        }
        Ok(steps)
    }
}
fn low(
    path: &std::path::Path,
    config: &moshi::MoshiConfig,
    residency: ExecutionResidency,
    demand: OutputDemand,
    diagnostics: bool,
    observed: bool,
) -> Vec<LowStep> {
    let source = moshi::prepare_selected_moshi_realtime_source(selected_with_residency(
        path, config, residency,
    ))
    .unwrap();
    source.artifact_identity().unwrap();
    moshi::visit_selected_moshi_realtime_architecture::<NumericBackend, State, _>(
        source,
        &NumericContext::default(),
        LowVisitor {
            demand,
            diagnostics,
            observed,
            config: config.clone(),
            widths: vec![3, 2, 1, 1],
            token_card: [config.audio_vocabulary_size(); 2],
        },
    )
    .unwrap()
}
