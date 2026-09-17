use super::*;
use eredu_runtime::PredictionTargetOperation;

struct StableOperation(u8);
impl PredictionTargetOperation<PathsFixture, FakeBackend, State> for StableOperation {
    type Output = i32;
    fn preserves_architecture_declarations(&self) -> bool {
        true
    }
    fn apply(
        self,
        _: &mut PathsFixture,
        state: &mut State,
        _: Option<&()>,
        _: &(),
    ) -> Result<i32, Error> {
        state.as_mut()[0].0 += 10;
        match self.0 {
            1 => Err(Error::backend("prediction failure after state work")),
            2 => panic!("prediction unwind after state work"),
            _ => Ok(state.as_ref()[0].0),
        }
    }
}
struct UnclassifiedOperation;
impl PredictionTargetOperation<PathsFixture, FakeBackend, State> for UnclassifiedOperation {
    type Output = ();
    fn apply(
        self,
        architecture: &mut PathsFixture,
        _: &mut State,
        _: Option<&()>,
        _: &(),
    ) -> Result<(), Error> {
        architecture.namespace = "changed";
        Ok(())
    }
}

#[test]
fn typed_prediction_preserves_declared_paths_across_success_failure_and_unwind() {
    macro_rules! check {
        ($runtime:expr) => {
            for mode in 0..3 {
                let mut runtime = $runtime;
                let paths = runtime.prepare_observation_paths().unwrap();
                let identity = paths.binding_identity();
                let declarations = runtime.architecture().declarations.get();
                let mut state = fixture_state();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    runtime.apply_prediction_target_operation(
                        StableOperation(mode),
                        &mut state,
                        None,
                        &(),
                    )
                }));
                match mode {
                    0 => assert!(result.unwrap().is_ok()),
                    1 => assert!(result.unwrap().is_err()),
                    _ => assert!(result.is_err()),
                }
                runtime.validate_observation_binding(&paths).unwrap();
                assert!(identity.matches(&paths));
                assert_eq!(runtime.architecture().declarations.get(), declarations);
                let input = prepared_composite_input(3);
                let mut observer = Observer::default();
                let (output, _) = runtime
                    .forward_with_prepared_observer_and_context_with_readout(
                        Some(&input),
                        &mut state,
                        &(),
                        &mut observer,
                        &paths,
                        eredu_core::OutputDemand::Sequence,
                    )
                    .unwrap();
                assert!(output.is_some());
                assert!(observer
                    .values
                    .iter()
                    .any(|(_, values)| values.0.iter().any(|value| *value != 0)));
                runtime
                    .apply_prediction_target_operation(UnclassifiedOperation, &mut state, None, &())
                    .unwrap();
                assert!(matches!(
                    runtime.validate_observation_binding(&paths),
                    Err(PreparedError::BindingMismatch)
                ));
            }
        };
    }
    check!(ResidentRuntime::<_, FakeBackend, State>::new(PathsFixture::new(), &()).unwrap());
    check!(layerwise());
}
