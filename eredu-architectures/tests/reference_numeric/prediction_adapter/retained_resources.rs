//! Cold inspection must borrow the same physical owners as parameter control.
use super::*;

#[derive(Default)]
struct Resources {
    modules: Vec<(usize, usize)>,
    parameters: Vec<NumericTensor>,
    prototypes: Vec<usize>,
    stop: bool,
}
impl Resources {
    fn module<M: Parameterized<NumericTensor>>(
        &mut self,
        ordinal: usize,
        module: &Module<M>,
    ) -> Result<(), &'static str> {
        self.modules
            .push((ordinal, std::ptr::from_ref(module).addr()));
        if self.stop {
            return Err("injected inventory failure");
        }
        eredu_nn::visit_parameter_values(&module.0, &mut |value| {
            self.parameters.push(value.clone());
        });
        Ok(())
    }
}
impl PredictionModuleVisitor<NumericBackend, Materializer> for Resources {
    type Error = &'static str;
    fn visit<M: Parameterized<NumericTensor>>(
        &mut self,
        ordinal: usize,
        module: &mut Module<M>,
    ) -> Result<(), Self::Error> {
        self.module(ordinal, module)
    }
}
impl PredictionResourceVisitor<NumericBackend, Materializer> for Resources {
    type Error = &'static str;
    fn module<M: Parameterized<NumericTensor>>(
        &mut self,
        ordinal: usize,
        module: &Module<M>,
    ) -> Result<(), Self::Error> {
        self.module(ordinal, module)
    }
    fn pooling_state(&mut self, state: &NumericPoolingCache) -> Result<(), Self::Error> {
        self.prototypes.push(std::ptr::from_ref(state).addr());
        Ok(())
    }
    fn model_state(&mut self, state: &PredictionState) -> Result<(), Self::Error> {
        self.prototypes.push(std::ptr::from_ref(state).addr());
        Ok(())
    }
}

pub(super) fn verify<A, E>(extension: &mut E, expected_prototypes: usize)
where
    E: MaterializedPredictionExecutor<A, NumericBackend, Materializer>,
{
    let mut writable = Resources::default();
    extension.visit_modules(&mut writable).unwrap();
    let mut borrowed = Resources::default();
    extension.visit_retained_resources(&mut borrowed).unwrap();
    assert_eq!(
        borrowed.modules, writable.modules,
        "same modules, addresses and ordinals"
    );
    assert_eq!(borrowed.prototypes.len(), expected_prototypes);
    assert_eq!(borrowed.parameters.len(), writable.parameters.len());
    for (borrowed, writable) in borrowed.parameters.iter().zip(&writable.parameters) {
        assert_tensor_exact(
            borrowed,
            writable,
            "cold traversal reads current parameters",
        );
    }
    let mut failed = Resources {
        stop: true,
        ..Resources::default()
    };
    assert_eq!(
        extension.visit_retained_resources(&mut failed),
        Err("injected inventory failure")
    );
    assert_eq!(
        failed.modules,
        borrowed.modules[..1],
        "failure prevents subsequent owner visits"
    );
    assert!(failed.prototypes.is_empty());
}

#[test]
fn v4_retained_resources_borrow_every_nonzero_pooling_prototype() {
    #[derive(Default)]
    struct Borrowed {
        modules: Vec<usize>,
        states: Vec<usize>,
        values: Vec<f32>,
        fail_state: bool,
    }
    impl PredictionResourceVisitor<NumericBackend, CaptureOnlyMaterializer> for Borrowed {
        type Error = &'static str;
        fn module<M: Parameterized<NumericTensor>>(
            &mut self,
            ordinal: usize,
            _: &CaptureOnlyModule<M>,
        ) -> Result<(), Self::Error> {
            self.modules.push(ordinal);
            Ok(())
        }
        fn pooling_state(&mut self, state: &NumericPoolingCache) -> Result<(), Self::Error> {
            self.states.push(std::ptr::from_ref(state).addr());
            if self.fail_state {
                return Err("pool inspection failed");
            }
            self.values
                .extend(state.local.as_ref().unwrap().data.iter().copied());
            Ok(())
        }
        fn model_state(&mut self, _: &CaptureOnlyModelState) -> Result<(), Self::Error> {
            panic!("pooling prototypes have their own typed storage mechanism")
        }
    }
    let mut args = tiny_v4_args();
    args.num_nextn_predict_layers = 2;
    args.attention_schedule = eredu_core::LayerSchedule::new(
        5,
        vec![
            deepseek::V4AttentionPolicy::Local,
            deepseek::V4AttentionPolicy::Compressed { ratio: 4 },
            deepseek::V4AttentionPolicy::Local,
            deepseek::V4AttentionPolicy::Local,
            deepseek::V4AttentionPolicy::Local,
        ],
    )
    .unwrap();
    args.target_capture_policy =
        Some(deepseek::config::V4TargetCapturePolicy::new(vec![0, 1], 2).unwrap());
    args.dspark = Some(deepseek::DsparkConfig {
        block_size: 3,
        noise_token_id: 0,
        markov_rank: 2,
    });
    for fused in [false, true] {
        let state = [3.0, 7.0]
            .into_iter()
            .map(|value| {
                let mut state = NumericPoolingCache::new(4, &[]);
                state
                    .append_local(
                        NumericTensor::new([1, 1, 1], vec![value]),
                        &NumericContext::default(),
                    )
                    .unwrap();
                state
            })
            .collect::<Vec<_>>();
        let units = vec![CaptureOnlyModule::new(), CaptureOnlyModule::new()];
        let extension = if fused {
            MaterializedDeepSeekV4Prediction::<NumericBackend, CaptureOnlyMaterializer>::Dspark {
                strategy: DsparkPredictionStrategy::from_args(&args).unwrap(),
                static_modules: CaptureOnlyModule::new(),
                units,
                state,
            }
        } else {
            MaterializedDeepSeekV4Prediction::Sequential { units, state }
        };
        let expected = match &extension {
            MaterializedDeepSeekV4Prediction::Sequential { state, .. }
            | MaterializedDeepSeekV4Prediction::Dspark { state, .. } => state
                .iter()
                .map(|state| std::ptr::from_ref(state).addr())
                .collect::<Vec<_>>(),
        };
        let mut borrowed = Borrowed::default();
        extension.visit_retained_resources(&mut borrowed).unwrap();
        assert_eq!(
            borrowed.modules,
            (0..if fused { 3 } else { 2 }).collect::<Vec<_>>()
        );
        assert_eq!(borrowed.states, expected, "borrow prototypes, never clones");
        assert_eq!(borrowed.values, [3.0, 7.0]);
        let mut failed = Borrowed {
            fail_state: true,
            ..Borrowed::default()
        };
        assert_eq!(
            extension.visit_retained_resources(&mut failed),
            Err("pool inspection failed")
        );
        assert_eq!(failed.states, expected[..1]);
    }
}
