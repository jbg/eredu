//! Static target equations through the real prepared partition session transport.
use super::*;
use eredu_architectures::composite_execution::{
    CompositeArchitecture, ExternalPredictionTargetOperation as Op, PreparedCompositeArchitecture,
};
use eredu_runtime::{
    PredictionTargetOperation, PredictionTargetTransport, PredictionTensorAssembly,
    PredictionTensorAssemblyPlan, PredictionTensorSource,
};

type State = DeviceState<NumericBackend, NumericHybridLayerState>;
pub(super) struct Operation<'a> {
    pub operation: Op<'a, NumericTensor>,
    pub plan: &'a PredictionTensorAssemblyPlan,
    pub fault: Option<&'static str>,
}
impl<A> PredictionTargetOperation<PreparedCompositeArchitecture<A>, NumericBackend, State>
    for Operation<'_>
where
    A: CompositeArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Output = NumericTensor;
    fn apply(
        self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        _state: &mut State,
        parallel: Option<&NumericParallelContext>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        eredu_architectures::external_assistant::PreparedExternalTargetOperation::new(
            self.operation,
            Ok(Some(self.plan.clone())),
        )
        .apply(architecture, _state, parallel, context)
    }
    fn apply_with_transport<T: PredictionTargetTransport<NumericBackend>>(
        self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        state: &mut State,
        _: Option<&NumericParallelContext>,
        transport: &mut T,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let rank = transport.rank();
        let assembly = if self.fault == Some("prepare") && rank == 1 {
            Err(Error::backend("injected local static preparation"))
        } else if self.fault == Some("geometry") && rank == self.plan.sources()[0].owner() {
            let sources = self
                .plan
                .sources()
                .iter()
                .map(|source| {
                    let mut shape = source.shape().to_vec();
                    shape[1] = 1;
                    PredictionTensorSource::new(source.owner(), shape)
                })
                .collect::<Result<Vec<_>, _>>()?;
            PredictionTensorAssemblyPlan::new(
                "invalid-static-geometry",
                sources,
                if self.operation.kind().is_lookup() {
                    PredictionTensorAssembly::Sum
                } else {
                    PredictionTensorAssembly::Concatenate { axis: 2 }
                },
            )
            .map(Some)
        } else {
            Ok(Some(self.plan.clone()))
        };
        let result = eredu_architectures::external_assistant::PreparedExternalTargetOperation::new(
            self.operation,
            assembly,
        )
        .apply_with_transport(architecture, state, None, transport, context)?;
        if self.fault == Some("finish") && rank == 1 {
            for layer in state.as_mut() {
                eredu_runtime::ResettableRuntimeLayerState::<NumericBackend>::reset(layer)
                    .map_err(Error::backend_source)?;
            }
            return Err(Error::backend("injected static finishing"));
        }
        Ok(result)
    }
}




#[test]
fn static_target_transport_allocates_bf16_destinations_from_completed_precision_header() {
    struct Receiver {
        publications: usize,
        expected: NumericTensor,
    }
    impl PredictionTargetTransport<NumericBackend> for Receiver {
        fn rank(&self) -> usize {
            1
        }
        fn agree(&mut self, ready: bool, _descriptor: [u64; 8]) -> Result<bool, Error> {
            assert!(ready);
            Ok(true)
        }
        fn publish(&mut self, value: NumericTensor, owner: usize) -> Result<NumericTensor, Error> {
            assert_eq!(owner, 0);
            self.publications += 1;
            match self.publications {
                1 => {
                    assert_eq!(value.element_type(), Some(eredu_nn::TensorElementType::F32));
                    assert_eq!(value.shape(), [1]);
                    Ok(NumericTensor::new([1], vec![2.]))
                }
                2 => {
                    assert_eq!(
                        value.element_type(),
                        Some(eredu_nn::TensorElementType::Bf16)
                    );
                    assert_eq!(value.shape(), self.expected.shape());
                    assert!(value.data.iter().all(|v| *v == 0.));
                    Ok(self.expected.clone())
                }
                _ => panic!("unexpected publication"),
            }
        }
        fn complete(&mut self, value: &NumericTensor) -> Result<(), Error> {
            assert_eq!(
                value.element_type(),
                Some(eredu_nn::TensorElementType::Bf16)
            );
            Ok(())
        }
    }
    let expected = NumericTensor::new(
        [1, 2, 3],
        [-1.137, 2.417, 0.0031, 4.831, -0.27, 9.97]
            .map(|x| half::bf16::from_f32(x).to_f32())
            .to_vec(),
    )
    .with_dtype(eredu_core::checkpoint::TensorDtype::Bf16);
    let plan = PredictionTensorAssemblyPlan::new(
        "BF16 receiver",
        vec![PredictionTensorSource::new(0, vec![1, 2, 3]).unwrap()],
        PredictionTensorAssembly::Concatenate { axis: 2 },
    )
    .unwrap();
    let mut receiver = Receiver {
        publications: 0,
        expected: expected.clone(),
    };
    let actual = plan
        .assemble::<NumericBackend, _>(Ok(None), &mut receiver, &NumericContext::default())
        .unwrap();
    assert_eq!(
        actual.element_type(),
        Some(eredu_nn::TensorElementType::Bf16)
    );
    assert_tensor_exact(
        &actual,
        &expected,
        "BF16 static destination preserves every published value",
    );
    assert_eq!(receiver.publications, 2);
}
