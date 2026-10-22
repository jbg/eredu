//! Shared target-static equation composition inside the existing session transaction.
use crate::composite_execution::{CompositeArchitecture, ExternalPredictionTargetOperation};
use eredu_nn::{Error, NeuralBackend, Tensor};
use eredu_runtime::{
    PredictionTargetOperation, PredictionTargetTransport, PredictionTensorAssemblyPlan,
    RuntimeState,
};

/// A target-owned operation and its fallible retained static-source preparation.
/// Preparation failure is carried to peer agreement, rather than returned early.
pub struct PreparedExternalTargetOperation<'a, T> {
    operation: ExternalPredictionTargetOperation<'a, T>,
    assembly: Result<Option<PredictionTensorAssemblyPlan>, Error>,
}
impl<'a, T> PreparedExternalTargetOperation<'a, T> {
    /// Pairs the request with its source-free construction/geometry result.
    pub fn new(
        operation: ExternalPredictionTargetOperation<'a, T>,
        assembly: Result<Option<PredictionTensorAssemblyPlan>, Error>,
    ) -> Self {
        Self {
            operation,
            assembly,
        }
    }
}
impl<A, B, S> PredictionTargetOperation<A, B, S> for PreparedExternalTargetOperation<'_, B::Tensor>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Output = B::Tensor;
    fn apply(
        self,
        architecture: &mut A,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, Error> {
        self.assembly?;
        architecture
            .external_prediction_target_operation(self.operation, parallel, context)?
            .ok_or_else(|| {
                Error::backend(
                    "architecture does not implement the selected external target operation",
                )
            })
    }
    fn apply_with_transport<X: PredictionTargetTransport<B>>(
        self,
        architecture: &mut A,
        _state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        transport: &mut X,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, Error> {
        let plan = self.assembly.and_then(|p| {
            p.ok_or_else(|| {
                Error::backend(
                    "partitioned target operation has no retained static source placement",
                )
            })
        });
        let plan = match plan {
            Ok(plan) => plan,
            Err(error) => {
                transport.agree(false, [0x4552_5354, 0, 0, 0, 0, 0, 0, 0])?;
                return Err(error);
            }
        };
        let local = if plan
            .sources()
            .iter()
            .any(|source| source.owner() == transport.rank())
        {
            architecture
                .external_prediction_target_contribution(self.operation, None, context)
                .and_then(|value| {
                    value.ok_or_else(|| {
                        Error::backend("architecture has no selected static contribution")
                    })
                })
                .map(Some)
        } else {
            Ok(None)
        };
        let complete = plan.assemble::<B, _>(local, transport, context)?;
        let result = architecture
            .finish_external_prediction_target_operation(self.operation, complete, context)?
            .ok_or_else(|| {
                Error::backend("architecture has no selected static finishing equation")
            })?;
        transport.complete(&result)?;
        Ok(result)
    }
}
