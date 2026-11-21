//! Source-free placement and invocation geometry for an external assistant.

use crate::{
    composite_execution::{CompositeArchitecture, ExternalPredictionTargetOperation},
    PredictionTargetOperation, PredictionTargetTransport, RuntimeState,
};
use crate::{
    executable_binding::discovery::{
        ExternalTargetCapturePlacement, ExternalTargetStaticPlacement, LoadedModelDiscovery,
    },
    speculative::ExternalPredictionTargetOperationKind,
    ForwardCaptureBundle, PredictionTensorAssemblyPlan,
};
use eredu_core::BackendFailure;
use eredu_nn::{Error, NeuralBackend, Tensor};
use std::{collections::BTreeMap, sync::Arc};

/// Retains source-free placement plans for a loaded external-assistant target.
#[derive(Default)]
pub struct ExternalTargetPlacements {
    discovery: Option<Arc<dyn LoadedModelDiscovery>>,
    captures: BTreeMap<Vec<String>, Option<Box<dyn ExternalTargetCapturePlacement>>>,
    operations: BTreeMap<
        ExternalPredictionTargetOperationKind,
        Option<Box<dyn ExternalTargetStaticPlacement>>,
    >,
}

impl ExternalTargetPlacements {
    /// Replaces target authority and invalidates any plans retained from it.
    pub(crate) fn bind(&mut self, discovery: Arc<dyn LoadedModelDiscovery>) {
        self.discovery = Some(discovery);
        self.captures.clear();
        self.operations.clear();
    }

    /// Resolve the exact target activation placement and local rank for an invocation.
    pub(crate) fn capture(
        &mut self,
        paths: &[String],
        decoder_shape: [u64; 2],
    ) -> Result<(Option<ForwardCaptureBundle>, usize), BackendFailure> {
        if !self.captures.contains_key(paths) {
            let placement = self
                .discovery
                .as_ref()
                .map(|discovery| discovery.prepare_target_capture_placement(paths))
                .transpose()?
                .flatten();
            self.captures.insert(paths.to_vec(), placement);
        }
        let placement = self.captures[paths].as_ref();
        let bundle = placement
            .map(|placement| placement.resolve(decoder_shape))
            .transpose()
            .map_err(BackendFailure::from_error)?;
        Ok((bundle, placement.map_or(0, |placement| placement.rank())))
    }

    /// Resolves static source placement for the current invocation geometry.
    pub(crate) fn operation(
        &mut self,
        kind: ExternalPredictionTargetOperationKind,
        input_shape: &[i32],
    ) -> Result<Option<PredictionTensorAssemblyPlan>, eredu_nn::Error> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.operations.entry(kind) {
            let placement = self
                .discovery
                .as_ref()
                .map(|discovery| discovery.prepare_target_static_placement(kind))
                .transpose()?
                .flatten();
            entry.insert(placement);
        }
        self.operations[&kind]
            .as_ref()
            .map(|placement| placement.resolve(kind, input_shape))
            .transpose()
    }
}

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
