//! Native binding of retained target capture and static-operation placement.
use super::*;
use eredu_architectures::{
    external_assistant::PreparedTargetCapturePlacement, media_plan::AdmittedCompositeInput,
    prepared_sources::PreparedModelDiscovery,
};
use eredu_nn::Tensor as _;
use eredu_runtime::{ForwardCaptureBundle, PreparedModelInput};
use std::{collections::BTreeMap, sync::Arc};

type PreparedInput<P> = (
    PreparedModelInput<MlxTensor>,
    AdmittedCompositeInput<P>,
    Option<eredu_runtime::PreparedInputCacheIdentity>,
);
#[derive(Default)]
pub(in crate::composition::mlx::replicated_text) struct ExternalTargetBinding {
    discovery: Option<Arc<PreparedModelDiscovery>>,
    placements: BTreeMap<Vec<String>, Option<PreparedTargetCapturePlacement>>,
    static_placements: BTreeMap<
        eredu_architectures::composite_execution::ExternalPredictionTargetOperationKind,
        Option<eredu_architectures::external_assistant::PreparedTargetStaticPlacement>,
    >,
}
impl ExternalTargetBinding {
    pub(in crate::composition::mlx::replicated_text) fn bind(
        &mut self,
        discovery: Arc<PreparedModelDiscovery>,
    ) {
        self.discovery = Some(discovery);
        self.placements.clear();
        self.static_placements.clear();
    }
    pub(in crate::composition::mlx::replicated_text) fn operation<'a>(
        &mut self,
        operation: ExternalPredictionTargetOperation<'a, MlxTensor>,
    ) -> eredu_architectures::external_assistant::PreparedExternalTargetOperation<'a, MlxTensor>
    {
        let assembly = (|| {
            let kind = operation.kind();
            if !self.static_placements.contains_key(&kind) {
                let placement = self
                    .discovery
                    .as_ref()
                    .map(|d| d.prepare_target_static_placement(kind))
                    .transpose()?
                    .flatten();
                self.static_placements.insert(kind, placement);
            }
            self.static_placements[&kind]
                .as_ref()
                .map(|p| p.resolve(kind, operation.input().shape()))
                .transpose()
        })();
        eredu_architectures::external_assistant::PreparedExternalTargetOperation::new(
            operation, assembly,
        )
    }
    pub(in crate::composition::mlx::replicated_text) fn prepare<P>(
        &mut self,
        paths: Result<Vec<String>, Error>,
        input: Result<PreparedInput<P>, Error>,
    ) -> PreparedCaptureInput<P> {
        let prepared: Result<_, Error> = (|| {
            let paths = paths?;
            let input = input?;
            if !self.placements.contains_key(&paths) {
                let placement = self
                    .discovery
                    .as_ref()
                    .map(|discovery| discovery.prepare_target_capture_placement(&paths))
                    .transpose()
                    .map_err(|e| Error::Other(Box::new(e)))?
                    .flatten();
                self.placements.insert(paths.clone(), placement);
            }
            let placement = self.placements[&paths].as_ref();
            let bundle = placement
                .map(|p| p.resolve(input.1.decoder_shape()))
                .transpose()
                .map_err(|e| Error::Other(Box::new(e)))?;
            let rank = placement.map_or(0, |p| p.topology().global_rank());
            let observer = ExactPredictionCaptureObserver::new(paths)?;
            Ok((input, bundle, rank, observer))
        })();
        match prepared {
            Ok((input, bundle, rank, observer)) => PreparedCaptureInput {
                input: Ok(input),
                bundle,
                rank,
                observer,
            },
            Err(error) => PreparedCaptureInput {
                input: Err(eredu_nn::Error::backend_source(error)),
                bundle: None,
                rank: 0,
                observer: ExactPredictionCaptureObserver {
                    paths: vec![],
                    values: std::rc::Rc::new(std::cell::RefCell::new(vec![])),
                },
            },
        }
    }
}
pub(in crate::composition::mlx::replicated_text) struct PreparedCaptureInput<P> {
    input: Result<PreparedInput<P>, eredu_nn::Error>,
    bundle: Option<ForwardCaptureBundle>,
    rank: usize,
    observer: ExactPredictionCaptureObserver,
}
impl<P> PreparedCaptureInput<P> {
    pub(in crate::composition::mlx::replicated_text) fn with_input<R>(
        mut self,
        run: impl FnOnce(
            Result<PreparedCompositeInput<'_, MlxTensor, P>, eredu_nn::Error>,
            Option<&ForwardCaptureBundle>,
            usize,
            &mut ExactPredictionCaptureObserver,
        ) -> R,
    ) -> R {
        match self.input {
            Ok((input, admitted, _)) => run(
                PreparedCompositeInput::new(&input, &admitted).map_err(eredu_nn::Error::backend),
                self.bundle.as_ref(),
                self.rank,
                &mut self.observer,
            ),
            Err(error) => run(Err(error), None, self.rank, &mut self.observer),
        }
    }
}
pub(in crate::composition::mlx::replicated_text) fn collect(
    observed: &std::rc::Rc<std::cell::RefCell<Vec<Option<MlxTensor>>>>,
    bundle: Option<&ForwardCaptureBundle>,
    rank: usize,
    stream: &Stream,
) -> Result<Vec<MlxTensor>, eredu_nn::Error> {
    observed
        .borrow()
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if let Some(slot) = bundle
                .map(|plan| &plan.slots()[index])
                .filter(|slot| slot.owner() != rank && value.is_none())
            {
                let dtype = match slot.placeholder_element() {
                    eredu_nn::TensorElementType::F16 => safemlx::Dtype::Float16,
                    eredu_nn::TensorElementType::Bf16 => safemlx::Dtype::Bfloat16,
                    eredu_nn::TensorElementType::F32 => safemlx::Dtype::Float32,
                    _ => {
                        return Err(eredu_nn::Error::backend(
                            "selected target capture has a nonfloating transport dtype",
                        ))
                    }
                };
                return safemlx::ops::zeros_dtype(slot.shape(), dtype, stream)
                    .map(MlxTensor::from_array)
                    .map_err(eredu_nn::Error::backend_source);
            }
            value.clone().ok_or_else(|| {
                eredu_nn::Error::backend(format!(
                    "external-assistant target did not reach capture path {index}"
                ))
            })
        })
        .collect()
}
