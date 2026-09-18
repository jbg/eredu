//! The existing target equations with exact external-assistant output custody.
use super::*;
use crate::composite_execution::{
    CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    ExternalPredictionTargetOperation,PreparedCompositeArchitecture,
};
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceMetadataError};
use eredu_runtime::speculative::external_occurrence::{ExternalInvocation, ExternalInvocationKind};
use eredu_runtime::{ActivationObserver, SharedLayeredObservationPaths};

#[derive(Clone, Copy)]
pub(super) enum ExternalTargetQuote<'a> {
    Capture { request: &'a ExternalPredictionCaptureRequest, paths: &'a SharedLayeredObservationPaths, input: &'a WorkspaceTensor },
    Static(ExternalPredictionTargetOperation<'a,WorkspaceTensor>),
}
impl<'a> ExternalTargetQuote<'a> {
    pub(super) fn paths(self)->Option<&'a SharedLayeredObservationPaths>{
        match self {Self::Capture{paths,..}=>Some(paths),Self::Static(_)=>None}
    }
}

pub(super) enum EquationCapture {
    Embedded(WorkspaceTensor),
    External(ExternalPredictionTargetCapture<WorkspaceTensor>),
}
impl EquationCapture {
    pub(super) fn visit_values(&self, visitor: &mut dyn FnMut(&WorkspaceTensor)) {
        match self {
            Self::Embedded(value) => visitor(value),
            Self::External(capture) => capture.visit_values(visitor),
        }
    }
    pub(super) fn root_count(&self) -> usize {
        let mut count = 0;
        self.visit_values(&mut |_| count += 1);
        count
    }
}

pub(super) struct CaptureObserver<'a> {
    paths: Vec<String>,
    values: Vec<Option<WorkspaceTensor>>,
    context: &'a WorkspaceContext,
}
impl<'a> CaptureObserver<'a> {
    pub fn new<A>(
        request: &ExternalPredictionCaptureRequest,
        context: &'a WorkspaceContext,
    ) -> Result<Self, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
    {
        context.charge_metadata(std::mem::size_of::<(Self, Result<Self, Error>)>())?;
        let paths = A::external_prediction_capture_paths_with_metadata(request, context)?
            .ok_or_else(|| {
                context.metadata_error(format_args!(
                    "external assistant capture differs from the selected target"
                ))
            })?;
        if paths.is_empty()
            || paths
                .iter()
                .enumerate()
                .any(|(i, path)| paths[..i].contains(path))
        {
            return Err(context.metadata_error(format_args!(
                "external capture paths are empty or duplicated"
            )));
        }
        let mut values = context.metadata_vec(paths.len())?;
        values.resize_with(paths.len(), || None);
        Ok(Self {
            paths,
            values,
            context,
        })
    }
    pub fn into_values(self) -> Result<Vec<WorkspaceTensor>, Error> {
        let mut output = self.context.metadata_vec(self.values.len())?;
        for value in self.values {
            output.push(value.ok_or_else(|| {
                self.context.metadata_error(format_args!(
                    "external target did not reach its selected capture path"
                ))
            })?);
        }
        Ok(output)
    }
}
impl ActivationObserver<WorkspaceTensor, Error> for CaptureObserver<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        self.paths
            .iter()
            .any(|path| path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
    }
    fn observe(&mut self, path: &str, value: &WorkspaceTensor) -> Result<(), Error> {
        if let Some(index) = self.paths.iter().position(|expected| expected == path) {
            if self.values[index].is_some() {
                return Err(self
                    .context
                    .metadata_error(format_args!("duplicate external capture path")));
            }
            self.context
                .charge_metadata(std::mem::size_of::<Option<WorkspaceTensor>>())?;
            self.values[index] = Some(value.clone());
        }
        Ok(())
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &WorkspaceTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        if self.paths.iter().any(|expected| expected == path) {
            self.observe(path, &generate()?)?;
        }
        Ok(())
    }
}

impl PreparedInferenceBlueprint {
    /// Quotes one actual external target prefill/verification through ordinary
    /// total construction, including its complete family-declared capture.
    /// The invocation is descriptive; native source binding and admission remain
    /// mandatory in the caller. Paths must be the target's retained source.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_external_target_invocation(
        &self,
        invocation: ExternalInvocation,
        request: &ExternalPredictionCaptureRequest,
        paths: &SharedLayeredObservationPaths,
        input: &WorkspaceTensor,
        media: Option<OriginalMediaWorkspaceInput>,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        trace: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        let geometry = invocation.geometry();
        let input_dtype = input.layout().dtype();
        context
            .charge_metadata(std::mem::size_of::<(
                ExternalInvocation,
                ExternalTargetQuote<'_>,
                EquationVisitor<'_, '_, '_>,
                EquationCapture,
                Result<EquationQuote, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        if !matches!(
            invocation.kind(),
            ExternalInvocationKind::TargetPrefill | ExternalInvocationKind::TargetVerification
        ) || !matches!(input_dtype, WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
            || geometry.max_output_tokens != 0
            || geometry.prefill_chunk_positions != geometry.input_positions
        {
            return Err(PreparedExecutionError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        geometry
            .validate()
            .map_err(|cause| preparation_message(context, format_args!("{cause}")))?;
        if self.selected().text_realization().state().policy() != &CacheResidencyPolicy::Device {
            return Err(preparation_message(
                context,
                format_args!("external target requires projected selected state"),
            ));
        }
        if parameters.is_some()
            && !matches!(
                self.selected().text_realization().residency(),
                eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
            )
        {
            return Err(preparation_message(
                context,
                format_args!("external parameter provider differs from selected residency"),
            ));
        }
        let trace = std::cell::RefCell::new(trace);
        context.charge_metadata(std::mem::size_of::<std::cell::RefCell<Option<OriginalMediaWorkspaceInput>>>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let has_media = media.is_some();
        let media = std::cell::RefCell::new(media);
        EquationVisitor {
            geometry,
            state,
            context,
            sampling: None,
            parameters,
            unpriced_execution: None,
            observation: None,
            trace: Some(EquationTraceRef(&trace)),
            media: has_media.then_some(MediaEquationRef { input: &media, intervals: None }),
            input_dtype: Some(input_dtype),
            target_capture: false,
                routed_pass: None,
            external_target: Some(ExternalTargetQuote::Capture { request, paths, input }),
        }
        .construct(self)
        .map(|(report, _)| report)
    }
}

mod static_target;
