//! Native partitioned parameter slots, tensor loans and overlay publication.
use super::*;
use eredu_core::parameters::native::{ParameterNative, ParameterReadFragment};

#[path = "partition/overlay.rs"]
mod overlay;
#[path = "partition/read.rs"]
mod read;

impl<Dispatch> ParameterNative<MlxBackend<'_, Dispatch>> for MlxSessionMechanisms {
    fn parameter_dtype(value: &MlxTensor) -> eredu_checkpoint::recipe::RecipeDtype {
        crate::backend::runtime::checkpoint::recipe::recipe_dtype_from_mlx(value.as_array().dtype())
    }

    fn prepared_parameter_dtype(
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
    ) -> Result<eredu_checkpoint::recipe::RecipeDtype, ParameterError> {
        use crate::backend::runtime::checkpoint::recipe::{mlx_dtype, recipe_dtype_from_mlx};
        mlx_dtype(dtype)
            .map(recipe_dtype_from_mlx)
            .map_err(|error| ParameterError::Unsupported(error.to_string()))
    }

    fn effective_parameter_supported(
        layout: &EffectiveParameterLayout,
        shape: &[u64],
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
    ) -> bool {
        crate::backend::runtime::checkpoint::recipe::mlx_dtype(dtype)
            .is_ok_and(|dtype| encoding::supported(layout, shape, dtype))
    }

    fn parameter_input_transform(
        layout: &EffectiveParameterLayout,
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
    ) -> ProjectionInputTransform {
        crate::backend::runtime::checkpoint::recipe::mlx_dtype(dtype)
            .map_or(ProjectionInputTransform::Unspecified, |dtype| {
                encoding::input_transform(layout, dtype)
            })
    }

    fn parameter_access_error(error: ParameterError) -> Error {
        Error::Other(Box::new(error))
    }

    fn fence_parameter_session(&self) {
        self.poison.set(true);
    }

    fn parameter_read_cost(
        layout: &EffectiveParameterLayout,
        fragments: &[ParameterReadFragment],
    ) -> Result<CaptureUsage, ParameterError> {
        read::parameter_read_cost(layout, fragments)
    }

    fn read_parameter_fragments(
        selected: &BTreeMap<String, MlxTensor>,
        layout: &EffectiveParameterLayout,
        parameter: &str,
        fragments: &[ParameterReadFragment],
        words: usize,
        stream: &Stream,
    ) -> Result<Vec<u32>, Error> {
        read::read_parameter_fragments(selected, layout, parameter, fragments, words, stream)
    }

    fn local_read_cost(
        layout: &EffectiveParameterLayout,
        source: u64,
        selected: u64,
        output: u64,
        directions: Option<u64>,
    ) -> Result<CaptureUsage, ParameterError> {
        read::local_read_cost(layout, source, selected, output, directions)
    }

    fn read_parameter_values(
        selected: &BTreeMap<String, MlxTensor>,
        layout: &EffectiveParameterLayout,
        parameter: &str,
        region: &ParameterRegion,
        projection: Option<&ParameterProjection>,
        context: &Stream,
    ) -> Result<Vec<f32>, Error> {
        read::read_parameter_values(selected, layout, parameter, region, projection, context)
    }

    fn local_overlay_cost(
        layout: &EffectiveParameterLayout,
        elements: u64,
    ) -> Result<(CaptureUsage, u64), ParameterError> {
        overlay::base_overlay_cost(layout, elements)
    }

    fn prepare_replacement<'a>(
        selected: &BTreeMap<String, MlxTensor>,
        layout: &EffectiveParameterLayout,
        parameter: &str,
        edits: impl Iterator<Item = (&'a ParameterRegion, &'a ParameterUpdate)>,
        stream: &Stream,
    ) -> Result<Result<(MlxTensor, MlxTensor), ParameterError>, Error> {
        overlay::prepare_replacement(selected, layout, parameter, edits, stream)
    }
}
