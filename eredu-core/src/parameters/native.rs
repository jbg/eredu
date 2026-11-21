//! Physical slots and native tensor operations for loaded parameter access.

use crate::{
    backend::{NativeTextValues, TextSessionNative, TextTransactionMechanisms},
    capture::CaptureUsage,
    consensus::BoundedConsensusTransport,
    Completion, TextGenerationBackend,
};
use eredu_checkpoint::LinearFormat;
use eredu_nn::LinearCompanionRole;
use std::collections::{BTreeMap, BTreeSet};

use super::{
    ParameterError, ParameterProjection, ParameterProjectionFragment, ParameterRegion,
    ParameterUpdate, ProjectionInputTransform,
};

#[derive(Clone)]
/// Layout and loan requirements for one locally available parameter.
pub struct EffectiveParameterLayout {
    /// Selected storage and arithmetic format.
    pub format: LinearFormat,
    /// Logical local shape of the decoded parameter.
    pub shape: Vec<u64>,
    /// Placement of logical rows within the encoded storage.
    pub row_layout: eredu_nn::LinearRowLayout,
    scale: Option<String>,
    affine_bias: Option<String>,
    global_scale: Option<String>,
}
impl EffectiveParameterLayout {
    /// Creates a layout from the selected logical format and shape.
    pub fn new(format: LinearFormat, shape: Vec<u64>) -> Self {
        Self {
            format,
            shape,
            row_layout: eredu_nn::LinearRowLayout::Contiguous,
            scale: None,
            affine_bias: None,
            global_scale: None,
        }
    }
    /// Records a companion declared by parameter traversal.
    pub fn bind_companion(&mut self, role: LinearCompanionRole, id: String) {
        match role {
            LinearCompanionRole::Scale => self.scale = Some(id),
            LinearCompanionRole::AffineBias => self.affine_bias = Some(id),
            LinearCompanionRole::GlobalScale => self.global_scale = Some(id),
        }
    }
    /// Adds the exact declared companion slots.
    pub fn extend_dependencies(&self, ids: &mut BTreeSet<String>) {
        ids.extend(self.scale.iter().cloned());
        ids.extend(self.affine_bias.iter().cloned());
        ids.extend(self.global_scale.iter().cloned());
    }
    /// Returns the binding name for the exact encoded companion.
    pub fn companion(&self, role: LinearCompanionRole) -> Option<&str> {
        match role {
            LinearCompanionRole::Scale => self.scale.as_deref(),
            LinearCompanionRole::AffineBias => self.affine_bias.as_deref(),
            LinearCompanionRole::GlobalScale => self.global_scale.as_deref(),
        }
    }
}

/// An admitted local tensor read or contraction.
pub struct ParameterReadFragment {
    /// Region of the rank-local effective weight.
    pub region: ParameterRegion,
    /// Optional contraction restricted to this region.
    pub projection: Option<ParameterProjectionFragment>,
    /// Number of values this fragment contributes to the result.
    pub output_elements: u64,
}

/// Native data and operations for loaded parameter access. The text driver owns
/// rank admission, transport ordering and delivery of globally assembled results.
pub trait ParameterNative<B: TextGenerationBackend>:
    TextSessionNative<
        B,
        TextValues: NativeTextValues<Tensor = Self::ParameterValue>,
        ParameterValue: eredu_nn::Tensor,
        ControlTransport: Clone
                              + BoundedConsensusTransport<
            Completion: Completion<Error: Send + Sync + 'static>,
        >,
    > + TextTransactionMechanisms<Error = B::Error>
{
    /// Reports the physical encoding of a native tensor.
    fn parameter_dtype(value: &Self::ParameterValue) -> eredu_checkpoint::recipe::RecipeDtype;
    /// Reports which physical encoding a selected checkpoint dtype uses on this backend.
    fn prepared_parameter_dtype(
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
    ) -> Result<eredu_checkpoint::recipe::RecipeDtype, ParameterError>;
    /// Reports support for one exact layout with its declared companions.
    fn effective_parameter_supported(
        layout: &EffectiveParameterLayout,
        shape: &[u64],
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
    ) -> bool;
    /// Reports the input arithmetic for the selected effective projection.
    fn parameter_input_transform(
        layout: &EffectiveParameterLayout,
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
    ) -> ProjectionInputTransform;

    /// Translates a rejected portable selection.
    fn parameter_access_error(error: ParameterError) -> B::Error;
    /// Marks native execution unusable after its communicator becomes terminal.
    fn fence_parameter_session(&self);
    /// Prices the local tensor loans, conversion and device work for one read.
    fn parameter_read_cost(
        layout: &EffectiveParameterLayout,
        fragments: &[ParameterReadFragment],
    ) -> Result<CaptureUsage, ParameterError>;
    /// Performs the admitted local read or contraction and returns F32 words.
    fn read_parameter_fragments(
        selected: &BTreeMap<String, Self::ParameterValue>,
        layout: &EffectiveParameterLayout,
        parameter: &str,
        fragments: &[ParameterReadFragment],
        words: usize,
        context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<Vec<u32>, B::Error>;
    /// Prices a bounded native read or contraction, excluding result serialization.
    fn local_read_cost(
        layout: &EffectiveParameterLayout,
        source: u64,
        selected: u64,
        output: u64,
        directions: Option<u64>,
    ) -> Result<CaptureUsage, ParameterError>;
    /// Evaluates an admitted region or contraction on native tensors.
    fn read_parameter_values(
        selected: &BTreeMap<String, Self::ParameterValue>,
        layout: &EffectiveParameterLayout,
        parameter: &str,
        region: &ParameterRegion,
        projection: Option<&ParameterProjection>,
        context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<Vec<f32>, B::Error>;
    /// Prices native overlay storage and the additional destination per edit.
    fn local_overlay_cost(
        layout: &EffectiveParameterLayout,
        elements: u64,
    ) -> Result<(CaptureUsage, u64), ParameterError>;
    /// Applies selected edits to the admitted tensor and returns its original and replacement.
    fn prepare_replacement<'a>(
        selected: &BTreeMap<String, Self::ParameterValue>,
        layout: &EffectiveParameterLayout,
        parameter: &str,
        edits: impl Iterator<Item = (&'a ParameterRegion, &'a ParameterUpdate)>,
        context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<Result<(Self::ParameterValue, Self::ParameterValue), ParameterError>, B::Error>;
}
