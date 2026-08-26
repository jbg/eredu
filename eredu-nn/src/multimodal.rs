//! Backend-neutral patch projection and multi-axis position operations.

use crate::{Error, Tensor};

/// One ordered token/embedding segment at decoder ingress.
#[derive(Debug, Clone, Copy)]
pub struct OrderedInputPart<'a, T> {
    /// Token identities shaped `[batch, sequence]`.
    pub token_ids: &'a T,
    /// Corresponding embeddings shaped `[batch, sequence, hidden]`.
    pub embeddings: &'a T,
}

/// Fully assembled decoder ingress tensors.
#[derive(Debug, Clone)]
pub struct OrderedModelInput<T> {
    /// Concatenated token identities.
    pub token_ids: T,
    /// Concatenated token and media embeddings.
    pub embeddings: T,
}

/// Validates and concatenates ordered text and media segments.
pub fn assemble_ordered_inputs<T: Tensor>(
    parts: &[OrderedInputPart<'_, T>],
    hidden_size: i32,
    context: &T::Context,
) -> Result<OrderedModelInput<T>, Error> {
    if parts.is_empty() || hidden_size <= 0 {
        return Err(Error::backend(
            "ordered input assembly requires parts and a positive hidden size",
        ));
    }
    let batch = parts[0].token_ids.shape().first().copied().unwrap_or(0);
    for (index, part) in parts.iter().enumerate() {
        let tokens = part.token_ids.shape();
        let embeddings = part.embeddings.shape();
        if tokens.len() != 2
            || embeddings.len() != 3
            || tokens[0] != batch
            || embeddings[0] != batch
            || tokens[1] != embeddings[1]
            || embeddings[2] != hidden_size
        {
            return Err(Error::backend(format!(
                "ordered input part {index} has incompatible token/embedding shapes {tokens:?} and {embeddings:?}; expected batch {batch} and hidden {hidden_size}"
            )));
        }
    }
    let token_ids = T::concatenate(
        &parts
            .iter()
            .map(|part| part.token_ids.clone())
            .collect::<Vec<_>>(),
        1,
        context,
    )?;
    let embeddings = T::concatenate(
        &parts
            .iter()
            .map(|part| part.embeddings.clone())
            .collect::<Vec<_>>(),
        1,
        context,
    )?;
    Ok(OrderedModelInput {
        token_ids,
        embeddings,
    })
}

/// Selected-vocabulary output projection request.
#[derive(Debug, Clone, Copy)]
pub struct MaskedOutputProjectionInput<'a, T> {
    /// Decoder states shaped `[batch, sequence, hidden]`.
    pub hidden: &'a T,
    /// Output embedding rows shaped `[vocabulary, hidden]`.
    pub output_weight: &'a T,
    /// Per-centroid scores shaped `[batch, sequence, centroids]`.
    pub centroid_logits: &'a T,
    /// Canonical token IDs grouped contiguously by centroid.
    pub token_ordering: &'a T,
    /// Selected centroid count.
    pub top_centroids: i32,
    /// Amount subtracted from the smallest selected logit for masked rows.
    pub mask_margin: f32,
}

impl<T: Tensor> MaskedOutputProjectionInput<'_, T> {
    /// Validates all exact, non-broadcast geometry.
    pub fn validate(&self) -> Result<(), Error> {
        let hidden = self.hidden.shape();
        let weight = self.output_weight.shape();
        let centroids = self.centroid_logits.shape();
        let ordering = self.token_ordering.shape();
        if hidden.len() != 3
            || weight.len() != 2
            || centroids.len() != 3
            || ordering.len() != 1
            || hidden[0] != centroids[0]
            || hidden[1] != centroids[1]
            || hidden[2] != weight[1]
            || ordering[0] != weight[0]
            || centroids[2] <= 0
            || weight[0] % centroids[2] != 0
            || self.top_centroids <= 0
            || self.top_centroids > centroids[2]
            || !self.mask_margin.is_finite()
            || self.mask_margin <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid masked-output geometry: hidden={hidden:?} weight={weight:?} centroids={centroids:?} ordering={ordering:?} top={} margin={}",
                self.top_centroids, self.mask_margin
            )));
        }
        Ok(())
    }
}

/// Projects only vocabulary rows selected by the highest-scoring centroids.
pub fn masked_output_projection<T: Tensor>(
    input: MaskedOutputProjectionInput<'_, T>,
    context: &T::Context,
) -> Result<T, Error> {
    input.validate()?;
    T::masked_output_projection(input, context)
}

/// Logical geometry of one flattened image or video patch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct FlattenedPatchSpec {
    /// Input channels.
    pub channels: i32,
    /// Temporal samples contained in one patch.
    pub temporal: i32,
    /// Patch height.
    pub height: i32,
    /// Patch width.
    pub width: i32,
    /// Projected feature width.
    pub output: i32,
}

impl FlattenedPatchSpec {
    /// Returns the flattened input width after validating the geometry.
    pub fn input_width(self) -> Result<i32, Error> {
        if self.channels <= 0
            || self.temporal <= 0
            || self.height <= 0
            || self.width <= 0
            || self.output <= 0
        {
            return Err(Error::backend(format!(
                "flattened patch dimensions must be positive, got {self:?}"
            )));
        }
        self.channels
            .checked_mul(self.temporal)
            .and_then(|value| value.checked_mul(self.height))
            .and_then(|value| value.checked_mul(self.width))
            .ok_or_else(|| Error::backend("flattened patch dimensions overflowed i32"))
    }
}

/// Projects a prepacked patch matrix through a dense or convolution-shaped kernel.
///
/// `input` is shaped `[patches, flattened_patch]`. `weight` may be the canonical
/// matrix `[output, flattened_patch]` or retain any checkpoint convolution axes
/// whose product is `output * flattened_patch`.
pub fn project_flattened_patches<T: Tensor>(
    input: &T,
    weight: &T,
    bias: Option<&T>,
    spec: FlattenedPatchSpec,
    context: &T::Context,
) -> Result<T, Error> {
    let input_width = spec.input_width()?;
    if input.shape().len() != 2 || input.shape()[1] != input_width {
        return Err(Error::backend(format!(
            "flattened patch input must have shape [patches, {input_width}], got {:?}",
            input.shape()
        )));
    }
    let expected_elements = spec
        .output
        .checked_mul(input_width)
        .ok_or_else(|| Error::backend("flattened patch weight dimensions overflowed i32"))?;
    let actual_elements = checked_elements(weight.shape())?;
    if weight.shape()[0] != spec.output || actual_elements != expected_elements {
        return Err(Error::backend(format!(
            "flattened patch weight must contain {expected_elements} values, got shape {:?}",
            weight.shape()
        )));
    }
    if let Some(bias) = bias {
        if bias.shape() != [spec.output] {
            return Err(Error::backend(format!(
                "flattened patch bias must have shape [{}], got {:?}",
                spec.output,
                bias.shape()
            )));
        }
    }
    let weight = weight.reshape(&[spec.output, input_width], context)?;
    T::linear(input, &weight, bias, context)
}

fn checked_elements(shape: &[i32]) -> Result<i32, Error> {
    if shape.is_empty() || shape.iter().any(|dimension| *dimension <= 0) {
        return Err(Error::backend(format!(
            "tensor dimensions must be positive, got {shape:?}"
        )));
    }
    shape.iter().try_fold(1_i32, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| Error::backend("tensor dimensions overflowed i32"))
    })
}

/// Canonical NHWC two-dimensional convolution used by patch subsampling.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PatchConvolution2dSpec {
    /// Kernel stride `(height, width)`.
    pub stride: (i32, i32),
    /// Symmetric input padding `(height, width)`.
    pub padding: (i32, i32),
    /// Kernel dilation `(height, width)`.
    pub dilation: (i32, i32),
    /// Convolution groups.
    pub groups: i32,
}

impl PatchConvolution2dSpec {
    fn validate(self) -> Result<(), Error> {
        if self.stride.0 <= 0
            || self.stride.1 <= 0
            || self.padding.0 < 0
            || self.padding.1 < 0
            || self.dilation.0 <= 0
            || self.dilation.1 <= 0
            || self.groups <= 0
        {
            return Err(Error::backend(format!(
                "invalid patch convolution geometry {self:?}"
            )));
        }
        Ok(())
    }
}

/// Arrangement of axis-specific frequencies in the final rotary feature axis.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MultiAxisRotaryLayout {
    /// Each axis occupies an independent rotary subspace: `x,x,y,y`.
    IndependentAxes,
    /// Axis frequencies form one half which is then repeated: `x,y,x,y`.
    SplitHalves,
    /// Global frequencies select axes round-robin while each axis has an
    /// explicit section width; exhausted secondary sections fall back to the
    /// first axis, and the completed half is repeated.
    RoundRobinSections,
}

/// One explicit position axis in a multi-axis rotary layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RotaryAxisSpec {
    /// Feature dimensions assigned to this axis.
    pub dimensions: i32,
    /// Constant added to caller-provided positions before clamping.
    pub position_offset: i32,
}

/// Explicit spatial/temporal rotary embedding policy.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiAxisRotarySpec {
    /// Axis policies in the same order as the last position-ID dimension.
    pub axes: Vec<RotaryAxisSpec>,
    /// Base wavelength.
    pub base: f32,
    /// Smallest position used for frequency construction. Padding values below
    /// this boundary map to the boundary position.
    pub minimum_position: i32,
    /// Final-axis frequency arrangement.
    pub layout: MultiAxisRotaryLayout,
}

impl MultiAxisRotarySpec {
    /// Validates the policy and returns the total rotated width.
    pub fn dimensions(&self) -> Result<i32, Error> {
        if self.axes.is_empty() || !self.base.is_finite() || self.base <= 0.0 {
            return Err(Error::backend(format!(
                "multi-axis rotary requires axes and a finite positive base, got {self:?}"
            )));
        }
        self.axes.iter().try_fold(0_i32, |total, axis| {
            if axis.dimensions <= 0 || axis.dimensions % 2 != 0 {
                return Err(Error::backend(format!(
                    "rotary axis dimensions must be positive and even, got {}",
                    axis.dimensions
                )));
            }
            total
                .checked_add(axis.dimensions)
                .ok_or_else(|| Error::backend("multi-axis rotary dimensions overflowed i32"))
        })
    }
}

/// Builds backend-native cosine and sine tensors for explicit multi-axis positions.
pub fn multi_axis_rotary_embeddings<T: Tensor>(
    position_ids: &T,
    spec: &MultiAxisRotarySpec,
    context: &T::Context,
) -> Result<(T, T), Error> {
    let _ = spec.dimensions()?;
    let shape = position_ids.shape();
    if shape.len() < 2 || shape.last().copied() != Some(spec.axes.len() as i32) {
        return Err(Error::backend(format!(
            "multi-axis position IDs must end in {} axes, got {shape:?}",
            spec.axes.len()
        )));
    }
    T::multi_axis_rotary_embeddings(position_ids, spec, context)
}

/// Scalar reference for flattened patch projection.
pub fn reference_flattened_patch_projection(
    input: &[f32],
    patches: usize,
    weight: &[f32],
    output: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, Error> {
    if patches == 0 || output == 0 || !input.len().is_multiple_of(patches) {
        return Err(Error::backend("invalid scalar patch projection geometry"));
    }
    let input_width = input.len() / patches;
    if input_width == 0 || weight.len() != output * input_width {
        return Err(Error::backend("scalar patch projection weight mismatch"));
    }
    if bias.is_some_and(|bias| bias.len() != output) {
        return Err(Error::backend("scalar patch projection bias mismatch"));
    }
    let mut result = vec![0.0; patches * output];
    for patch in 0..patches {
        for feature in 0..output {
            let mut value = bias.map_or(0.0, |bias| bias[feature]);
            for input_feature in 0..input_width {
                value += input[patch * input_width + input_feature]
                    * weight[feature * input_width + input_feature];
            }
            result[patch * output + feature] = value;
        }
    }
    Ok(result)
}

/// Scalar batch-one reference for ordered token and embedding assembly.
pub fn reference_ordered_input_assembly(
    parts: &[(&[u32], &[f32])],
    hidden_size: usize,
) -> Result<(Vec<u32>, Vec<f32>), Error> {
    if parts.is_empty() || hidden_size == 0 {
        return Err(Error::backend(
            "scalar ordered assembly requires parts and hidden size",
        ));
    }
    let mut tokens = Vec::new();
    let mut embeddings = Vec::new();
    for (index, (part_tokens, part_embeddings)) in parts.iter().enumerate() {
        if part_tokens.is_empty() || part_embeddings.len() != part_tokens.len() * hidden_size {
            return Err(Error::backend(format!(
                "scalar ordered assembly part {index} has mismatched token and embedding lengths"
            )));
        }
        tokens.extend_from_slice(part_tokens);
        embeddings.extend_from_slice(part_embeddings);
    }
    Ok((tokens, embeddings))
}

/// Scalar reference for canonical NHWC two-dimensional convolution.
pub fn reference_patch_convolution_2d(
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 4],
    spec: PatchConvolution2dSpec,
) -> Result<(Vec<f32>, [usize; 4]), Error> {
    spec.validate()?;
    let [batch, input_height, input_width, input_channels] = input_shape;
    let [output_channels, kernel_height, kernel_width, weight_channels] = weight_shape;
    let groups =
        usize::try_from(spec.groups).map_err(|_| Error::backend("negative convolution groups"))?;
    if input.len() != input_shape.into_iter().product::<usize>()
        || weight.len() != weight_shape.into_iter().product::<usize>()
        || input_channels % groups != 0
        || output_channels % groups != 0
        || weight_channels * groups != input_channels
    {
        return Err(Error::backend(
            "scalar patch convolution tensor geometry mismatch",
        ));
    }
    let output_axis = |input: usize, kernel: usize, stride: i32, padding: i32, dilation: i32| {
        let effective = dilation as i64 * (kernel.saturating_sub(1)) as i64 + 1;
        let numerator = input as i64 + 2 * padding as i64 - effective;
        (numerator >= 0).then_some((numerator / stride as i64 + 1) as usize)
    };
    let output_height = output_axis(
        input_height,
        kernel_height,
        spec.stride.0,
        spec.padding.0,
        spec.dilation.0,
    )
    .ok_or_else(|| Error::backend("convolution kernel does not overlap input height"))?;
    let output_width = output_axis(
        input_width,
        kernel_width,
        spec.stride.1,
        spec.padding.1,
        spec.dilation.1,
    )
    .ok_or_else(|| Error::backend("convolution kernel does not overlap input width"))?;
    let mut output = vec![0.0; batch * output_height * output_width * output_channels];
    let outputs_per_group = output_channels / groups;
    for n in 0..batch {
        for oy in 0..output_height {
            for ox in 0..output_width {
                for oc in 0..output_channels {
                    let group = oc / outputs_per_group;
                    let mut value = 0.0;
                    for ky in 0..kernel_height {
                        let iy = oy as i64 * spec.stride.0 as i64
                            + ky as i64 * spec.dilation.0 as i64
                            - spec.padding.0 as i64;
                        if !(0..input_height as i64).contains(&iy) {
                            continue;
                        }
                        for kx in 0..kernel_width {
                            let ix = ox as i64 * spec.stride.1 as i64
                                + kx as i64 * spec.dilation.1 as i64
                                - spec.padding.1 as i64;
                            if !(0..input_width as i64).contains(&ix) {
                                continue;
                            }
                            for wc in 0..weight_channels {
                                let ic = group * weight_channels + wc;
                                let input_index = ((n * input_height + iy as usize) * input_width
                                    + ix as usize)
                                    * input_channels
                                    + ic;
                                let weight_index = ((oc * kernel_height + ky) * kernel_width + kx)
                                    * weight_channels
                                    + wc;
                                value += input[input_index] * weight[weight_index];
                            }
                        }
                    }
                    let output_index =
                        ((n * output_height + oy) * output_width + ox) * output_channels + oc;
                    output[output_index] = value;
                }
            }
        }
    }
    Ok((
        output,
        [batch, output_height, output_width, output_channels],
    ))
}

/// Scalar reference for centroid-selected vocabulary projection.
#[allow(clippy::too_many_arguments)] // Reference oracle mirrors the primitive's full contract.
pub fn reference_masked_output_projection(
    hidden: &[f32],
    rows: usize,
    hidden_size: usize,
    output_weight: &[f32],
    vocabulary: usize,
    centroid_logits: &[f32],
    centroids: usize,
    token_ordering: &[usize],
    top_centroids: usize,
    mask_margin: f32,
) -> Result<Vec<f32>, Error> {
    if rows == 0
        || hidden_size == 0
        || centroids == 0
        || !vocabulary.is_multiple_of(centroids)
        || top_centroids == 0
        || top_centroids > centroids
        || hidden.len() != rows * hidden_size
        || output_weight.len() != vocabulary * hidden_size
        || centroid_logits.len() != rows * centroids
        || token_ordering.len() != vocabulary
        || !mask_margin.is_finite()
        || mask_margin <= 0.0
    {
        return Err(Error::backend("invalid scalar masked-output geometry"));
    }
    let per_centroid = vocabulary / centroids;
    let mut output = Vec::with_capacity(rows * vocabulary);
    for row in 0..rows {
        let mut ranked = (0..centroids).collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            centroid_logits[row * centroids + *right]
                .total_cmp(&centroid_logits[row * centroids + *left])
                .then_with(|| left.cmp(right))
        });
        let mut selected = Vec::with_capacity(top_centroids * per_centroid);
        for centroid in ranked.into_iter().take(top_centroids) {
            for slot in 0..per_centroid {
                let token = token_ordering[centroid * per_centroid + slot];
                if token >= vocabulary || selected.iter().any(|(existing, _)| *existing == token) {
                    return Err(Error::backend(
                        "scalar masked-output ordering is not a vocabulary permutation",
                    ));
                }
                let logit = (0..hidden_size)
                    .map(|feature| {
                        hidden[row * hidden_size + feature]
                            * output_weight[token * hidden_size + feature]
                    })
                    .sum::<f32>();
                selected.push((token, logit));
            }
        }
        let masked = selected
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f32::INFINITY, f32::min)
            - mask_margin;
        let start = output.len();
        output.resize(start + vocabulary, masked);
        for (token, logit) in selected {
            output[start + token] = logit;
        }
    }
    Ok(output)
}

/// Scalar reference for explicit multi-axis rotary cosine and sine values.
pub fn reference_multi_axis_rotary_embeddings(
    positions: &[i32],
    rows: usize,
    spec: &MultiAxisRotarySpec,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let dimensions = usize::try_from(spec.dimensions()?)
        .map_err(|_| Error::backend("negative rotary dimensions"))?;
    let axis_count = spec.axes.len();
    if rows == 0 || positions.len() != rows * axis_count {
        return Err(Error::backend(
            "scalar multi-axis position geometry mismatch",
        ));
    }
    let mut cosine = Vec::with_capacity(rows * dimensions);
    let mut sine = Vec::with_capacity(rows * dimensions);
    for row in 0..rows {
        let mut axis_angles = Vec::with_capacity(axis_count);
        for (axis_index, axis) in spec.axes.iter().enumerate() {
            let position = positions[row * axis_count + axis_index]
                .saturating_add(axis.position_offset)
                .max(spec.minimum_position) as f32;
            let angles = (0..axis.dimensions)
                .step_by(2)
                .map(|index| position / spec.base.powf(index as f32 / axis.dimensions as f32))
                .collect::<Vec<_>>();
            axis_angles.push(angles);
        }
        let angles = match spec.layout {
            MultiAxisRotaryLayout::IndependentAxes => axis_angles
                .into_iter()
                .flat_map(|axis| axis.clone().into_iter().chain(axis))
                .collect::<Vec<_>>(),
            MultiAxisRotaryLayout::SplitHalves => {
                let half = axis_angles.into_iter().flatten().collect::<Vec<_>>();
                half.clone().into_iter().chain(half).collect()
            }
            MultiAxisRotaryLayout::RoundRobinSections => {
                let half_width = dimensions / 2;
                let mut half = Vec::with_capacity(half_width);
                for frequency in 0..half_width {
                    let candidate = frequency % axis_count;
                    let section = usize::try_from(spec.axes[candidate].dimensions / 2)
                        .map_err(|_| Error::backend("negative rotary section"))?;
                    let axis = if candidate != 0 && frequency < section * axis_count {
                        candidate
                    } else {
                        0
                    };
                    let position = positions[row * axis_count + axis]
                        .saturating_add(spec.axes[axis].position_offset)
                        .max(spec.minimum_position) as f32;
                    half.push(
                        position / spec.base.powf(2.0 * frequency as f32 / dimensions as f32),
                    );
                }
                half.clone().into_iter().chain(half).collect()
            }
        };
        cosine.extend(angles.iter().map(|angle| angle.cos()));
        sine.extend(angles.iter().map(|angle| angle.sin()));
    }
    Ok((cosine, sine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattened_patch_reference_projects_each_row() {
        let output = reference_flattened_patch_projection(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            &[1.0, 0.5, -1.0, 2.0],
            2,
            Some(&[0.25, -0.5]),
        )
        .unwrap();
        assert_eq!(output, vec![2.25, 2.5, 5.25, 4.5]);
    }

    #[test]
    fn ordered_assembly_preserves_part_and_token_order() {
        let (tokens, embeddings) = reference_ordered_input_assembly(
            &[
                (&[7, 8], &[1.0, 2.0, 3.0, 4.0]),
                (&[99], &[5.0, 6.0]),
                (&[9], &[7.0, 8.0]),
            ],
            2,
        )
        .unwrap();
        assert_eq!(tokens, [7, 8, 99, 9]);
        assert_eq!(embeddings, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(reference_ordered_input_assembly(&[(&[1], &[1.0])], 2).is_err());
    }

    #[test]
    fn convolution_reference_obeys_nhwc_stride_and_padding() {
        let (output, shape) = reference_patch_convolution_2d(
            &[1.0, 2.0, 3.0, 4.0],
            [1, 2, 2, 1],
            &[1.0, 1.0, 1.0, 1.0],
            [1, 2, 2, 1],
            PatchConvolution2dSpec {
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
                groups: 1,
            },
        )
        .unwrap();
        assert_eq!(shape, [1, 3, 3, 1]);
        assert_eq!(output, vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
    }

    #[test]
    fn masked_output_reference_selects_centroids_and_scatter_rows() {
        let output = reference_masked_output_projection(
            &[2.0, 1.0],
            1,
            2,
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
            4,
            &[0.1, 0.9],
            2,
            &[2, 0, 3, 1],
            1,
            1.0,
        )
        .unwrap();
        assert_eq!(output, vec![-2.0, 1.0, -2.0, -1.0]);
    }

    #[test]
    fn multi_axis_layouts_preserve_distinct_rotation_pairing() {
        let mut spec = MultiAxisRotarySpec {
            axes: vec![
                RotaryAxisSpec {
                    dimensions: 4,
                    position_offset: 0,
                },
                RotaryAxisSpec {
                    dimensions: 4,
                    position_offset: 1,
                },
            ],
            base: 100.0,
            minimum_position: 0,
            layout: MultiAxisRotaryLayout::IndependentAxes,
        };
        let (_, independent) = reference_multi_axis_rotary_embeddings(&[-1, 2], 1, &spec).unwrap();
        spec.layout = MultiAxisRotaryLayout::SplitHalves;
        let (_, split) = reference_multi_axis_rotary_embeddings(&[-1, 2], 1, &spec).unwrap();
        assert_eq!(independent.len(), 8);
        assert_eq!(split.len(), 8);
        assert_eq!(independent[0], 0.0);
        assert_eq!(independent[1], 0.0);
        assert_eq!(independent[2], 0.0);
        assert_eq!(independent[3], 0.0);
        assert_eq!(split[0], 0.0);
        assert_eq!(split[1], 0.0);
        assert_eq!(split[4], 0.0);
        assert_eq!(split[5], 0.0);
        assert_eq!(independent[4], split[2]);
        assert_eq!(independent[6], split[6]);
        assert_ne!(independent, split);
    }

    #[test]
    fn multi_axis_rejects_odd_width_and_bad_position_shape() {
        let spec = MultiAxisRotarySpec {
            axes: vec![RotaryAxisSpec {
                dimensions: 3,
                position_offset: 0,
            }],
            base: 10_000.0,
            minimum_position: 0,
            layout: MultiAxisRotaryLayout::SplitHalves,
        };
        assert!(spec.dimensions().is_err());
        assert!(reference_multi_axis_rotary_embeddings(&[], 1, &spec).is_err());
    }
}
