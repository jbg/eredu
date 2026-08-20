use std::cell::Cell;

use eredu_architectures::qwen;
use eredu_nn::{
    AttentionCache, AttentionMask, EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator,
    LinearSpec, NeuralBackend, NormalizationOperator, NormalizationSpec, PadMode,
    ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized,
    RotaryOperator, RotaryPosition, RotarySpec, RoutedNeuralBackend, RoutingOperator,
    RoutingResult, SwiGluExpertBankOperator, SwiGluExpertBankSpec, SwiGluExpertLayout, Tensor,
    TopKRouterSpec, TopKRoutingSpec,
};

#[derive(Debug, Clone)]
struct NumericTensor {
    shape: Vec<i32>,
    data: Vec<f32>,
}

impl NumericTensor {
    fn new(shape: impl Into<Vec<i32>>, data: Vec<f32>) -> Self {
        let shape = shape.into();
        assert_eq!(elements(&shape), data.len());
        Self { shape, data }
    }

    fn zeros(shape: impl Into<Vec<i32>>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![0.0; elements(&shape)],
            shape,
        }
    }

    fn token_ids(ids: &[usize]) -> Self {
        Self::new(
            vec![1, i32::try_from(ids.len()).unwrap()],
            ids.iter().map(|id| *id as f32).collect(),
        )
    }

    fn axis_slice(&self, axis: usize, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.shape[axis] as usize);
        let mut shape = self.shape.clone();
        shape[axis] = (end - start) as i32;
        let mut output = Self::zeros(shape);
        for output_index in 0..output.data.len() {
            let mut coordinate = unravel(output_index, &output.shape);
            coordinate[axis] += start;
            output.data[output_index] = self.data[offset(&coordinate, &self.shape)];
        }
        output
    }

    fn map(&self, operation: impl Fn(f32) -> f32) -> Self {
        Self::new(
            self.shape.clone(),
            self.data.iter().copied().map(operation).collect(),
        )
    }

    fn zip(&self, rhs: &Self, operation: impl Fn(f32, f32) -> f32) -> Result<Self, Error> {
        if self.shape != rhs.shape {
            return Err(Error::backend(format!(
                "numeric tensor shape mismatch: {:?} versus {:?}",
                self.shape, rhs.shape
            )));
        }
        Ok(Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .copied()
                .zip(rhs.data.iter().copied())
                .map(|(left, right)| operation(left, right))
                .collect(),
        ))
    }
}

fn elements(shape: &[i32]) -> usize {
    shape
        .iter()
        .map(|dimension| usize::try_from(*dimension).unwrap())
        .product()
}

fn strides(shape: &[i32]) -> Vec<usize> {
    let mut stride = 1;
    let mut result = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        result[axis] = stride;
        stride *= usize::try_from(*dimension).unwrap();
    }
    result
}

fn unravel(mut index: usize, shape: &[i32]) -> Vec<usize> {
    let tensor_strides = strides(shape);
    tensor_strides
        .iter()
        .map(|stride| {
            let coordinate = index / stride;
            index %= stride;
            coordinate
        })
        .collect()
}

fn offset(coordinate: &[usize], shape: &[i32]) -> usize {
    coordinate
        .iter()
        .zip(strides(shape))
        .map(|(coordinate, stride)| coordinate * stride)
        .sum()
}

fn axis(axis: i32, rank: usize, insertion: bool) -> Result<usize, Error> {
    let limit = if insertion { rank + 1 } else { rank };
    let normalized = if axis < 0 {
        i32::try_from(limit).unwrap() + axis
    } else {
        axis
    };
    let normalized = usize::try_from(normalized).map_err(Error::backend)?;
    if normalized >= limit {
        return Err(Error::backend(format!(
            "axis {axis} is outside rank {rank}"
        )));
    }
    Ok(normalized)
}

fn unsupported<T>(operation: &str) -> Result<T, Error> {
    Err(Error::backend(format!(
        "numeric Qwen reference does not use {operation}"
    )))
}

impl Tensor for NumericTensor {
    type Context = NumericContext;

    fn shape(&self) -> &[i32] {
        &self.shape
    }

    fn unloaded_f32(shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::zeros(shape.to_vec()))
    }

    fn from_f32_slice(values: &[f32], shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(shape.to_vec(), values.to_vec()))
    }

    fn add(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left + right)
    }

    fn subtract(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left - right)
    }

    fn multiply(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left * right)
    }

    fn multiply_scalar(&self, rhs: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value * rhs))
    }

    fn divide(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left / right)
    }

    fn square(&self, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value * value))
    }

    fn maximum_scalar(&self, rhs: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value.max(rhs)))
    }

    fn reshape(&self, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        let mut shape = shape.to_vec();
        let inferred = shape.iter().position(|dimension| *dimension == -1);
        if let Some(inferred) = inferred {
            if shape.iter().filter(|dimension| **dimension == -1).count() != 1 {
                return Err(Error::backend("numeric reshape has multiple inferred axes"));
            }
            let known = shape
                .iter()
                .filter(|dimension| **dimension != -1)
                .map(|dimension| usize::try_from(*dimension).unwrap())
                .product::<usize>();
            shape[inferred] = i32::try_from(self.data.len() / known).map_err(Error::backend)?;
        }
        if elements(&shape) != self.data.len() {
            return Err(Error::backend("numeric reshape changes element count"));
        }
        Ok(Self::new(shape, self.data.clone()))
    }

    fn transpose_axes(&self, axes: &[i32], _: &NumericContext) -> Result<Self, Error> {
        if axes.len() != self.shape.len() {
            return Err(Error::backend("numeric transpose rank mismatch"));
        }
        let axes = axes
            .iter()
            .map(|selected| axis(*selected, self.shape.len(), false))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = vec![false; axes.len()];
        for selected in &axes {
            if std::mem::replace(&mut seen[*selected], true) {
                return Err(Error::backend("numeric transpose repeats an axis"));
            }
        }
        let shape = axes
            .iter()
            .map(|selected| self.shape[*selected])
            .collect::<Vec<_>>();
        let mut output = Self::zeros(shape);
        for output_index in 0..output.data.len() {
            let output_coordinate = unravel(output_index, &output.shape);
            let mut input_coordinate = vec![0; axes.len()];
            for (output_axis, input_axis) in axes.iter().enumerate() {
                input_coordinate[*input_axis] = output_coordinate[output_axis];
            }
            output.data[output_index] = self.data[offset(&input_coordinate, &self.shape)];
        }
        Ok(output)
    }

    fn swap_axes(&self, left: i32, right: i32, context: &NumericContext) -> Result<Self, Error> {
        let left = axis(left, self.shape.len(), false)?;
        let right = axis(right, self.shape.len(), false)?;
        let mut axes = (0..self.shape.len())
            .map(|axis| axis as i32)
            .collect::<Vec<_>>();
        axes.swap(left, right);
        self.transpose_axes(&axes, context)
    }

    fn transpose(&self, context: &NumericContext) -> Result<Self, Error> {
        if self.shape.len() != 2 {
            return Err(Error::backend("numeric transpose requires rank two"));
        }
        self.transpose_axes(&[1, 0], context)
    }

    fn expand_dims(&self, selected: i32, _: &NumericContext) -> Result<Self, Error> {
        let selected = axis(selected, self.shape.len(), true)?;
        let mut shape = self.shape.clone();
        shape.insert(selected, 1);
        Ok(Self::new(shape, self.data.clone()))
    }

    fn squeeze_axes(&self, axes: &[i32], _: &NumericContext) -> Result<Self, Error> {
        let mut axes = axes
            .iter()
            .map(|selected| axis(*selected, self.shape.len(), false))
            .collect::<Result<Vec<_>, _>>()?;
        axes.sort_unstable();
        axes.dedup();
        let mut shape = self.shape.clone();
        for selected in axes.into_iter().rev() {
            if shape[selected] != 1 {
                return Err(Error::backend("numeric squeeze selected a non-unit axis"));
            }
            shape.remove(selected);
        }
        Ok(Self::new(shape, self.data.clone()))
    }

    fn index(&self, _: &[Index], _: &NumericContext) -> Result<Self, Error> {
        unsupported("index")
    }

    fn take_axis(&self, _: &Self, _: i32, _: &NumericContext) -> Result<Self, Error> {
        unsupported("take_axis")
    }

    fn concatenate(values: &[Self], selected: i32, _: &NumericContext) -> Result<Self, Error> {
        let first = values
            .first()
            .ok_or_else(|| Error::backend("numeric concatenate requires inputs"))?;
        let selected = axis(selected, first.shape.len(), false)?;
        let mut shape = first.shape.clone();
        shape[selected] = 0;
        for value in values {
            if value.shape.len() != shape.len()
                || value
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(current, dimension)| current != selected && *dimension != shape[current])
            {
                return Err(Error::backend("numeric concatenate shape mismatch"));
            }
            shape[selected] += value.shape[selected];
        }
        let mut output = Self::zeros(shape);
        let mut axis_base = 0;
        for value in values {
            for input_index in 0..value.data.len() {
                let input_coordinate = unravel(input_index, &value.shape);
                let mut output_coordinate = input_coordinate.clone();
                output_coordinate[selected] += axis_base;
                let output_index = offset(&output_coordinate, &output.shape);
                output.data[output_index] = value.data[input_index];
            }
            axis_base += value.shape[selected] as usize;
        }
        Ok(output)
    }

    fn stack(values: &[Self], selected: i32, context: &NumericContext) -> Result<Self, Error> {
        let first = values
            .first()
            .ok_or_else(|| Error::backend("numeric stack requires inputs"))?;
        let selected = axis(selected, first.shape.len(), true)?;
        let expanded = values
            .iter()
            .map(|value| value.expand_dims(selected as i32, context))
            .collect::<Result<Vec<_>, _>>()?;
        Self::concatenate(&expanded, selected as i32, context)
    }

    fn matmul(lhs: &Self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
            return Err(Error::backend(
                "numeric matmul requires compatible matrices",
            ));
        }
        let rows = lhs.shape[0] as usize;
        let inner_width = lhs.shape[1] as usize;
        let columns = rhs.shape[1] as usize;
        let mut output = Self::zeros(vec![rows as i32, columns as i32]);
        for row in 0..rows {
            for column in 0..columns {
                output.data[row * columns + column] = (0..inner_width)
                    .map(|inner_index| {
                        lhs.data[row * inner_width + inner_index]
                            * rhs.data[inner_index * columns + column]
                    })
                    .sum();
            }
        }
        Ok(output)
    }

    fn sum_axis(
        value: &Self,
        selected: i32,
        keep_dims: bool,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        let selected = axis(selected, value.shape.len(), false)?;
        let mut shape = value.shape.clone();
        if keep_dims {
            shape[selected] = 1;
        } else {
            shape.remove(selected);
        }
        let mut output = Self::zeros(shape);
        for input_index in 0..value.data.len() {
            let mut coordinate = unravel(input_index, &value.shape);
            if keep_dims {
                coordinate[selected] = 0;
            } else {
                coordinate.remove(selected);
            }
            let output_index = offset(&coordinate, &output.shape);
            output.data[output_index] += value.data[input_index];
        }
        Ok(output)
    }

    fn argmin_axis(_: &Self, _: i32, _: bool, _: &NumericContext) -> Result<Self, Error> {
        unsupported("argmin_axis")
    }

    fn pad(_: &Self, _: &[(i32, i32)], _: PadMode, _: &NumericContext) -> Result<Self, Error> {
        unsupported("pad")
    }

    fn conv1d(
        _: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        unsupported("conv1d")
    }

    fn conv_transpose1d(
        _: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        unsupported("conv_transpose1d")
    }

    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        linear(input, weight, bias)
    }

    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        let width = *input
            .shape
            .last()
            .ok_or_else(|| Error::backend("numeric layer norm requires rank"))?
            as usize;
        let mut output = input.clone();
        for (input_row, output_row) in input
            .data
            .chunks_exact(width)
            .zip(output.data.chunks_exact_mut(width))
        {
            let mean = input_row.iter().sum::<f32>() / width as f32;
            let variance = input_row
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f32>()
                / width as f32;
            for index in 0..width {
                let scale = weight.map_or(1.0, |weight| weight.data[index]);
                let shift = bias.map_or(0.0, |bias| bias.data[index]);
                output_row[index] =
                    (input_row[index] - mean) / (variance + epsilon).sqrt() * scale + shift;
            }
        }
        Ok(output)
    }

    fn gelu(input: &Self, _: &NumericContext) -> Result<Self, Error> {
        Ok(input.map(|value| {
            0.5 * value
                * (1.0
                    + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value.powi(3)))
                        .tanh())
        }))
    }

    fn elu(input: &Self, alpha: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(input.map(|value| {
            if value >= 0.0 {
                value
            } else {
                alpha * value.exp_m1()
            }
        }))
    }

    fn rope(
        input: &Self,
        dimensions: i32,
        traditional: bool,
        base: f32,
        _: f32,
        position_offset: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        rotary_offset(input, dimensions, traditional, base, position_offset)
    }

    fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask<'_, Self>,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        let mask = match mask {
            AttentionMask::Tensor(mask) => Some(mask),
            AttentionMask::None | AttentionMask::Causal => None,
        };
        attention(queries, keys, values, scale, mask, None, 0)
    }
}

fn deterministic_values(spec: &ParameterSpec, length: usize, norm: bool) -> Vec<f32> {
    let mut hash = 2_166_136_261_u32;
    for byte in spec.id.as_str().bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    (0..length)
        .map(|index| {
            let mixed = hash
                .wrapping_add((index as u32).wrapping_mul(747_796_405))
                .rotate_left((index % 23) as u32);
            let centered = (mixed % 2001) as f32 - 1000.0;
            if norm {
                1.0 + centered / 50_000.0
            } else {
                centered / 25_000.0
            }
        })
        .collect()
}

fn parameter(spec: &ParameterSpec, shape: Vec<i32>, norm: bool) -> NumericTensor {
    NumericTensor::new(
        shape.clone(),
        deterministic_values(spec, elements(&shape), norm),
    )
}

fn visit<'a, V>(metadata: &ParameterMetadata, value: &'a NumericTensor, visitor: &mut V)
where
    V: ParameterVisitor<'a, NumericTensor>,
{
    visitor.visit(metadata.clone(), value);
}

fn visit_mut<'a, V>(metadata: &ParameterMetadata, value: &'a mut NumericTensor, visitor: &mut V)
where
    V: ParameterVisitorMut<'a, NumericTensor>,
{
    visitor.visit_mut(metadata.clone(), value);
}

fn linear(
    input: &NumericTensor,
    weight: &NumericTensor,
    bias: Option<&NumericTensor>,
) -> Result<NumericTensor, Error> {
    if weight.shape.len() != 2
        || input.shape.last() != weight.shape.get(1)
        || bias.is_some_and(|bias| bias.shape != [weight.shape[0]])
    {
        return Err(Error::backend(format!(
            "numeric linear geometry mismatch: input={:?}, weight={:?}, bias={:?}",
            input.shape,
            weight.shape,
            bias.map(|bias| bias.shape.as_slice())
        )));
    }
    let input_width = weight.shape[1] as usize;
    let output_width = weight.shape[0] as usize;
    let rows = input.data.len() / input_width;
    let mut shape = input.shape.clone();
    *shape.last_mut().unwrap() = weight.shape[0];
    let mut output = NumericTensor::zeros(shape);
    for row in 0..rows {
        for output_column in 0..output_width {
            let mut value = bias.map_or(0.0, |bias| bias.data[output_column]);
            for input_column in 0..input_width {
                value += input.data[row * input_width + input_column]
                    * weight.data[output_column * input_width + input_column];
            }
            output.data[row * output_width + output_column] = value;
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct NumericLinear {
    weight: NumericTensor,
    weight_metadata: ParameterMetadata,
    bias: Option<(NumericTensor, ParameterMetadata)>,
}

impl Parameterized<NumericTensor> for NumericLinear {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.weight_metadata, &self.weight, visitor);
        if let Some((bias, metadata)) = &self.bias {
            visit(metadata, bias, visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.weight_metadata, &mut self.weight, visitor);
        if let Some((bias, metadata)) = &mut self.bias {
            visit_mut(metadata, bias, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.weight_metadata.trainable = trainable;
        if let Some((_, metadata)) = &mut self.bias {
            metadata.trainable = trainable;
        }
    }
}

impl LinearOperator<NumericTensor> for NumericLinear {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        linear(
            input,
            &self.weight,
            self.bias.as_ref().map(|(bias, _)| bias),
        )
    }
}

#[derive(Debug, Clone)]
struct NumericEmbedding {
    weight: NumericTensor,
    metadata: ParameterMetadata,
}

impl Parameterized<NumericTensor> for NumericEmbedding {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.metadata, &self.weight, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.metadata, &mut self.weight, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl EmbeddingOperator<NumericTensor> for NumericEmbedding {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let vocabulary = self.weight.shape[0] as usize;
        let dimensions = self.weight.shape[1] as usize;
        let mut shape = input.shape.clone();
        shape.push(dimensions as i32);
        let mut output = NumericTensor::zeros(shape);
        for (token_index, token) in input.data.iter().enumerate() {
            let token = *token as usize;
            if token >= vocabulary || token as f32 != input.data[token_index] {
                return Err(Error::backend("numeric embedding token is invalid"));
            }
            output.data[token_index * dimensions..(token_index + 1) * dimensions]
                .copy_from_slice(&self.weight.data[token * dimensions..(token + 1) * dimensions]);
        }
        Ok(output)
    }

    fn as_linear(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        linear(input, &self.weight, None)
    }
}

#[derive(Debug, Clone)]
struct NumericNorm {
    weight: NumericTensor,
    metadata: ParameterMetadata,
    epsilon: f32,
}

impl Parameterized<NumericTensor> for NumericNorm {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.metadata, &self.weight, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.metadata, &mut self.weight, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl NormalizationOperator<NumericTensor> for NumericNorm {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let dimensions = self.weight.data.len();
        if input.shape.last().copied() != Some(dimensions as i32) {
            return Err(Error::backend("numeric RMSNorm geometry mismatch"));
        }
        let mut output = input.clone();
        for (input_row, output_row) in input
            .data
            .chunks_exact(dimensions)
            .zip(output.data.chunks_exact_mut(dimensions))
        {
            let rms = (input_row.iter().map(|value| value * value).sum::<f32>()
                / dimensions as f32
                + self.epsilon)
                .sqrt();
            for index in 0..dimensions {
                output_row[index] = input_row[index] / rms * self.weight.data[index];
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct NumericRotary {
    dimensions: i32,
    traditional: bool,
    base: f32,
}

impl Parameterized<NumericTensor> for NumericRotary {
    fn visit_parameters<'a, V>(&'a self, _: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, _: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
    }

    fn set_trainable(&mut self, _: bool) {}
}

impl RotaryOperator<NumericTensor> for NumericRotary {
    fn forward(
        &mut self,
        input: &NumericTensor,
        position: RotaryPosition<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        match position {
            RotaryPosition::Offset(position_offset) => rotary_offset(
                input,
                self.dimensions,
                self.traditional,
                self.base,
                position_offset,
            ),
            RotaryPosition::Embeddings { cosine, sine } => {
                rotary_embeddings(input, self.dimensions, self.traditional, cosine, sine)
            }
        }
    }
}

fn rotary_offset(
    input: &NumericTensor,
    dimensions: i32,
    traditional: bool,
    base: f32,
    position_offset: i32,
) -> Result<NumericTensor, Error> {
    let sequence = input.shape.get(input.shape.len().wrapping_sub(2)).copied();
    let Some(sequence) = sequence else {
        return Err(Error::backend(
            "numeric rotary requires sequence and feature axes",
        ));
    };
    let half = dimensions as usize / 2;
    let mut cosine = NumericTensor::zeros(vec![sequence, half as i32]);
    let mut sine = cosine.clone();
    for position in 0..sequence as usize {
        for frequency in 0..half {
            let theta = (position_offset as f32 + position as f32)
                / base.powf(2.0 * frequency as f32 / dimensions as f32);
            cosine.data[position * half + frequency] = theta.cos();
            sine.data[position * half + frequency] = theta.sin();
        }
    }
    rotary_embeddings(input, dimensions, traditional, &cosine, &sine)
}

fn rotary_embeddings(
    input: &NumericTensor,
    dimensions: i32,
    traditional: bool,
    cosine: &NumericTensor,
    sine: &NumericTensor,
) -> Result<NumericTensor, Error> {
    if dimensions <= 0 || dimensions % 2 != 0 || input.shape.last().copied() != Some(dimensions) {
        return Err(Error::backend("numeric rotary geometry mismatch"));
    }
    let sequence = input.shape[input.shape.len() - 2] as usize;
    let half = dimensions as usize / 2;
    if cosine.shape != [sequence as i32, half as i32] || sine.shape != cosine.shape {
        return Err(Error::backend(
            "numeric explicit rotary embedding shape mismatch",
        ));
    }
    let dimensions = dimensions as usize;
    let rows = input.data.len() / dimensions;
    let mut output = input.clone();
    for row in 0..rows {
        let position = row % sequence;
        for frequency in 0..half {
            let (left, right) = if traditional {
                (2 * frequency, 2 * frequency + 1)
            } else {
                (frequency, frequency + half)
            };
            let cosine = cosine.data[position * half + frequency];
            let sine = sine.data[position * half + frequency];
            let left_value = input.data[row * dimensions + left];
            let right_value = input.data[row * dimensions + right];
            output.data[row * dimensions + left] = left_value * cosine - right_value * sine;
            output.data[row * dimensions + right] = right_value * cosine + left_value * sine;
        }
    }
    Ok(output)
}

#[derive(Default)]
struct NumericContext {
    sliding_attention_calls: Cell<usize>,
}

struct NumericBackend;

impl NeuralBackend for NumericBackend {
    type Tensor = NumericTensor;
    type Linear = NumericLinear;
    type Embedding = NumericEmbedding;
    type Normalization = NumericNorm;
    type Rotary = NumericRotary;
    type ParallelContext = ();

    fn linear(spec: LinearSpec, _: &NumericContext) -> Result<Self::Linear, Error> {
        let weight = parameter(&spec.weight, vec![spec.output, spec.input], false);
        let weight_metadata = ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable);
        let bias = spec.bias.map(|bias| {
            let value = parameter(&bias, vec![spec.output], false);
            let metadata = ParameterMetadata::from_spec(&bias, bias.trainable);
            (value, metadata)
        });
        Ok(NumericLinear {
            weight,
            weight_metadata,
            bias,
        })
    }

    fn embedding(spec: EmbeddingSpec, _: &NumericContext) -> Result<Self::Embedding, Error> {
        Ok(NumericEmbedding {
            weight: parameter(&spec.weight, vec![spec.vocabulary, spec.dimensions], false),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
        })
    }

    fn rms_norm(spec: NormalizationSpec, _: &NumericContext) -> Result<Self::Normalization, Error> {
        Ok(NumericNorm {
            weight: parameter(&spec.weight, vec![spec.dimensions], true),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            epsilon: spec.epsilon,
        })
    }

    fn rotary(spec: RotarySpec<'_>, _: &NumericContext) -> Result<Self::Rotary, Error> {
        Ok(NumericRotary {
            dimensions: spec.dimensions,
            traditional: spec.traditional,
            base: spec.base,
        })
    }

    fn silu(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(|value| value / (1.0 + (-value).exp())))
    }

    fn attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        mask: Option<&Self::Tensor>,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        attention(&queries, &keys, &values, scale, mask, None, 0)
    }

    fn sliding_window_attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        context
            .sliding_attention_calls
            .set(context.sliding_attention_calls.get() + 1);
        let batch = queries.shape[0];
        let sequence = queries.shape[2];
        let attended = attention(
            &queries,
            &keys,
            &values,
            scale,
            None,
            Some(window),
            position_offset,
        )?;
        attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)
    }

    fn causal_mask(
        sequence: i32,
        position_offset: i32,
        window: Option<i32>,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let keys = position_offset + sequence;
        let mut mask = NumericTensor::zeros(vec![sequence, keys]);
        for query in 0..sequence {
            let query_position = position_offset + query;
            for key in 0..keys {
                let too_new = key > query_position;
                let too_old = window.is_some_and(|window| key <= query_position - window);
                if too_new || too_old {
                    mask.data[(query * keys + key) as usize] = -1.0e9;
                }
            }
        }
        Ok(mask)
    }

    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        _: &(),
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        linear.forward(input, context)
    }
}

fn attention(
    queries: &NumericTensor,
    keys: &NumericTensor,
    values: &NumericTensor,
    scale: f32,
    mask: Option<&NumericTensor>,
    window: Option<i32>,
    query_position_offset: i32,
) -> Result<NumericTensor, Error> {
    if queries.shape.len() != 4
        || keys.shape.len() != 4
        || values.shape != keys.shape
        || queries.shape[0] != keys.shape[0]
        || queries.shape[3] != keys.shape[3]
        || queries.shape[1] % keys.shape[1] != 0
    {
        return Err(Error::backend("numeric attention geometry mismatch"));
    }
    let batch = queries.shape[0] as usize;
    let query_heads = queries.shape[1] as usize;
    let key_heads = keys.shape[1] as usize;
    let query_sequence = queries.shape[2] as usize;
    let key_sequence = keys.shape[2] as usize;
    let dimensions = queries.shape[3] as usize;
    if let Some(mask) = mask {
        if mask.shape != [query_sequence as i32, key_sequence as i32] {
            return Err(Error::backend("numeric attention mask shape mismatch"));
        }
    }
    let key_position_start = query_position_offset + query_sequence as i32 - key_sequence as i32;
    let groups = query_heads / key_heads;
    let mut output = NumericTensor::zeros(queries.shape.clone());
    for batch_index in 0..batch {
        for query_head in 0..query_heads {
            let key_head = query_head / groups;
            for query_position in 0..query_sequence {
                let global_query = query_position_offset + query_position as i32;
                let mut scores = vec![f32::NEG_INFINITY; key_sequence];
                for (key_position, score) in scores.iter_mut().enumerate() {
                    let global_key = key_position_start + key_position as i32;
                    let causal = global_key <= global_query;
                    let local = window.is_none_or(|window| global_key > global_query - window);
                    if causal && local {
                        let query_base = ((batch_index * query_heads + query_head)
                            * query_sequence
                            + query_position)
                            * dimensions;
                        let key_base = ((batch_index * key_heads + key_head) * key_sequence
                            + key_position)
                            * dimensions;
                        *score = (0..dimensions)
                            .map(|dimension| {
                                queries.data[query_base + dimension]
                                    * keys.data[key_base + dimension]
                            })
                            .sum::<f32>()
                            * scale
                            + mask.map_or(0.0, |mask| {
                                mask.data[query_position * key_sequence + key_position]
                            });
                    }
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probabilities = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = probabilities.iter().sum::<f32>();
                for probability in &mut probabilities {
                    *probability /= denominator;
                }
                for dimension in 0..dimensions {
                    let output_index = (((batch_index * query_heads + query_head)
                        * query_sequence
                        + query_position)
                        * dimensions)
                        + dimension;
                    output.data[output_index] = (0..key_sequence)
                        .map(|key_position| {
                            let value_index = (((batch_index * key_heads + key_head)
                                * key_sequence
                                + key_position)
                                * dimensions)
                                + dimension;
                            probabilities[key_position] * values.data[value_index]
                        })
                        .sum();
                }
            }
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct NumericRouter {
    linear: NumericLinear,
    routing: TopKRoutingSpec,
}

impl Parameterized<NumericTensor> for NumericRouter {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        self.linear.visit_parameters(visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        self.linear.visit_parameters_mut(visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.linear.set_trainable(trainable);
    }
}

impl RoutingOperator<NumericTensor> for NumericRouter {
    fn route(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<RoutingResult<NumericTensor>, Error> {
        let logits = self.linear.forward(input, context)?;
        let experts = self.routing.expert_count() as usize;
        let top_k = self.routing.top_k() as usize;
        let tokens = logits.data.len() / experts;
        let route_shape = vec![tokens as i32, top_k as i32];
        let mut expert_ids = NumericTensor::zeros(route_shape.clone());
        let mut selected_scores = NumericTensor::zeros(route_shape.clone());
        let mut route_weights = NumericTensor::zeros(route_shape);
        for token in 0..tokens {
            let row = &logits.data[token * experts..(token + 1) * experts];
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials = row
                .iter()
                .map(|value| (*value - maximum).exp())
                .collect::<Vec<_>>();
            let sum = exponentials.iter().sum::<f32>();
            let probabilities = exponentials
                .iter()
                .map(|value| *value / sum)
                .collect::<Vec<_>>();
            let mut order = (0..experts).collect::<Vec<_>>();
            order.sort_by(|left, right| probabilities[*right].total_cmp(&probabilities[*left]));
            let selected_sum = order[..top_k]
                .iter()
                .map(|expert| probabilities[*expert])
                .sum::<f32>();
            for (route, expert) in order.iter().copied().take(top_k).enumerate() {
                let selected = probabilities[expert];
                let index = token * top_k + route;
                expert_ids.data[index] = expert as f32;
                selected_scores.data[index] = selected;
                route_weights.data[index] = if self.routing.normalize_selected() {
                    selected / selected_sum
                } else {
                    selected
                };
            }
        }
        Ok(RoutingResult {
            expert_ids,
            selected_scores,
            route_weights,
        })
    }
}

#[derive(Debug, Clone)]
struct NumericExpert {
    gate: NumericTensor,
    up: NumericTensor,
    down: NumericTensor,
}

#[derive(Debug, Clone)]
struct NumericExpertBank {
    experts: Vec<NumericExpert>,
    parameters: Vec<(NumericTensor, ParameterMetadata)>,
}

impl Parameterized<NumericTensor> for NumericExpertBank {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        for (parameter, metadata) in &self.parameters {
            visit(metadata, parameter, visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        for (parameter, metadata) in &mut self.parameters {
            visit_mut(metadata, parameter, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        for (_, metadata) in &mut self.parameters {
            metadata.trainable = trainable;
        }
    }
}

impl SwiGluExpertBankOperator<NumericTensor> for NumericExpertBank {
    fn forward_routed(
        &mut self,
        input: &NumericTensor,
        routes: &RoutingResult<NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let hidden = input.shape.last().copied().unwrap() as usize;
        let tokens = input.data.len() / hidden;
        let top_k = routes.expert_ids.shape[1] as usize;
        if routes.expert_ids.shape != [tokens as i32, top_k as i32]
            || routes.route_weights.shape != routes.expert_ids.shape
        {
            return Err(Error::backend("numeric expert route geometry mismatch"));
        }
        let mut output = NumericTensor::zeros(input.shape.clone());
        for token in 0..tokens {
            let token_input = NumericTensor::new(
                vec![1, hidden as i32],
                input.data[token * hidden..(token + 1) * hidden].to_vec(),
            );
            for route in 0..top_k {
                let route_index = token * top_k + route;
                let expert_id = routes.expert_ids.data[route_index] as usize;
                let expert = self
                    .experts
                    .get(expert_id)
                    .ok_or_else(|| Error::backend("numeric expert id is out of range"))?;
                let gate = linear(&token_input, &expert.gate, None)?
                    .map(|value| value / (1.0 + (-value).exp()));
                let up = linear(&token_input, &expert.up, None)?;
                let activated = gate.zip(&up, |left, right| left * right)?;
                let expert_output = linear(&activated, &expert.down, None)?;
                let weight = routes.route_weights.data[route_index];
                for dimension in 0..hidden {
                    output.data[token * hidden + dimension] +=
                        weight * expert_output.data[dimension];
                }
            }
        }
        Ok(output)
    }
}

impl RoutedNeuralBackend for NumericBackend {
    type Router = NumericRouter;
    type SwiGluExpertBank = NumericExpertBank;

    fn top_k_router(spec: TopKRouterSpec, context: &NumericContext) -> Result<Self::Router, Error> {
        spec.validate()?;
        let routing = spec.routing;
        let linear = Self::linear(
            LinearSpec {
                input: spec.input_dimensions,
                output: routing.expert_count(),
                weight: spec.weight,
                bias: None,
                quantization: spec.quantization,
            },
            context,
        )?;
        Ok(NumericRouter { linear, routing })
    }

    fn swiglu_expert_bank(
        spec: SwiGluExpertBankSpec,
        _: &NumericContext,
    ) -> Result<Self::SwiGluExpertBank, Error> {
        spec.validate()?;
        let expert_count = spec.expert_count as usize;
        let hidden = spec.input_dimensions;
        let intermediate = spec.intermediate_dimensions;
        let mut experts = Vec::with_capacity(expert_count);
        let mut parameters = Vec::new();
        match spec.layout {
            SwiGluExpertLayout::Packed { gate_up, down } => {
                let packed_gate_up = parameter(
                    &gate_up.weight,
                    vec![spec.expert_count, 2 * intermediate, hidden],
                    false,
                );
                let packed_down = parameter(
                    &down.weight,
                    vec![spec.expert_count, spec.output_dimensions, intermediate],
                    false,
                );
                let gate_up_per_expert = (2 * intermediate * hidden) as usize;
                let projection_per_expert = (intermediate * hidden) as usize;
                let down_per_expert = (spec.output_dimensions * intermediate) as usize;
                for expert in 0..expert_count {
                    let gate_up_start = expert * gate_up_per_expert;
                    let down_start = expert * down_per_expert;
                    experts.push(NumericExpert {
                        gate: NumericTensor::new(
                            vec![intermediate, hidden],
                            packed_gate_up.data
                                [gate_up_start..gate_up_start + projection_per_expert]
                                .to_vec(),
                        ),
                        up: NumericTensor::new(
                            vec![intermediate, hidden],
                            packed_gate_up.data[gate_up_start + projection_per_expert
                                ..gate_up_start + 2 * projection_per_expert]
                                .to_vec(),
                        ),
                        down: NumericTensor::new(
                            vec![spec.output_dimensions, intermediate],
                            packed_down.data[down_start..down_start + down_per_expert].to_vec(),
                        ),
                    });
                }
                parameters.push((
                    packed_gate_up,
                    ParameterMetadata::from_spec(&gate_up.weight, gate_up.weight.trainable),
                ));
                parameters.push((
                    packed_down,
                    ParameterMetadata::from_spec(&down.weight, down.weight.trainable),
                ));
            }
            SwiGluExpertLayout::Independent(specs) => {
                for expert_spec in specs {
                    let gate =
                        parameter(&expert_spec.gate.weight, vec![intermediate, hidden], false);
                    let up = parameter(&expert_spec.up.weight, vec![intermediate, hidden], false);
                    let down = parameter(
                        &expert_spec.down.weight,
                        vec![spec.output_dimensions, intermediate],
                        false,
                    );
                    experts.push(NumericExpert {
                        gate: gate.clone(),
                        up: up.clone(),
                        down: down.clone(),
                    });
                    parameters.extend([
                        (
                            gate,
                            ParameterMetadata::from_spec(
                                &expert_spec.gate.weight,
                                expert_spec.gate.weight.trainable,
                            ),
                        ),
                        (
                            up,
                            ParameterMetadata::from_spec(
                                &expert_spec.up.weight,
                                expert_spec.up.weight.trainable,
                            ),
                        ),
                        (
                            down,
                            ParameterMetadata::from_spec(
                                &expert_spec.down.weight,
                                expert_spec.down.weight.trainable,
                            ),
                        ),
                    ]);
                }
            }
        }
        Ok(NumericExpertBank {
            experts,
            parameters,
        })
    }
}

#[derive(Debug, Clone)]
struct NumericCache {
    offset: i32,
    window: Option<i32>,
    keys: Option<NumericTensor>,
    values: Option<NumericTensor>,
}

impl NumericCache {
    fn new(window: Option<i32>) -> Self {
        Self {
            offset: 0,
            window,
            keys: None,
            values: None,
        }
    }

    fn retained(&self) -> i32 {
        self.keys.as_ref().map_or(0, |keys| keys.shape[2])
    }
}

impl AttentionCache<NumericTensor> for NumericCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        self.window
    }

    fn update_for_attention(
        &mut self,
        keys: NumericTensor,
        values: NumericTensor,
        context: &NumericContext,
    ) -> Result<(NumericTensor, NumericTensor), Error> {
        let added = keys.shape[2];
        let attention_keys = if let Some(previous) = &self.keys {
            NumericTensor::concatenate(&[previous.clone(), keys], 2, context)?
        } else {
            keys
        };
        let attention_values = if let Some(previous) = &self.values {
            NumericTensor::concatenate(&[previous.clone(), values], 2, context)?
        } else {
            values
        };
        self.offset += added;
        let retained_start =
            self.window
                .map_or(0, |window| (attention_keys.shape[2] - window).max(0)) as usize;
        self.keys =
            Some(attention_keys.axis_slice(2, retained_start, attention_keys.shape[2] as usize));
        self.values = Some(attention_values.axis_slice(
            2,
            retained_start,
            attention_values.shape[2] as usize,
        ));
        Ok((attention_keys, attention_values))
    }

    fn attention(
        &mut self,
        queries: NumericTensor,
        keys: NumericTensor,
        values: NumericTensor,
        scale: f32,
        mask: Option<&NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let query_offset = self.offset - queries.shape[2];
        attention(
            &queries,
            &keys,
            &values,
            scale,
            mask,
            self.window,
            query_offset,
        )
    }
}

fn config(model_type: &str, tied: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": model_type,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 17,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tied
    });
    if model_type == "qwen2" {
        config["attention_bias"] = true.into();
        config["use_sliding_window"] = true.into();
        config["sliding_window"] = 2.into();
        config["max_window_layers"] = 0.into();
    }
    if model_type == "qwen3_moe" {
        config["intermediate_size"] = 0.into();
        config["moe_intermediate_size"] = 6.into();
        config["num_experts"] = 4.into();
        config["num_experts_per_tok"] = 2.into();
        config["norm_topk_prob"] = true.into();
    }
    config
}

struct ForwardResult {
    prefill: Vec<f32>,
    decode: Vec<f32>,
    retained: Vec<i32>,
    sliding_calls: usize,
}

fn forward(model_type: &str, tied: bool) -> Result<ForwardResult, Error> {
    let args =
        qwen::model_args_from_config_value(&config(model_type, tied)).map_err(Error::backend)?;
    let context = NumericContext::default();
    let mut decoder = qwen::new_decoder::<NumericBackend>(&args, &context)?;
    let mut head = (!tied)
        .then(|| {
            NumericBackend::linear(
                LinearSpec {
                    input: args.hidden_size,
                    output: args.vocab_size,
                    weight: ParameterSpec::trainable("lm_head.weight").map_err(Error::backend)?,
                    bias: None,
                    quantization: None,
                },
                &context,
            )
        })
        .transpose()?;
    let mut caches = qwen::create_caches(&args, |_, window| NumericCache::new(window))?;

    let prefill_tokens = NumericTensor::token_ids(&[1, 4, 2]);
    let mask = NumericBackend::causal_mask(3, 0, None, &context)?;
    let hidden = decoder.embed(&prefill_tokens, &context)?;
    let hidden = decoder.forward_embedded(hidden, Some(&mask), true, &mut caches, &context)?;
    let prefill_logits = match &mut head {
        Some(head) => head.forward(&hidden, &context)?,
        None => decoder.embeddings.as_linear(&hidden, &context)?,
    };

    let decode_tokens = NumericTensor::token_ids(&[3]);
    let hidden = decoder.forward(&decode_tokens, None, &mut caches, &context)?;
    let decode_logits = match &mut head {
        Some(head) => head.forward(&hidden, &context)?,
        None => decoder.embeddings.as_linear(&hidden, &context)?,
    };
    let cache_state = caches
        .iter()
        .map(|cache| cache.as_ref().unwrap().retained())
        .collect();
    Ok(ForwardResult {
        prefill: prefill_logits.data,
        decode: decode_logits.data,
        retained: cache_state,
        sliding_calls: context.sliding_attention_calls.get(),
    })
}

fn checksum(values: &[f32]) -> [f32; 3] {
    [
        values.iter().sum(),
        values
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f32 * value)
            .sum(),
        values.iter().map(|value| value * value).sum(),
    ]
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2.0e-5,
        "expected {expected:.9}, got {actual:.9}"
    );
}

#[test]
fn numerical_qwen2_qwen3_and_moe_prefill_decode_goldens() {
    let cases = [
        (
            "qwen2",
            false,
            [0.362_308_77, 8.959_988, 0.310_638_3],
            [-0.036_465_22, 0.943_931_04, 0.095_674_14],
        ),
        (
            "qwen3",
            true,
            [0.331_483_6, 9.507_395, 0.279_646_04],
            [0.250_913_5, 2.956_005, 0.086_268_68],
        ),
        (
            "qwen3_moe",
            false,
            [0.327_792_88, 7.773_554, 0.309_266_42],
            [-0.042_136_565, 0.978_026_4, 0.092_903_41],
        ),
    ];
    for (model_type, tied, expected_prefill, expected_decode) in cases {
        let result = forward(model_type, tied).unwrap();
        let actual_prefill = checksum(&result.prefill);
        let actual_decode = checksum(&result.decode);
        for (actual, expected) in actual_prefill.into_iter().zip(expected_prefill) {
            assert_close(actual, expected);
        }
        for (actual, expected) in actual_decode.into_iter().zip(expected_decode) {
            assert_close(actual, expected);
        }
        assert_eq!(
            result.retained,
            if model_type == "qwen2" {
                vec![2]
            } else {
                vec![4]
            }
        );
        assert_eq!(result.sliding_calls, usize::from(model_type == "qwen2"));
    }
}

#[test]
fn explicit_rotary_embeddings_match_offset_positions() {
    let context = NumericContext::default();
    let input = NumericTensor::new(
        vec![1, 1, 2, 4],
        vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
    );
    let mut rotary = NumericRotary {
        dimensions: 4,
        traditional: false,
        base: 10_000.0,
    };
    let offset_output = rotary
        .forward(&input, RotaryPosition::Offset(3), &context)
        .unwrap();
    let mut cosine = NumericTensor::zeros(vec![2, 2]);
    let mut sine = cosine.clone();
    for position in 0..2 {
        for frequency in 0..2 {
            let theta = (3 + position) as f32 / 10_000_f32.powf(frequency as f32 / 2.0);
            cosine.data[position * 2 + frequency] = theta.cos();
            sine.data[position * 2 + frequency] = theta.sin();
        }
    }
    let embedded_output = rotary
        .forward(
            &input,
            RotaryPosition::Embeddings {
                cosine: &cosine,
                sine: &sine,
            },
            &context,
        )
        .unwrap();
    for (offset, embedded) in offset_output.data.iter().zip(embedded_output.data) {
        assert_close(*offset, embedded);
    }
}

#[test]
fn reference_attention_obeys_causal_and_sliding_semantics() {
    let queries = NumericTensor::new(vec![1, 1, 2, 1], vec![1.0, 1.0]);
    let keys = NumericTensor::new(vec![1, 1, 2, 1], vec![0.0, 3.0_f32.ln()]);
    let values = NumericTensor::new(vec![1, 1, 2, 1], vec![2.0, 10.0]);

    let causal = attention(&queries, &keys, &values, 1.0, None, None, 0).unwrap();
    assert_close(causal.data[0], 2.0);
    assert_close(causal.data[1], 8.0);

    let sliding = attention(&queries, &keys, &values, 1.0, None, Some(1), 0).unwrap();
    assert_close(sliding.data[0], 2.0);
    assert_close(sliding.data[1], 10.0);
}

#[test]
fn reference_router_and_packed_experts_match_analytical_values() {
    let context = NumericContext::default();
    let router_spec = ParameterSpec::trainable("router.weight").unwrap();
    let mut router = NumericRouter {
        linear: NumericLinear {
            weight: NumericTensor::new(
                vec![3, 2],
                vec![0.0, 0.0, 2.0_f32.ln(), 0.0, 4.0_f32.ln(), 0.0],
            ),
            weight_metadata: ParameterMetadata::from_spec(&router_spec, true),
            bias: None,
        },
        routing: TopKRoutingSpec::new(3, 2, eredu_nn::RoutingScoring::Softmax, true).unwrap(),
    };
    let input = NumericTensor::new(vec![1, 2], vec![1.0, 0.0]);
    let routes = router.route(&input, &context).unwrap();
    assert_eq!(routes.expert_ids.data, [2.0, 1.0]);
    assert_close(routes.selected_scores.data[0], 4.0 / 7.0);
    assert_close(routes.selected_scores.data[1], 2.0 / 7.0);
    assert_close(routes.route_weights.data[0], 2.0 / 3.0);
    assert_close(routes.route_weights.data[1], 1.0 / 3.0);

    let mut experts = NumericExpertBank {
        experts: vec![
            NumericExpert {
                gate: NumericTensor::new(vec![1, 2], vec![1.0, 0.0]),
                up: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                down: NumericTensor::new(vec![2, 1], vec![1.0, 2.0]),
            },
            NumericExpert {
                gate: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                up: NumericTensor::new(vec![1, 2], vec![1.0, 0.0]),
                down: NumericTensor::new(vec![2, 1], vec![2.0, -1.0]),
            },
        ],
        parameters: Vec::new(),
    };
    let routes = RoutingResult {
        expert_ids: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
        selected_scores: NumericTensor::new(vec![1, 2], vec![0.25, 0.75]),
        route_weights: NumericTensor::new(vec![1, 2], vec![0.25, 0.75]),
    };
    let output = experts
        .forward_routed(
            &NumericTensor::new(vec![1, 2], vec![2.0, 1.0]),
            &routes,
            &context,
        )
        .unwrap();
    assert_close(output.data[0], 2.633_574_2);
    assert_close(output.data[1], -0.215_790_8);
}
