use super::dispatch::validate_activation_dtype;
use super::storage::*;
use super::*;

pub(super) fn packed_linear_cpu(
    input: &Array,
    view: &NativeQuantizedTensor,
    encoding: GgmlType,
    stream: &Stream,
) -> Result<Array, Exception> {
    use rayon::prelude::*;
    let dtype = validate_activation_dtype(input)?;
    let input = input.contiguous(false, stream)?;
    let storage = view.storage.bytes();
    eval([&input, storage])?;
    let evaluated = input.evaluated()?;
    let width = view.columns as usize;
    let rows = input.size() / width;
    let activations = eredu_gguf::packed_dot::PackedDotActivations::new(
        encoding,
        width,
        evaluated
            .try_as_slice::<f32>()
            .map_err(|error| Exception::custom(error.to_string()))?,
    )
    .map_err(block_decode_error)?;
    let storage = storage.evaluated()?;
    let raw = storage
        .try_as_slice::<u8>()
        .map_err(|error| Exception::custom(error.to_string()))?;
    let (block_width, block_bytes) = view.format().block_geometry();
    let row_bytes = width / block_width as usize * block_bytes as usize;
    let output_width = view.rows as usize;
    let count = rows
        .checked_mul(output_width)
        .ok_or_else(|| Exception::custom("packed projection output overflows"))?;
    let mut output = vec![0f32; count];
    output
        .par_iter_mut()
        .with_min_len(256)
        .enumerate()
        .try_for_each(|(i, out)| {
            let row = i / output_width;
            let weight = (view.row_start as usize + i % output_width) * row_bytes;
            *out = activations
                .dot(row, &raw[weight..][..row_bytes], row < rows / 4 * 4)
                .map_err(block_decode_error)?;
            Ok::<_, Exception>(())
        })?;
    let mut shape = input.shape().to_vec();
    *shape.last_mut().expect("validated projection axis") = view.rows;
    Array::from_slice(&output, &shape)
        .copy(stream)?
        .as_dtype(dtype, stream)
}

pub(super) fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .fold(0.0f32, |sum, (&left, &right)| left.mul_add(right, sum))
}

impl NativeQuantizedTensor {
    pub(super) fn packed_rows<'a>(
        &self,
        raw: &'a [u8],
    ) -> Result<eredu_gguf::PackedRows<'a>, Exception> {
        let physical_rows = (self.matrix_count as usize)
            .checked_mul(self.physical_rows as usize)
            .ok_or_else(|| Exception::custom("native packed row count overflows"))?;
        let encoding = self.format().ggml_type().expect("native block encoding");
        eredu_gguf::BlockDecoder::new(encoding, self.storage.endian)
            .and_then(|decoder| decoder.packed_rows(raw, physical_rows, self.columns as usize))
            .map_err(block_decode_error)
    }

    pub(super) fn physical_row(&self, matrix: i32, row: i32) -> Result<usize, Exception> {
        if !(0..self.matrix_count).contains(&matrix) || !(0..self.rows).contains(&row) {
            return Err(Exception::custom(format!(
                "native row ({matrix}, {row}) is outside {:?}",
                self.shape()
            )));
        }
        Ok(matrix as usize * self.physical_rows as usize + (self.row_start + row) as usize)
    }

    pub(super) fn decode_row(
        &self,
        packed: &eredu_gguf::PackedRows<'_>,
        matrix: i32,
        row: i32,
        output: &mut [f32],
    ) -> Result<(), Exception> {
        packed
            .decode_row(self.physical_row(matrix, row)?, output)
            .map_err(block_decode_error)
    }
}

pub(super) fn block_decode_error(error: eredu_gguf::Error) -> Exception {
    Exception::custom(error.to_string())
}

pub(super) fn native_grouped_linear_cpu(
    input: &Array,
    weight: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let input = input.as_dtype(Dtype::Float32, stream)?;
    let group_ids = group_ids.as_dtype(Dtype::Int32, stream)?;
    eval([&input, &group_ids, weight.storage.bytes()])?;
    let evaluated_input = input.evaluated()?;
    let input_values = evaluated_input.as_slice::<f32>();
    let evaluated_ids = group_ids.evaluated()?;
    let ids = evaluated_ids.as_slice::<i32>();
    let evaluated_storage = weight.storage.bytes.evaluated()?;
    let raw = evaluated_storage.as_slice::<u8>();
    let packed = weight.packed_rows(raw)?;
    let mut weights = vec![0.0; weight.columns as usize];
    let routes = input.dim(0);
    let mut output = vec![0.0f32; routes as usize * weight.rows as usize];
    for route in 0..routes as usize {
        let group = ids[route];
        if group < 0 || group >= weight.matrix_count {
            return Err(Exception::custom(format!(
                "native group {group} is outside 0..{}",
                weight.matrix_count
            )));
        }
        let input_row =
            &input_values[route * weight.columns as usize..(route + 1) * weight.columns as usize];
        for output_row in 0..weight.rows {
            weight.decode_row(&packed, group, output_row, &mut weights)?;
            output[route * weight.rows as usize + output_row as usize] =
                dot_f32(input_row, &weights);
        }
    }
    Array::from_slice(&output, &[routes, weight.rows]).copy(stream)
}
