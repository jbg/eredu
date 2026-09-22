use super::*;
use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use safemlx::{
    ops::indexing::{IntoStrideBy, TryIndexOp},
    Array, Device, DeviceType, Dtype, Stream,
};

struct Bind<'a> {
    weight: &'a MlxTensor,
    scale: Option<&'a MlxTensor>,
    affine: Option<&'a MlxTensor>,
    bias: &'a MlxTensor,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        *value = match metadata.id().as_str() {
            "matrix.weight" => self.weight,
            "matrix.scales" => self.scale.unwrap(),
            "matrix.biases" => self.affine.unwrap(),
            "matrix.bias" => self.bias,
            name => panic!("unexpected binding {name}"),
        }
        .clone();
    }
}
fn strided(array: Array, yes: bool, s: &Stream) -> Array {
    if !yes || array.size() == 0 {
        return array;
    }
    // Interleave with zeros, then retain only the original entries. This
    // preserves every logical value while forcing row compaction where needed.
    let mut shape = array.shape().to_vec();
    let axis = shape.len() - 1;
    shape[axis] *= 2;
    let expanded = array.expand_dims(-1, s).unwrap();
    let zeros = safemlx::ops::zeros_like(&expanded, s).unwrap();
    let joined = safemlx::ops::concatenate_axis(&[expanded, zeros], -1, s)
        .unwrap()
        .reshape(&shape, s)
        .unwrap();
    match shape.len() {
        1 => joined.try_index_device((..).stride_by(2), s),
        2 => joined.try_index_device((.., (..).stride_by(2)), s),
        3 => joined.try_index_device((.., .., (..).stride_by(2)), s),
        _ => panic!("fixture rank"),
    }
    .unwrap()
}
fn floats(values: &[f32], shape: &[i32], dtype: Dtype, view: bool, s: &Stream) -> MlxTensor {
    MlxTensor::from_array(strided(
        Array::from_slice(values, shape).as_dtype(dtype, s).unwrap(),
        view,
        s,
    ))
}
fn fixture_affine_bias(row: i32, column: i32, group: i32) -> f32 {
    ((row + column / group) % 5 - 2) as f32 / 16.0
}
struct Packed {
    encoding: LinearFormat,
    k: i32,
    n: i32,
    weight: MlxTensor,
    scale: Option<MlxTensor>,
    affine: Option<MlxTensor>,
    dense: Vec<f32>,
}
impl Packed {
    fn evaluate(&self) {
        let mut arrays = vec![self.weight.as_array()];
        arrays.extend(self.scale.iter().map(MlxTensor::as_array));
        arrays.extend(self.affine.iter().map(MlxTensor::as_array));
        safemlx::transforms::eval(arrays).unwrap();
    }
    fn bind<'a>(&'a self, bias: &'a MlxTensor) -> Bind<'a> {
        Bind {
            weight: &self.weight,
            scale: self.scale.as_ref(),
            affine: self.affine.as_ref(),
            bias,
        }
    }
}
fn affine(encoding: LinearFormat, k: i32, n: i32, dtype: Dtype, view: bool, s: &Stream) -> Packed {
    let (group, bits) = match encoding {
        LinearFormat::Affine(c) => (c.group_size, c.bits),
        LinearFormat::MxFp4 => (32, 4),
        _ => unreachable!(),
    };
    let words = k * bits / 32;
    let mut weights = vec![0u32; (n * words) as usize];
    let mut dense = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mx = encoding == LinearFormat::MxFp4;
    const FP4: [f32; 16] = [
        0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
    ];
    for row in 0..n {
        for column in 0..k {
            let q = ((row * 7 + column * 3) % (1 << bits)) as u32;
            let bit = column * bits;
            let index = (row * words + bit / 32) as usize;
            weights[index] |= q << (bit % 32);
            if bit % 32 + bits > 32 {
                weights[index + 1] |= q >> (32 - bit % 32);
            }
            let scale = if mx {
                if (row + column / group) % 2 == 0 {
                    0.5
                } else {
                    1.0
                }
            } else {
                ((row + column / group) % 3 + 1) as f32 / 32.0
            };
            let bias = if mx {
                0.
            } else {
                fixture_affine_bias(row, column, group)
            };
            if column % group == 0 {
                scales.push(scale);
                biases.push(bias);
            }
            dense.push(if mx {
                FP4[q as usize] * scale
            } else {
                q as f32 * scale + bias
            });
        }
    }
    let weight = MlxTensor::from_array(strided(Array::from_slice(&weights, &[n, words]), view, s));
    let scale = if mx {
        let scales = scales
            .iter()
            .map(|v| if *v == 0.5 { 126u8 } else { 127 })
            .collect::<Vec<_>>();
        MlxTensor::from_array(strided(
            Array::from_slice(&scales, &[n, k / group]),
            view,
            s,
        ))
    } else {
        floats(&scales, &[n, k / group], dtype, view, s)
    };
    let affine = (!mx).then(|| floats(&biases, &[n, k / group], dtype, view, s));
    Packed {
        encoding,
        k,
        n,
        weight,
        scale: Some(scale),
        affine,
        dense,
    }
}
fn check_projection(
    p: &Packed,
    shape: &[i32],
    dtype: Dtype,
    view: bool,
    tied: bool,
    s: &Stream,
) -> (u64, u64) {
    let count = shape.iter().product::<i32>() as usize;
    let fp8 = matches!(p.encoding, LinearFormat::E4M3BlockFp8(_));
    let values = (0..count)
        .map(|i| {
            if fp8 && (i % p.k as usize) % 128 == 0 {
                448.
            } else {
                ((i * 3 + 1) % 9) as f32 - 4.
            }
        })
        .collect::<Vec<_>>();
    let input = floats(&values, shape, dtype, view, s);
    let bias_values = (0..p.n)
        .map(|i| (i % 3 - 1) as f32 / 8.)
        .collect::<Vec<_>>();
    let bias = floats(&bias_values, &[p.n], Dtype::Float32, view, s);
    let mut projection = linear::<MlxNeuralBackend>(p.encoding, p.k, p.n, !tied, s);
    projection.visit_parameters_mut(&mut p.bind(&bias));
    let mut table = tied.then(|| embedding::<MlxNeuralBackend>(p.encoding, p.k, p.n, s));
    if let Some(table) = &mut table {
        table.visit_parameters_mut(&mut p.bind(&bias));
    }
    p.evaluate();
    safemlx::transforms::eval([input.as_array(), bias.as_array()]).unwrap();
    let c = WorkspaceContext::new(selected());
    let meta = WorkspaceTensor::existing(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
        &c,
    )
    .unwrap();
    let mut model = linear::<WorkspaceBackend>(p.encoding, p.k, p.n, !tied, &c);
    c.begin_span();
    let out = model.forward(&meta, &c).unwrap();
    let quote = c.report(&[out]).unwrap();
    let bound = quote.tensor_buffers.total_bytes.unwrap();
    assert!(quote.total_bytes.is_some());
    s.synchronize().unwrap();
    let before = safemlx::memory::active_memory().unwrap();
    safemlx::memory::reset_peak_memory().unwrap();
    let output = if let Some(table) = &mut table {
        table.as_linear(&input, s).unwrap()
    } else {
        projection.forward(&input, s).unwrap()
    };
    safemlx::transforms::eval([output.as_array()]).unwrap();
    s.synchronize().unwrap();
    let peak = safemlx::memory::peak_memory()
        .unwrap()
        .saturating_sub(before) as u64;
    assert!(
        peak <= bound,
        "{:?} {shape:?} {dtype:?} view={view} tied={tied}: peak {peak} > {bound}",
        p.encoding
    );
    let actual = output.to_f32_vec(s).unwrap();
    let mut max_error = 0f64;
    let mut max_rounding_allowance = 0f64;
    for (i, &actual) in actual.iter().enumerate() {
        let row = i / p.n as usize;
        let col = i % p.n as usize;
        let expected = (0..p.k as usize)
            .map(|k| {
                f64::from(values[row * p.k as usize + k])
                    * f64::from(p.dense[col * p.k as usize + k])
            })
            .sum::<f64>()
            + if tied {
                0.
            } else {
                f64::from(bias_values[col])
            };
        let error = (f64::from(actual) - expected).abs();
        let rounding = packed_rounding_allowance(
            p,
            &values[row * p.k as usize..(row + 1) * p.k as usize],
            col,
            dtype,
            count / p.k as usize,
        );
        max_error = max_error.max(error);
        max_rounding_allowance = max_rounding_allowance.max(rounding);
        assert!(
            actual.is_finite() && error <= 0.03 + 0.02 * expected.abs() + rounding,
            "{:?} {shape:?} {dtype:?} i={i}: {actual} != {expected}, rounding={rounding}",
            p.encoding
        );
    }
    eprintln!("PACKED_PROJECTION format={:?} shape={shape:?} dtype={dtype:?} scale_dtype={:?} affine_dtype={:?} view={view} tied={tied} peak={peak} bound={bound} values={} max_error={max_error} max_rounding_allowance={max_rounding_allowance}",p.encoding,p.scale.as_ref().map(|v| v.as_array().dtype()),p.affine.as_ref().map(|v| v.as_array().dtype()),actual.len());
    (peak, bound)
}

fn packed_rounding_allowance(
    p: &Packed,
    input: &[f32],
    row: usize,
    dtype: Dtype,
    positions: usize,
) -> f64 {
    let group = match p.encoding {
        LinearFormat::Affine(config) => config.group_size as usize,
        LinearFormat::MxFp4 => 32,
        _ => return 0.,
    };
    if dtype == Dtype::Float32 || group == 16 || positions < 6 {
        return 0.;
    }
    let round = |value: f64| match dtype {
        Dtype::Float16 => f64::from(half::f16::from_f64(value).to_f32()),
        Dtype::Bfloat16 => f64::from(half::bf16::from_f64(value).to_f32()),
        _ => unreachable!(),
    };
    // Independently decoded weights and F64 dot products remain the oracle.
    // Native QMM rounds decoded tiles and each split-K partial to the input
    // dtype, then reduces them in that dtype. These 512-column fixtures have
    // at most 16 partials: native col_reduce_small uses up to eight cyclic
    // lanes and rounds each addition. Affine multiply/add may round separately.
    // Bound those realizations over every legal partition and lane count,
    // independently of this machine's performance-based dispatch choice.
    let k = p.k as usize;
    let weights = &p.dense[row * k..(row + 1) * k];
    let exact = input
        .iter()
        .zip(weights)
        .map(|(&x, &w)| f64::from(x) * f64::from(w))
        .sum::<f64>();
    let groups = k / group;
    let mut allowance = 0f64;
    for separate in [false, true] {
        let mut prefix = vec![0f64];
        for (g, (x, w)) in input
            .chunks_exact(group)
            .zip(weights.chunks_exact(group))
            .enumerate()
        {
            let bias = if separate && matches!(p.encoding, LinearFormat::Affine(_)) {
                f64::from(fixture_affine_bias(
                    row as i32,
                    (g * group) as i32,
                    group as i32,
                ))
            } else {
                0.
            };
            let dot = x
                .iter()
                .zip(w)
                .map(|(&x, &w)| {
                    let decoded = if separate {
                        round(round(f64::from(w) - bias) + bias)
                    } else {
                        round(f64::from(w))
                    };
                    f64::from(x) * decoded
                })
                .sum::<f64>();
            prefix.push(prefix.last().unwrap() + dot);
        }
        for parts in 1..=groups.min(512) {
            if groups % parts != 0 {
                continue;
            }
            let width = groups / parts;
            let partials = (0..parts)
                .map(|part| round(prefix[(part + 1) * width] - prefix[part * width]))
                .collect::<Vec<_>>();
            for lanes in 1..=parts.min(8) {
                let mut sum = 0.;
                for lane in 0..lanes {
                    let mut subtotal = 0.;
                    for i in (lane..parts).step_by(lanes) {
                        subtotal = round(subtotal + partials[i]);
                    }
                    sum = round(sum + subtotal);
                }
                allowance = allowance.max((sum - exact).abs());
            }
        }
    }
    allowance
}
fn check_embedding(p: &Packed, view: bool, policy: EmbeddingLookupPolicy, s: &Stream) {
    let values = [
        p.n - 1,
        0,
        if policy == EmbeddingLookupPolicy::Strict {
            1
        } else {
            -1
        },
        1,
        p.n - 1,
        0,
    ];
    let ids = MlxTensor::from_array(strided(Array::from_slice(&values, &[2, 3]), view, s));
    let bias = floats(&vec![0.; p.n as usize], &[p.n], Dtype::Float32, false, s);
    let mut table = embedding::<MlxNeuralBackend>(p.encoding, p.k, p.n, s);
    table.visit_parameters_mut(&mut p.bind(&bias));
    p.evaluate();
    safemlx::transforms::eval([ids.as_array()]).unwrap();
    let c = WorkspaceContext::new(selected());
    let mut model = embedding::<WorkspaceBackend>(p.encoding, p.k, p.n, &c);
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2, 3], WorkspaceDtype::Int32).unwrap(),
        &c,
    )
    .unwrap();
    c.begin_span();
    let output = model.lookup(&input, policy, &c).unwrap();
    let quote = c.report(&[output]).unwrap();
    let bound = quote.tensor_buffers.total_bytes.unwrap();
    assert_eq!(quote.host_workspace_bytes, Some(0));
    s.synchronize().unwrap();
    let before = safemlx::memory::active_memory().unwrap();
    safemlx::memory::reset_peak_memory().unwrap();
    let output = table.lookup(&ids, policy, s).unwrap();
    safemlx::transforms::eval([output.as_array()]).unwrap();
    s.synchronize().unwrap();
    let peak = safemlx::memory::peak_memory()
        .unwrap()
        .saturating_sub(before) as u64;
    assert!(
        peak <= bound,
        "{:?} embedding view={view}: {peak} > {bound}",
        p.encoding
    );
    let actual = output.to_f32_vec(s).unwrap();
    let mut max_error = 0f32;
    for (i, &actual) in actual.iter().enumerate() {
        let id = values[i / p.k as usize];
        let expected = if id < 0 {
            0.
        } else {
            p.dense[id as usize * p.k as usize + i % p.k as usize]
        };
        let error = (actual - expected).abs();
        max_error = max_error.max(error);
        assert!(
            actual.is_finite() && error <= 0.002 + 0.01 * expected.abs(),
            "{:?} embedding i={i}: {actual} != {expected}",
            p.encoding
        );
    }
    eprintln!("PACKED_EMBEDDING format={:?} scale_dtype={:?} affine_dtype={:?} view={view} policy={policy:?} peak={peak} bound={bound} values={} max_error={max_error}",p.encoding,p.scale.as_ref().map(|v| v.as_array().dtype()),p.affine.as_ref().map(|v| v.as_array().dtype()),actual.len());
}

#[test]
#[ignore = "requires exclusive Metal allocator measurements"]
fn affine_and_mxfp4_peaks_and_values_cover_readout_and_selected_rows() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for encoding in encodings()
        .into_iter()
        .filter(|f| matches!(f, LinearFormat::Affine(_) | LinearFormat::MxFp4))
    {
        for view in [false, true] {
            for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                let p = affine(encoding, 512, 37, dtype, view, &s);
                for shape in [&[512][..], &[2, 7, 512], &[65, 512]] {
                    check_projection(&p, shape, dtype, view, false, &s);
                }
                check_projection(&p, &[2, 1, 512], dtype, view, true, &s);
                for policy in [
                    EmbeddingLookupPolicy::Strict,
                    EmbeddingLookupPolicy::ZeroSentinel(-1),
                ] {
                    check_embedding(&p, view, policy, &s);
                }
            }
        }
    }
    // Select both actual split-K and its own two-pass reduction on this GPU.
    for k in [16384, 16416] {
        let p = affine(
            LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
            k,
            1,
            Dtype::Float32,
            false,
            &s,
        );
        check_projection(&p, &[32, k], Dtype::Float32, false, false, &s);
    }
    // Independent companion dtypes require promotion of selected rows before
    // the native dequantizer reads them. Cover all mixed floating pairs and
    // activation promotion, without expanding the complete table.
    for scale_dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for bias_dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            if scale_dtype == bias_dtype {
                continue;
            }
            for view in [false, true] {
                let mut p = affine(
                    LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
                    512,
                    37,
                    scale_dtype,
                    view,
                    &s,
                );
                let bias = p.affine.take().unwrap();
                p.affine = Some(MlxTensor::from_array(strided(
                    bias.as_array().as_dtype(bias_dtype, &s).unwrap(),
                    view,
                    &s,
                )));
                for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                    check_projection(&p, &[65, 512], dtype, view, false, &s);
                }
                for policy in [
                    EmbeddingLookupPolicy::Strict,
                    EmbeddingLookupPolicy::ZeroSentinel(-1),
                ] {
                    check_embedding(&p, view, policy, &s);
                }
            }
        }
    }
}

fn ggml(ty: GgmlType, endian: Endian, view: bool, s: &Stream) -> Packed {
    let (values, bytes) = ty.block_and_bytes().unwrap();
    let bytes = bytes as usize;
    let mut block = (0..bytes).map(|i| (i % 7 + 1) as u8).collect::<Vec<_>>();
    if endian == Endian::Little && ty.is_iq() {
        for line in include_str!("../../../fixtures/llama-c0bc8591-iq.oracle").lines() {
            let mut fields = line.split('|');
            if GgmlType::from_code(fields.next().unwrap().parse().unwrap()) == ty {
                let hex = fields.next().unwrap();
                block = (0..bytes)
                    .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
                    .collect();
                break;
            }
        }
    }
    // Finite, nonzero scales for ordinary GGML and big-endian IQ blocks.
    let half = |value: f32| {
        let bits = half::f16::from_f32(value).to_bits();
        if endian == Endian::Little {
            bits.to_le_bytes()
        } else {
            bits.to_be_bytes()
        }
    };
    match ty {
        GgmlType::Q4K | GgmlType::Q5K => {
            block[..2].copy_from_slice(&half(0.03125));
            block[2..4].copy_from_slice(&half(0.015625));
        }
        GgmlType::Q6K => {
            block[208..210].copy_from_slice(&half(0.03125));
        }
        GgmlType::Q5_1 => {
            block[..2].copy_from_slice(&half(0.03125));
            block[2..4].copy_from_slice(&half(-0.125));
        }
        GgmlType::Q8_0 => {
            block[..2].copy_from_slice(&half(0.03125));
        }
        GgmlType::IQ1M if endian == Endian::Big => {
            // IQ1M stores its half scale in the high nibbles of four words.
            let scale = half::f16::from_f32(0.03125).to_bits();
            for i in 0..4 {
                let offset = 48 + 2 * i;
                let word = u16::from_be_bytes([block[offset], block[offset + 1]]);
                let word = (word & 0x0fff) | (((scale >> (4 * i)) & 0xf) << 12);
                block[offset..offset + 2].copy_from_slice(&word.to_be_bytes());
            }
        }
        _ if ty.is_iq() && endian == Endian::Big => {
            block[..2].copy_from_slice(&half(0.03125));
        }
        _ => {}
    }
    let n = 7;
    let k = values as i32 * 2;
    // Distinct row scales make wrong-row lookup observable. The first little-
    // endian IQ row retains the pinned oracle block exactly.
    let read_word = |bytes: &[u8]| {
        let word = [bytes[0], bytes[1]];
        if endian == Endian::Little {
            u16::from_le_bytes(word)
        } else {
            u16::from_be_bytes(word)
        }
    };
    let write_word = |value: u16| {
        if endian == Endian::Little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        }
    };
    let mut raw = Vec::new();
    for row in 0..n {
        let mut row_block = block.clone();
        // Keep the pinned row's dynamic range: larger factors can overflow
        // legitimate F16 outputs even though the packed weights remain finite.
        let row_scale = (8 - row) as f32 / 8.0;
        if ty == GgmlType::IQ1M {
            let mut words = (0..4)
                .map(|i| read_word(&row_block[48 + 2 * i..]))
                .collect::<Vec<_>>();
            let bits = words
                .iter()
                .enumerate()
                .fold(0u16, |bits, (i, word)| bits | ((word >> 12) << (4 * i)));
            let scaled =
                half::f16::from_f32(half::f16::from_bits(bits).to_f32() * row_scale).to_bits();
            for (i, word) in words.iter_mut().enumerate() {
                *word = (*word & 0xfff) | (((scaled >> (4 * i)) & 0xf) << 12);
                row_block[48 + 2 * i..50 + 2 * i].copy_from_slice(&write_word(*word));
            }
        } else {
            let offset = if ty == GgmlType::Q6K { 208 } else { 0 };
            let scale = half::f16::from_bits(read_word(&row_block[offset..])).to_f32();
            row_block[offset..offset + 2].copy_from_slice(&half(scale * row_scale));
        }
        raw.extend(row_block.repeat(2));
    }
    let dense = eredu_gguf::IQuantTensor {
        shape: vec![n as u64, k as u64],
        ggml_type: ty,
        endian,
        data: raw.clone(),
    }
    .dequantize_f32()
    .unwrap();
    assert!(
        dense.iter().all(|v| v.is_finite()) && dense.iter().any(|v| *v != 0.),
        "{ty:?} {endian:?} fixture"
    );
    assert!(
        dense
            .chunks_exact(k as usize)
            .skip(1)
            .all(|row| row != &dense[..k as usize]),
        "{ty:?} {endian:?} distinct rows"
    );
    Packed {
        encoding: LinearFormat::GgufIQuant {
            ggml_type: ty,
            endian,
        },
        k,
        n,
        weight: MlxTensor::from_array(strided(
            Array::from_slice(&raw, &[n, 2 * bytes as i32]),
            view,
            s,
        )),
        scale: None,
        affine: None,
        dense,
    }
}

#[test]
#[ignore = "requires exclusive Metal allocator measurements"]
fn native_ggml_peaks_and_values_cover_all_formats_and_byte_orders() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for ty in ggml_types() {
        for endian in [Endian::Little, Endian::Big] {
            for view in [false, true] {
                let p = ggml(ty, endian, view, &s);
                for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                    for rows in [1, 7, 9, 33] {
                        check_projection(&p, &[rows, p.k], dtype, view, false, &s);
                    }
                    check_projection(&p, &[2, 1, p.k], dtype, view, true, &s);
                }
                for policy in [
                    EmbeddingLookupPolicy::Strict,
                    EmbeddingLookupPolicy::ZeroSentinel(-1),
                ] {
                    check_embedding(&p, view, policy, &s);
                }
            }
        }
    }
}

fn fp8(
    k: i32,
    n: i32,
    encoding: BlockFp8ScaleEncoding,
    dtype: Dtype,
    view: bool,
    s: &Stream,
) -> Packed {
    const CODES: [u8; 6] = [0x30, 0xb0, 0x38, 0xb8, 0x20, 0xa0];
    const VALUES: [f32; 6] = [0.5, -0.5, 1., -1., 0.125, -0.125];
    let scale_cols = (k as usize).div_ceil(128);
    let scale_rows = (n as usize).div_ceil(128);
    let scales = (0..scale_cols * scale_rows)
        .map(|i| if i % 2 == 0 { 0.5 } else { 0.25 })
        .collect::<Vec<_>>();
    let mut raw = Vec::new();
    let mut dense = Vec::new();
    for row in 0..n {
        for col in 0..k {
            let i = ((row * 3 + col) as usize) % 6;
            raw.push(CODES[i]);
            dense.push(VALUES[i] * scales[(row as usize / 128) * scale_cols + col as usize / 128]);
        }
    }
    let scale = match encoding {
        BlockFp8ScaleEncoding::FloatingPoint => floats(
            &scales,
            &[scale_rows as i32, scale_cols as i32],
            dtype,
            view,
            s,
        ),
        BlockFp8ScaleEncoding::Ue8m0 => {
            let bytes = scales
                .iter()
                .map(|v| if *v == 0.5 { 126u8 } else { 125 })
                .collect::<Vec<_>>();
            MlxTensor::from_array(strided(
                Array::from_slice(&bytes, &[scale_rows as i32, scale_cols as i32]),
                view,
                s,
            ))
        }
    };
    Packed {
        encoding: LinearFormat::E4M3BlockFp8(BlockFp8Format::new(128, 128, encoding).unwrap()),
        k,
        n,
        weight: MlxTensor::from_array(strided(Array::from_slice(&raw, &[n, k]), view, s)),
        scale: Some(scale),
        affine: None,
        dense,
    }
}

#[test]
#[ignore = "requires exclusive Metal allocator measurements"]
fn fp8_projection_peaks_include_activation_quantization_and_native_scale_lookup() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for scale in [
        BlockFp8ScaleEncoding::FloatingPoint,
        BlockFp8ScaleEncoding::Ue8m0,
    ] {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for view in [false, true] {
                for (k, n) in [(16, 7), (128, 128), (129, 137), (384, 37)] {
                    let p = fp8(k, n, scale, dtype, view, &s);
                    for rows in [1, 7, 65, 257] {
                        check_projection(&p, &[rows, k], dtype, view, false, &s);
                    }
                }
            }
        }
    }
}
