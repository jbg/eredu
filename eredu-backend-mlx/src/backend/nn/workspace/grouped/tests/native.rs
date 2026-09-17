use super::*;
use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized};
use safemlx::{
    ops::indexing::{IntoStrideBy, TryIndexOp},
    Array, Device, DeviceType, Dtype, Stream,
};
use std::collections::BTreeMap;

fn strided(a: Array, yes: bool, s: &Stream) -> Array {
    if !yes || a.size() == 0 {
        return a;
    }
    let shape = a.shape().to_vec();
    let flat = a.reshape(&[-1, 1], s).unwrap();
    let zero = safemlx::ops::zeros_like(&flat, s).unwrap();
    safemlx::ops::concatenate_axis(&[flat, zero], 1, s)
        .unwrap()
        .reshape(&[-1], s)
        .unwrap()
        .try_index_device((..).stride_by(2), s)
        .unwrap()
        .reshape(&shape, s)
        .unwrap()
}
fn floats(data: &[f32], shape: &[i32], dtype: Dtype, view: bool, s: &Stream) -> MlxTensor {
    MlxTensor::from_array(strided(
        Array::from_slice(data, shape).as_dtype(dtype, s).unwrap(),
        view,
        s,
    ))
}
fn rounded(x: f32, dtype: Dtype) -> f32 {
    match dtype {
        Dtype::Float16 => half::f16::from_f32(x).to_f32(),
        Dtype::Bfloat16 => half::bf16::from_f32(x).to_f32(),
        _ => x,
    }
}
fn fp8(code: u8) -> f64 {
    let sign = if code & 128 != 0 { -1. } else { 1. };
    let exp = (code >> 3) & 15;
    let frac = code & 7;
    if exp == 0 {
        sign * f64::from(frac) * 2f64.powi(-9)
    } else if exp == 15 && frac == 7 {
        f64::NAN
    } else {
        sign * (1. + f64::from(frac) / 8.) * 2f64.powi(i32::from(exp) - 7)
    }
}
fn quantize(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for block in values.chunks(128) {
        let max = block.iter().map(|v| v.abs()).fold(1e-4f64, f64::max);
        let scale = (max as f32 / 448.) as f64;
        for &value in block {
            let x = value.abs() / scale;
            let best = (0..127u8)
                .min_by(|a, b| {
                    let da = (fp8(*a) - x).abs();
                    let db = (fp8(*b) - x).abs();
                    da.total_cmp(&db).then_with(|| (a & 1).cmp(&(b & 1)))
                })
                .unwrap();
            out.push(fp8(best) * scale * if value < 0. { -1. } else { 1. });
        }
    }
    out
}
struct Fixture {
    values: BTreeMap<String, MlxTensor>,
    dense: BTreeMap<String, Vec<f32>>,
    bias: BTreeMap<String, Vec<f32>>,
}
struct NativeScale<'a>(Scale<'a, MlxTensor>);
impl GroupedUnitObserver<MlxTensor> for NativeScale<'_> {
    fn observe(&mut self, batch: &GroupedUnitBatch<'_, MlxTensor>) -> Result<(), Error> {
        self.0.observe(batch)
    }
    fn intervene(
        &mut self,
        batch: &GroupedUnitBatch<'_, MlxTensor>,
    ) -> Result<Option<MlxTensor>, Error> {
        let value = self.0.intervene(batch)?.unwrap();
        // A replacement must retain the original activation dtype. The cold
        // multiply envelope includes input promotion, result and restoration.
        Ok(Some(MlxTensor::from_array(
            value
                .as_array()
                .as_dtype(batch.values.as_array().dtype(), self.0.c)
                .map_err(Error::backend)?,
        )))
    }
    fn observe_effective(&mut self, batch: &GroupedUnitBatch<'_, MlxTensor>) -> Result<(), Error> {
        self.0.observe_effective(batch)
    }
}
impl Fixture {
    fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            dense: BTreeMap::new(),
            bias: BTreeMap::new(),
        }
    }
    fn projection(
        &mut self,
        name: &str,
        groups: usize,
        rows: usize,
        columns: usize,
        encoding: LinearFormat,
        bias: bool,
        dtype: Dtype,
        view: bool,
        s: &Stream,
    ) {
        let mut dense = Vec::new();
        let shape = [groups as i32, rows as i32, columns as i32];
        let weight = match encoding {
            LinearFormat::Dense => {
                dense = (0..groups * rows * columns)
                    .map(|i| rounded(((i * 11 + 7) % 29) as f32 / 256. - 0.05, dtype))
                    .collect();
                floats(&dense, &shape, dtype, view, s)
            }
            LinearFormat::Affine(config) => {
                let mut packed = vec![0u32; groups * rows * columns / 8];
                for i in 0..groups * rows * columns {
                    let q = ((i * 7 + i / columns * 3) % 16) as u32;
                    packed[i / 8] |= q << (i % 8 * 4);
                    dense.push(q as f32 / 64. - 0.125);
                }
                let scale_shape = [
                    groups as i32,
                    rows as i32,
                    columns as i32 / config.group_size,
                ];
                let n = scale_shape.iter().product::<i32>() as usize;
                self.values.insert(
                    format!("{name}.scales"),
                    floats(&vec![1. / 64.; n], &scale_shape, dtype, view, s),
                );
                self.values.insert(
                    format!("{name}.biases"),
                    floats(&vec![-0.125; n], &scale_shape, dtype, view, s),
                );
                MlxTensor::from_array(strided(
                    Array::from_slice(&packed, &[groups as i32, rows as i32, columns as i32 / 8]),
                    view,
                    s,
                ))
            }
            LinearFormat::MxFp4 => {
                const FP4: [f32; 16] = [
                    0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
                ];
                let mut packed = vec![0u32; groups * rows * columns / 8];
                for i in 0..groups * rows * columns {
                    let q = ((i * 7 + i / columns * 3) % 16) as u32;
                    packed[i / 8] |= q << (i % 8 * 4);
                    dense.push(FP4[q as usize] / 32.);
                }
                self.values.insert(
                    format!("{name}.scales"),
                    MlxTensor::from_array(strided(
                        Array::from_slice(
                            &vec![122u8; groups * rows * columns / 32],
                            &[groups as i32, rows as i32, columns as i32 / 32],
                        ),
                        view,
                        s,
                    )),
                );
                MlxTensor::from_array(strided(
                    Array::from_slice(&packed, &[groups as i32, rows as i32, columns as i32 / 8]),
                    view,
                    s,
                ))
            }
            LinearFormat::GgufIQuant { .. } => {
                let mut bytes = Vec::new();
                for block in 0..groups * rows * columns / 32 {
                    bytes
                        .extend_from_slice(&half::f16::from_f32(1. / 128.).to_bits().to_le_bytes());
                    for i in 0..32 {
                        let q = ((block * 3 + i * 7) % 17) as i8 - 8;
                        bytes.push(q as u8);
                        dense.push(q as f32 / 128.);
                    }
                }
                MlxTensor::from_array(strided(
                    Array::from_slice(
                        &bytes,
                        &[groups as i32, rows as i32, columns as i32 / 32 * 34],
                    ),
                    view,
                    s,
                ))
            }
            LinearFormat::E4M3BlockFp8(config) => {
                let bytes = (0..groups * rows * columns)
                    .map(|i| 0x20 + (i % 16) as u8 + if i % 3 == 0 { 128 } else { 0 })
                    .collect::<Vec<_>>();
                dense = bytes.iter().map(|v| (fp8(*v) * 0.25) as f32).collect();
                let scale_shape = [
                    groups as i32,
                    rows.div_ceil(128) as i32,
                    columns.div_ceil(128) as i32,
                ];
                let n = scale_shape.iter().product::<i32>() as usize;
                let scale = if config.scale_encoding == BlockFp8ScaleEncoding::Ue8m0 {
                    MlxTensor::from_array(strided(
                        Array::from_slice(&vec![125u8; n], &scale_shape),
                        view,
                        s,
                    ))
                } else {
                    floats(&vec![0.25; n], &scale_shape, dtype, view, s)
                };
                self.values.insert(format!("{name}.scales"), scale);
                MlxTensor::from_array(strided(Array::from_slice(&bytes, &shape), view, s))
            }
        };
        self.values.insert(format!("{name}.weight"), weight);
        self.dense.insert(name.into(), dense);
        if bias {
            let data = (0..groups * rows)
                .map(|i| rounded(((i * 3) % 7) as f32 / 128. - 0.02, dtype))
                .collect::<Vec<_>>();
            self.values.insert(
                format!("{name}.bias"),
                floats(&data, &[groups as i32, rows as i32], dtype, view, s),
            );
            self.bias.insert(name.into(), data);
        }
    }
    fn bind(&self, executable: &mut Executable<MlxNeuralBackend>) {
        struct Bind<'a>(&'a BTreeMap<String, MlxTensor>);
        impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
            fn visit_mut(&mut self, m: ParameterMetadata, t: &'a mut MlxTensor) {
                *t = self.0[m.id.as_str()].clone();
            }
        }
        let mut bind = Bind(&self.values);
        match executable {
            Executable::Linear(m) => m.visit_parameters_mut(&mut bind),
            Executable::Gated(m) => m.visit_parameters_mut(&mut bind),
            Executable::Relu2(m) => m.visit_parameters_mut(&mut bind),
        }
    }
}
fn check(
    kind: Kind,
    tokens: i32,
    width: i32,
    units: i32,
    encoding: LinearFormat,
    dtype: Dtype,
    view: bool,
    observe: bool,
    parallel: bool,
    s: &Stream,
) {
    let groups = 4;
    let k = 2;
    let gated = matches!(kind, Kind::Gated(..));
    let linear = matches!(kind, Kind::Linear(_));
    let bias = !matches!(kind, Kind::Relu2);
    let spec = specification(kind, groups, width, units, encoding, bias);
    let shape = if linear {
        vec![tokens, width]
    } else {
        vec![1, tokens, width]
    };
    let (quote, deliveries) = quote_with_deliveries(&spec, &shape, k, parallel, observe);
    let allowed = quote
        .tensor_buffers
        .total_bytes
        .expect("complete tested operator trace");
    let mut executable = Executable::<MlxNeuralBackend>::new(&spec, s);
    let mut fixture = Fixture::new();
    fixture.projection(
        "read",
        groups as usize,
        units as usize * if gated { 2 } else { 1 },
        width as usize,
        encoding,
        bias,
        dtype,
        view,
        s,
    );
    if !linear {
        fixture.projection(
            "write",
            groups as usize,
            width as usize,
            units as usize,
            encoding,
            bias,
            dtype,
            view,
            s,
        );
    }
    fixture.bind(&mut executable);
    let input_values = (0..tokens * width)
        .map(|i| rounded(((i * 13 + 5) % 31) as f32 / 128. - 0.11, dtype))
        .collect::<Vec<_>>();
    let input = floats(&input_values, &shape, dtype, view, s);
    // Includes duplicate routes and a long skewed prefix, exercising full-bank
    // and uneven selected-group distributions with nonzero coefficients.
    let ids = (0..tokens * k)
        .map(|i| {
            if i / k < tokens / 2 {
                0
            } else {
                (i * 3 + i / k) % groups
            }
        })
        .collect::<Vec<i32>>();
    let coefficients = (0..tokens * k)
        .map(|i| rounded(0.25 + (i % 5) as f32 / 16., dtype))
        .collect::<Vec<_>>();
    let coefficients_native = floats(&coefficients, &[tokens, k], dtype, view, s);
    let selections = GroupSelection::new(
        MlxTensor::from_array(strided(Array::from_slice(&ids, &[tokens, k]), view, s)),
        coefficients_native.clone(),
        coefficients_native,
    );
    let mut arrays = fixture
        .values
        .values()
        .map(MlxTensor::as_array)
        .collect::<Vec<_>>();
    arrays.extend([
        input.as_array(),
        selections.group_indices().as_array(),
        selections.coefficients().as_array(),
    ]);
    safemlx::transforms::eval(arrays).unwrap();
    s.synchronize().unwrap();
    let before = safemlx::memory::active_memory().unwrap();
    safemlx::memory::reset_peak_memory().unwrap();
    let mut observer = NativeScale(Scale {
        c: s,
        deliveries: vec![],
    });
    let out = executable.forward(
        &input,
        &selections,
        parallel.then_some(2),
        observe.then_some(&mut observer),
        s,
    );
    assert_eq!(
        observer.0.deliveries, deliveries,
        "cold and native callback coordinates/order"
    );
    safemlx::transforms::eval(out.iter().map(MlxTensor::as_array)).unwrap();
    s.synchronize().unwrap();
    let observed = safemlx::memory::peak_memory()
        .unwrap()
        .saturating_sub(before) as u64;
    assert!(observed<=allowed,"{kind:?} tokens={tokens} {encoding:?} {dtype:?} view={view} observe={observe} parallel={parallel}: {observed}>{allowed}");
    let expected = reference(
        &spec,
        &fixture,
        &input_values,
        &ids,
        &coefficients,
        observe,
        parallel,
    );
    for (output, expected) in out.iter().zip(expected) {
        let actual = output.to_f32_vec(s).unwrap();
        assert_eq!(actual.len(), expected.len());
        let (atol, rtol) = if dtype == Dtype::Float32 {
            (4e-5, 4e-4)
        } else {
            (0.006, 0.06)
        };
        for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((*actual as f64-expected).abs()<=atol+rtol*expected.abs(),"{kind:?} {encoding:?} {dtype:?} tokens={tokens} observe={observe} i={i}: {actual}!={expected}");
        }
    }
}
fn reference(
    spec: &WorkspaceGroupedBank,
    f: &Fixture,
    input: &[f32],
    ids: &[i32],
    weights: &[f32],
    observe: bool,
    parallel: bool,
) -> Vec<Vec<f64>> {
    let b = Bank::new(spec).unwrap().unwrap();
    let (width, units, out_width) = (b.input as usize, b.units as usize, b.output as usize);
    let tokens = input.len() / width;
    let k = 2;
    let read_rows = if matches!(b.activation, Activation::Gated(_)) {
        units * 2
    } else {
        units
    };
    let mut out = vec![0.; tokens * out_width];
    let mut bias = vec![0.; tokens * out_width];
    let fp8 = matches!(b.first.format().encoding(), LinearFormat::E4M3BlockFp8(_));
    let project = |x: &[f64], name: &str, group: usize, rows: usize, columns: usize| {
        (0..rows)
            .map(|row| {
                x.iter()
                    .enumerate()
                    .map(|(col, v)| v * f.dense[name][(group * rows + row) * columns + col] as f64)
                    .sum::<f64>()
                    + f.bias
                        .get(name)
                        .map_or(0., |v| v[group * rows + row] as f64)
            })
            .collect::<Vec<_>>()
    };
    for token in 0..tokens {
        let x = input[token * width..(token + 1) * width]
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>();
        let x = if fp8 { quantize(&x) } else { x };
        for slot in 0..k {
            let group = ids[token * k + slot] as usize;
            let read = project(&x, "read", group, read_rows, width);
            let mut activated = match b.activation {
                Activation::Linear(GroupedLinearActivation::Identity) => read,
                Activation::Linear(GroupedLinearActivation::Silu) => {
                    read.iter().map(|v| v / (1. + (-v).exp())).collect()
                }
                Activation::Relu2 => read.iter().map(|v| v.max(0.).powi(2)).collect(),
                Activation::Gated(p) => (0..units)
                    .map(|i| {
                        let gate = p
                            .gate_upper_bound()
                            .map_or(read[i], |v| read[i].min(v as f64));
                        let up = p.up_absolute_bound().map_or(read[units + i], |v| {
                            read[units + i].clamp(-f64::from(v), f64::from(v))
                        }) + f64::from(p.up_offset());
                        let activated = match p.activation() {
                            eredu_nn::GatedProductActivation::Silu => {
                                gate / (1. + (-gate * f64::from(p.sigmoid_multiplier())).exp())
                            }
                            eredu_nn::GatedProductActivation::GeluApproximate => {
                                0.5 * gate
                                    * (1.
                                        + ((2. / std::f64::consts::PI).sqrt()
                                            * (gate + 0.044715 * gate.powi(3)))
                                        .tanh())
                            }
                            _ => unreachable!(),
                        };
                        activated * up
                    })
                    .collect(),
            };
            if observe {
                for v in &mut activated {
                    *v *= 0.5;
                }
            }
            let result = if b.down.is_some() {
                let x = if fp8 { quantize(&activated) } else { activated };
                project(&x, "write", group, out_width, units)
            } else {
                activated
            };
            for (row, v) in result.into_iter().enumerate() {
                let w = weights[token * k + slot] as f64;
                out[token * out_width + row] += v * w;
                bias[token * out_width + row] += f
                    .bias
                    .get("write")
                    .map_or(0., |v| v[group * out_width + row] as f64)
                    * w;
            }
        }
    }
    if parallel && b.down.is_some_and(|s| s.bias().is_some()) {
        for (v, b) in out.iter_mut().zip(&bias) {
            *v -= b;
        }
        vec![out, bias]
    } else {
        vec![out]
    }
}

#[test]
#[ignore = "requires exclusive native Metal allocator measurements; run with --test-threads=1"]
fn metal_grouped_workspace_bounds_cover_packed_equations_chunks_and_bias_partials() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mut cases = 0;
    let encodings = [
        LinearFormat::Dense,
        LinearFormat::Affine(AffineQuantization::new(16, 4).unwrap()),
        LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
        LinearFormat::MxFp4,
        LinearFormat::GgufIQuant {
            ggml_type: eredu_gguf::GgmlType::Q8_0,
            endian: eredu_gguf::Endian::Little,
        },
        LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
        ),
        LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0).unwrap(),
        ),
    ];
    for encoding in encodings {
        for kind in [
            Kind::Linear(true),
            Kind::Gated(0, false),
            Kind::Gated(1, true),
            Kind::Gated(2, false),
            Kind::Relu2,
        ] {
            if matches!(kind, Kind::Relu2) && matches!(encoding, LinearFormat::E4M3BlockFp8(_)) {
                continue;
            }
            for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                for view in [false, true] {
                    for tokens in [0, 1, 65] {
                        check(
                            kind,
                            tokens,
                            32,
                            32,
                            encoding,
                            dtype,
                            view,
                            false,
                            !matches!(kind, Kind::Linear(_)),
                            &s,
                        );
                        cases += 1;
                    }
                }
            }
            if !matches!(kind, Kind::Linear(_)) {
                for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                    for tokens in [0, 3, 65, 129] {
                        check(
                            kind,
                            tokens,
                            32,
                            32,
                            encoding,
                            dtype,
                            true,
                            true,
                            tokens >= 65,
                            &s,
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    for kind in [Kind::Linear(true), Kind::Gated(0, true)] {
        check(
            kind,
            1025,
            5,
            7,
            LinearFormat::Dense,
            Dtype::Float32,
            true,
            false,
            false,
            &s,
        );
        cases += 1;
    }
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        check(
            Kind::Gated(0, false),
            64,
            32,
            32,
            LinearFormat::Dense,
            dtype,
            true,
            true,
            true,
            &s,
        );
        cases += 1;
    }
    eprintln!("GROUPED_WORKSPACE_NATIVE_CASES={cases}");
}
