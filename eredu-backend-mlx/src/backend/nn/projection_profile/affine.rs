//! Bounded real-operand capture and isolated affine projection replay.
use safemlx::{
    error::Exception,
    memory,
    ops::{self, QuantizationMode},
    Array, Dtype, Stream,
};
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    rc::Rc,
    time::Instant,
};

thread_local! { static CAPTURE: RefCell<Option<Collector>> = const { RefCell::new(None) }; }
struct Collector {
    operands: Vec<Operands>,
    limit: usize,
    bytes: usize,
    max_bytes: usize,
    overflow: bool,
}
#[derive(Debug)]
struct Operands {
    input: Array,
    weight: Array,
    scales: Array,
    biases: Array,
    output: Array,
    group: i32,
    bits: i32,
    site: &'static str,
    stream: Stream,
}
impl Operands {
    fn arrays(&self) -> [&Array; 5] {
        [
            &self.input,
            &self.weight,
            &self.scales,
            &self.biases,
            &self.output,
        ]
    }
    fn parameter_bytes(&self) -> usize {
        self.weight.nbytes() + self.scales.nbytes() + self.biases.nbytes()
    }
    fn project(&self) -> Result<Array, Exception> {
        ops::quantized_matmul_with_mode(
            &self.input,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            self.group,
            self.bits,
            QuantizationMode::Affine,
            &self.stream,
        )
    }
    fn padded(&self) -> Result<Option<Self>, Exception> {
        if !cfg!(all(feature = "metal", not(feature = "cuda")))
            || self.stream.get_device()?.get_type()? != safemlx::DeviceType::Gpu
        {
            return Ok(None);
        }
        let width = self.input.dim(-1);
        if self.bits != 8
            || self.group != 64
            || self.input.size() != width as usize
            || self.weight.dim(0) % 8 != 0
            || width % 512 == 0
            || width % 64 != 0
        {
            return Ok(None);
        }
        let padded = width
            .checked_add(511)
            .ok_or_else(|| Exception::custom("projection padding overflow"))?
            / 512
            * 512;
        let extra = padded - width;
        let mut input_pads = vec![(0, 0); self.input.ndim()];
        input_pads.last_mut().unwrap().1 = extra;
        let input = ops::pad(&self.input, input_pads.as_slice(), None, None, &self.stream)?;
        let weight = ops::pad(
            &self.weight,
            &[(0, 0), (0, extra / 4)],
            None,
            None,
            &self.stream,
        )?;
        let scales = ops::pad(
            &self.scales,
            &[(0, 0), (0, extra / 64)],
            None,
            None,
            &self.stream,
        )?;
        let biases = ops::pad(
            &self.biases,
            &[(0, 0), (0, extra / 64)],
            None,
            None,
            &self.stream,
        )?;
        safemlx::transforms::eval([&input, &weight, &scales, &biases])?;
        Ok(Some(Self {
            input,
            weight,
            scales,
            biases,
            output: self.output.clone(),
            group: self.group,
            bits: self.bits,
            site: self.site,
            stream: self.stream.clone(),
        }))
    }
}

/// Thread-affine diagnostic capture. Hooks never evaluate arrays or alter dispatch.
/// Retained-byte admission counts every invocation conservatively, including aliases.
#[derive(Debug)]
pub struct AffineCapture(PhantomData<Rc<()>>);
impl AffineCapture {
    /// Reject nesting and zero limits. Overflow fails at finish rather than truncating.
    pub fn begin(max_invocations: usize, max_retained_bytes: usize) -> Result<Self, Exception> {
        CAPTURE.with(|cell| {
            let mut capture = cell.borrow_mut();
            if capture.is_some() || max_invocations == 0 || max_retained_bytes == 0 {
                return Err(Exception::custom("invalid or nested affine capture"));
            }
            *capture = Some(Collector {
                operands: Vec::new(),
                limit: max_invocations,
                bytes: 0,
                max_bytes: max_retained_bytes,
                overflow: false,
            });
            Ok(Self(PhantomData))
        })
    }
    /// Settle captured roots and group by actual layouts, preserving one real operand
    /// set per distinct parameter identity for rotating-weight replay.
    pub fn finish(self) -> Result<Vec<AffineCase>, Exception> {
        let collector = CAPTURE
            .with(|cell| cell.borrow_mut().take())
            .ok_or_else(|| Exception::custom("affine capture inactive"))?;
        if collector.overflow {
            return Err(Exception::custom(
                "affine capture invocation or byte limit exceeded",
            ));
        }
        let mut cases = BTreeMap::<String, AffineCase>::new();
        for operands in collector.operands {
            safemlx::transforms::eval(operands.arrays())?;
            let key = format!(
                "{}:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
                format!("{}:{:?}", operands.site, operands.stream),
                operands.group,
                operands.bits,
                operands.input.shape(),
                operands.input.strides(),
                operands.input.dtype(),
                operands.weight.shape(),
                operands.weight.strides(),
                operands.scales.strides(),
                operands.scales.dtype(),
                operands.biases.strides(),
                operands.biases.dtype(),
                operands.output.dtype()
            );
            let id = (
                operands.weight.graph_identity(),
                operands.scales.graph_identity(),
                operands.biases.graph_identity(),
            );
            let case = cases.entry(key).or_insert_with(|| AffineCase {
                operands: Vec::new(),
                identities: BTreeSet::new(),
                invocations: 0,
            });
            case.invocations += 1;
            if case.identities.insert(id) {
                case.operands.push(operands);
            }
        }
        Ok(cases.into_values().collect())
    }
}
impl Drop for AffineCapture {
    fn drop(&mut self) {
        CAPTURE.with(|cell| {
            cell.borrow_mut().take();
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    output: &Array,
    group: i32,
    bits: i32,
    mode: QuantizationMode,
    site: &'static str,
    stream: &Stream,
) {
    CAPTURE.with(|cell| {
        let mut capture = cell.borrow_mut();
        let Some(c) = capture.as_mut() else { return };
        let Some(biases) = biases else { return };
        if mode != QuantizationMode::Affine || weight.ndim() != 2 || input.ndim() == 0 {
            return;
        }
        let bytes = [input, weight, scales, biases, output]
            .iter()
            .try_fold(0usize, |n, a| n.checked_add(a.nbytes()));
        let total = bytes.and_then(|n| c.bytes.checked_add(n));
        if c.operands.len() >= c.limit || total.is_none_or(|n| n > c.max_bytes) {
            c.overflow = true;
            return;
        }
        c.bytes = total.unwrap();
        c.operands.push(Operands {
            input: input.clone(),
            weight: weight.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
            output: output.clone(),
            group,
            bits,
            site,
            stream: stream.clone(),
        });
    });
}

/// One settled affine projection class with distinct real parameters retained.
#[derive(Debug)]
pub struct AffineCase {
    operands: Vec<Operands>,
    identities: BTreeSet<(usize, usize, usize)>,
    invocations: usize,
}
impl AffineCase {
    /// Layout, parameter payload and work estimates. Payload is not measured DRAM traffic.
    pub fn describe(&self) -> Value {
        let o = &self.operands[0];
        let rows = o.input.size() / o.input.dim(-1) as usize;
        let n = o.weight.dim(0) as usize;
        let k = o.input.dim(-1) as usize;
        json!({"site":o.site,"stream":format!("{:?}",o.stream),"invocations":self.invocations,"distinct_parameter_sets":self.operands.len(),"input_shape":o.input.shape(),"input_strides":o.input.strides(),"input_dtype":format!("{:?}",o.input.dtype()),"weight_shape":o.weight.shape(),"weight_strides":o.weight.strides(),"scale_strides":o.scales.strides(),"scale_dtype":format!("{:?}",o.scales.dtype()),"bias_strides":o.biases.strides(),"bias_dtype":format!("{:?}",o.biases.dtype()),"output_dtype":format!("{:?}",o.output.dtype()),"bits":o.bits,"group_size":o.group,"logical_width":k,"logical_outputs":n,"rows":rows,"parameter_payload_bytes":o.parameter_bytes(),"logical_dot_flops":2.0*rows as f64*n as f64*k as f64,"rotating_parameter_payload_bytes":self.operands.iter().map(Operands::parameter_bytes).sum::<usize>()})
    }
    /// Replay native dispatch and eligible zero-padded native fast Q8, with fixed
    /// and rotating real weights. Padding is settled before timing and charged separately.
    /// Metal native timing sums command-buffer intervals that may overlap or include
    /// waits, and can exceed wall time. These are not elapsed GPU, per-shader or
    /// full-model times; compare wall time and full-conversation controls.
    pub fn replay(&self, samples: usize, label: &str) -> Result<Value, Exception> {
        if samples == 0 || samples > 1000 {
            return Err(Exception::custom("replay samples must be in 1..=1000"));
        }
        let mut results = Vec::new();
        for rotating in [false, true] {
            let sources = if rotating {
                self.operands.as_slice()
            } else {
                &self.operands[..1]
            };
            for padded in [false, true] {
                let padded_sources = if padded {
                    sources
                        .iter()
                        .map(Operands::padded)
                        .collect::<Result<Option<Vec<_>>, _>>()?
                } else {
                    None
                };
                if padded && padded_sources.is_none() {
                    continue;
                }
                let sources = padded_sources.as_deref().unwrap_or(sources);
                let phase = if padded {
                    "native_padded_fast_q8"
                } else {
                    "native"
                };
                let stream = &sources[0].stream;
                let expected = sources
                    .iter()
                    .map(|o| super::bits(&o.output.as_dtype(Dtype::Float32, &o.stream)?))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut measurements = Vec::new();
                let mut comparison = Value::Null;
                for iteration in 0..samples + 2 {
                    stream.synchronize()?;
                    let baseline = memory::active_memory()?;
                    memory::reset_peak_memory()?;
                    eprintln!("[affine-phase-begin] {label} {phase} rotating={rotating} iteration={iteration}");
                    let start = Instant::now();
                    let outputs = sources
                        .iter()
                        .map(Operands::project)
                        .collect::<Result<Vec<_>, _>>()?;
                    let build_ms = start.elapsed().as_secs_f64() * 1000.;
                    let timing = safemlx::transforms::async_eval_timed(outputs.iter(), stream)?;
                    let stats = timing.summary()?;
                    let gpu_ms = stats.sum.as_secs_f64() * 1000.;
                    stream.synchronize()?;
                    let wall_ms = start.elapsed().as_secs_f64() * 1000.;
                    let peak = memory::peak_memory()?.saturating_sub(baseline);
                    eprintln!("[affine-phase-end] {label} {phase} rotating={rotating} iteration={iteration}");
                    if iteration == 0 {
                        comparison = compare(&outputs, &expected, stream)?;
                        if !padded && comparison["bitwise_equal"] != true {
                            return Err(Exception::custom(
                                "native affine replay differs from capture",
                            ));
                        }
                    }
                    if iteration != 1 {
                        measurements.push(json!({"first_sample":iteration==0,"projection_count":sources.len(),"build_wall_ms":build_ms,"total_wall_ms":wall_ms,"gpu_evaluation_ms":gpu_ms,"native_interval_union_ms":stats.occupied.as_secs_f64()*1000.,"native_interval_span_ms":stats.span.as_secs_f64()*1000.,"native_interval_count":stats.interval_count,"missing_timestamps":stats.missing_timestamps,"gpu_ms_per_projection":gpu_ms/sources.len() as f64,"baseline_active_bytes":baseline,"peak_growth_bytes":peak}));
                    }
                }
                results.push(json!({"phase":phase,"rotating_weights":rotating,"parameter_payload_bytes":sources.iter().map(Operands::parameter_bytes).sum::<usize>(),"padded_width":sources[0].input.dim(-1),"comparison":comparison,"samples":measurements}));
            }
        }
        Ok(
            json!({"label":label,"captured":self.describe(),"phases":results,"measurement":"isolated_evaluation_not_shader_or_end_to_end","native_timing_note":"On Metal, native timing sums command-buffer intervals that can overlap and include waits; the sum can exceed wall time and is not elapsed GPU execution time.","native_padded_fast_q8_replayed":results.iter().any(|phase| phase["phase"]=="native_padded_fast_q8"),"parameter_payload_is_not_measured_memory_traffic":true}),
        )
    }
}
fn compare(outputs: &[Array], expected: &[Vec<u32>], stream: &Stream) -> Result<Value, Exception> {
    let mut equal = true;
    let mut finite = true;
    let mut max_abs = 0f64;
    let mut max_scaled = 0f64;
    let mut fingerprints = Vec::new();
    for (output, expected) in outputs.iter().zip(expected) {
        let actual = super::bits(&output.as_dtype(Dtype::Float32, stream)?)?;
        if actual.len() != expected.len() {
            return Err(Exception::custom("affine replay output geometry differs"));
        }
        fingerprints.push(super::fingerprint(&actual));
        equal &= actual == *expected;
        for (&a, &b) in actual.iter().zip(expected) {
            let a = f32::from_bits(a) as f64;
            let b = f32::from_bits(b) as f64;
            finite &= a.is_finite() && b.is_finite();
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_scaled = max_scaled.max(d / (1. + b.abs()));
        }
    }
    Ok(
        json!({"bitwise_equal":equal,"finite":finite,"max_abs_error":max_abs,"max_error_over_one_plus_abs_reference":max_scaled,"output_f32_bits_sha256":fingerprints}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_overflow_is_explicit_and_drop_releases_capture() {
        let stream = crate::test_stream();
        let input = Array::from_slice(&[1f32; 64], &[64]);
        let weight = Array::from_slice(&[0u32; 16], &[1, 16]);
        let scale = Array::from_slice(&[1f32], &[1, 1]);
        let output = Array::from_slice(&[0f32], &[1]);
        let capture = AffineCapture::begin(10, 1).unwrap();
        record(
            &input,
            &weight,
            &scale,
            Some(&scale),
            &output,
            64,
            8,
            QuantizationMode::Affine,
            "linear",
            stream,
        );
        assert!(capture
            .finish()
            .unwrap_err()
            .to_string()
            .contains("byte limit exceeded"));
        drop(AffineCapture::begin(10, 1024).unwrap());
        assert!(AffineCapture::begin(10, 1024)
            .unwrap()
            .finish()
            .unwrap()
            .is_empty());
    }
    #[test]
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    fn real_strided_operands_rotate_and_padded_fast_matches_independent_dot() {
        use safemlx::{ops::indexing::TryIndexOp, Device, DeviceType};
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let width = 576;
        let rows = 16;
        let x = (0..width * 2)
            .map(|c| (c % 17) as f32 / 16. - 0.5)
            .collect::<Vec<_>>();
        let input = Array::from_slice(&x, &[width, 2])
            .try_index_device((.., 0), &stream)
            .unwrap();
        let capture = AffineCapture::begin(3, 1024 * 1024).unwrap();
        assert!(AffineCapture::begin(1, 1).is_err());
        for variant in 0..2 {
            let words = (0..rows * width / 4)
                .map(|i| {
                    let mut v = 0u32;
                    for byte in 0..4 {
                        v |= ((i + byte * 7 + variant * 11) as u32 & 255) << (byte * 8);
                    }
                    v
                })
                .collect::<Vec<_>>();
            let scales = vec![1f32 / 256.; (rows * width / 64) as usize];
            let biases = vec![-0.5f32; scales.len()];
            let w = Array::from_slice(&words, &[rows, width / 4]);
            let s = Array::from_slice(&scales, &[rows, width / 64]);
            let b = Array::from_slice(&biases, &[rows, width / 64]);
            let y = ops::quantized_matmul_with_mode(
                &input,
                &w,
                &s,
                Some(&b),
                true,
                64,
                8,
                QuantizationMode::Affine,
                &stream,
            )
            .unwrap();
            let observed = y.clone().into_evaluated().unwrap();
            for row in 0..rows as usize {
                let mut expected = 0f64;
                for c in 0..width as usize {
                    let q =
                        ((words[row * width as usize / 4 + c / 4] >> (8 * (c % 4))) & 255) as f64;
                    expected += x[c * 2] as f64 * (q / 256. - 0.5);
                }
                assert!(
                    (observed.as_slice::<f32>()[row] as f64 - expected).abs()
                        < 1e-4 * (1. + expected.abs())
                );
            }
            record(
                &input,
                &w,
                &s,
                Some(&b),
                &y,
                64,
                8,
                QuantizationMode::Affine,
                "linear",
                &stream,
            );
            if variant == 0 {
                record(
                    &input,
                    &w,
                    &s,
                    Some(&b),
                    &y,
                    64,
                    8,
                    QuantizationMode::Affine,
                    "linear",
                    &stream,
                );
            }
        }
        let cases = capture.finish().unwrap();
        assert_eq!(cases.len(), 1);
        let replay = cases[0].replay(1, "fixture").unwrap();
        let phases = replay["phases"].as_array().unwrap();
        assert_eq!(phases.len(), 4);
        for phase in phases {
            assert_eq!(phase["comparison"]["finite"], true);
            assert!(
                phase["comparison"]["max_error_over_one_plus_abs_reference"]
                    .as_f64()
                    .unwrap()
                    < 1e-4
            );
            if phase["rotating_weights"] == true {
                assert_eq!(phase["samples"][0]["projection_count"], 2);
            }
        }
    }
}
