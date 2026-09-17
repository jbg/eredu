use super::*;
use eredu_nn::{AttentionMask, AttentionRequest, NeuralBackend, Tensor};

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
#[derive(Clone, Copy, Debug)]
struct Case {
    b: i32,
    h: i32,
    kv: i32,
    q: i32,
    k: i32,
    d: i32,
    v: i32,
    mask: u8,
    sinks: bool,
    cap: bool,
    input: bool,
    window: Option<(i32, i32)>,
}
impl Case {
    fn shapes(self) -> [[i32; 4]; 3] {
        [
            [self.b, self.h, self.q, self.d],
            [self.b, self.kv, self.k, self.d],
            [self.b, self.kv, self.k, self.v],
        ]
    }
    fn equation<B: NeuralBackend>(
        self,
        t: &[B::Tensor],
        mask: Option<&B::Tensor>,
        sinks: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let scale = (self.d as f32).sqrt().recip();
        if self.mask == 1 && self.window.is_none() {
            return B::Tensor::scaled_dot_product_attention(
                &t[0],
                &t[1],
                &t[2],
                scale,
                AttentionMask::Causal,
                context,
            );
        }
        let request = AttentionRequest {
            queries: t[0].clone(),
            keys: t[1].clone(),
            values: t[2].clone(),
            scale,
            mask,
            sinks,
            softcap: self.cap.then_some(0.5),
            arithmetic: if self.input {
                AttentionArithmetic::InputScores
            } else {
                AttentionArithmetic::Fused
            },
        };
        if let Some((window, offset)) = self.window {
            B::sliding_window_attention_with_sinks(request, window, offset, context)
        } else {
            B::attention_with_sinks(request, context)
        }
    }
    fn quote(self, selected: MlxMetalWorkspaceMechanisms) -> Result<WorkspaceTraceReport, Error> {
        let context = WorkspaceContext::new(selected);
        let tensor = |shape: &[i32], dtype| {
            WorkspaceTensor::existing(WorkspaceLayout::new(shape, dtype)?, &context)
        };
        let t = self
            .shapes()
            .iter()
            .map(|s| tensor(s, WorkspaceDtype::Float32))
            .collect::<Result<Vec<_>, _>>()?;
        let mask = if self.mask >= 2 {
            Some(tensor(
                &[self.q, self.k],
                if matches!(self.mask, 2 | 4) {
                    WorkspaceDtype::Bool
                } else {
                    WorkspaceDtype::Float32
                },
            )?)
        } else {
            None
        };
        let sinks = if self.sinks {
            Some(tensor(&[self.h], WorkspaceDtype::Float32)?)
        } else {
            None
        };
        let output =
            self.equation::<WorkspaceBackend>(&t, mask.as_ref(), sinks.as_ref(), &context)?;
        let expected = if self.window.is_some() {
            vec![self.b, self.q, self.h * self.v]
        } else {
            vec![self.b, self.h, self.q, self.v]
        };
        assert_eq!(output.shape(), expected);
        context.report(&[output])
    }
}
fn base() -> Case {
    Case {
        b: 1,
        h: 4,
        kv: 2,
        q: 1,
        k: 1024,
        d: 64,
        v: 64,
        mask: 0,
        sinks: false,
        cap: false,
        input: false,
        window: None,
    }
}
#[test]
fn attention_prices_selected_kernels_copies_masks_and_retained_scratch_override() {
    let short = Case { k: 1023, ..base() }
        .quote(mechanisms())
        .unwrap()
        .tensor_buffers
        .total_bytes
        .unwrap();
    let long = base()
        .quote(mechanisms())
        .unwrap()
        .tensor_buffers
        .total_bytes
        .unwrap();
    assert!(long > short);
    let mut override_facts = mechanisms();
    override_facts.sdpa_blocks = Some(2048);
    assert!(
        base()
            .quote(override_facts)
            .unwrap()
            .tensor_buffers
            .total_bytes
            .unwrap()
            > long
    );
    assert_eq!(
        Case { k: 1023, ..base() }
            .quote(override_facts)
            .unwrap()
            .tensor_buffers
            .total_bytes,
        Some(short)
    );
    for q in [1, 8, 9, 256, 257, 3000] {
        for input in [false, true] {
            for cap in [false, true] {
                let case = Case {
                    q,
                    k: q.max(37),
                    input,
                    cap,
                    window: Some((17, 36)),
                    ..base()
                };
                assert!(case
                    .quote(mechanisms())
                    .unwrap()
                    .tensor_buffers
                    .total_bytes
                    .is_some());
            }
        }
    }
    // A fused full kernel never prices a square score tensor when geometry
    // selects that implementation, even for a long prompt.
    let g = Geometry {
        b: 1,
        h: 4,
        kv: 2,
        q: 3000,
        k: 3000,
        d: 64,
        v: 64,
    };
    assert!(
        fused_cost(g, Mask::None, false, mechanisms().allocation, None).unwrap()
            < capacity(mechanisms().allocation, g.scores().unwrap()).unwrap()
    );
}
#[test]
fn malformed_attention_cannot_authorize_a_trace() {
    assert!(Case {
        q: 3,
        k: 2,
        window: Some((7, 1)),
        ..base()
    }
    .quote(mechanisms())
    .is_err());
    assert!(Case {
        q: 3,
        k: 3,
        window: Some((7, i32::MAX)),
        ..base()
    }
    .quote(mechanisms())
    .is_err());
    let shape = WorkspaceLayout::new(&[1, 4, 3, 64], WorkspaceDtype::Float32).unwrap();
    let bad = WorkspaceOperation {
        kind: WorkspaceOperationKind::Attention {
            causal: false,
            window: None,
            sinks: false,
            softcap: false,
            arithmetic: AttentionArithmetic::Fused,
        },
        inputs: vec![shape.clone(); 3],
        outputs: vec![WorkspaceLayout::new(&[1, 3, 256], WorkspaceDtype::Float32).unwrap()],
    };
    assert!(mechanisms().operation_bound(&bad).is_err());
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_attention_peaks_fit_bounds_and_values_match_independent_host_equation() {
    use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use safemlx::{
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let mut geometries = vec![];
    for (b, h, kv, q, k, d, v) in [
        (1, 4, 2, 1, 7, 64, 64),
        (1, 8, 2, 8, 17, 96, 96),
        (1, 8, 1, 8, 17, 96, 96),
        (2, 4, 2, 9, 19, 80, 80),
        (1, 4, 2, 9, 19, 96, 96),
        (1, 4, 2, 3, 5, 192, 128),
        (2, 4, 2, 3, 5, 32, 17),
        (1, 4, 2, 1, 1024, 64, 64),
        (1, 4, 2, 1, 8193, 64, 64),
        (1, 4, 2, 2, 4097, 32, 17),
        (1, 2, 2, 9, 65, 128, 128),
        (1, 1, 1, 1, 1024, 256, 256),
        (1, 4, 2, 1, 8192, 32, 17),
        (1, 4, 2, 3, 8448, 32, 17),
        (2, 2, 2, 2, 16385, 7, 5),
    ] {
        geometries.push(Case {
            b,
            h,
            kv,
            q,
            k,
            d,
            v,
            ..base()
        });
    }
    geometries.push(Case {
        q: 2,
        k: 9002,
        d: 32,
        v: 17,
        window: Some((9000, 10000)),
        ..base()
    });
    geometries.extend([
        Case {
            q: 17,
            k: 17,
            window: Some((32, 0)),
            ..base()
        },
        Case {
            q: 259,
            k: 277,
            window: Some((17, 18)),
            ..base()
        },
        Case {
            q: 3,
            k: 19,
            window: Some((1, 16)),
            ..base()
        },
    ]);
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for strided in [false, true] {
            for geometry in &geometries {
                let combinations = if geometry.window.is_some() {
                    vec![
                        (0, false, false, false),
                        (0, true, true, false),
                        (0, true, false, true),
                    ]
                } else if geometry.k > 8192 {
                    vec![
                        (0, false, false, false),
                        (2, true, false, false),
                        (3, true, false, false),
                        (0, false, false, true),
                        (2, true, false, true),
                        (3, true, true, true),
                        (4, false, false, true),
                        (5, true, true, true),
                    ]
                } else {
                    vec![
                        (0, false, false, false),
                        (1, false, false, false),
                        (2, true, false, false),
                        (3, true, false, false),
                        (2, true, true, false),
                        (3, true, true, true),
                    ]
                };
                for (mask_kind, sinks_enabled, cap, input) in combinations {
                    let case = Case {
                        mask: mask_kind,
                        sinks: sinks_enabled,
                        cap,
                        input,
                        ..*geometry
                    };
                    let values = case
                        .shapes()
                        .iter()
                        .enumerate()
                        .map(|(which, shape)| {
                            (0..shape.iter().product::<i32>() as usize)
                                .map(|i| (((i * 7 + which * 3) % 23) as f32 - 11.0) / 16.0)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let t = case
                        .shapes()
                        .iter()
                        .zip(&values)
                        .map(|(shape, values)| {
                            let array = if strided {
                                let mut storage = vec![0.0; values.len() * 2];
                                for (i, x) in values.iter().enumerate() {
                                    storage[2 * i] = *x;
                                }
                                let mut physical = *shape;
                                physical[3] *= 2;
                                let a = Array::from_slice(&storage, &physical)
                                    .as_dtype(dtype, &stream)
                                    .unwrap();
                                a.try_index_device((.., .., .., (..).stride_by(2)), &stream)
                                    .unwrap()
                            } else {
                                Array::from_slice(values, shape)
                                    .as_dtype(dtype, &stream)
                                    .unwrap()
                            };
                            MlxTensor::from_array(array)
                        })
                        .collect::<Vec<_>>();
                    let step = if strided { 2 } else { 1 };
                    let mask = if mask_kind >= 2 {
                        let shape = [case.q, case.k * step];
                        let size = (case.q * case.k * step) as usize;
                        let a = if matches!(mask_kind, 2 | 4) {
                            Array::from_slice(
                                &(0..size)
                                    .map(|i| {
                                        mask_kind != 4
                                            && (i / step as usize) % case.k as usize % 5 != 4
                                    })
                                    .collect::<Vec<_>>(),
                                &shape,
                            )
                        } else {
                            Array::from_slice(
                                &(0..size)
                                    .map(|i| {
                                        let logical = i / step as usize;
                                        if mask_kind == 5 || logical % case.k as usize % 5 == 4 {
                                            f32::NEG_INFINITY
                                        } else {
                                            -((logical % 3) as f32) / 8.0
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                                &shape,
                            )
                            .as_dtype(dtype, &stream)
                            .unwrap()
                        };
                        let a = if strided {
                            a.try_index_device((.., (..).stride_by(2)), &stream)
                                .unwrap()
                        } else {
                            a
                        };
                        Some(MlxTensor::from_array(a))
                    } else {
                        None
                    };
                    let sink = if sinks_enabled {
                        let a = Array::from_slice(
                            &(0..case.h * step)
                                .map(|h| -0.25 * (h / step) as f32)
                                .collect::<Vec<_>>(),
                            &[case.h * step],
                        )
                        .as_dtype(dtype, &stream)
                        .unwrap();
                        let a = if strided {
                            a.try_index_device((..).stride_by(2), &stream).unwrap()
                        } else {
                            a
                        };
                        Some(MlxTensor::from_array(a))
                    } else {
                        None
                    };
                    safemlx::transforms::eval(
                        t.iter()
                            .chain(mask.iter())
                            .chain(sink.iter())
                            .map(|t| t.as_array()),
                    )
                    .unwrap();
                    let allowed = case
                        .quote(selected)
                        .unwrap()
                        .tensor_buffers
                        .total_bytes
                        .unwrap();
                    stream.synchronize().unwrap();
                    let before = safemlx::memory::active_memory().unwrap();
                    safemlx::memory::reset_peak_memory().unwrap();
                    let mut ownership = safemlx::SubmissionScope::begin().unwrap();
                    let output = case
                        .equation::<MlxNeuralBackend>(&t, mask.as_ref(), sink.as_ref(), &stream)
                        .unwrap();
                    ownership.seal();
                    if case.input && case.k > INPUT_SCORE_ROW_BUDGET {
                        let status = ownership.status();
                        assert!(
                            status.is_settled() && !status.failed() && !status.blocked(),
                            "completed key blocks still retain native work: {status:?}"
                        );
                    }
                    safemlx::transforms::eval([output.as_array()]).unwrap();
                    stream.synchronize().unwrap();
                    let observed = safemlx::memory::peak_memory()
                        .unwrap()
                        .saturating_sub(before) as u64;
                    assert!(
                        observed <= allowed,
                        "{case:?} {dtype:?} strided={strided}: peak {observed}>{allowed}"
                    );
                    let actual = output.to_f32_vec(&stream).unwrap();
                    let mut error = 0.0_f64;
                    for b in 0..case.b {
                        for h in 0..case.h {
                            for q in 0..case.q {
                                let mut scores = vec![];
                                let mut max = f64::NEG_INFINITY;
                                for k in 0..case.k {
                                    let mut score = 0.0;
                                    for d in 0..case.d {
                                        let qi =
                                            (((b * case.h + h) * case.q + q) * case.d + d) as usize;
                                        let ki = (((b * case.kv + h / (case.h / case.kv)) * case.k
                                            + k)
                                            * case.d
                                            + d)
                                            as usize;
                                        score +=
                                            f64::from(values[0][qi]) * f64::from(values[1][ki]);
                                    }
                                    score /= (case.d as f64).sqrt();
                                    if cap {
                                        score = 0.5 * (score / 0.5).tanh();
                                    }
                                    let allowed = if let Some((window, offset)) = case.window {
                                        let abs_key = offset + case.q - case.k + k;
                                        abs_key <= offset + q && abs_key > offset + q - window
                                    } else if mask_kind == 1 {
                                        k <= case.k - case.q + q
                                    } else {
                                        mask_kind < 4 && (mask_kind < 2 || k % 5 != 4)
                                    };
                                    if !allowed {
                                        score = f64::NEG_INFINITY;
                                    } else if mask_kind == 3 {
                                        score -= f64::from((q * case.k + k) % 3) / 8.0;
                                    }
                                    max = max.max(score);
                                    scores.push(score);
                                }
                                if sinks_enabled {
                                    max = max.max(-0.25 * f64::from(h));
                                }
                                let mut sum = if sinks_enabled {
                                    (-0.25 * f64::from(h) - max).exp()
                                } else {
                                    0.0
                                };
                                for score in &mut scores {
                                    *score = if max.is_finite() {
                                        (*score - max).exp()
                                    } else {
                                        0.0
                                    };
                                    sum += *score;
                                }
                                for v in 0..case.v {
                                    let expected = scores
                                        .iter()
                                        .enumerate()
                                        .map(|(k, score)| {
                                            let vi = (((b * case.kv + h / (case.h / case.kv))
                                                * case.k
                                                + k as i32)
                                                * case.v
                                                + v)
                                                as usize;
                                            if sum > 0.0 {
                                                score / sum * f64::from(values[2][vi])
                                            } else {
                                                0.0
                                            }
                                        })
                                        .sum::<f64>();
                                    let index = if case.window.is_some() {
                                        (((b * case.q + q) * case.h + h) * case.v + v) as usize
                                    } else {
                                        (((b * case.h + h) * case.q + q) * case.v + v) as usize
                                    };
                                    let difference = (f64::from(actual[index]) - expected).abs();
                                    error = error.max(difference);
                                    assert!(difference<=0.003+0.03*expected.abs(),"{case:?} {dtype:?}: index {index} actual={} expected={expected} difference={difference}",actual[index]);
                                }
                            }
                        }
                    }
                    println!("ATTENTION_MEASUREMENT case={case:?} dtype={dtype:?} strided={strided} blocks={:?} peak={observed} bound={allowed} max_abs={error}",selected.sdpa_blocks);
                }
            }
        }
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_retained_scratch_overrides_fit_the_same_native_quote() {
    use safemlx::{Array, Device, DeviceType, Stream};
    let Ok(value) = std::env::var("EREDU_SDPA_TEST_BLOCKS") else {
        for blocks in [32, 256, 2048] {
            let child=std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact","backend::nn::workspace::attention::tests::metal_retained_scratch_overrides_fit_the_same_native_quote","--ignored","--nocapture"])
                .env("MLX_SDPA_BLOCKS",blocks.to_string()).env("EREDU_SDPA_TEST_BLOCKS",blocks.to_string()).output().unwrap();
            assert!(
                child.status.success(),
                "override {blocks}: {} {}",
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            );
            print!("{}", String::from_utf8_lossy(&child.stdout));
        }
        return;
    };
    let blocks = value.parse::<u32>().unwrap();
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    assert_eq!(selected.sdpa_blocks, Some(blocks));
    let case = base();
    let allowed = case
        .quote(selected)
        .unwrap()
        .tensor_buffers
        .total_bytes
        .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let t = case
        .shapes()
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            Array::from_slice(
                &vec![
                    if i == 2 { 0.25_f32 } else { 0.125 };
                    shape.iter().product::<i32>() as usize
                ],
                shape,
            )
        })
        .collect::<Vec<_>>();
    safemlx::transforms::eval(t.iter()).unwrap();
    stream.synchronize().unwrap();
    let before = safemlx::memory::active_memory().unwrap();
    safemlx::memory::reset_peak_memory().unwrap();
    let output = safemlx::fast::scaled_dot_product_attention(
        t[0].clone(),
        t[1].clone(),
        t[2].clone(),
        0.125,
        None,
        None,
        &stream,
    )
    .unwrap();
    safemlx::transforms::eval([&output]).unwrap();
    stream.synchronize().unwrap();
    let observed = safemlx::memory::peak_memory()
        .unwrap()
        .saturating_sub(before) as u64;
    assert!(
        observed <= allowed,
        "override {blocks}: {observed}>{allowed}"
    );
    assert!(crate::MlxTensor::from_array(output)
        .to_f32_vec(&stream)
        .unwrap()
        .iter()
        .all(|x| (*x - 0.25).abs() < 1e-5));
    println!("ATTENTION_OVERRIDE blocks={blocks} peak={observed} bound={allowed}");
}

#[test]
fn completed_key_blocks_bound_transients_independently_of_context_length() {
    for q in [1, 2, 17] {
        for mask in [0, 2, 3] {
            for sinks in [false, true] {
                for cap in [false, true] {
                    let case = Case {
                        q,
                        k: 8193,
                        mask,
                        sinks,
                        cap,
                        input: true,
                        ..base()
                    };
                    let first = case
                        .quote(mechanisms())
                        .unwrap()
                        .tensor_buffers
                        .total_bytes
                        .unwrap();
                    for k in [16385, 32769, 65537] {
                        let later = Case { k, ..case }
                            .quote(mechanisms())
                            .unwrap()
                            .tensor_buffers
                            .total_bytes
                            .unwrap();
                        assert_eq!(
                            first, later,
                            "context growth must not accumulate completed block scratch"
                        );
                    }
                }
            }
        }
    }
}
