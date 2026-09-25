//! Opt-in diagnostic capture. Holding samples changes graph retention: never use
//! capture-run memory or time as ordinary inference measurements. Replay settles
//! inputs first and measures one operation at a time in an otherwise idle process.
use std::{cell::RefCell, collections::BTreeMap, time::Instant};

use safemlx::{error::Exception, memory, ops::indexing::TryIndexOp, Array, Dtype, Stream};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

thread_local! {
    static CAPTURE: RefCell<Option<Collector>> = const { RefCell::new(None) };
}
struct Collector {
    cases: Vec<ProjectionCase>,
    limit: usize,
    overflow: bool,
}

/// A bounded, same-thread diagnostic capture. Dropping it cancels collection.
/// No native evaluation or synchronization occurs in the collection hooks.
#[derive(Debug)]
pub struct ProjectionCapture(std::marker::PhantomData<std::rc::Rc<()>>);
impl ProjectionCapture {
    /// Begin collection; reject nested captures and a zero invocation limit.
    pub fn begin(max_invocations: usize) -> Result<Self, Exception> {
        CAPTURE.with(|cell| {
            let mut capture = cell.borrow_mut();
            if capture.is_some() || max_invocations == 0 {
                return Err(Exception::custom("invalid or nested projection capture"));
            }
            *capture = Some(Collector {
                cases: Vec::new(),
                limit: max_invocations,
                overflow: false,
            });
            Ok(Self(std::marker::PhantomData))
        })
    }

    /// Stop collection, returning the first real input/weight/output per class.
    /// Arrays are settled here before grouping by actual shapes AND strides.
    /// An exceeded limit is an error, never silently incomplete coverage.
    pub fn finish(self) -> Result<Vec<ProjectionCase>, Exception> {
        let collector = CAPTURE
            .with(|cell| cell.borrow_mut().take())
            .ok_or_else(|| Exception::custom("projection capture is not active"))?;
        if collector.overflow {
            return Err(Exception::custom(
                "projection capture invocation limit exceeded",
            ));
        }
        let mut classes: BTreeMap<String, ProjectionCase> = BTreeMap::new();
        for case in collector.cases {
            safemlx::transforms::eval([&case.input, &case.weight, &case.output])?;
            let key = format!(
                "{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}",
                case.site,
                case.path,
                case.input.shape(),
                case.input.strides(),
                case.weight.shape(),
                case.weight.strides(),
                case.weight.dtype()
            );
            if let Some(previous) = classes.get_mut(&key) {
                previous.invocations += 1;
            } else {
                classes.insert(key, case);
            }
        }
        Ok(classes.into_values().collect())
    }
}
impl Drop for ProjectionCapture {
    fn drop(&mut self) {
        CAPTURE.with(|cell| {
            cell.borrow_mut().take();
        });
    }
}

/// One observed mixed-width projection class, retaining its first actual arrays.
#[derive(Debug)]
pub struct ProjectionCase {
    input: Array,
    weight: Array,
    output: Array,
    site: &'static str,
    path: &'static str,
    invocations: usize,
    stream: Stream,
}

pub(crate) fn record(
    input: &Array,
    weight: &Array,
    output: &Array,
    site: &'static str,
    path: &'static str,
    stream: &Stream,
) {
    CAPTURE.with(|cell| {
        let mut capture = cell.borrow_mut();
        let Some(capture) = capture.as_mut() else {
            return;
        };
        if input.dtype() != Dtype::Float32
            || !matches!(weight.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        {
            return;
        }
        // Lazy strides are provisional. Keep the bounded invocation set, then
        // group by real settled layouts at finish, without changing submission.
        if capture.cases.len() < capture.limit {
            capture.cases.push(ProjectionCase {
                input: input.clone(),
                weight: weight.clone(),
                output: output.clone(),
                site,
                path,
                invocations: 1,
                stream: stream.clone(),
            });
        } else {
            capture.overflow = true;
        }
    });
}

fn bits(array: &Array) -> Result<Vec<u32>, Exception> {
    Ok(array
        .clone()
        .into_evaluated()?
        .try_as_slice::<f32>()
        .map_err(|error| Exception::custom(error.to_string()))?
        .iter()
        .map(|x| x.to_bits())
        .collect())
}
fn fingerprint(bits: &[u32]) -> String {
    let mut hash = Sha256::new();
    for value in bits {
        hash.update(value.to_le_bytes());
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl ProjectionCase {
    /// Number of actual logical input rows in this captured invocation.
    pub fn rows(&self) -> usize {
        self.input.size() / self.input.dim(-1) as usize
    }

    /// Settled shapes/layouts and counts. The representative is the first
    /// invocation, not an assertion that every equal-shaped parameter is identical.
    pub fn describe(&self) -> Result<Value, Exception> {
        safemlx::transforms::eval([&self.input, &self.weight, &self.output])?;
        Ok(json!({
            "site": self.site, "observed_path": self.path, "invocations": self.invocations,
            "input_shape": self.input.shape(), "input_strides": self.input.strides(),
            "weight_shape": self.weight.shape(), "weight_strides": self.weight.strides(),
            "input_dtype": format!("{:?}", self.input.dtype()), "weight_dtype": format!("{:?}", self.weight.dtype()),
            "output_shape": self.output.shape(), "output_strides": self.output.strides(),
            "narrow_weight_bytes": self.weight.nbytes(), "f32_conversion_payload_bytes": self.weight.size() * 4,
        }))
    }

    /// Replay a real input (or its leading rows) through cast, preconverted GEMM,
    /// native mixed matmul and current Eredu dispatch. First replay samples are
    /// reported separately; collection/reference work may already warm kernels.
    /// Later samples follow one additional unreported warmup. Each output is
    /// bit-compared with the captured projection, or the explicit-F32 reference
    /// for a new row count. GPU duration excludes eager work during graph building;
    /// build and total wall durations retain that cost.
    pub fn replay(&self, rows: usize, samples: usize, label: &str) -> Result<Value, Exception> {
        self.replay_impl(rows, samples, label, false)
    }

    /// Replay baseline phases plus the loader-only native GEMM prototype.
    /// Unsupported cases are errors: this diagnostic must not silently benchmark
    /// the fallback while claiming to validate the new loader.
    pub fn replay_with_mixed_storage(
        &self,
        rows: usize,
        samples: usize,
        label: &str,
    ) -> Result<Value, Exception> {
        self.replay_impl(rows, samples, label, true)
    }

    fn replay_impl(
        &self,
        rows: usize,
        samples: usize,
        label: &str,
        mixed_storage: bool,
    ) -> Result<Value, Exception> {
        let stream = &self.stream;
        if rows == 0 || rows > self.rows() || samples == 0 {
            return Err(Exception::custom("invalid projection replay rows/samples"));
        }
        let description = self.describe()?;
        let input = if rows == self.rows() {
            self.input.clone()
        } else {
            self.input
                .reshape(&[-1, self.input.dim(-1)], stream)?
                .try_index_device((..rows as i32, ..), stream)?
        };
        safemlx::transforms::eval([&input])?;
        stream.synchronize()?;
        let converted = self.weight.as_dtype(Dtype::Float32, stream)?;
        safemlx::transforms::eval([&converted])?;
        let expected = if rows == self.rows() {
            bits(&self.output)?
        } else {
            bits(&safemlx::ops::matmul(
                &input,
                converted.transpose(stream)?,
                stream,
            )?)?
        };
        // Do not keep the reference conversion live during cast/mixed phases.
        drop(converted);
        stream.synchronize()?;
        let prototype_input = if mixed_storage {
            let flattened = input.reshape(&[-1, input.dim(-1)], stream)?;
            safemlx::transforms::eval([&flattened])?;
            Some(flattened)
        } else {
            None
        };
        let embedding = crate::nn::Embedding {
            weight: crate::module::PhysicalParam::new(self.weight.clone()),
        };
        let mut phases = Vec::new();
        let mut phase_names = vec![
            "cast",
            "preconverted_gemm",
            "native_mixed",
            "eredu_dispatch",
        ];
        if mixed_storage {
            phase_names.push("mixed_storage_prototype");
        }
        for phase in phase_names {
            let preconverted = if phase == "preconverted_gemm" {
                let weight = self.weight.as_dtype(Dtype::Float32, stream)?;
                safemlx::transforms::eval([&weight])?;
                Some(weight)
            } else {
                None
            };
            let mut measurements = Vec::new();
            for iteration in 0..samples + 2 {
                stream.synchronize()?;
                let baseline = memory::active_memory()?;
                memory::reset_peak_memory()?;
                eprintln!("[projection-phase-begin] {label} {rows} {phase} {iteration}");
                let start = Instant::now();
                let output = match phase {
                    "cast" => self.weight.as_dtype(Dtype::Float32, stream)?,
                    "preconverted_gemm" => safemlx::ops::matmul(
                        &input,
                        preconverted.as_ref().unwrap().transpose(stream)?,
                        stream,
                    )?,
                    "mixed_storage_prototype" => safemlx::fast::try_mixed_storage_gemm(
                        prototype_input.as_ref().unwrap(),
                        &self.weight,
                        stream,
                    )?
                    .ok_or_else(|| {
                        Exception::custom("mixed-storage prototype does not support this replay")
                    })?,
                    "native_mixed" => {
                        safemlx::ops::matmul(&input, self.weight.transpose(stream)?, stream)?
                    }
                    _ if self.site == "tied_embedding" => embedding.as_linear(&input, stream)?,
                    _ => super::linear::dense_projection(&input, &self.weight, stream)?,
                };
                let build_ms = start.elapsed().as_secs_f64() * 1000.0;
                let timing = safemlx::transforms::async_eval_timed([&output], stream)?;
                let gpu_ms = timing.elapsed()?.as_secs_f64() * 1000.0;
                stream.synchronize()?;
                let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
                let peak = memory::peak_memory()?.saturating_sub(baseline);
                let active_with_output = memory::active_memory()?.saturating_sub(baseline);
                eprintln!("[projection-phase-end] {label} {rows} {phase} {iteration}");
                if phase != "cast" && bits(&output)? != expected {
                    return Err(Exception::custom(format!(
                        "projection bits differ: {label} rows={rows} phase={phase}"
                    )));
                }
                drop(output);
                stream.synchronize()?;
                if iteration != 1 {
                    measurements.push(json!({
                        "first_sample": iteration == 0, "build_wall_ms": build_ms, "total_wall_ms": wall_ms,
                        "gpu_evaluation_ms": gpu_ms, "baseline_active_bytes": baseline,
                        "peak_growth_bytes": peak, "active_with_output_growth_bytes": active_with_output,
                        "active_after_drop_bytes": memory::active_memory()?, "cached_after_drop_bytes": memory::cache_memory()?,
                    }));
                }
            }
            phases.push(json!({"phase": phase, "samples": measurements,
                "preconverted_weight_bytes": preconverted.as_ref().map_or(0, Array::nbytes)}));
        }
        Ok(
            json!({"label": label, "captured": description, "rows": rows,
            "replay_input_shape": input.shape(), "replay_input_strides": input.strides(),
            "output_f32_bits_sha256": fingerprint(&expected), "bitwise_equal": true, "phases": phases}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_is_bounded_counts_classes_and_cancels_on_drop() {
        let input = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let stream = crate::test_stream();
        let weight = input.as_dtype(Dtype::Bfloat16, stream).unwrap();
        let capture = ProjectionCapture::begin(2).unwrap();
        assert!(ProjectionCapture::begin(1).is_err());
        record(&input, &weight, &input, "dense", "native_fallback", stream);
        record(&input, &weight, &input, "dense", "native_fallback", stream);
        let cases = capture.finish().unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].invocations, 2);
        let capture = ProjectionCapture::begin(1).unwrap();
        record(&input, &weight, &input, "dense", "native_fallback", stream);
        record(
            &input,
            &weight,
            &input,
            "tied_embedding",
            "native_fallback",
            stream,
        );
        assert!(capture.finish().is_err());
        drop(ProjectionCapture::begin(1).unwrap());
        assert!(ProjectionCapture::begin(1)
            .unwrap()
            .finish()
            .unwrap()
            .is_empty());
    }
}
