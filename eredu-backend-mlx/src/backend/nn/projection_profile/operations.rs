//! Bounded capture of immutable native attention products and cache concatenations.
use safemlx::{error::Exception, memory, ops, Array, Dtype, Stream};
use serde_json::{json, Value};
use std::{cell::RefCell, collections::BTreeMap, marker::PhantomData, rc::Rc, time::Instant};

thread_local! { static CAPTURE: RefCell<Option<Collector>> = const { RefCell::new(None) }; }
struct Collector {
    records: Vec<Operands>,
    bytes: usize,
    limit: usize,
    max_bytes: usize,
    overflow: bool,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Operation {
    AttentionProduct,
    Concatenate(i32),
}
#[derive(Debug)]
struct Operands {
    operation: Operation,
    role: &'static str,
    lhs: Array,
    rhs: Array,
    output: Array,
    stream: Stream,
}
impl Operands {
    fn project(&self) -> Result<Array, Exception> {
        match self.operation {
            Operation::AttentionProduct => {
                super::super::attention::grouped_head_product(&self.lhs, &self.rhs, &self.stream)
            }
            Operation::Concatenate(axis) => {
                ops::concatenate_axis(&[&self.lhs, &self.rhs], axis, &self.stream)
            }
        }
    }
}
/// Thread-affine capture of native immutable operands. Overflow is explicit.
#[derive(Debug)]
pub struct OperationCapture(PhantomData<Rc<()>>);
impl OperationCapture {
    /// Bounds invocation count and conservative retained payload, including aliases.
    pub fn begin(limit: usize, max_bytes: usize) -> Result<Self, Exception> {
        CAPTURE.with(|cell| {
            if limit == 0 || max_bytes == 0 || cell.borrow().is_some() {
                return Err(Exception::custom("invalid or nested operation capture"));
            }
            *cell.borrow_mut() = Some(Collector {
                records: Vec::new(),
                bytes: 0,
                limit,
                max_bytes,
                overflow: false,
            });
            Ok(Self(PhantomData))
        })
    }
    /// Settles immutable operands and groups actual native layouts without deduplicating calls.
    pub fn finish(self) -> Result<Vec<OperationCase>, Exception> {
        let collector = CAPTURE
            .with(|cell| cell.borrow_mut().take())
            .ok_or_else(|| Exception::custom("operation capture inactive"))?;
        if collector.overflow {
            return Err(Exception::custom(
                "operation capture invocation or byte limit exceeded",
            ));
        }
        let mut cases = BTreeMap::<String, OperationCase>::new();
        for operands in collector.records {
            safemlx::transforms::eval([&operands.lhs, &operands.rhs, &operands.output])?;
            let key = format!(
                "{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
                operands.role,
                operands.operation,
                operands.stream,
                operands.lhs.shape(),
                operands.lhs.strides(),
                operands.lhs.dtype(),
                operands.rhs.shape(),
                operands.rhs.strides(),
                operands.rhs.dtype(),
                operands.output.shape(),
                operands.output.dtype()
            );
            cases
                .entry(key)
                .or_insert_with(|| OperationCase(Vec::new()))
                .0
                .push(operands);
        }
        Ok(cases.into_values().collect())
    }
}
impl Drop for OperationCapture {
    fn drop(&mut self) {
        CAPTURE.with(|cell| {
            cell.borrow_mut().take();
        });
    }
}

pub(crate) fn record(
    operation: Operation,
    role: &'static str,
    lhs: &Array,
    rhs: &Array,
    output: &Array,
    stream: &Stream,
) {
    CAPTURE.with(|cell| {
        let mut capture = cell.borrow_mut();
        let Some(c) = capture.as_mut() else { return };
        let total = [lhs, rhs, output]
            .iter()
            .try_fold(c.bytes, |n, a| n.checked_add(a.nbytes()));
        if c.records.len() >= c.limit || total.is_none_or(|n| n > c.max_bytes) {
            c.overflow = true;
            return;
        }
        c.bytes = total.unwrap();
        c.records.push(Operands {
            operation,
            role,
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            output: output.clone(),
            stream: stream.clone(),
        });
    });
}

/// A native operation layout class, retaining every real invocation's operands.
#[derive(Debug)]
pub struct OperationCase(Vec<Operands>);
impl OperationCase {
    /// Actual logical payload/layout. Bytes are not measured DRAM traffic.
    pub fn describe(&self) -> Value {
        let o = &self.0[0];
        let logical = self
            .0
            .iter()
            .map(|o| o.lhs.nbytes() + o.rhs.nbytes() + o.output.nbytes())
            .sum::<usize>();
        json!({"operation":format!("{:?}",o.operation),"role":o.role,"stream":format!("{:?}",o.stream),"invocations":self.0.len(),"lhs_shape":o.lhs.shape(),"lhs_strides":o.lhs.strides(),"lhs_dtype":format!("{:?}",o.lhs.dtype()),"rhs_shape":o.rhs.shape(),"rhs_strides":o.rhs.strides(),"rhs_dtype":format!("{:?}",o.rhs.dtype()),"output_shape":o.output.shape(),"output_dtype":format!("{:?}",o.output.dtype()),"logical_input_output_payload_bytes":logical,"payload_is_not_measured_dram_traffic":true})
    }
    /// Replays actual independent operands after settling. Exact output bits are checked.
    /// Union/span remove interval overlap/include gaps respectively, but retain stalls.
    /// These isolated intervals are not additive whole-model or shader attribution.
    pub fn replay(&self, samples: usize) -> Result<Value, Exception> {
        if samples == 0 || samples > 1000 {
            return Err(Exception::custom("operation samples must be in 1..=1000"));
        }
        let stream = &self.0[0].stream;
        let expected = self
            .0
            .iter()
            .map(|o| super::bits(&o.output.as_dtype(Dtype::Float32, stream)?))
            .collect::<Result<Vec<_>, _>>()?;
        let mut measurements = Vec::new();
        for iteration in 0..samples + 2 {
            stream.synchronize()?;
            let baseline = memory::active_memory()?;
            memory::reset_peak_memory()?;
            let start = Instant::now();
            let outputs = self
                .0
                .iter()
                .map(Operands::project)
                .collect::<Result<Vec<_>, _>>()?;
            let build_ms = start.elapsed().as_secs_f64() * 1000.;
            let timed = safemlx::transforms::async_eval_timed(outputs.iter(), stream)?;
            let stats = timed.summary()?;
            stream.synchronize()?;
            let wall_ms = start.elapsed().as_secs_f64() * 1000.;
            let peak = memory::peak_memory()?.saturating_sub(baseline);
            if iteration == 0 {
                for (output, reference) in outputs.iter().zip(&expected) {
                    if super::bits(&output.as_dtype(Dtype::Float32, stream)?)? != *reference {
                        return Err(Exception::custom(
                            "native operation replay differs from capture",
                        ));
                    }
                }
            }
            if iteration != 1 {
                measurements.push(json!({"first_sample":iteration==0,"operation_count":outputs.len(),"build_wall_ms":build_ms,"total_wall_ms":wall_ms,"native_interval_sum_ms":stats.sum.as_secs_f64()*1000.,"native_interval_union_ms":stats.occupied.as_secs_f64()*1000.,"native_interval_span_ms":stats.span.as_secs_f64()*1000.,"native_interval_count":stats.interval_count,"missing_timestamps":stats.missing_timestamps,"peak_growth_bytes":peak}));
            }
        }
        Ok(
            json!({"captured":self.describe(),"bitwise_equal":true,"samples":measurements,"measurement":"isolated_real_operands_not_shader_or_end_to_end"}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_operation_capture_replays_nonzero_products_and_append() {
        let stream = crate::test_stream();
        let lhs = Array::from_slice(&[1f32, 2., 3., 4.], &[1, 1, 1, 4]);
        let rhs = Array::from_slice(&[1f32, -1., 2., 0., 3., 1., 4., 2.], &[1, 1, 4, 2]);
        let product =
            super::super::super::attention::grouped_head_product(&lhs, &rhs, stream).unwrap();
        assert_eq!(
            product.clone().into_evaluated().unwrap().as_slice::<f32>(),
            &[30., 10.]
        );
        let failed = OperationCapture::begin(1, 1).unwrap();
        record(
            Operation::AttentionProduct,
            "product",
            &lhs,
            &rhs,
            &product,
            stream,
        );
        assert!(failed
            .finish()
            .unwrap_err()
            .to_string()
            .contains("byte limit exceeded"));
        drop(OperationCapture::begin(2, 4096).unwrap());
        let capture = OperationCapture::begin(2, 4096).unwrap();
        assert!(OperationCapture::begin(2, 4096).is_err());
        record(
            Operation::AttentionProduct,
            "product",
            &lhs,
            &rhs,
            &product,
            stream,
        );
        let appended = Array::from_slice(&[5f32, 6., 7., 8.], &[1, 1, 1, 4]);
        let output = ops::concatenate_axis(&[&lhs, &appended], -2, stream).unwrap();
        assert_eq!(
            output.clone().into_evaluated().unwrap().as_slice::<f32>(),
            &[1., 2., 3., 4., 5., 6., 7., 8.]
        );
        record(
            Operation::Concatenate(-2),
            "append",
            &lhs,
            &appended,
            &output,
            stream,
        );
        let cases = capture.finish().unwrap();
        assert_eq!(cases.len(), 2);
        for case in cases {
            assert_eq!(case.replay(1).unwrap()["bitwise_equal"], true);
        }
        assert_eq!(
            lhs.into_evaluated().unwrap().as_slice::<f32>(),
            &[1., 2., 3., 4.]
        );
    }
}
