//! Application-selected diagnostic capture; replay occurs after conversation drain.
use anyhow::{ensure, Result};
use eredu_backend_mlx::backend::nn::projection_profile::{
    affine::{AffineCapture, AffineCase},
    operations::{OperationCapture, OperationCase},
    GpuCapture,
};
use serde_json::json;
use std::{fs::OpenOptions, path::PathBuf};

#[derive(Clone)]
pub struct Options {
    pub output: PathBuf,
    pub advance: u64,
    pub samples: usize,
    pub metal: Option<PathBuf>,
    pub metal_case: usize,
}
pub struct Profiler {
    options: Options,
    capture: Option<AffineCapture>,
    cases: Option<Vec<AffineCase>>,
    operation_capture: Option<OperationCapture>,
    operations: Option<Vec<OperationCase>>,
}
impl Profiler {
    pub fn new(options: Options) -> Result<Self> {
        ensure!(
            options.advance > 0 && (1..=1000).contains(&options.samples),
            "invalid profiling advance/samples"
        );
        ensure!(
            !options.output.try_exists()?,
            "projection profile output already exists"
        );
        if let Some(path) = &options.metal {
            ensure!(!path.try_exists()?, "Metal capture already exists");
        }
        Ok(Self {
            options,
            capture: None,
            cases: None,
            operation_capture: None,
            operations: None,
        })
    }
    pub fn before(&mut self, advance: u64) -> Result<()> {
        if advance == self.options.advance {
            self.capture = Some(AffineCapture::begin(2048, 32 * 1024 * 1024 * 1024)?);
            self.operation_capture = Some(OperationCapture::begin(2048, 4 * 1024 * 1024 * 1024)?);
        }
        Ok(())
    }
    pub fn after(&mut self, advance: u64) -> Result<()> {
        if advance == self.options.advance {
            self.cases = Some(self.capture.take().expect("capture begun").finish()?);
            self.operations = Some(
                self.operation_capture
                    .take()
                    .expect("operation capture begun")
                    .finish()?,
            );
        }
        Ok(())
    }
    pub fn finish(self) -> Result<()> {
        let cases = self
            .cases
            .ok_or_else(|| anyhow::anyhow!("profile advance not reached"))?;
        ensure!(
            !cases.is_empty(),
            "no affine projections captured at requested advance"
        );
        ensure!(
            self.options.metal.is_none() || self.options.metal_case < cases.len(),
            "Metal capture class index out of range"
        );
        let mut replays = Vec::new();
        for (index, case) in cases.iter().enumerate() {
            // Export one explicitly selected class; rotating sets can make captures large.
            let capture = if index == self.options.metal_case {
                self.options
                    .metal
                    .as_ref()
                    .map(GpuCapture::begin)
                    .transpose()?
            } else {
                None
            };
            replays.push(case.replay(self.options.samples, &format!("case-{index}"))?);
            if let Some(capture) = capture {
                capture.finish()?;
            }
        }
        let operation_replays = self
            .operations
            .ok_or_else(|| anyhow::anyhow!("operation profile advance not reached"))?
            .iter()
            .map(|case| case.replay(self.options.samples))
            .collect::<Result<Vec<_>, _>>()?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.options.output)?;
        serde_json::to_writer_pretty(
            file,
            &json!({"schema_version":1,"advance":self.options.advance,"samples":self.options.samples,"metal_capture":self.options.metal,"metal_capture_case":self.options.metal_case,"capture_run_timings_valid_for_ordinary_inference":false,"replays":replays,"operation_replays":operation_replays,"notes":["Capture retains additional real array roots and settles them at its explicit finish boundary; conversation timings/memory are perturbed.","Replay happens after drain, with settled inputs; fixed and rotating real parameters expose different cache reuse.","Metal interval sum can overlap and exceed wall time; union removes overlap and span includes gaps. All can include buffer waits and are not individual shader duration. Use wall time and full-conversation controls for speedup claims.","Padded parameters are prepared outside timing and their full payload is charged. Payload bytes are not measured DRAM traffic.","Optional Metal capture instruments the selected class only; its timings are perturbed further. Inspect dispatches in Xcode."]}),
        )?;
        Ok(())
    }
}
