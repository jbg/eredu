use super::display::Display;
use anyhow::Result;
use serde_json::Value;
use std::{
    io::{self, IsTerminal, Write},
    time::Instant,
};

pub struct Events {
    sink: Box<dyn Write>,
    start: Instant,
    pending: Vec<u8>,
    live: u64,
    timings: Vec<f64>,
    limit: u64,
    display: Display<io::Stderr>,
    interactive: bool,
}
impl Events {
    pub fn new(sink: Box<dyn Write>, limit: u64, show_transcription: bool) -> Result<Self> {
        let interactive = io::stdout().is_terminal() && io::stderr().is_terminal();
        Ok(Self {
            sink,
            start: Instant::now(),
            pending: Vec::new(),
            live: 0,
            timings: Vec::new(),
            limit,
            display: Display::new(io::stderr(), show_transcription, interactive),
            interactive,
        })
    }
    pub fn summary(&self) {
        if self.timings.is_empty() {
            return;
        }
        let mut sorted = self.timings.clone();
        sorted.sort_by(f64::total_cmp);
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let percentile = |p: f64| {
            sorted[((sorted.len() as f64 * p).ceil() as usize)
                .saturating_sub(1)
                .min(sorted.len() - 1)]
        };
        eprintln!(
            "warm live execution (first 10 frames excluded): mean={mean:.3} ms p95={:.3} ms p99={:.3} ms headroom={:.3} ms misses={}/{}; audio pacing excluded",
            percentile(0.95),
            percentile(0.99),
            80.0 - mean,
            sorted.iter().filter(|v| **v > 80.0).count(),
            sorted.len()
        );
    }
    fn line(&mut self) -> io::Result<()> {
        let mut value: Value = serde_json::from_slice(&self.pending)?;
        value["wall_ms"] = serde_json::json!(self.start.elapsed().as_secs_f64() * 1000.0);
        if value["event"] == "frame" && value["step"] == "LiveAudio" {
            self.live += 1;
            if self.live > 10 && (self.timings.len() as u64) < self.limit {
                if let Some(ms) = value["execution_ms"].as_f64() {
                    self.timings.push(ms);
                }
            }
        }
        if !self.interactive && value["event"] == "agent_text" {
            if let Some(text) = value["text"].as_str() {
                print!("{text}");
                io::stdout().flush()?;
            }
        }
        self.display.observe(&value)?;
        serde_json::to_writer(&mut self.sink, &value)?;
        self.sink.write_all(b"\n")?;
        self.pending.clear();
        Ok(())
    }
}
impl Write for Events {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        for &byte in bytes {
            if byte == b'\n' {
                self.line()?;
            } else {
                if self.pending.len() >= 1_048_576 {
                    return Err(io::Error::other("event line exceeds 1 MiB"));
                }
                self.pending.push(byte);
            }
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.display.finish()?;
        self.sink.flush()
    }
}
