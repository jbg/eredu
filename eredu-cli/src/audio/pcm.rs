//! Common host PCM files, playback sinks, and execution contexts.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{ensure, Result};
use eredu::api::LocalDevice;
use eredu_backend_mlx::native::ExecutionContext;
#[cfg(feature = "system-audio")]
use eredu_media::audio::system::SystemAudioOutput;
use eredu_media::audio::wav::{WavReader, WavWriter};

pub type WavInput = WavReader<BufReader<File>>;
pub type WavOutput = WavWriter<BufWriter<File>>;

pub fn open_wav(path: &Path, rate: u32) -> Result<WavInput> {
    let wav = WavReader::new(BufReader::new(File::open(path)?))?;
    ensure!(
        wav.sample_rate() == rate,
        "input must be mono {rate} Hz WAV"
    );
    Ok(wav)
}

pub fn create_wav(path: &Path, rate: u32) -> Result<WavOutput> {
    Ok(WavWriter::new(
        BufWriter::new(crate::common::output_file(path)?),
        rate,
    )?)
}

pub fn finish_wav(wav: WavOutput) -> Result<()> {
    wav.finish()?.flush()?;
    Ok(())
}

pub struct PcmOutput {
    pub wav: Option<WavOutput>,
    #[cfg(feature = "system-audio")]
    pub speaker: Option<SystemAudioOutput>,
}

impl PcmOutput {
    pub fn write(&mut self, samples: &[f32]) -> Result<()> {
        if let Some(wav) = &mut self.wav {
            wav.write_samples(samples)?;
        }
        #[cfg(feature = "system-audio")]
        if let Some(speaker) = &mut self.speaker {
            speaker.write(samples)?;
        }
        Ok(())
    }
}

pub fn execution_contexts(device: LocalDevice) -> Result<(ExecutionContext, ExecutionContext)> {
    let weights = ExecutionContext::cpu()?;
    let execution = match device {
        LocalDevice::Cpu => ExecutionContext::cpu()?,
        LocalDevice::Accelerator(index) => ExecutionContext::accelerator(usize::try_from(index)?)?,
    };
    Ok((weights, execution))
}

pub(crate) fn decode_wav(path: &Path) -> Result<eredu_core::Audio> {
    let mut wav = WavReader::new(BufReader::new(File::open(path)?))?;
    let mut samples = Vec::new();
    let mut block = [0.0; 4096];
    loop {
        let count = wav.read_samples(&mut block)?;
        if count == 0 {
            break;
        }
        samples.extend_from_slice(&block[..count]);
    }
    Ok(eredu_core::Audio::new(samples, wav.sample_rate())?)
}
