//! JSON and file ingress for the public decision task.
use anyhow::{ensure, Context, Result};
use eredu::api::decision::{DecisionItem, DecisionQuestion, DecisionRequest};
use eredu_core::{Audio, Media};
use serde::Deserialize;
use std::{
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestFile {
    state: Option<String>,
    #[serde(default)]
    media: Vec<MediaFile>,
    questions: Vec<QuestionFile>,
    refinement_steps: Option<u32>,
    seed: Option<u64>,
    assets: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuestionFile {
    id: String,
    question: DecisionQuestion,
    spoken: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum MediaFile {
    Image { path: PathBuf },
    Video { frames: PathBuf, fps: f64 },
    Audio { path: PathBuf },
}

fn audio(path: &Path) -> Result<Audio> {
    #[cfg(feature = "realtime")]
    {
        crate::audio::pcm::decode_wav(path)
            .with_context(|| format!("decoding WAV {}", path.display()))
    }
    #[cfg(not(feature = "realtime"))]
    anyhow::bail!(
        "audio input {} requires the realtime build feature",
        path.display()
    );
}

impl MediaFile {
    fn decode(self, root: &Path) -> Result<Media> {
        match self {
            Self::Image { path } => {
                #[cfg(feature = "vision")]
                return Ok(Media::Image(crate::media::decode_image(&root.join(path))?));
                #[cfg(not(feature = "vision"))]
                anyhow::bail!(
                    "image input {} requires the vision build feature",
                    path.display()
                );
            }
            Self::Video { frames, fps } => {
                ensure!(
                    fps.is_finite() && fps > 0.0,
                    "video fps must be finite and positive"
                );
                #[cfg(feature = "vision")]
                return Ok(Media::Video(crate::media::decode_video(
                    &root.join(frames),
                    Some(fps),
                )?));
                #[cfg(not(feature = "vision"))]
                anyhow::bail!(
                    "video input {} requires the vision build feature",
                    frames.display()
                );
            }
            Self::Audio { path } => Ok(Media::Audio(audio(&root.join(path))?)),
        }
    }
}

pub(crate) fn read(path: &Path) -> Result<(DecisionRequest, Option<PathBuf>)> {
    let stdin = path == Path::new("-");
    let root = if stdin {
        Path::new(".")
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let reader: Box<dyn Read> = if stdin {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?)
    };
    let mut data = Vec::new();
    reader.take(1_048_577).read_to_end(&mut data)?;
    ensure!(data.len() <= 1_048_576, "decision request exceeds 1 MiB");
    let file: RequestFile = serde_json::from_slice(&data).context("parsing decision request")?;
    let media = file
        .media
        .into_iter()
        .map(|m| m.decode(root))
        .collect::<Result<_>>()?;
    let questions = file
        .questions
        .into_iter()
        .map(|q| {
            Ok(DecisionItem {
                id: q.id,
                question: q.question,
                spoken: q.spoken.map(|path| audio(&root.join(path))).transpose()?,
            })
        })
        .collect::<Result<_>>()?;
    Ok((
        DecisionRequest {
            state: file.state,
            media,
            questions,
            refinement_steps: file.refinement_steps,
            seed: file.seed,
        },
        file.assets.map(|path| root.join(path)),
    ))
}
