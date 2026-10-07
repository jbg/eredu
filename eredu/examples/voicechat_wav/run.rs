//! Recorded-time WAV delivery through the ordinary bound session.

use std::{
    collections::BTreeMap,
    io::{Read, Seek, Write},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use eredu::api::realtime::{
    BoundVoiceChatSession, FinalPcmFrame, PcmFinishPolicy, VoiceChatAdvance, VoiceChatBlocked,
    VoiceChatConversationError, VoiceChatConversationOutput, VoiceChatExecutor,
    VoiceChatInitialReminder, VoiceChatInputError, VoiceChatInputStatus, VoiceChatPublicationError,
    VoiceChatReminderPlayback, VoiceChatSessionError, VoiceChatSessionProgress,
    VoiceChatSessionStatus, VoiceChatToolError, VoiceChatToolTicket, VoiceChatToolWaitMessages,
    VoiceChatToolWaitPolicy, VoiceChatTurnEvent,
};
use eredu_core::scheduler::SemanticStateTransaction;
use eredu_media::audio::{
    streaming::AudioStreamError,
    wav::{WavReader, WavWriter},
};
use serde::Deserialize;
use serde_json::{Value, json};

const CADENCE: Duration = Duration::from_millis(80);
const INPUT_FRAME_SAMPLES: usize = 1_280;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptedToolResponse {
    /// Zero-based ordinal of a complete published tool batch, across turns.
    pub call_index: usize,
    /// Delay relative to publication, measured on the recorded presentation clock.
    pub after_ms: u64,
    /// Complete response text, passed verbatim to the public text preparation API.
    pub response: String,
    /// Optional recorded choice of the initial reminder phrase.
    #[serde(default)]
    pub initial: Option<String>,
    /// Optional recorded generic phrases (at most two distinct phrases).
    #[serde(default)]
    pub generic: Vec<String>,
}

#[derive(Clone, Copy)]
pub struct RunOptions {
    pub chunk_samples: usize,
    pub tail_frames: u64,
    pub max_steps: u64,
}

/// Streams bounded host chunks; neither the WAV nor generated speech is retained
/// in full. The caller finalizes the WAV header after successful drain.
pub fn run<S, D, E, R, W>(
    session: &mut BoundVoiceChatSession<S, D, E>,
    input: &mut WavReader<R>,
    speech: &mut WavWriter<W>,
    events: &mut impl Write,
    options: RunOptions,
    scripts: Vec<ScriptedToolResponse>,
) -> Result<()>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
    E: VoiceChatExecutor<S, D>,
    R: Read + Seek,
    W: Write + Seek,
{
    ensure!(
        input.sample_rate() == 16_000,
        "input WAV must be mono 16 kHz"
    );
    drive(
        session,
        |buffer| Ok(input.read_samples(buffer)?),
        |samples| Ok(speech.write_samples(samples)?),
        events,
        options,
        scripts,
    )
}

// This adapter exposes application actions for a small host-driver fixture. The
// production implementation calls only the existing facade; work selection,
// transactional publication and tool policy remain inside that session.
trait Session {
    type Diagnostics;
    fn input_status(&self) -> VoiceChatInputStatus;
    fn submit_pcm(&mut self, samples: &[f32]) -> Result<usize>;
    fn finish(&mut self, policy: PcmFinishPolicy) -> Result<()>;
    fn status(&self) -> VoiceChatSessionStatus;
    fn has_tool_work(&self) -> bool;
    fn advance(&mut self, at: Duration) -> Result<VoiceChatSessionProgress>;
    fn poll(&mut self, at: Duration) -> Result<VoiceChatSessionProgress>;
    fn output(&mut self) -> Result<Option<VoiceChatConversationOutput<Self::Diagnostics>>>;
    fn start_wait(
        &mut self,
        ticket: VoiceChatToolTicket,
        at: Duration,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<()>;
    fn response(&mut self, ticket: VoiceChatToolTicket, text: &str) -> Result<bool>;
    fn finish_playback(&mut self, receipt: VoiceChatReminderPlayback, at: Duration) -> Result<()>;
    fn cancel(&mut self, at: Duration) -> Result<()>;
}

impl<S, D, E> Session for BoundVoiceChatSession<S, D, E>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
    E: VoiceChatExecutor<S, D>,
{
    type Diagnostics = D;
    fn input_status(&self) -> VoiceChatInputStatus {
        self.input().status()
    }
    fn submit_pcm(&mut self, samples: &[f32]) -> Result<usize> {
        match self.input_mut().submit(samples, 16_000) {
            Ok(()) => Ok(samples.len()),
            Err(VoiceChatInputError::Audio(AudioStreamError::Backpressure {
                available, ..
            })) => {
                if available != 0 {
                    self.input_mut().submit(&samples[..available], 16_000)?;
                }
                Ok(available)
            }
            Err(error) => Err(error.into()),
        }
    }
    fn finish(&mut self, policy: PcmFinishPolicy) -> Result<()> {
        Ok(self.input_mut().finish(policy)?)
    }
    fn status(&self) -> VoiceChatSessionStatus {
        self.status()
    }
    fn has_tool_work(&self) -> bool {
        self.session().has_tool_work()
    }
    fn advance(&mut self, at: Duration) -> Result<VoiceChatSessionProgress> {
        Ok(self.advance(VoiceChatAdvance {
            at,
            ..Default::default()
        })?)
    }
    fn poll(&mut self, at: Duration) -> Result<VoiceChatSessionProgress> {
        Ok(self.poll(at)?)
    }
    fn output(&mut self) -> Result<Option<VoiceChatConversationOutput<D>>> {
        Ok(self.output_mut().try_pop_output()?)
    }
    fn start_wait(
        &mut self,
        ticket: VoiceChatToolTicket,
        at: Duration,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<()> {
        Ok(self.session_mut().start_wait(
            ticket,
            at,
            VoiceChatToolWaitPolicy::default(),
            messages,
        )?)
    }
    fn response(&mut self, ticket: VoiceChatToolTicket, text: &str) -> Result<bool> {
        match self.session_mut().submit_response(ticket, text) {
            Ok(()) => Ok(true),
            Err(VoiceChatSessionError::Publication(VoiceChatPublicationError::Conversation(
                VoiceChatConversationError::Tool(VoiceChatToolError::StaleResult),
            ))) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }
    fn finish_playback(&mut self, receipt: VoiceChatReminderPlayback, at: Duration) -> Result<()> {
        Ok(self.session_mut().finish_playback(receipt, at)?)
    }
    fn cancel(&mut self, at: Duration) -> Result<()> {
        self.cancel(at)?;
        Ok(())
    }
}

struct PendingResponse {
    ticket: VoiceChatToolTicket,
    call_index: usize,
    due: Duration,
    text: String,
}

fn drive<S: Session>(
    session: &mut S,
    mut read: impl FnMut(&mut [f32]) -> Result<usize>,
    mut speech: impl FnMut(&[f32]) -> Result<()>,
    events: &mut impl Write,
    options: RunOptions,
    scripts: Vec<ScriptedToolResponse>,
) -> Result<()> {
    ensure!(
        options.chunk_samples > 0 && options.chunk_samples <= 1_048_576,
        "chunk samples must be in 1..=1048576"
    );
    ensure!(options.max_steps > 0, "max steps must be positive");
    let mut scripts = validated_scripts(scripts)?;
    let mut buffer = vec![0.0; options.chunk_samples];
    let (mut filled, mut consumed) = (0, 0);
    let mut finished_input = false;
    let mut source_finished = false;
    let mut source_samples = 0_u64;
    let mut tail_remaining = options.tail_frames;
    let mut had_tool_work = session.has_tool_work();
    let silence = [0.0; INPUT_FRAME_SAMPLES];
    let mut at = Duration::ZERO;
    let mut steps = 0;
    let mut output_samples = 0_u64;
    let mut call_index = 0;
    // Each admitted script schedules at most one external result. Interrupting
    // the conversation does not cancel work already running outside the session.
    let mut pending: Vec<PendingResponse> = Vec::with_capacity(scripts.len());
    let mut playback: Option<(VoiceChatReminderPlayback, Duration)> = None;

    let outcome = (|| {
        loop {
            // Pending completion does not advance recorded time or consume an
            // application step. Native execution speed cannot alter tool deadlines.
            if session.status() == VoiceChatSessionStatus::Pending {
                session.poll(at)?;
                if session.status() == VoiceChatSessionStatus::Pending {
                    std::thread::yield_now();
                    continue;
                }
            }
            while let Some(output) = session.output()? {
                let tool_work = session.has_tool_work();
                if had_tool_work && !tool_work {
                    // The post-tool listening/response window belongs after
                    // response injection, including the committed EOTR token.
                    tail_remaining = options.tail_frames;
                }
                had_tool_work = tool_work;
                if source_finished
                    && !tool_work
                    && output.frame.step
                        == eredu_architectures::voicechat::schedule::Step::LiveAudio
                    && output.frame.before.audio_frames
                        >= source_samples.div_ceil(INPUT_FRAME_SAMPLES as u64)
                {
                    tail_remaining = tail_remaining.saturating_sub(1);
                }
                publish(&output, at, &mut output_samples, &mut speech, events)?;
                if output.turn == Some(VoiceChatTurnEvent::Interrupted) {
                    playback = None;
                }
                if let Some(batch) = output.tools {
                    let script = scripts.remove(&call_index);
                    let generic: Vec<&str> = script
                        .as_ref()
                        .map(|script| script.generic.iter().map(String::as_str).collect())
                        .unwrap_or_default();
                    session.start_wait(
                        batch.ticket,
                        at,
                        VoiceChatToolWaitMessages {
                            initial: script
                                .as_ref()
                                .and_then(|script| script.initial.as_deref())
                                .map_or(
                                    VoiceChatInitialReminder::Configured,
                                    VoiceChatInitialReminder::Text,
                                ),
                            generic: &generic,
                            max_phrase_tokens: 4_096,
                        },
                    )?;
                    record(
                        events,
                        json!({"event":"tool_batch", "at_ms":millis(at)?, "call_index":call_index, "ticket":ticket(batch.ticket), "events":batch.events}),
                    )?;
                    if let Some(script) = script {
                        pending.push(PendingResponse {
                            ticket: batch.ticket,
                            call_index,
                            due: at
                                .checked_add(Duration::from_millis(script.after_ms))
                                .context("tool response time overflow")?,
                            text: script.response,
                        });
                    }
                    call_index += 1;
                }
                if let Some(receipt) = output.playback {
                    // WAV presentation is simulated explicitly. A phrase ends no
                    // earlier than the final emitted sample or its committed tick.
                    playback = Some((
                        receipt,
                        at.max(Duration::from_secs_f64(output_samples as f64 / 22_050.0)),
                    ));
                }
            }
            match session.status() {
                VoiceChatSessionStatus::Finished => {
                    let undelivered = pending.len();
                    for response in pending.drain(..) {
                        record(
                            events,
                            json!({"event":"tool_response_discarded", "at_ms":millis(at)?, "due_ms":millis(response.due)?, "call_index":response.call_index, "ticket":ticket(response.ticket), "reason":"finished"}),
                        )?;
                    }
                    record(
                        events,
                        json!({"event":"finished", "at_ms":millis(at)?, "input_samples":session.input_status().accepted_samples, "source_samples":source_samples, "silence_samples":session.input_status().accepted_samples - source_samples, "output_samples":output_samples, "unused_script_entries":scripts.len(), "undelivered_script_responses":undelivered}),
                    )?;
                    events.flush()?;
                    return Ok(());
                }
                VoiceChatSessionStatus::Cancelled { .. }
                | VoiceChatSessionStatus::Failed { .. } => {
                    bail!("session ended without drain: {:?}", session.status())
                }
                _ => {}
            }
            at = at.max(
                CADENCE
                    .checked_mul(
                        u32::try_from(session.input_status().committed_frames)
                            .context("input exceeds recorded clock range")?,
                    )
                    .context("input clock overflow")?,
            );
            if playback.as_ref().is_some_and(|(_, due)| at >= *due) {
                let (receipt, _) = playback.take().expect("due playback");
                session.finish_playback(receipt, at)?;
                record(
                    events,
                    json!({"event":"playback_finished", "at_ms":millis(at)?}),
                )?;
            }
            while let Some(index) = pending.iter().position(|response| at >= response.due) {
                let response = pending.remove(index);
                let accepted = session.response(response.ticket, &response.text)?;
                record(
                    events,
                    json!({"event":if accepted {"tool_response_submitted"} else {"tool_response_discarded"}, "at_ms":millis(at)?, "call_index":response.call_index, "ticket":ticket(response.ticket), "reason":if accepted {"script"} else {"stale"}}),
                )?;
            }
            if !finished_input {
                if !source_finished
                    && filled == consumed
                    && session.input_status().queued_samples < INPUT_FRAME_SAMPLES
                {
                    filled = read(&mut buffer)?;
                    ensure!(
                        filled <= buffer.len(),
                        "PCM reader returned an invalid sample count"
                    );
                    consumed = 0;
                    if filled == 0 {
                        source_finished = true;
                    }
                }
                if consumed < filled {
                    let accepted = session.submit_pcm(&buffer[consumed..filled])?;
                    consumed += accepted;
                    source_samples = source_samples
                        .checked_add(accepted as u64)
                        .context("source sample count overflow")?;
                }
                if source_finished {
                    let input = session.input_status();
                    let tool_work = session.has_tool_work();
                    if input.queued_samples == 0 && !tool_work && tail_remaining == 0 {
                        session.finish(PcmFinishPolicy {
                            final_frame: FinalPcmFrame::ZeroPad,
                            trailing_silence_frames: 0,
                        })?;
                        finished_input = true;
                    } else if input.queued_samples < INPUT_FRAME_SAMPLES {
                        let count = INPUT_FRAME_SAMPLES - input.queued_samples;
                        let padding_end = source_samples.div_ceil(INPUT_FRAME_SAMPLES as u64)
                            * INPUT_FRAME_SAMPLES as u64;
                        let purpose = if input.accepted_samples < padding_end {
                            "padding"
                        } else if tool_work {
                            "tool_work"
                        } else {
                            "response_tail"
                        };
                        let accepted = session.submit_pcm(&silence[..count])?;
                        if accepted != 0 {
                            record(
                                events,
                                json!({"event":"input_silence", "at_ms":millis(at)?, "start_sample":input.accepted_samples, "samples":accepted, "purpose":purpose}),
                            )?;
                        }
                    }
                }
            }
            ensure!(
                steps < options.max_steps,
                "VoiceChat reached --max-steps={} before drain; increase the explicit bound",
                options.max_steps
            );
            steps += 1;
            let progress = session.advance(at)?;
            if matches!(
                progress.status,
                VoiceChatSessionStatus::Blocked(
                    VoiceChatBlocked::WaitingForTime | VoiceChatBlocked::WaitingForTool
                )
            ) {
                at = at.checked_add(CADENCE).context("recorded time overflow")?;
            }
        }
    })();
    match outcome {
        Ok(()) => Ok(()),
        Err(error) => match session.cancel(at) {
            Ok(()) => Err(error),
            Err(cancel) => Err(error.context(format!(
                "cancellation at {} ms also failed: {cancel}",
                at.as_millis()
            ))),
        },
    }
}

pub fn validate_scripts(scripts: &[ScriptedToolResponse]) -> Result<()> {
    let mut indices = std::collections::BTreeSet::new();
    for script in scripts {
        ensure!(
            script.response.is_ascii(),
            "script response {} must be ASCII",
            script.call_index
        );
        ensure!(
            script.generic.len() <= 2
                && (script.generic.len() != 2 || script.generic[0] != script.generic[1]),
            "script {} requires at most two distinct generic phrases",
            script.call_index
        );
        ensure!(
            script
                .initial
                .iter()
                .chain(script.generic.iter())
                .all(|text| text.is_ascii()),
            "script reminder {} must be ASCII",
            script.call_index
        );
        for (kind, text) in std::iter::once(("response", script.response.as_str()))
            .chain(
                script
                    .initial
                    .iter()
                    .map(|text| ("reminder", text.as_str())),
            )
            .chain(
                script
                    .generic
                    .iter()
                    .map(|text| ("reminder", text.as_str())),
            )
        {
            ensure!(
                !text.trim().is_empty(),
                "script {kind} {} must contain non-whitespace text",
                script.call_index
            );
            ensure!(
                text.len() <= 65_536,
                "script {kind} {} exceeds the 65536-byte text limit",
                script.call_index
            );
        }
        let index = script.call_index;
        ensure!(
            indices.insert(index),
            "duplicate scripted call_index {index}"
        );
    }
    Ok(())
}

fn validated_scripts(
    scripts: Vec<ScriptedToolResponse>,
) -> Result<BTreeMap<usize, ScriptedToolResponse>> {
    validate_scripts(&scripts)?;
    Ok(scripts
        .into_iter()
        .map(|script| (script.call_index, script))
        .collect())
}

fn ticket(ticket: VoiceChatToolTicket) -> Value {
    json!({"request":ticket.request().value(), "turn":ticket.turn(), "batch":ticket.batch()})
}

fn millis(at: Duration) -> Result<u64> {
    Ok(u64::try_from(at.as_millis())?)
}

fn record(writer: &mut impl Write, value: Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, &value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn publish<D>(
    output: &VoiceChatConversationOutput<D>,
    at: Duration,
    output_samples: &mut u64,
    speech: &mut impl FnMut(&[f32]) -> Result<()>,
    events: &mut impl Write,
) -> Result<()> {
    let frame = &output.frame;
    let at_ms = millis(at)?;
    record(
        events,
        json!({"event":"frame", "at_ms":at_ms, "invocation":frame.invocation_index, "step":format!("{:?}", frame.step), "audio_frames":frame.after.audio_frames, "backbone_positions":frame.after.backbone_positions}),
    )?;
    if let Some(chunk) = &frame.speech {
        ensure!(
            chunk.sample_rate == 22_050,
            "VoiceChat output sample rate differs from 22050 Hz"
        );
        ensure!(
            chunk.start_sample == *output_samples,
            "VoiceChat speech sample timeline is not contiguous"
        );
        speech(&chunk.samples)?;
        *output_samples = output_samples
            .checked_add(u64::try_from(chunk.samples.len())?)
            .context("output sample clock overflow")?;
        record(
            events,
            json!({"event":"speech", "at_ms":at_ms, "invocation":frame.invocation_index, "start_sample":chunk.start_sample, "samples":chunk.samples.len(), "sample_rate":chunk.sample_rate, "delay_samples":chunk.delay_samples, "tail":chunk.is_tail}),
        )?;
    }
    if let Some(text) = &output.agent_text {
        record(
            events,
            json!({"event":"agent_text", "at_ms":at_ms, "invocation":frame.invocation_index, "audio_frame":frame.before.audio_frames, "model_position":frame.before.backbone_positions, "text":text.text}),
        )?;
    }
    if let Some(transcription) = &frame.transcription {
        record(
            events,
            json!({"event":"transcription", "at_ms":at_ms, "invocation":frame.invocation_index, "start_sample":transcription.start_sample, "sample_rate":transcription.sample_rate, "valid_samples":transcription.valid_samples, "source":format!("{:?}",transcription.source), "first_blank":transcription.first_blank, "tokens":transcription.emitted, "text":output.transcription.as_ref().map(|text|text.text.as_str()).unwrap_or("")}),
        )?;
    }
    if let Some(turn) = output.turn {
        record(
            events,
            json!({"event":"turn", "at_ms":at_ms, "invocation":frame.invocation_index, "kind":format!("{turn:?}")}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_architectures::voicechat::{
        observation::{OwnedHostOutput, OwnedSpeechChunk},
        schedule::{AudioSource, Progress, Step},
    };
    use std::collections::VecDeque;

    struct Fixture {
        input: VecDeque<f32>,
        accepted: u64,
        consumed: Vec<f32>,
        finish: Option<PcmFinishPolicy>,
        frames: u64,
        continuation: u64,
        progress: Progress,
        prefilled: bool,
        tool_steps: usize,
        tool_active: bool,
        response_ready: bool,
        response_steps: usize,
        invocations: u64,
        staged: Option<VoiceChatConversationOutput<()>>,
        delivered: Option<VoiceChatConversationOutput<()>>,
        polled: bool,
        times: Vec<(Step, Duration)>,
        batch: Option<eredu::api::realtime::VoiceChatToolBatch>,
        interrupt: bool,
        stale_ticket: Option<VoiceChatToolTicket>,
        waits: Vec<(VoiceChatToolTicket, Duration)>,
        responses: Vec<(VoiceChatToolTicket, String)>,
        cancellations: Vec<Duration>,
    }

    impl Default for Fixture {
        fn default() -> Self {
            Self {
                input: VecDeque::new(),
                accepted: 0,
                consumed: Vec::new(),
                finish: None,
                frames: 0,
                continuation: 0,
                progress: Progress::default(),
                prefilled: false,
                tool_steps: 0,
                tool_active: false,
                response_ready: false,
                response_steps: 0,
                invocations: 0,
                staged: None,
                delivered: None,
                polled: false,
                times: Vec::new(),
                batch: None,
                interrupt: false,
                stale_ticket: None,
                waits: Vec::new(),
                responses: Vec::new(),
                cancellations: Vec::new(),
            }
        }
    }

    impl Session for Fixture {
        type Diagnostics = ();
        fn has_tool_work(&self) -> bool {
            self.tool_active
        }
        fn input_status(&self) -> VoiceChatInputStatus {
            VoiceChatInputStatus {
                accepted_samples: self.accepted,
                committed_samples: self.consumed.len() as u64,
                committed_frames: self.frames,
                committed_continuation_frames: self.continuation,
                remaining_continuation_frames: self.finish.map_or(0, |policy| {
                    policy
                        .trailing_silence_frames
                        .saturating_sub(self.continuation)
                }),
                queued_samples: self.input.len(),
                reserved_samples: 0,
                pending: self.staged.is_some(),
                finished: self.finish.is_some(),
                finish_declared: self.finish.is_some(),
                finish_committed: self.finish.is_some() && self.input.is_empty(),
                closed: false,
            }
        }
        fn submit_pcm(&mut self, samples: &[f32]) -> Result<usize> {
            let count = samples.len().min(INPUT_FRAME_SAMPLES - self.input.len());
            self.input.extend(&samples[..count]);
            self.accepted += count as u64;
            Ok(count)
        }
        fn finish(&mut self, policy: PcmFinishPolicy) -> Result<()> {
            self.finish = Some(policy);
            Ok(())
        }
        fn status(&self) -> VoiceChatSessionStatus {
            if self.staged.is_some() {
                VoiceChatSessionStatus::Pending
            } else if self.progress.codec_finished {
                VoiceChatSessionStatus::Finished
            } else {
                VoiceChatSessionStatus::Ready
            }
        }
        fn advance(&mut self, at: Duration) -> Result<VoiceChatSessionProgress> {
            assert!(self.staged.is_none());
            let step = if !self.prefilled {
                self.prefilled = true;
                Step::Prefill { backbone_tokens: 2 }
            } else if self.frames == 1 && self.tool_steps < 2 {
                self.tool_steps += 1;
                Step::Tool {
                    audio: AudioSource::Silence,
                }
            } else if self.tool_active && self.response_ready {
                self.response_steps += 1;
                Step::Tool {
                    audio: AudioSource::Silence,
                }
            } else if self.input.len() >= INPUT_FRAME_SAMPLES
                || (!self.input.is_empty() && self.finish.is_some())
            {
                let count = self.input.len().min(INPUT_FRAME_SAMPLES);
                self.consumed.extend(self.input.drain(..count));
                self.frames += 1;
                if self.tool_active {
                    Step::PerceptionOnly
                } else {
                    Step::LiveAudio
                }
            } else if self
                .finish
                .is_some_and(|policy| self.continuation < policy.trailing_silence_frames)
            {
                self.frames += 1;
                self.continuation += 1;
                Step::LiveAudio
            } else if self.finish.is_some() {
                Step::Drain
            } else {
                return Ok(VoiceChatSessionProgress {
                    submitted: 0,
                    committed: 0,
                    status: VoiceChatSessionStatus::Blocked(VoiceChatBlocked::NeedsInput),
                });
            };
            self.times.push((step, at));
            let before = self.progress;
            let mut after = before;
            after.audio_frames = self.frames;
            after.backbone_positions += if matches!(step, Step::Prefill { .. }) {
                2
            } else if matches!(step, Step::PerceptionOnly | Step::Drain) {
                0
            } else {
                1
            };
            after.codec_finished = step == Step::Drain;
            let speech = if matches!(step, Step::LiveAudio | Step::Drain) {
                after.output_samples += 2;
                Some(OwnedSpeechChunk {
                    samples: vec![0.25, -0.5],
                    sample_rate: 22_050,
                    start_sample: before.output_samples,
                    delay_samples: 7,
                    is_tail: step == Step::Drain,
                })
            } else {
                None
            };
            self.staged = Some(VoiceChatConversationOutput {
                frame: OwnedHostOutput {
                    invocation_index: self.invocations,
                    step,
                    reminder_token: None,
                    before,
                    after,
                    decisions: None,
                    generated_codes: None,
                    speech,
                    transcription: None,
                    diagnostics: vec![],
                },
                turn: (self.interrupt && matches!(step, Step::Tool { .. }) && self.tool_steps == 2)
                    .then_some(VoiceChatTurnEvent::Interrupted),
                tools: if matches!(step, Step::Tool { .. }) && self.tool_steps == 1 {
                    self.batch.take()
                } else {
                    None
                },
                playback: None,
                agent_text: None,
                transcription: None,
            });
            self.invocations += 1;
            self.polled = false;
            Ok(VoiceChatSessionProgress {
                submitted: 1,
                committed: 0,
                status: VoiceChatSessionStatus::Pending,
            })
        }
        fn poll(&mut self, at: Duration) -> Result<VoiceChatSessionProgress> {
            assert_eq!(
                at,
                self.times.last().unwrap().1,
                "completion polling must retain the invocation timestamp"
            );
            if !self.polled {
                self.polled = true;
                return Ok(VoiceChatSessionProgress {
                    submitted: 0,
                    committed: 0,
                    status: VoiceChatSessionStatus::Pending,
                });
            }
            let output = self.staged.take().unwrap();
            self.progress = output.frame.after;
            if output.tools.is_some() {
                self.tool_active = true;
            }
            if self.response_steps == 3 {
                // The third response position models the committed EOTR.
                self.tool_active = false;
            }
            if output.turn == Some(VoiceChatTurnEvent::Interrupted) {
                self.stale_ticket = self.waits.last().map(|(ticket, _)| *ticket);
                self.tool_active = false;
            }
            self.delivered = Some(output);
            Ok(VoiceChatSessionProgress {
                submitted: 0,
                committed: 1,
                status: self.status(),
            })
        }
        fn output(&mut self) -> Result<Option<VoiceChatConversationOutput<()>>> {
            Ok(self.delivered.take())
        }
        fn start_wait(
            &mut self,
            ticket: VoiceChatToolTicket,
            at: Duration,
            _: VoiceChatToolWaitMessages<'_>,
        ) -> Result<()> {
            self.waits.push((ticket, at));
            Ok(())
        }
        fn response(&mut self, ticket: VoiceChatToolTicket, text: &str) -> Result<bool> {
            self.responses.push((ticket, text.to_owned()));
            let accepted = self.stale_ticket != Some(ticket);
            self.response_ready = accepted;
            Ok(accepted)
        }
        fn finish_playback(&mut self, _: VoiceChatReminderPlayback, _: Duration) -> Result<()> {
            unreachable!()
        }
        fn cancel(&mut self, at: Duration) -> Result<()> {
            self.cancellations.push(at);
            Ok(())
        }
    }

    #[test]
    fn fragmented_and_backpressured_wav_drivers_keep_audio_and_tool_clocks() {
        let input: Vec<f32> = (0..1_301).map(|i| i as f32 / 2_048.0).collect();
        let mut runs = Vec::new();
        for chunk_samples in [127, 4_096] {
            let mut fixture = Fixture::default();
            let mut read_offset = 0;
            let mut speech = Vec::new();
            let mut events = Vec::new();
            drive(
                &mut fixture,
                |buffer| {
                    let count = buffer.len().min(input.len() - read_offset);
                    buffer[..count].copy_from_slice(&input[read_offset..read_offset + count]);
                    read_offset += count;
                    Ok(count)
                },
                |samples| {
                    speech.extend_from_slice(samples);
                    Ok(())
                },
                &mut events,
                RunOptions {
                    chunk_samples,
                    tail_frames: 1,
                    max_steps: 100,
                },
                vec![],
            )
            .unwrap();
            assert_eq!(&fixture.consumed[..input.len()], input);
            assert!(
                fixture.consumed[input.len()..]
                    .iter()
                    .all(|&sample| sample == 0.0)
            );
            assert_eq!(speech, [0.25, -0.5, 0.25, -0.5, 0.25, -0.5, 0.25, -0.5]);
            let lines: Vec<Value> = String::from_utf8(events)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let final_event = lines.last().unwrap();
            assert_eq!(final_event["event"], "finished");
            assert_eq!(final_event["input_samples"], 3_840);
            assert_eq!(final_event["source_samples"], 1_301);
            assert_eq!(final_event["silence_samples"], 2_539);
            assert_eq!(final_event["output_samples"], 8);
            assert_eq!(
                fixture
                    .times
                    .iter()
                    .map(|(_, at)| at.as_millis())
                    .collect::<Vec<_>>(),
                [0, 0, 80, 80, 80, 160, 240]
            );
            runs.push(lines);
        }
        assert_eq!(runs[0], runs[1]);
    }

    #[test]
    fn scripts_reject_ambiguous_ordinals_and_invalid_text_before_loading() {
        let duplicate: Vec<ScriptedToolResponse> = serde_json::from_value(json!([
            {"call_index":0,"after_ms":0,"response":"one"},
            {"call_index":0,"after_ms":80,"response":"two"}
        ]))
        .unwrap();
        assert!(
            validate_scripts(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        let unicode: Vec<ScriptedToolResponse> = serde_json::from_value(json!([
            {"call_index":1,"after_ms":0,"response":"caf\u{e9}"}
        ]))
        .unwrap();
        assert!(
            validate_scripts(&unicode)
                .unwrap_err()
                .to_string()
                .contains("ASCII")
        );
        for (field, value, expected) in [
            ("response", json!(" \t\n"), "non-whitespace"),
            ("response", json!("x".repeat(65_537)), "65536-byte"),
            ("initial", json!(""), "non-whitespace"),
            ("generic", json!(["x".repeat(65_537)]), "65536-byte"),
        ] {
            let mut script = json!({"call_index":0,"after_ms":0,"response":"ok"});
            script[field] = value;
            let scripts = vec![serde_json::from_value(script).unwrap()];
            assert!(
                validate_scripts(&scripts)
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }
    }

    // Mint the opaque ticket through public text/tool admission rather than
    // fabricating an identity that the production session cannot construct.
    fn tool_batch() -> eredu::api::realtime::VoiceChatToolBatch {
        use eredu::api::realtime::{
            RequestId, VoiceChatTextLimits, VoiceChatTextPreparation, VoiceChatToolChannel,
            VoiceChatToolLimits,
        };
        use eredu_architectures::voicechat::{
            host::{PreparedHostConditioning, TextControls},
            tts_conditioning::{CharacterFlags, CharacterMapping},
            tts_feedback::FeedbackControls,
        };
        let mut vocabulary: BTreeMap<u32, String> =
            (0..32).map(|id| (id, format!("token_{id}"))).collect();
        vocabulary.insert(0, "[UNK]".to_owned());
        vocabulary.insert(30, r#"{"name":"weather","arguments":{}}"#.to_owned());
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                vocabulary
                    .into_iter()
                    .map(|(id, text)| (text, id))
                    .collect(),
            )
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let tokenizer =
            eredu_text::tokenizer::Tokenizer::from_tokenizer(tokenizers::Tokenizer::new(model));
        let host = PreparedHostConditioning {
            tokenizer,
            characters: CharacterMapping::new(
                &[(0, "a".to_owned())].into(),
                CharacterFlags {
                    continuation: vec![0, 0],
                    continuation_pad: 1,
                    special: vec![0],
                    special_pad: 0,
                },
            )
            .unwrap(),
            feedback: FeedbackControls::new(8, 2, 12, true, &[8], &[0]).unwrap(),
            tokens: TextControls {
                bos: 1,
                eos: 2,
                pad: 12,
                listening: 11,
                user_bos: 3,
                user_eos: 4,
                tool_call_start: 20,
                tool_call_end: 21,
                tool_response_end: 22,
            },
        };
        let text = VoiceChatTextPreparation::new(
            &host,
            VoiceChatTextLimits {
                max_bytes: 8_192,
                max_tokens: 2_048,
            },
        )
        .unwrap();
        let mut channel = VoiceChatToolChannel::new(text, RequestId::new(91), vec![json!({"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{}}}})], VoiceChatToolLimits { max_payload_tokens:32, max_payload_bytes:2_048, max_calls:1 }).unwrap();
        for token in [20, 30] {
            assert!(
                channel
                    .observe_committed_function(channel.turn_identity(), token)
                    .unwrap()
                    .is_none()
            );
        }
        channel
            .observe_committed_function(channel.turn_identity(), 21)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn delayed_script_arrives_after_interruption_or_is_reported_at_finish() {
        for (interrupt, after_ms) in [(false, 160), (true, 160), (true, 1_000)] {
            let batch = tool_batch();
            let expected = batch.ticket;
            let mut fixture = Fixture {
                batch: Some(batch),
                interrupt,
                ..Default::default()
            };
            let mut remaining = 1_301;
            let mut events = Vec::new();
            drive(
                &mut fixture,
                |buffer| {
                    let count = remaining.min(buffer.len());
                    buffer[..count].fill(0.25);
                    remaining -= count;
                    Ok(count)
                },
                |_| Ok(()),
                &mut events,
                RunOptions {
                    chunk_samples: 4_096,
                    tail_frames: 1,
                    max_steps: 100,
                },
                vec![ScriptedToolResponse {
                    call_index: 0,
                    after_ms,
                    response: "<TOOL_RESPONSE>ok</TOOL_RESPONSE>".to_owned(),
                    initial: None,
                    generic: Vec::new(),
                }],
            )
            .unwrap();
            assert_eq!(fixture.waits, [(expected, Duration::from_millis(80))]);
            let lines: Vec<Value> = String::from_utf8(events)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            if after_ms == 1_000 {
                assert!(fixture.responses.is_empty());
                let discarded = lines
                    .iter()
                    .find(|event| event["event"] == "tool_response_discarded")
                    .unwrap();
                assert_eq!(discarded["at_ms"], 240);
                assert_eq!(discarded["due_ms"], 1_080);
                assert_eq!(discarded["reason"], "finished");
                assert_eq!(lines.last().unwrap()["undelivered_script_responses"], 1);
            } else {
                assert_eq!(
                    fixture.responses,
                    [(expected, "<TOOL_RESPONSE>ok</TOOL_RESPONSE>".to_owned())]
                );
                let result = lines
                    .iter()
                    .find(|event| {
                        event["event"] == "tool_response_submitted"
                            || event["event"] == "tool_response_discarded"
                    })
                    .unwrap();
                assert_eq!(result["at_ms"], 240);
                assert_eq!(result["ticket"]["request"], 91);
                assert_eq!(
                    result["event"],
                    if interrupt {
                        "tool_response_discarded"
                    } else {
                        "tool_response_submitted"
                    }
                );
                assert_eq!(result["reason"], if interrupt { "stale" } else { "script" });
                assert_eq!(lines.last().unwrap()["undelivered_script_responses"], 0);
            }
        }
    }

    #[test]
    fn late_tool_result_keeps_a_full_live_tail_after_response_end() {
        let mut fixture = Fixture {
            batch: Some(tool_batch()),
            ..Default::default()
        };
        let mut remaining = INPUT_FRAME_SAMPLES;
        let mut events = Vec::new();
        drive(
            &mut fixture,
            |buffer| {
                let count = remaining.min(buffer.len());
                buffer[..count].fill(0.25);
                remaining -= count;
                Ok(count)
            },
            |_| Ok(()),
            &mut events,
            RunOptions {
                chunk_samples: INPUT_FRAME_SAMPLES,
                tail_frames: 2,
                max_steps: 100,
            },
            vec![ScriptedToolResponse {
                call_index: 0,
                after_ms: 800,
                response: "<TOOL_RESPONSE>ok</TOOL_RESPONSE>".to_owned(),
                initial: None,
                generic: Vec::new(),
            }],
        )
        .unwrap();
        assert_eq!(fixture.response_steps, 3);
        let response_end = fixture
            .times
            .iter()
            .rposition(|(step, _)| matches!(step, Step::Tool { .. }))
            .unwrap();
        assert_eq!(
            fixture.times[response_end + 1..]
                .iter()
                .map(|(step, _)| *step)
                .collect::<Vec<_>>(),
            [Step::LiveAudio, Step::LiveAudio, Step::Drain]
        );
        assert!(
            fixture.times[..response_end]
                .iter()
                .filter(|(step, _)| *step == Step::PerceptionOnly)
                .count()
                >= 10
        );
        assert!(fixture.times[response_end].1 >= Duration::from_millis(880));
        assert_eq!(fixture.finish.unwrap().trailing_silence_frames, 0);
        let lines: Vec<Value> = String::from_utf8(events)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(
            lines.iter().any(|event| {
                event["event"] == "input_silence" && event["purpose"] == "tool_work"
            })
        );
        assert_eq!(lines.last().unwrap()["source_samples"], INPUT_FRAME_SAMPLES);
        assert_eq!(lines.last().unwrap()["undelivered_script_responses"], 0);
    }

    #[test]
    fn step_limit_cancels_at_recorded_time_without_hiding_failure() {
        let mut fixture = Fixture::default();
        let error = drive(
            &mut fixture,
            |buffer| {
                buffer.fill(0.25);
                Ok(buffer.len())
            },
            |_| Ok(()),
            &mut Vec::new(),
            RunOptions {
                chunk_samples: 1_280,
                tail_frames: 0,
                max_steps: 3,
            },
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("max-steps=3"));
        assert_eq!(fixture.cancellations, [Duration::from_millis(80)]);
    }
}
