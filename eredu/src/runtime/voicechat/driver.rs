//! Transactional invocation selection for the ordinary shared scheduler.

use std::{cell::RefCell, convert::Infallible, time::Duration};

use eredu_architectures::voicechat::{
    channel_decision::ChannelDecisionPolicy,
    input::PcmInputState,
    schedule::{AudioSource, Step},
};
use eredu_core::scheduler::WorkDescriptor;

use super::{
    conversation::VoiceChatConversationTransition, VoiceChatInputPermit,
    VoiceChatPublicationAdmission, VoiceChatPublicationBranch, VoiceChatPublicationError,
    VoiceChatPublicationPermit, VoiceChatReminderStep,
};

/// Recorded advancement inputs. The driver never reads the wall clock.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoiceChatAdvance {
    /// Monotonic presentation time, committed with accepted neural work.
    pub at: Duration,
    /// Recorded function-channel acoustic policy; the released default is silence.
    pub tool_audio: VoiceChatToolAudioPolicy,
}

/// Acoustic conditioning during asynchronous tool serialization and responses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VoiceChatToolAudioPolicy {
    /// Released default: process fresh perception separately and condition tool
    /// backbone steps on the adapter's prepared silence embedding.
    #[default]
    Silence,
    /// Explicit reference override: condition on fresh or retained user features.
    RealAudio,
}

/// Why this advancement cannot yet prepare an invocation. No state is consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceChatBlocked {
    /// Explicit prompt/Aria prefill has not committed.
    NeedsPrefill,
    /// More PCM or an explicit input finish declaration is needed.
    NeedsInput,
    /// External tool response, playback completion or wait configuration is needed.
    WaitingForTool,
    /// Recorded time has not reached the next reminder cadence or wait deadline.
    WaitingForTime,
    /// Codec drain already committed.
    Finished,
}

/// Portable intent supplied to the architecture executor by its composition adapter.
/// Native prompt assets and speech rendering policy remain separately prepared;
/// this descriptor neither materializes parameters nor invents an executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceChatWorkKind {
    /// Explicit prompt/Aria preparation with the architecture-declared token count.
    Prefill {
        /// Number of framed backbone prompt positions.
        backbone_tokens: u64,
    },
    /// One ordinary live audio invocation using the admission's turn resolver.
    Live,
    /// Fresh perception/RNNT without a backbone or speech position.
    PerceptionOnly,
    /// Feed one previously perceived frame into the backbone without advancing audio.
    ReplayAudio,
    /// Function serialization or response injection with independent audio time.
    Tool {
        /// Fresh PCM, prepared silence, or the last retained perception result.
        audio: AudioSource,
        /// Selected forced-response or greedy tool policy.
        decisions: ChannelDecisionPolicy,
    },
    /// One supplied reminder token, with its separately declared speech policy.
    Reminder(VoiceChatReminderStep),
    /// Terminal codec overlap, after explicit input finish.
    Drain,
}

impl VoiceChatWorkKind {
    /// Architecture step whose attribution the publication boundary validates.
    pub const fn step(self) -> Step {
        match self {
            Self::Prefill { backbone_tokens } => Step::Prefill { backbone_tokens },
            Self::Live => Step::LiveAudio,
            Self::PerceptionOnly => Step::PerceptionOnly,
            Self::ReplayAudio => Step::ReplayAudio,
            Self::Tool { audio, .. } => Step::Tool { audio },
            Self::Reminder(_) => Step::Reminder,
            Self::Drain => Step::Drain,
        }
    }
}

/// Result of preparing one ordinary scheduler work item.
pub enum VoiceChatPreparation<D> {
    /// Enqueue this move-only item in the existing scheduler.
    Ready(VoiceChatWork<D>),
    /// No neural or host-policy successor was adopted.
    Blocked(VoiceChatBlocked),
}

struct Admission<D> {
    transition: VoiceChatConversationTransition,
    output: VoiceChatPublicationPermit<D>,
    input: Option<VoiceChatInputPermit>,
}

/// Reserved invocation with frozen host policy from preparation through retirement.
/// Dropping unsubmitted work releases its admission and both capacity reservations.
/// An uninterrupted driver repeatedly prepares and executes these same work items;
/// explicit control changes only when the existing scheduler is advanced.
#[must_use = "enqueue or drop this reserved invocation"]
pub struct VoiceChatWork<D> {
    kind: VoiceChatWorkKind,
    at: Duration,
    admission: RefCell<Option<Admission<D>>>,
    descriptor: Vec<u32>,
}

impl<D> VoiceChatWork<D> {
    pub(super) fn new(
        kind: VoiceChatWorkKind,
        at: Duration,
        transition: VoiceChatConversationTransition,
        output: VoiceChatPublicationPermit<D>,
        input: Option<VoiceChatInputPermit>,
    ) -> Self {
        let mut work = Self {
            kind,
            at,
            admission: RefCell::new(Some(Admission {
                transition,
                output,
                input,
            })),
            descriptor: Vec::new(),
        };
        let mut descriptor = Vec::new();
        work.describe(&mut descriptor);
        work.descriptor = descriptor;
        work
    }

    /// Immutable selected intent, including effective forced function feedback.
    pub const fn kind(&self) -> VoiceChatWorkKind {
        self.kind
    }

    /// Explicit timestamp used to select this work.
    pub const fn at(&self) -> Duration {
        self.at
    }

    /// Transfers retained policy and PCM into the scheduler's unpublished branch.
    /// Call once in the scheduler's submission closure, then attach the returned
    /// admission to its ordinary prepublication gate. Any failure requires branch
    /// discard; input and host policy remain uncommitted.
    pub fn admit<B: PcmInputState>(
        &self,
        branch: &mut VoiceChatPublicationBranch<B, D>,
    ) -> Result<VoiceChatPublicationAdmission<D>, VoiceChatPublicationError> {
        let admission =
            self.admission
                .borrow_mut()
                .take()
                .ok_or(VoiceChatPublicationError::Transaction(
                    "prepared work already admitted",
                ))?;
        branch.admit_prepared(
            self.kind.step(),
            admission.transition,
            admission.output,
            admission.input,
        )
    }
}

impl<D> VoiceChatWork<D> {
    fn describe(&self, output: &mut Vec<u32>) {
        // Versioned semantic descriptor. Scheduler identity and source values
        // accompany intent so consensus does not compare only an enum tag.
        output.push(3);
        encode_u64(output, self.at.as_secs());
        output.push(self.at.subsec_nanos());
        match self.kind {
            VoiceChatWorkKind::Prefill { backbone_tokens } => {
                output.push(0);
                encode_u64(output, backbone_tokens);
            }
            VoiceChatWorkKind::Live => output.push(1),
            VoiceChatWorkKind::PerceptionOnly => output.push(2),
            VoiceChatWorkKind::Tool { audio, decisions } => {
                output.push(3);
                output.push(match audio {
                    AudioSource::Fresh => 0,
                    AudioSource::Silence => 1,
                    AudioSource::Retained => 2,
                });
                let values = match decisions {
                    ChannelDecisionPolicy::Main {
                        agent_feedback,
                        function_feedback,
                        spoken_text,
                    } => {
                        output.push(0);
                        [agent_feedback, function_feedback, spoken_text]
                    }
                    ChannelDecisionPolicy::Tool {
                        function_feedback,
                        spoken_text,
                    } => {
                        output.push(1);
                        [None, function_feedback, spoken_text]
                    }
                };
                for value in values {
                    output.push(u32::from(value.is_some()));
                    output.push(value.unwrap_or_default());
                }
            }
            VoiceChatWorkKind::Reminder(receipt) => {
                output.push(4);
                receipt.encode_descriptor(output);
            }
            VoiceChatWorkKind::Drain => output.push(5),
            VoiceChatWorkKind::ReplayAudio => output.push(6),
        }
        let admission = self.admission.borrow();
        if let Some(admission) = admission.as_ref() {
            output.push(1);
            admission.output.encode_descriptor(output);
            if let Some(input) = &admission.input {
                output.push(1);
                input.encode_descriptor(output);
            } else {
                output.push(0);
            }
        } else {
            output.push(0);
        }
    }
}

impl<D> WorkDescriptor for VoiceChatWork<D> {
    type Error = Infallible;
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Infallible> {
        output.extend_from_slice(&self.descriptor);
        Ok(())
    }
}

pub(super) fn encode_u64(output: &mut Vec<u32>, value: u64) {
    output.extend([value as u32, (value >> 32) as u32]);
}
