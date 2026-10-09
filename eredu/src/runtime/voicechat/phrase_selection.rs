//! Recorded phrase selection for function-call acknowledgments and tool reminders.

use std::{num::NonZeroUsize, sync::Arc};

use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate},
    BackendFailure,
};
use eredu_runtime::HostRandomState;

use super::{PreparedVoiceChatPhrase, VoiceChatTextError, VoiceChatTextPreparation};

/// Speech selection applied once when a committed function call starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceChatPhrasePolicy {
    /// No phrase is selected for this speech role.
    Disabled,
    /// Every function call selects this ASCII phrase.
    Fixed(String),
    /// Selects uniformly from the released streaming phrase pool with replacement.
    /// The explicit host seed is independent of neural sampling randomness.
    Streaming {
        /// Seed for the conversation's retained host phrase-selection stream.
        seed: u64,
    },
}

// NVIDIA-NeMo/Speech 097dfe9e2f55baf653b83035868bdc89849f1b47,
// nemo/collections/speechlm2/inference/pipelines/data/fc_ack_messages.json.
pub(super) const ACK_PHRASES: [&str; 20] = [
    "Happy to help with that.",
    "My pleasure to help.",
    "Great question, looking now.",
    "Lovely, on it now.",
    "Glad you asked.",
    "Happy to take a look.",
    "Wonderful, looking now.",
    "Happy to look into that.",
    "Glad to help with that.",
    "Happy to look that up.",
    "Pleased to help out.",
    "Great, happy to help.",
    "Lovely question.",
    "Glad to check.",
    "Happy to check that.",
    "Wonderful question.",
    "Pleased to look into that.",
    "Happy to assist.",
    "Great, on it now.",
    "Lovely, looking now.",
];

// NVIDIA-NeMo/Speech 097dfe9e2f55baf653b83035868bdc89849f1b47,
// nemo/collections/speechlm2/inference/pipelines/data/fc_reminder_messages.json.
pub(super) const INITIAL_REMINDER_PHRASES: [&str; 10] = [
    "Still working on that.",
    "Just a moment longer.",
    "Almost there, one second.",
    "Bear with me just a bit.",
    "Still looking that up for you.",
    "Give me just another second.",
    "Working on it, almost done.",
    "Still checking, won't be long.",
    "Hang tight, almost there.",
    "Nearly done, one more moment.",
];

/// Immutable prepared speech plus transaction-local selection and random carry.
#[derive(Clone, Default)]
pub(super) struct PreparedPhraseSelection {
    phrases: Option<Arc<[PreparedVoiceChatPhrase]>>,
    random: Option<HostRandomState>,
    selected: Option<usize>,
    cursor: usize,
}

impl PreparedPhraseSelection {
    pub(super) fn prepare(
        policy: &VoiceChatPhrasePolicy,
        preparation: &VoiceChatTextPreparation,
        max_tokens: usize,
        streaming_phrases: &[&str],
    ) -> Result<Self, VoiceChatTextError> {
        let (texts, random): (&[&str], _) = match policy {
            VoiceChatPhrasePolicy::Disabled => return Ok(Self::default()),
            VoiceChatPhrasePolicy::Fixed(text) => (&[text.as_str()], None),
            VoiceChatPhrasePolicy::Streaming { seed } => {
                (streaming_phrases, Some(HostRandomState::with_seed(*seed)))
            }
        };
        let phrases = texts
            .iter()
            .map(|text| PreparedVoiceChatPhrase::prepare(preparation, text, max_tokens))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            phrases: Some(phrases.into()),
            random,
            selected: None,
            cursor: 0,
        })
    }

    pub(super) fn begin_call(&mut self) {
        self.clear();
        let Some(phrases) = &self.phrases else {
            return;
        };
        let count = NonZeroUsize::new(phrases.len()).expect("prepared nonempty phrase pool");
        self.selected = Some(self.random.as_mut().map_or(0, |random| random.index(count)));
    }

    pub(super) fn selected_phrase(&self) -> Option<&PreparedVoiceChatPhrase> {
        self.phrases.as_ref()?.get(self.selected?)
    }

    pub(super) fn token(&self) -> Option<u32> {
        self.selected_phrase()?
            .tokens()
            .get(self.cursor)
            .map(|token| *token as u32)
    }

    pub(super) fn advance(&mut self) {
        self.cursor += 1;
    }

    pub(super) fn clear(&mut self) {
        self.selected = None;
        self.cursor = 0;
    }

    pub(super) fn snapshot_retention(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let mut retained_bytes = 0;
        if let Some(phrases) = &self.phrases {
            retained_bytes = (2 * std::mem::size_of::<usize>()) as u64;
            for phrase in phrases.iter() {
                retained_bytes = retained_bytes
                    .checked_add(phrase.snapshot_retention()?.retained_bytes)
                    .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
            }
        }
        // The pool is immutable and shared; its handles and mutable RNG are
        // inline in the enclosing conversation's already-priced value.
        Ok(SnapshotEstimate {
            retained_bytes,
            copy_bytes: 0,
        })
    }

    #[cfg(test)]
    pub(super) fn selected(&self) -> Option<usize> {
        self.selected
    }
}
