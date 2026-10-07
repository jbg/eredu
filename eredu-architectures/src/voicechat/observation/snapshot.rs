//! Independent host publication copies for bounded session snapshots.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use super::{OwnedHostOutput, OwnedSpeechChunk, TranscriptionOutput};

impl SnapshotValue for OwnedSpeechChunk {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([self.samples.estimate_snapshot_value()])
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            samples: self.samples.copy_snapshot_value(retention)?,
            sample_rate: self.sample_rate,
            start_sample: self.start_sample,
            delay_samples: self.delay_samples,
            is_tail: self.is_tail,
        })
    }
}

impl SnapshotValue for TranscriptionOutput {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([self.emitted.estimate_snapshot_value()])
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            frame_index: self.frame_index,
            sample_rate: self.sample_rate,
            start_sample: self.start_sample,
            valid_samples: self.valid_samples,
            source: self.source,
            emitted: self.emitted.copy_snapshot_value(retention)?,
            first_blank: self.first_blank,
            ended_blank: self.ended_blank,
            hit_symbol_limit: self.hit_symbol_limit,
        })
    }
}

impl<D: SnapshotValue> SnapshotValue for OwnedHostOutput<D> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.generated_codes.estimate_snapshot_value(),
            self.speech.estimate_snapshot_value(),
            self.transcription.estimate_snapshot_value(),
            self.diagnostics.estimate_snapshot_value(),
        ])
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            invocation_index: self.invocation_index,
            step: self.step,
            reminder_token: self.reminder_token,
            before: self.before,
            after: self.after,
            decisions: self.decisions,
            generated_codes: self.generated_codes.copy_snapshot_value(retention)?,
            speech: self.speech.copy_snapshot_value(retention)?,
            transcription: self.transcription.copy_snapshot_value(retention)?,
            diagnostics: self.diagnostics.copy_snapshot_value(retention)?,
        })
    }
}
