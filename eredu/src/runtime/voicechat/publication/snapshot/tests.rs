use std::{cell::Cell, convert::Infallible};

use eredu_architectures::voicechat::{observation::OwnedHostOutput, schedule::Progress};
use eredu_core::execution_control::{SnapshotEstimate, SnapshotLimits};

use super::*;
use crate::runtime::voicechat::{
    VoiceChatConversationOutput,
    conversation::tests::{capacity, conversation, frame},
    tools::tests::{channel, limits},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Callback {
    DiagnosticEstimate,
    DiagnosticCopy,
    NativeEstimate,
    NativeCopy,
    NativeRestore,
    DiagnosticDrop,
}

type Hooks = Rc<RefCell<Option<Box<dyn Fn(Callback)>>>>;

fn invoke(hooks: &Hooks, callback: Callback) {
    if let Some(hook) = hooks.borrow().as_ref() {
        hook(callback);
    }
}

struct Diagnostic(Hooks);

impl SnapshotValue for Diagnostic {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        invoke(&self.0, Callback::DiagnosticEstimate);
        snapshot_value_estimate::<Self>([])
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        invoke(&self.0, Callback::DiagnosticCopy);
        Ok(Self(self.0.clone()))
    }
}

impl Drop for Diagnostic {
    fn drop(&mut self) {
        invoke(&self.0, Callback::DiagnosticDrop);
    }
}

struct Neural(Hooks);

impl eredu_architectures::voicechat::state::InterruptionFeedback for Neural {
    fn reset_interruption_feedback(&mut self) {
        // This snapshot callback fixture has no token feedback or sampling history.
    }
}

impl SemanticStateTransaction for Neural {
    type Branch = Self;
    type Error = BackendFailure;
    fn branch(&self) -> Result<Self, Self::Error> {
        Ok(Self(self.0.clone()))
    }
    fn commit_branch(&mut self, branch: Self) -> Result<(), Self::Error> {
        *self = branch;
        Ok(())
    }
}

impl IsolatedStateCopy for Neural {
    type Error = Infallible;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Self::Error> {
        invoke(&self.0, Callback::NativeEstimate);
        Ok(SnapshotEstimate {
            retained_bytes: std::mem::size_of::<Self>() as u64,
            copy_bytes: std::mem::size_of::<Self>() as u64,
        })
    }

    fn copy_isolated(&self, _: &SnapshotReservation) -> Result<Self, Self::Error> {
        invoke(&self.0, Callback::NativeCopy);
        Ok(Self(self.0.clone()))
    }

    fn validate_isolated_restore(&self, _: &Self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn restore_isolated(&mut self, saved: Self) -> Result<(), Self::Error> {
        invoke(&self.0, Callback::NativeRestore);
        *self = saved;
        Ok(())
    }
}

#[test]
fn snapshot_callbacks_cannot_admit_pcm_or_deliver_output_and_retired_drops_see_restored_state() {
    let hooks: Hooks = Rc::default();
    let mut allowance = capacity();
    allowance.diagnostics = 1;
    let (mut state, sender, receiver) = VoiceChatPublicationState::new_with_input(
        Neural(hooks.clone()),
        conversation(channel(90, limits())),
        VoiceChatInputConfig::default(),
        VoiceChatPublicationLimits {
            max_publications: 2,
            capacity: allowance,
        },
    )
    .unwrap();
    let sender = Rc::new(RefCell::new(sender));
    let receiver = Rc::new(RefCell::new(receiver));
    sender.borrow_mut().submit(&[0.25, -0.5], 16000).unwrap();
    let base = frame(
        0,
        Progress::default(),
        Step::Prefill { backbone_tokens: 0 },
        None,
    );
    let output = VoiceChatConversationOutput {
        frame: OwnedHostOutput {
            invocation_index: base.invocation_index,
            step: base.step,
            reminder_token: base.reminder_token,
            before: base.before,
            after: base.after,
            decisions: base.decisions,
            generated_codes: base.generated_codes,
            speech: base.speech,
            transcription: base.transcription,
            diagnostics: vec![Diagnostic(hooks.clone())],
        },
        turn: None,
        tools: None,
        playback: None,
        agent_text: None,
        transcription: None,
    };
    let permit = Queue::reserve(&state.queue, allowance).unwrap();
    let charge = permit.prepare(&output).unwrap();
    permit.commit(output, charge);

    let seen = Rc::new(RefCell::new(Vec::new()));
    let allow_retired_drop = Rc::new(Cell::new(false));
    let weak_sender = Rc::downgrade(&sender);
    let weak_receiver = Rc::downgrade(&receiver);
    let callbacks = seen.clone();
    let drop_enabled = allow_retired_drop.clone();
    *hooks.borrow_mut() = Some(Box::new(move |callback| {
        if callback == Callback::DiagnosticDrop {
            if drop_enabled.replace(false) {
                weak_sender
                    .upgrade()
                    .unwrap()
                    .borrow_mut()
                    .submit(&[0.75], 16000)
                    .expect("retired output drops after host installation");
                callbacks.borrow_mut().push(callback);
            }
            return;
        }
        let sender = weak_sender.upgrade().unwrap();
        assert!(matches!(
            sender.borrow_mut().submit(&[0.75], 16000),
            Err(VoiceChatInputError::Busy)
        ));
        assert!(matches!(
            sender
                .borrow_mut()
                .finish(eredu_media::audio::streaming::PcmFinishPolicy {
                    final_frame: eredu_media::audio::streaming::FinalPcmFrame::ZeroPad,
                    trailing_silence_frames: 0
                }),
            Err(VoiceChatInputError::Busy)
        ));
        let receiver = weak_receiver.upgrade().unwrap();
        assert!(matches!(
            receiver.borrow_mut().try_pop_output(),
            Err(VoiceChatPublicationError::SnapshotBusy)
        ));
        assert!(receiver.borrow_mut().pop_output().is_none());
        callbacks.borrow_mut().push(callback);
    }));

    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 16 << 20,
        cumulative_copy_bytes: 64 << 20,
    });
    let saved = state.capture_snapshot(&budget, 0).unwrap();
    for callback in [
        Callback::DiagnosticEstimate,
        Callback::DiagnosticCopy,
        Callback::NativeEstimate,
        Callback::NativeCopy,
    ] {
        assert!(seen.borrow().contains(&callback), "capture {callback:?}");
    }
    seen.borrow_mut().clear();
    allow_retired_drop.set(true);
    state.restore_snapshot(&saved).unwrap();
    for callback in [
        Callback::DiagnosticEstimate,
        Callback::DiagnosticCopy,
        Callback::NativeEstimate,
        Callback::NativeCopy,
        Callback::NativeRestore,
        Callback::DiagnosticDrop,
    ] {
        assert!(seen.borrow().contains(&callback), "restore {callback:?}");
    }
    assert_eq!(sender.borrow().status().accepted_samples, 3);
    assert_eq!(sender.borrow().status().queued_samples, 3);
    assert!(!sender.borrow().status().finished);
    assert_eq!(receiver.borrow().usage().queued_publications, 1);
    assert!(receiver.borrow_mut().try_pop_output().unwrap().is_some());
    assert!(receiver.borrow_mut().try_pop_output().unwrap().is_none());
}
