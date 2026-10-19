use eredu_media::audio::streaming::{
    AudioStreamError, FinalPcmFrame, PcmFinishPolicy, PcmFrameInfo, PcmFrameQueue, PcmFrameSource,
};

#[test]
fn pcm_peek_retains_capacity_and_head_while_more_audio_arrives() {
    let mut queue = PcmFrameQueue::new(16_000, 4, 8).unwrap();
    queue.submit(&[1.0, 2.0, 3.0], 16_000).unwrap();
    assert!(queue.peek_frame().is_none());
    assert!(queue.consume_frame().is_none());
    queue.submit(&[4.0], 16_000).unwrap();
    let pending = queue.peek_frame().unwrap();
    assert_eq!(pending.samples, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(pending.start_sample, 0);
    assert_eq!(pending.valid_samples, 4);
    queue.submit(&[5.0, 6.0, 7.0], 16_000).unwrap();
    assert_eq!(queue.peek_frame().unwrap(), pending);
    assert!(matches!(
        queue.submit(&[8.0, 9.0], 16_000),
        Err(AudioStreamError::Backpressure { available: 1, .. })
    ));
    assert_eq!(queue.queued_samples(), 7);
    assert_eq!(
        queue.consume_frame(),
        Some(PcmFrameInfo {
            source: PcmFrameSource::Input,
            start_sample: 0,
            valid_samples: 4,
        })
    );
    queue
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 0,
        })
        .unwrap();
    let final_frame = queue.peek_frame().unwrap();
    assert_eq!(final_frame.samples, [5.0, 6.0, 7.0, 0.0]);
    assert_eq!(final_frame.start_sample, 4);
    assert_eq!(final_frame.valid_samples, 3);
    assert_eq!(queue.queued_samples(), 3);
    assert_eq!(
        queue.consume_frame(),
        Some(PcmFrameInfo {
            source: PcmFrameSource::Input,
            start_sample: 4,
            valid_samples: 3,
        })
    );
    assert!(queue.is_drained());
    assert!(queue.peek_frame().is_none());
    assert!(queue.consume_frame().is_none());
}

#[test]
fn pcm_partial_chunks_backpressure_finish_and_fork_preserve_every_sample() {
    let pcm: Vec<_> = (0..11).map(|n| n as f32 / 16.0).collect();
    for split in 0..=pcm.len() {
        let mut queue = PcmFrameQueue::new(16_000, 4, 12).unwrap();
        queue.submit(&pcm[..split], 16_000).unwrap();
        let mut delivered = Vec::new();
        while let Some(frame) = queue.next_frame() {
            delivered.extend(frame.samples);
        }
        queue.submit(&pcm[split..], 16_000).unwrap();
        queue
            .finish(PcmFinishPolicy {
                final_frame: FinalPcmFrame::Partial,
                trailing_silence_frames: 0,
            })
            .unwrap();
        while let Some(frame) = queue.next_frame() {
            assert_eq!(frame.start_sample as usize, delivered.len());
            assert_eq!(frame.valid_samples, frame.samples.len());
            delivered.extend(frame.samples);
        }
        assert_eq!(delivered, pcm);
        assert!(queue.is_drained());
    }
    let mut queue = PcmFrameQueue::new(16_000, 4, 4).unwrap();
    queue.submit(&pcm[..3], 16_000).unwrap();
    let mut fork = queue.snapshot_copy().unwrap();
    assert!(matches!(
        queue.submit(&pcm[3..5], 16_000),
        Err(AudioStreamError::Backpressure { available: 1, .. })
    ));
    assert!(matches!(
        queue.submit(&[1.0], 8_000),
        Err(AudioStreamError::Invalid(_))
    ));
    assert!(matches!(
        queue.finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::RequireComplete,
            trailing_silence_frames: 0
        }),
        Err(AudioStreamError::NeedMoreSamples { available: 3, .. })
    ));
    queue.submit(&pcm[3..4], 16_000).unwrap();
    queue
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::RequireComplete,
            trailing_silence_frames: 0,
        })
        .unwrap();
    assert_eq!(queue.next_frame().unwrap().samples, pcm[..4]);
    fork.finish(PcmFinishPolicy {
        final_frame: FinalPcmFrame::ZeroPad,
        trailing_silence_frames: 0,
    })
    .unwrap();
    let final_frame = fork.next_frame().unwrap();
    assert_eq!(final_frame.valid_samples, 3);
    assert_eq!(final_frame.samples, [pcm[0], pcm[1], pcm[2], 0.0]);
    assert!(matches!(
        fork.submit(&[0.0], 16_000),
        Err(AudioStreamError::Finished)
    ));
}

#[test]
fn pcm_finish_continuation_is_lazy_replayable_and_attributed_after_padding() {
    let mut queue = PcmFrameQueue::new(16_000, 4, 4).unwrap();
    queue.submit(&[0.25, -0.5], 16_000).unwrap();
    let policy = PcmFinishPolicy {
        final_frame: FinalPcmFrame::ZeroPad,
        trailing_silence_frames: 2,
    };
    queue.finish(policy).unwrap();
    let original_storage = queue.snapshot_storage_bytes();
    let mut restored = queue.snapshot_copy().unwrap();
    for owner in [&mut queue, &mut restored] {
        let input = owner.next_frame().unwrap();
        assert_eq!(input.samples, [0.25, -0.5, 0.0, 0.0]);
        assert_eq!(input.valid_samples, 2);
        assert_eq!(input.source, PcmFrameSource::Input);
        assert_eq!(owner.consumed_samples(), 2);
        assert_eq!(owner.queued_samples(), 0);
        assert!(!owner.is_drained());
        for (start_sample, remaining) in [(4, 2), (8, 1)] {
            let pending = owner.peek_frame().unwrap();
            assert_eq!(pending.start_sample, start_sample);
            assert_eq!(pending.source, PcmFrameSource::Continuation);
            assert_eq!(pending.samples, [0.0; 4]);
            assert_eq!(pending.valid_samples, 0);
            drop(pending);
            assert_eq!(owner.remaining_continuation_frames(), remaining);
            assert_eq!(
                owner.snapshot_copy().unwrap().next_frame(),
                owner.peek_frame()
            );
            // A replayed finish receipt must not restart or extend the suffix.
            owner.finish(policy).unwrap();
            assert_eq!(
                owner.consume_frame(),
                Some(PcmFrameInfo {
                    start_sample,
                    source: PcmFrameSource::Continuation,
                    valid_samples: 0,
                })
            );
            assert_eq!(owner.remaining_continuation_frames(), remaining - 1);
            assert_eq!(owner.consumed_samples(), 2);
        }
        assert!(owner.is_drained());
        owner.finish(policy).unwrap();
        assert!(owner.peek_frame().is_none());
        assert_eq!(owner.finish_policy(), Some(policy));
        assert!(matches!(
            owner.finish(PcmFinishPolicy {
                trailing_silence_frames: 3,
                ..policy
            }),
            Err(AudioStreamError::FinishPolicyMismatch { .. })
        ));
        assert!(owner.is_drained());
    }
    assert_eq!(queue.snapshot_storage_bytes(), original_storage);
}

#[test]
fn pcm_finish_rejects_overflow_before_closing_input_and_keeps_large_tails_virtual() {
    let mut queue = PcmFrameQueue::new(16_000, 4, 4).unwrap();
    let storage = queue.snapshot_storage_bytes();
    let policy = PcmFinishPolicy {
        final_frame: FinalPcmFrame::ZeroPad,
        trailing_silence_frames: u64::MAX / 4 + 1,
    };
    assert_eq!(
        queue.finish(policy),
        Err(AudioStreamError::TimelineOverflow)
    );
    assert_eq!(queue.finish_policy(), None);
    queue.submit(&[1.0], 16_000).unwrap();
    // Tail multiplication fits, but the padded final input frame does not.
    assert_eq!(
        queue.finish(PcmFinishPolicy {
            trailing_silence_frames: u64::MAX / 4,
            ..policy
        }),
        Err(AudioStreamError::TimelineOverflow)
    );
    assert!(!queue.is_finished());
    assert_eq!(queue.queued_samples(), 1);
    let mut empty = PcmFrameQueue::new(16_000, 4, 4).unwrap();
    empty
        .finish(PcmFinishPolicy {
            trailing_silence_frames: u64::MAX / 4,
            ..policy
        })
        .unwrap();
    assert_eq!(empty.snapshot_storage_bytes(), storage);
    assert_eq!(
        empty
            .snapshot_copy()
            .unwrap()
            .remaining_continuation_frames(),
        u64::MAX / 4
    );
    assert_eq!(
        empty.next_frame().unwrap().source,
        PcmFrameSource::Continuation
    );
    assert_eq!(empty.remaining_continuation_frames(), u64::MAX / 4 - 1);
}

fn config() -> eredu_media::audio::streaming::CenteredSlaneyLogMelConfig {
    eredu_media::audio::streaming::CenteredSlaneyLogMelConfig {
        sample_rate: 16_000,
        window_length: 400,
        fft_length: 512,
        hop_length: 160,
        mel_bins: 128,
        min_frequency: 0.0,
        max_frequency: 8_000.0,
        preemphasis: 0.97,
        log_floor: 2f32.powi(-24),
        capacity_samples: 5120,
    }
}

#[cfg(not(feature = "audio"))]
#[test]
fn optional_fft_is_unavailable_without_audio_feature() {
    assert!(matches!(
        eredu_media::audio::streaming::StreamingLogMelWindow::new(config()),
        Err(AudioStreamError::AudioFeatureDisabled)
    ));
}

#[cfg(feature = "audio")]
mod mel {
    use super::*;
    use eredu_media::audio::streaming::{MelWindowBoundary, StreamingLogMelWindow};

    #[test]
    fn retained_coefficients_control_centered_stft_and_survive_stream_forks() {
        let config = eredu_media::audio::streaming::CenteredSlaneyLogMelConfig {
            sample_rate: 8,
            window_length: 2,
            fft_length: 4,
            hop_length: 1,
            mel_bins: 2,
            min_frequency: 0.0,
            max_frequency: 4.0,
            preemphasis: 0.0,
            log_floor: 1.0,
            capacity_samples: 8,
        };
        let filters = vec![1.0, 0.0, 0.0, 0.0, 2.0, 1.0];
        let mut window = StreamingLogMelWindow::with_coefficients(
            config.clone(),
            vec![2.0, 3.0],
            filters.clone(),
        )
        .unwrap();
        window.append(&[1.0, 2.0, 3.0], 8).unwrap();
        let mut fork = window.clone();
        window.append(&[4.0], 8).unwrap();
        fork.append(&[4.0, 5.0], 8).unwrap();
        // The centered, reflected first frames become [0,4,3,0] and [0,2,6,0].
        // Their real-FFT power bins are [49,25,1] and [64,40,16].
        let expected = [50f32.ln(), 52f32.ln(), 65f32.ln(), 97f32.ln()];
        for branch in [&mut window, &mut fork] {
            let features = branch.extract(0..2, MelWindowBoundary::Reflect).unwrap();
            for (actual, expected) in features.values.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
        assert!(matches!(
            StreamingLogMelWindow::with_coefficients(config.clone(), vec![2.0], filters.clone(),),
            Err(AudioStreamError::Invalid(_))
        ));
        assert!(matches!(
            StreamingLogMelWindow::with_coefficients(
                config.clone(),
                vec![2.0, f32::NAN],
                filters.clone(),
            ),
            Err(AudioStreamError::Invalid(_))
        ));
        let mut negative = filters.clone();
        negative[0] = -1.0;
        assert!(matches!(
            StreamingLogMelWindow::with_coefficients(config.clone(), vec![2.0, 3.0], negative,),
            Err(AudioStreamError::Invalid(_))
        ));
        assert!(matches!(
            StreamingLogMelWindow::with_coefficients(
                config.clone(),
                vec![2.0, 3.0],
                filters[..5].to_vec(),
            ),
            Err(AudioStreamError::Invalid(_))
        ));
    }

    #[test]
    fn mel_validation_and_lookahead_rejection_preserve_pending_pcm() {
        let mut window = StreamingLogMelWindow::new(config()).unwrap();
        window.append(&[0.0; 256], 16_000).unwrap();
        assert!(matches!(
            window.extract(0..1, MelWindowBoundary::Reflect),
            Err(AudioStreamError::NeedMoreSamples { required: 257, .. })
        ));
        assert!(matches!(
            window.append(&[f32::NAN], 16_000),
            Err(AudioStreamError::Invalid(_))
        ));
        assert_eq!(window.retained_samples(), 256);
        window.append(&[0.0], 16_000).unwrap();
        let silence = window
            .extract(0..1, MelWindowBoundary::RequireLookahead)
            .unwrap();
        assert!(silence
            .values
            .iter()
            .all(|v| (*v - config().log_floor.ln()).abs() < 1e-6));
        assert!(matches!(
            window.discard_before(258),
            Err(AudioStreamError::Invalid(_))
        ));
        assert_eq!(window.start_sample(), 0);
    }
}
