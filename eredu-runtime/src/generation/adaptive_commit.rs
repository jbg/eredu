//! Actual adaptive policy copy and commit, independent of numerical permission.
use super::*;
use eredu_core::HostPreparationAuthority;
use std::mem::{size_of, size_of_val};

/// Fixed failure before publishing a provisional adaptive policy.
#[derive(Debug, thiserror::Error)]
pub enum PreparedAdaptiveCommitError {
    /// The source or destination has no declared prepared worker.
    #[error("adaptive commitment requires its prepared source and host destination")]
    Unknown,
    /// Same probability predicate and text as ordinary Mirostat commitment.
    #[error("accepted Mirostat V2 token probability must be finite and in (0, 1]")]
    Probability,
    /// The actual pending choice or durable controller rejected commitment.
    #[error(transparent)]
    Controller(#[from] PreparedControllerError),
}

/// Borrows a known policy and its complete provisional copy/growth destination.
/// It does not certify a probability or supply allocation authority. A backend
/// must observe the actual processed token probability under its admitted
/// numerical mechanism, pay this host extent, then replace only on success.
#[derive(Debug)]
pub struct PreparedAdaptiveCommit<'a, S> {
    source: &'a S,
    bytes: usize,
    validate: fn(&S, u32) -> Result<(), PreparedAdaptiveCommitError>,
    copy: fn(&S, HostPreparationAuthority) -> Result<S, PreparedAdaptiveCommitError>,
    apply: fn(&mut S, u32, f32) -> Result<(), PreparedAdaptiveCommitError>,
}
impl<S> PreparedAdaptiveCommit<'_, S> {
    /// Actual copy and potential history-growth allocations plus worker controls.
    pub fn metadata_bytes(&self) -> usize {
        self.bytes
    }
    /// Checks pending controller/domain state before requesting a probability.
    pub fn validate_token(&self, token: u32) -> Result<(), PreparedAdaptiveCommitError> {
        (self.validate)(self.source, token)
    }
    /// Preserves the source on every failure, including controller rejection.
    /// The caller retains host custody through this returned policy's lifetime.
    pub fn commit(
        self,
        token: u32,
        probability: f32,
        host: HostPreparationAuthority,
    ) -> Result<S, PreparedAdaptiveCommitError> {
        if host.is_unmanaged() {
            return Err(PreparedAdaptiveCommitError::Unknown);
        }
        self.validate_token(token)?;
        validate_probability(probability)?;
        let mut provisional = (self.copy)(self.source, host)?;
        (self.apply)(&mut provisional, token, probability)?;
        Ok(provisional)
    }
}
pub(super) fn validate_probability(probability: f32) -> Result<(), PreparedAdaptiveCommitError> {
    if !probability.is_finite() || probability <= 0.0 || probability > 1.0 {
        Err(PreparedAdaptiveCommitError::Probability)
    } else {
        Ok(())
    }
}
fn controls<S>(copy: usize, growth: usize) -> Option<usize> {
    let parts = [
        copy,
        growth,
        size_of::<S>(),
        size_of::<Result<S, PreparedAdaptiveCommitError>>(),
        size_of::<PreparedAdaptiveCommit<'_, S>>(),
        size_of::<Option<PreparedAdaptiveCommit<'_, S>>>(),
        size_of::<PreparedAdaptiveCommitError>(),
        size_of::<Result<(), PreparedAdaptiveCommitError>>(),
        size_of::<Result<S, PreparedControllerError>>(),
        size_of::<PreparedControllerError>(),
        size_of::<history::TokenHistory>(),
        size_of::<Vec<u32>>(),
        size_of::<Box<[u32]>>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<[f32; 4]>(),
        size_of::<u32>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn adaptive(
    source: &MirostatV2Sampler,
) -> Option<PreparedAdaptiveCommit<'_, MirostatV2Sampler>> {
    Some(PreparedAdaptiveCommit {
        source,
        bytes: controls::<MirostatV2Sampler>(
            speculative_copy::adaptive(source)?.metadata_bytes(),
            usize::try_from(source.penalties.generated_tokens.push_payload_bytes()?).ok()?,
        )?,
        validate: |_, _| Ok(()),
        copy: |source, _| Ok(source.clone()),
        apply: MirostatV2Sampler::accept_token_fixed,
    })
}
pub(super) fn configured(
    source: &ConfiguredTextSampler,
) -> Option<PreparedAdaptiveCommit<'_, ConfiguredTextSampler>> {
    let ConfiguredTextSampler::MirostatV2(policy) = source else {
        return None;
    };
    Some(PreparedAdaptiveCommit {
        source,
        bytes: controls::<ConfiguredTextSampler>(
            speculative_copy::configured(source)?.metadata_bytes(),
            usize::try_from(policy.penalties.generated_tokens.push_payload_bytes()?).ok()?,
        )?,
        validate: |_, _| Ok(()),
        // This is the existing SamplerCopyPlan, preserving tau/eta/mu and the
        // entire history allocation, including cleared and unused slots.
        copy: |source, _| Ok(source.clone()),
        apply: |source, token, probability| match source {
            ConfiguredTextSampler::MirostatV2(policy) => {
                policy.accept_token_fixed(token, probability)
            }
            _ => Err(PreparedAdaptiveCommitError::Unknown),
        },
    })
}
pub(super) fn constrained<B, P, C>(
    source: &ConstrainedSampler<P, C>,
) -> Option<PreparedAdaptiveCommit<'_, ConstrainedSampler<P, C>>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let policy = source.policy.prepared_adaptive_commit()?;
    let controller = source
        .controller
        .prepared_copy_bytes(source.controller.fixed_history_len()?.checked_add(1)?)?;
    Some(PreparedAdaptiveCommit {
        source,
        bytes: controls::<ConstrainedSampler<P, C>>(policy.metadata_bytes(), controller)?
            .checked_add(size_of::<
                PreparedSpeculativeController<'_, ConstrainedSampler<P, C>>,
            >())?,
        validate: |source, token| {
            source
                .controller
                .validate_fixed_commit(token)
                .map_err(Into::into)
        },
        copy: |source, host| {
            prepared_controller::copied::<B, P, C>(source, 1, host).map_err(Into::into)
        },
        apply: |source, token, probability| {
            // The same policy-then-controller sequence as ordinary commitment,
            // operating only on the independently copied provisional source.
            commit_components(
                &mut source.policy,
                &mut source.controller,
                token,
                |policy, token| {
                    let apply = policy
                        .prepared_adaptive_commit()
                        .ok_or(PreparedAdaptiveCommitError::Unknown)?
                        .apply;
                    apply(policy, token, probability)
                },
                |controller, token| controller.commit_prepared_fixed(token).map_err(Into::into),
            )
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::working_memory::WorkspaceSamplingBackend;
    type B = WorkspaceSamplingBackend;
    fn host() -> HostPreparationAuthority {
        HostPreparationAuthority::retain(())
    }
    fn adaptive_state(policy: &ConfiguredTextSampler) -> &MirostatV2Sampler {
        let ConfiguredTextSampler::MirostatV2(policy) = policy else {
            panic!("adaptive source")
        };
        policy
    }
    #[test]
    fn prepared_adaptive_commit_preserves_snapshot_and_matches_history_growth() {
        // Exercise empty, spare, full and cleared boxes with nondefault mu.
        for len in [0, 1, 4, 5, 8, 9] {
            let mut original = MirostatV2Sampler::new(3.5, 0.2).unwrap();
            for token in 0..len {
                original.accept_token(token, 0.125).unwrap();
            }
            for clear in [false, true] {
                let mut expected = original.clone();
                if clear {
                    expected.reset();
                }
                let source = ConfiguredTextSampler::MirostatV2(expected.clone());
                let snapshot =
                    <ConfiguredTextSampler as SpeculativeSampler<B>>::prepared_host_copy(&source)
                        .unwrap()
                        .copy();
                let plan =
                    <ConfiguredTextSampler as SpeculativeSampler<B>>::prepared_adaptive_commit(
                        &source,
                    )
                    .unwrap();
                let old_capacity = source.history_capacity();
                let new_allocation = if source.history_len() == old_capacity {
                    next_history_capacity(old_capacity).unwrap() * size_of::<u32>()
                } else {
                    0
                };
                assert!(plan.metadata_bytes() >= old_capacity * size_of::<u32>() + new_allocation);
                expected.accept_token(31, 0.25).unwrap();
                let actual = plan.commit(31, 0.25, host()).unwrap();
                assert_eq!(
                    adaptive_state(&actual).mu().to_bits(),
                    expected.mu().to_bits()
                );
                assert_eq!(
                    adaptive_state(&actual).generated_tokens(),
                    expected.generated_tokens()
                );
                assert_eq!(
                    adaptive_state(&source).mu().to_bits(),
                    adaptive_state(&snapshot).mu().to_bits()
                );
                assert_eq!(
                    adaptive_state(&source).generated_tokens(),
                    adaptive_state(&snapshot).generated_tokens()
                );
                assert_eq!(source.history_capacity(), snapshot.history_capacity());
                assert_eq!(
                    actual.history_capacity(),
                    expected.penalties.generated_tokens.capacity()
                );
            }
        }
    }
    #[test]
    fn adaptive_probability_failure_and_unknown_callback_leave_source_untouched() {
        let mut source = MirostatV2Sampler::default();
        source.accept_token(7, 0.125).unwrap();
        for probability in [0.0, -1.0, 1.1, f32::INFINITY, f32::NAN] {
            let mu = source.mu().to_bits();
            let copies = crate::generation::payload_copy_count();
            let failure = adaptive(&source)
                .unwrap()
                .commit(3, probability, host())
                .unwrap_err();
            assert!(matches!(failure, PreparedAdaptiveCommitError::Probability));
            assert_eq!(crate::generation::payload_copy_count(), copies);
            assert_eq!(source.mu().to_bits(), mu);
            assert_eq!(source.generated_tokens(), &[7]);
            assert_eq!(
                failure.to_string(),
                source
                    .clone()
                    .accept_token(3, probability)
                    .unwrap_err()
                    .to_string()
            );
        }
        #[derive(Clone)]
        struct Unknown;
        impl SpeculativeSampler<B> for Unknown {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
            fn process_logits(
                &mut self,
                _: &<B as SamplingBackend>::Logits,
                _: f32,
                _: &[u32],
                _: &<B as SamplingBackend>::Context,
            ) -> Result<<B as SamplingBackend>::Logits, <B as SamplingBackend>::Error> {
                panic!("must not invoke callback")
            }
        }
        assert!(Unknown.prepared_host_copy().is_none());
        assert!(Unknown.prepared_adaptive_commit().is_none());
        assert!(
            <MirostatV2Sampler as SpeculativeSampler<B>>::prepared_greedy_policy(&source).is_none()
        );
    }
    #[derive(Clone)]
    struct Plain {
        history: eredu_core::speculative::PlainControllerHistory,
        filters: [eredu_core::SharedTokenFilter; 1],
        reject_commit: bool,
    }
    impl eredu_core::TokenFilterController for Plain {
        type Error = eredu_core::speculative::PlainControllerError;
        fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
            Ok(self.filters[0].as_ref().clone())
        }
        fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
            self.history.try_push(token)
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    impl SpeculativeTokenFilterController for Plain {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
        fn prepared_plain_source(
            &self,
        ) -> Option<eredu_core::speculative::PlainControllerSource<'_>> {
            Some(eredu_core::speculative::PlainControllerSource::new(
                &self.history,
                &self.filters[0],
                eredu_core::TextControllerStorage::RunOwnedWithSharedFilters(&self.filters),
            ))
        }
        fn prepared_plain_copy_bytes(&self, capacity: usize) -> Option<usize> {
            eredu_core::speculative::PlainControllerHistory::copy_metadata_bytes(capacity)?
                .checked_add(size_of::<Self>())
        }
        fn copy_prepared_plain(
            &self,
            capacity: usize,
            host: HostPreparationAuthority,
        ) -> Result<Self, Self::Error> {
            Ok(Self {
                history: self.history.copy_prepared(capacity, host)?,
                filters: self.filters.clone(),
                reject_commit: self.reject_commit,
            })
        }
        fn prepared_plain_history_mut(
            &mut self,
        ) -> Option<&mut eredu_core::speculative::PlainControllerHistory> {
            (!self.reject_commit).then_some(&mut self.history)
        }
        fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error> {
            self.prepared_plain_source()
                .unwrap()
                .validate_history(history)?;
            Ok(self.filters[0].as_ref().clone())
        }
        fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
            self.prepared_plain_source()
                .unwrap()
                .validate_history(history)?;
            Ok(false)
        }
    }
    #[test]
    fn adaptive_forced_commit_keeps_controller_and_policy_snapshot_atomic() {
        type S = ConstrainedSampler<ConfiguredTextSampler, Plain>;
        for reject_commit in [false, true] {
            let mut source = S::new(
                ConfiguredTextSampler::MirostatV2(MirostatV2Sampler::new(3.5, 0.2).unwrap()),
                Plain {
                    history: Default::default(),
                    filters: [eredu_core::SharedTokenFilter::new(TokenFilter::Allowed(
                        vec![true; 16],
                    ))],
                    reject_commit,
                },
            );
            <S as SpeculativeSampler<B>>::control_force_next(
                &mut source,
                7,
                TokenDomain::new(16),
                0,
            )
            .unwrap();
            let snapshot = <S as SpeculativeSampler<B>>::prepared_controller(&source)
                .unwrap()
                .copy(host())
                .unwrap();
            let plan = <S as SpeculativeSampler<B>>::prepared_adaptive_commit(&source).unwrap();
            assert!(plan.validate_token(3).is_err());
            let result = plan.commit(7, 1.0, host());
            assert_eq!(
                <S as SpeculativeSampler<B>>::control_pending_forced(&source),
                Some(7)
            );
            assert_eq!(
                <S as SpeculativeSampler<B>>::control_pending_forced(&snapshot),
                Some(7)
            );
            assert!(
                adaptive_state(source.policy())
                    .generated_tokens()
                    .is_empty()
            );
            assert!(snapshot.controller().history.is_empty());
            if reject_commit {
                assert!(result.is_err());
            } else {
                let actual = result.unwrap();
                assert_eq!(&*actual.controller().history, &[7]);
                assert_eq!(adaptive_state(actual.policy()).generated_tokens(), &[7]);
                let mut expected = MirostatV2Sampler::new(3.5, 0.2).unwrap();
                expected.accept_token(7, 1.0).unwrap();
                assert_eq!(
                    adaptive_state(actual.policy()).mu().to_bits(),
                    expected.mu().to_bits()
                );
                assert_eq!(
                    <S as SpeculativeSampler<B>>::control_pending_forced(&actual),
                    None
                );
            }
        }
    }
}
