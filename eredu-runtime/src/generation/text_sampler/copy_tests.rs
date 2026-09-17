use super::*;

const PREFIX: [u32; 5] = [3, 11, 7, 19, 5];

fn standard() -> GenerationSampler {
    let mut sampler = GenerationSampler::new()
        .top_k(17)
        .top_p(0.83)
        .min_p(0.07)
        .penalties(1.13, 23, 0.17, 0.29);
    for token in PREFIX {
        sampler.accept_token(token);
    }
    sampler
}

fn adaptive() -> MirostatV2Sampler {
    let mut sampler = MirostatV2Sampler::new(3.7, 0.23)
        .unwrap()
        .penalties(1.19, 31, 0.13, 0.41);
    for (token, probability) in PREFIX.into_iter().zip([0.2, 0.3, 0.1, 0.4, 0.05]) {
        sampler.accept_token(token, probability).unwrap();
    }
    sampler
}

fn assert_standard_state(source: &GenerationSampler, copied: &GenerationSampler) {
    assert_eq!(source.top_k, copied.top_k);
    assert_eq!(source.top_p.to_bits(), copied.top_p.to_bits());
    assert_eq!(source.min_p.to_bits(), copied.min_p.to_bits());
    assert_eq!(source.penalty_config(), copied.penalty_config());
    assert_eq!(source.generated_tokens(), copied.generated_tokens());
    assert_eq!(
        source.generated_tokens.capacity(),
        copied.generated_tokens.capacity()
    );
}

#[test]
fn standard_copy_prices_spare_box_capacity_and_retains_independent_history() {
    let source = ConfiguredTextSampler::Standard(standard());
    let plan = source.prepare_copy().unwrap();
    assert_eq!(plan.history_len(), 5);
    assert_eq!(plan.history_capacity(), 8);
    assert_eq!(plan.history_bytes(), 32);
    assert_eq!(
        plan.retained_bytes(),
        std::mem::size_of::<ConfiguredTextSampler>() as u64 + 32
    );
    let copied = plan.copy();
    let ConfiguredTextSampler::Standard(original) = &source else {
        panic!("standard source changed variant")
    };
    let ConfiguredTextSampler::Standard(mut child) = copied else {
        panic!("standard copy changed variant")
    };
    assert_standard_state(original, &child);
    assert_ne!(
        original.generated_tokens().as_ptr(),
        child.generated_tokens().as_ptr()
    );
    child.accept_token(29);
    child.top_k = 2;
    assert_eq!(original.generated_tokens(), PREFIX);
    assert_eq!(original.top_k, 17);
    drop(source);
    child.accept_token(31);
    assert_eq!(child.generated_tokens(), &[3, 11, 7, 19, 5, 29, 31]);
    assert_eq!(child.generated_tokens.capacity(), 8);
}

#[test]
fn cleared_histories_keep_their_nonzero_destination_payload() {
    for mut source in [
        ConfiguredTextSampler::Standard(standard()),
        ConfiguredTextSampler::MirostatV2(adaptive()),
    ] {
        match &mut source {
            ConfiguredTextSampler::Standard(sampler) => sampler.clear_generated_tokens(),
            ConfiguredTextSampler::MirostatV2(sampler) => sampler.reset(),
        }
        let plan = source.prepare_copy().unwrap();
        assert_eq!(plan.history_len(), 0);
        assert_eq!(plan.history_capacity(), 8);
        assert_eq!(plan.history_bytes(), 32);
        let mut copied = plan.copy();
        assert_eq!(copied.history_len(), 0);
        assert_eq!(copied.history_capacity(), 8);
        assert_ne!(
            source.history().as_slice().as_ptr(),
            copied.history().as_slice().as_ptr()
        );
        match &mut copied {
            ConfiguredTextSampler::Standard(sampler) => sampler.accept_token(43),
            ConfiguredTextSampler::MirostatV2(sampler) => sampler.accept_token(43, 0.25).unwrap(),
        }
        assert_eq!(source.history_len(), 0);
        assert_eq!(source.history_capacity(), 8);
        assert_eq!(copied.history().as_slice(), &[43]);
        assert_eq!(copied.history_capacity(), 8);
    }
}

#[test]
fn adaptive_copy_preserves_updated_mu_and_allows_independent_continuations() {
    let source = ConfiguredTextSampler::MirostatV2(adaptive());
    let plan = source.prepare_copy().unwrap();
    assert_eq!((plan.history_len(), plan.history_capacity()), (5, 8));
    assert_eq!(plan.history_bytes(), 32);
    let copied = plan.copy();
    let ConfiguredTextSampler::MirostatV2(original) = &source else {
        panic!("adaptive source changed variant")
    };
    let ConfiguredTextSampler::MirostatV2(mut child) = copied else {
        panic!("adaptive copy changed variant")
    };
    assert_ne!(original.mu().to_bits(), (2.0 * original.tau()).to_bits());
    assert_eq!(original.tau().to_bits(), child.tau().to_bits());
    assert_eq!(original.eta().to_bits(), child.eta().to_bits());
    assert_eq!(original.mu().to_bits(), child.mu().to_bits());
    assert_standard_state(&original.penalties, &child.penalties);
    assert_ne!(
        original.generated_tokens().as_ptr(),
        child.generated_tokens().as_ptr()
    );
    let source_mu = original.mu();
    child.accept_token(47, 0.125).unwrap();
    assert_ne!(source_mu.to_bits(), child.mu().to_bits());
    assert_eq!(original.mu().to_bits(), source_mu.to_bits());
    assert_eq!(original.generated_tokens(), PREFIX);
    let copied_mu = child.mu();
    drop(source);
    child.accept_token(53, 0.5).unwrap();
    assert_ne!(copied_mu.to_bits(), child.mu().to_bits());
    assert_eq!(child.generated_tokens(), &[3, 11, 7, 19, 5, 47, 53]);
}

#[test]
fn existing_clone_paths_preserve_the_planned_capacity_and_policy() {
    let standard = standard();
    assert_standard_state(&standard, &standard.clone());
    let adaptive = adaptive();
    let direct = adaptive.clone();
    assert_standard_state(&adaptive.penalties, &direct.penalties);
    assert_eq!(adaptive.mu().to_bits(), direct.mu().to_bits());

    for source in [
        ConfiguredTextSampler::Standard(standard),
        ConfiguredTextSampler::MirostatV2(adaptive),
    ] {
        let plan = source.prepare_copy().unwrap();
        let expected = plan.retained_bytes();
        // Dropping an inspected plan allocates no destination and leaves the
        // source available to the existing Clone surface.
        drop(plan);
        let copied = source.clone();
        assert_eq!(copied.prepare_copy().unwrap().retained_bytes(), expected);
        assert_eq!(source.history().as_slice(), copied.history().as_slice());
        assert_ne!(
            source.history().as_slice().as_ptr(),
            copied.history().as_slice().as_ptr()
        );
    }
}

#[test]
fn empty_new_samplers_have_only_the_inline_managed_copy_cost() {
    for source in [
        ConfiguredTextSampler::Standard(GenerationSampler::default()),
        ConfiguredTextSampler::MirostatV2(MirostatV2Sampler::default()),
    ] {
        let plan = source.prepare_copy().unwrap();
        assert_eq!(plan.history_len(), 0);
        assert_eq!(plan.history_capacity(), 0);
        assert_eq!(plan.history_bytes(), 0);
        assert_eq!(
            plan.retained_bytes(),
            std::mem::size_of::<ConfiguredTextSampler>() as u64
        );
        let copied = plan.copy();
        assert_eq!(copied.history_capacity(), 0);
        assert_eq!(source.history_len(), 0);
    }
}
