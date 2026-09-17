use super::*;
use eredu_core::SharedControllerBytes;

struct MixedController {
    filter: SharedController,
    bytes: [SharedControllerBytes; 1],
    replace_on_decision: bool,
}

impl MixedController {
    fn new(bytes: SharedControllerBytes, calls: Rc<Cell<(usize, usize)>>) -> Self {
        Self {
            filter: SharedController::new(mask(193), calls),
            bytes: [bytes],
            replace_on_decision: false,
        }
    }
}

impl TokenFilterController for MixedController {
    type Error = std::convert::Infallible;

    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithSharedStorage {
            filters: &self.filter.masks,
            bytes: &self.bytes,
        }
    }

    fn inference_workspace(&self, outputs: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        let mut workspace = self.filter.inference_workspace(outputs)?;
        // Include the shared source plus a possible controller-owned replacement
        // overlap. Only the fixed shared source can receive existing-root credit.
        workspace.additional_host_bytes = workspace
            .additional_host_bytes
            .checked_add(self.bytes[0].capacity_bytes()?.checked_mul(2)?)?;
        Some(workspace)
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.filter.current_filter()
    }

    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Self::Error> {
        if self.replace_on_decision {
            self.bytes[0] = SharedControllerBytes::new(payload(257));
            self.replace_on_decision = false;
        }
        let filter = self.filter.current_filter()?;
        Ok(TokenSamplingDecision::new(filter)
            .with_shared_tokenizer_validity(&self.filter.masks[0])
            .with_controller_storage(self.inference_storage()))
    }

    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert_eq!(token, u32::from(self.bytes[0].as_ref()[0]));
        self.filter.commit_token(token)
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.filter.is_complete()
    }
}

fn payload(capacity: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&[11, 17, 29, 3]);
    bytes
}

fn run(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    controller: MixedController,
    controlled: bool,
) -> Vec<MlxTextToken> {
    if controlled {
        ControlledTextGeneration::from_input(
            runtime,
            TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
            config(Some(u64::MAX)),
            controller,
        )
        .unwrap()
        .map(|token| token.unwrap().into_output())
        .collect()
    } else {
        let mut driver = TextGenerationDriver::new(runtime);
        let mut continuation = driver
            .start_input(
                TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
                config(Some(u64::MAX)),
                controller,
            )
            .unwrap();
        let mut outputs = Vec::new();
        while let Some(token) = driver.advance(&mut continuation).unwrap() {
            outputs.push(token.into_output());
            assert!(driver
                .take_completed_step(&mut continuation)
                .unwrap()
                .is_none());
        }
        outputs
    }
}

#[test]
fn mixed_sources_preserve_each_domain_charge_in_ordinary_and_controlled_runs() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let source = SharedControllerBytes::new(payload(257));
    let identity = source.identity().clone();
    let pointer = source.as_ref().as_ptr();
    let bytes = source.capacity_bytes().unwrap();
    let mut pools = Vec::new();
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&stream, &pool);
        // This alias exists before the run attaches its domain's funding.
        let controller_source = source.clone();
        let calls = Rc::new(Cell::new((0, 0)));
        let outputs = run(
            &mut runtime,
            MixedController::new(controller_source, calls.clone()),
            controlled,
        );
        assert_eq!(
            outputs
                .iter()
                .map(|token| token.token_id().unwrap())
                .collect::<Vec<_>>(),
            [11, 11, 11]
        );
        assert_eq!(calls.get(), (3, 3));
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(7)
            .unwrap();
        assert_eq!(source.as_ref().as_ptr(), pointer);
        assert_eq!(source.identity(), &identity);
        drop((outputs, runtime, artifact));
        settle(&pool, bytes);
        pools.push(pool);
    }
    assert_eq!(source.as_ref(), &[11, 17, 29, 3]);
    for pool in &pools {
        assert_eq!(pool.used_bytes().unwrap(), bytes);
    }
    drop(source);
    for pool in &pools {
        settle(pool, 0);
    }
    // Keeping identity metadata cannot keep either domain charge alive.
    drop(identity);
}

#[test]
fn byte_loading_hook_publishes_before_inference_and_rejects_an_active_run_factory() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&stream, &pool);
    let initial = pool.used_bytes().unwrap();
    let factories = Cell::new(0);
    let original_pointer = Cell::new(std::ptr::null());
    let source = MlxBackend::prepare_shared_controller_bytes(&runtime, || {
        factories.set(factories.get() + 1);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        let bytes = payload(513);
        original_pointer.set(bytes.as_ptr());
        bytes
    })
    .unwrap();
    assert_eq!(factories.get(), 1);
    assert_eq!(source.capacity_bytes(), Some(513));
    assert_eq!(source.as_ref().as_ptr(), original_pointer.get());
    assert_eq!(pool.used_bytes().unwrap(), initial + 513);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let external = source.clone();
    let calls = Rc::new(Cell::new((0, 0)));
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
            config(Some(u64::MAX)),
            MixedController::new(source, calls.clone()),
        )
        .unwrap();
    let before = NoWork::capture(driver.runtime(), &pool);
    let error = MlxBackend::prepare_shared_controller_bytes(driver.runtime(), || {
        factories.set(factories.get() + 1);
        payload(1025)
    })
    .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert_eq!(factories.get(), 1);
    assert_eq!(calls.get(), (0, 0));
    before.assert_unchanged(driver.runtime(), &pool);

    let mut outputs = Vec::new();
    while let Some(token) = driver.advance(&mut continuation).unwrap() {
        outputs.push(token.into_output());
        assert!(driver
            .take_completed_step(&mut continuation)
            .unwrap()
            .is_none());
    }
    assert_eq!(calls.get(), (3, 3));
    drop((continuation, driver, outputs));
    drop((runtime, artifact));
    settle(&pool, 513);
    assert_eq!(external.as_ref(), &[11, 17, 29, 3]);
    drop(external);
    settle(&pool, 0);
}

#[test]
fn same_size_byte_owner_replacement_rejects_at_preflight_or_after_decision_before_native_work() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for during_decision in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&stream, &pool);
        let source = SharedControllerBytes::new(payload(257));
        let external = source.clone();
        let calls = Rc::new(Cell::new((0, 0)));
        let mut controller = MixedController::new(source, calls.clone());
        controller.replace_on_decision = during_decision;
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut continuation = driver
            .start_input(
                TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
                config(Some(u64::MAX)),
                controller,
            )
            .unwrap();
        if !during_decision {
            let replacement = SharedControllerBytes::new(payload(257));
            assert_eq!(replacement.capacity_bytes(), external.capacity_bytes());
            assert_ne!(replacement.identity(), external.identity());
            continuation.controller_mut().bytes[0] = replacement;
        }
        let before = NoWork::capture(driver.runtime(), &pool);
        let error = driver.advance(&mut continuation).err().unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(calls.get(), (usize::from(during_decision), 0));
        before.assert_unchanged(driver.runtime(), &pool);
        drop((error, continuation, driver));
        drop((runtime, artifact));
        settle(&pool, 257);
        drop(external);
        settle(&pool, 0);
    }
}
