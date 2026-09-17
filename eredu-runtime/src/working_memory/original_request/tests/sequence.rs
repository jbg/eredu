use super::*;
#[derive(Debug)]
pub(super) struct OutputPayload {
    slots: [u32; OUTPUTS],
    committed: usize,
    eos: [u32; 1],
    owner: AcceptedOwner,
}
impl GenerationTokenIdStorage for OutputPayload {
    fn token_ids(&self) -> &[u32] {
        &self.slots[..self.committed]
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug)]
pub(super) struct FixedSequence {
    payload: Option<Arc<OutputPayload>>,
}
impl FixedSequence {
    pub fn new(owner: AcceptedOwner) -> Self {
        Self {
            payload: Some(Arc::new(OutputPayload {
                slots: [0; OUTPUTS],
                committed: 0,
                eos: [99],
                owner,
            })),
        }
    }
    // Concrete move retires the Box allocation before its source/account value.
    #[inline(never)]
    fn unbox(value: Box<Self>) -> Self {
        *value
    }
}
impl Drop for FixedSequence {
    fn drop(&mut self) {
        if let Some(value) = self.payload.take() {
            drop(Arc::into_inner(value));
        }
    }
}
impl RetainedGenerationStorage for FixedSequence {
    fn max_tokens(&self) -> usize {
        OUTPUTS
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.payload.as_ref().unwrap().eos
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn token_slots(&self) -> &[u32] {
        &self.payload.as_ref().unwrap().slots
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        &mut Arc::get_mut(self.payload.as_mut().unwrap()).unwrap().slots
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let mut source = Self::unbox(self);
        let mut payload = source.payload.take().unwrap();
        Arc::get_mut(&mut payload).unwrap().committed = committed;
        GenerationTokenIds::from_owner(payload)
    }
    fn retire(self: Box<Self>) {
        drop(Self::unbox(self));
    }
}
