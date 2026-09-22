//! Common concrete original file-byte destination, private to the two fresh
//! tokenizer/chat constructors. No public adoption, extraction or callback API.
use super::{MemoryLedger, WorkingMemoryError, loaded_decode_source::Allowance};
use eredu_checkpoint::artifact::{ArtifactFileReadFailure, PreparedArtifactFileRead};
use std::{collections::TryReserveError, mem::size_of};

#[derive(Clone, Copy, Debug)]
pub(super) enum Destination {
    Tokenizer,
    Chat,
}

#[derive(Debug)]
pub(super) struct Input {
    pub(super) read: Option<PreparedArtifactFileRead>,
    pub(super) bytes: Vec<u8>,
    pub(super) filled: usize,
    pub(super) allowance: Allowance,
}
#[derive(Debug)]
pub(super) enum Cause {
    Accounting(WorkingMemoryError),
    Reserve(TryReserveError),
    Read(ArtifactFileReadFailure),
}
#[derive(Debug)]
pub(super) struct Failure {
    pub(super) cause: Cause,
    pub(super) settlement: Option<WorkingMemoryError>,
    pub(super) input: Option<Input>,
}
impl Failure {
    fn rejected(cause: WorkingMemoryError) -> Self {
        Self {
            cause: Cause::Accounting(cause),
            settlement: None,
            input: None,
        }
    }
    fn terminal(cause: Cause, mut input: Input) -> Self {
        let settlement = input.allowance.end_compilation().err();
        Self {
            cause,
            settlement,
            input: Some(input),
        }
    }
}
/// Fixed common local/return population, added to each concrete destination's
/// original comparison. Every caller is a private named fresh constructor.
pub(super) fn control_bytes() -> Option<usize> {
    [
        PreparedArtifactFileRead::control_bytes()?,
        size_of::<Destination>(),
        size_of::<u64>(),
        size_of::<Result<u64, WorkingMemoryError>>(),
        size_of::<Input>(),
        size_of::<Option<Input>>(),
        size_of::<Cause>(),
        size_of::<Failure>(),
        size_of::<Result<Input, Failure>>(),
        size_of::<Vec<u8>>(),
        size_of::<usize>(), // sealed length
        size_of::<usize>(), // actual target reserve request
        size_of::<Allowance>(),
        size_of::<Result<Allowance, WorkingMemoryError>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<(), ArtifactFileReadFailure>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
/// Admit the caller's concrete checked I population and execute the one actual
/// reserve/read. Returns idle retained bytes so fresh J/C coexist in the pool.
pub(super) fn read(
    pool: &MemoryLedger,
    read: PreparedArtifactFileRead,
    destination: Destination,
    after_admission: impl FnOnce(&mut usize),
) -> Result<Input, Failure> {
    let required = match destination {
        Destination::Tokenizer => MemoryLedger::tokenizer_file_required_bytes(&read),
        Destination::Chat => MemoryLedger::chat_template_file_required_bytes(&read),
    }
    .map_err(Failure::rejected)?;
    let allowance = pool
        .admit_source_compiler(required)
        .map_err(Failure::rejected)?;
    let length = read.byte_len();
    let mut input = Input {
        read: Some(read),
        bytes: Vec::new(),
        filled: 0,
        allowance,
    };
    let mut requested = length;
    // Both production callers supply a zero-sized no-op. Private tests select
    // overflow on this actual target or observe installed original custody.
    after_admission(&mut requested);
    if let Err(error) = input.bytes.try_reserve_exact(requested) {
        return Err(Failure::terminal(Cause::Reserve(error), input));
    }
    input.bytes.resize(length, 0);
    let result = input
        .read
        .take()
        .expect("one prepared file")
        .read_into(&mut input.bytes);
    match result {
        Ok(()) => input.filled = length,
        Err(error) => {
            input.filled = error.filled_bytes();
            return Err(Failure::terminal(Cause::Read(error), input));
        }
    }
    if let Err(error) = input.allowance.end_compilation() {
        return Err(Failure {
            cause: Cause::Accounting(error),
            settlement: None,
            input: Some(input),
        });
    }
    Ok(input)
}
