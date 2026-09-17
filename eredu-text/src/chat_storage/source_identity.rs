//! Original decoded template identity survives source normalization.
use sha2::{Digest, Sha256};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourceIdentity {
    bytes: usize,
    sha256: [u8; 32],
}
impl SourceIdentity {
    pub(super) fn new(input: impl Iterator<Item = u8>) -> Self {
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 256];
        let (mut used, mut bytes) = (0usize, 0usize);
        // Inputs are finite views of representable UTF-8/config byte slices.
        for value in input {
            buffer[used] = value;
            used += 1;
            bytes += 1;
            if used == buffer.len() {
                hash.update(buffer);
                used = 0;
            }
        }
        hash.update(&buffer[..used]);
        Self {
            bytes,
            sha256: hash.finalize().into(),
        }
    }
}
pub(super) fn control_bytes<I>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<I>(),
        size_of::<SourceIdentity>(),
        size_of::<Sha256>(),
        size_of::<[u8; 256]>(),
        size_of::<[usize; 3]>(),
        size_of::<Option<u8>>(),
        size_of::<[u8; 32]>(),
        size_of::<sha2::digest::Output<Sha256>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
