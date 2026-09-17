//! Ordinary generated call identifiers with an allocation-free writer seam.
use std::{
    fmt,
    mem::{size_of, size_of_val},
};
const DIGITS: usize = usize::MAX.ilog10() as usize + 1;
/// Maximum actual bytes in the ordinary `call_` plus usize decimal spelling.
pub const GENERATED_CALL_ID_BYTES: usize = 5 + DIGITS;
/// Writes the ordinary exact identifier into the caller's own destination.
/// The caller retains source policy and pays any destination allocation.
pub fn write_generated_call_id<W: fmt::Write>(
    destination: &mut W,
    mut index: usize,
) -> fmt::Result {
    let mut digits = [0u8; DIGITS];
    let mut start = DIGITS;
    loop {
        start -= 1;
        digits[start] = b'0' + (index % 10) as u8;
        index /= 10;
        if index == 0 {
            break;
        }
    }
    destination.write_str("call_")?;
    destination.write_str(std::str::from_utf8(&digits[start..]).expect("decimal digits"))
}
/// Fixed writer loan, exact digit array and failure controls.
pub fn generated_call_id_control_bytes<W: fmt::Write>() -> Option<usize> {
    let parts = [
        size_of::<[u8; DIGITS]>(),
        size_of::<(usize, usize)>(),
        size_of::<(&mut W, usize)>(),
        size_of::<Result<&str, std::str::Utf8Error>>(),
        size_of::<fmt::Result>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
