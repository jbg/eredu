//! Unicode uppercase traversal shared by ordinary and bounded string filters.
use std::fmt::{self, Write};
pub(crate) fn upper<W: Write>(out: &mut W, value: &str) -> fmt::Result {
    for character in value.chars() {
        for mapped in character.to_uppercase() {
            out.write_char(mapped)?;
        }
    }
    Ok(())
}
pub(crate) fn control_bytes<W>() -> Option<usize> {
    use std::mem::size_of;
    size_of::<(&mut W, &str)>()
        .checked_add(size_of::<std::str::Chars<'static>>())?
        .checked_add(size_of::<std::char::ToUppercase>())?
        .checked_add(size_of::<[char; 2]>())?
        .checked_add(size_of::<[u8; 4]>())?
        .checked_add(size_of::<fmt::Result>())
}

pub(crate) fn ascii_fold_cmp(left:&str,right:&str)->std::cmp::Ordering {
    left.bytes().map(|byte|byte.to_ascii_lowercase())
        .cmp(right.bytes().map(|byte|byte.to_ascii_lowercase()))
}
pub(crate) fn ascii_fold_control_bytes()->Option<usize> {
    use std::mem::{size_of,size_of_val};
    // Name the exact iterator expression used by the comparator; each closure
    // captures no state and only changes ASCII bytes, preserving other UTF-8.
    let iter="".bytes().map(|byte|byte.to_ascii_lowercase());
    size_of_val(&iter).checked_mul(2)?
        .checked_add(size_of::<(&str,&str)>())?
        .checked_add(size_of::<[Option<u8>;2]>())?
        .checked_add(size_of::<std::cmp::Ordering>())
}
