//! The ordinary chat normalization as a cloneable, allocation-free byte stream.
//! Matching deliberately preserves the existing raw tag scan, including tags
//! inside comments/strings and an unterminated final tag.
#[derive(Clone)]
pub(crate) struct Bytes<I> {
    input: I,
    copy: usize,
    skip: usize,
    replacement: std::slice::Iter<'static, u8>,
    opaque: bool,
}
impl<I> Bytes<I> {
    pub(crate) fn new(input: I) -> Self {
        Self {
            input,
            copy: 0,
            skip: 0,
            replacement: [].iter(),
            opaque: false,
        }
    }
}
impl<I: Iterator<Item = u8> + Clone> Iterator for Bytes<I> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        if self.copy != 0 {
            self.copy -= 1;
            return self.input.next();
        }
        while self.skip != 0 {
            self.input.next()?;
            self.skip -= 1;
        }
        if let Some(value) = self.replacement.next() {
            return Some(*value);
        }
        let value = self.input.next()?;
        if self.opaque || value != b'{' {
            return Some(value);
        }
        let mut probe = self.input.clone();
        if probe.next() != Some(b'%') {
            return Some(value);
        }
        let mut previous = b'%';
        let mut length = 1usize;
        let mut closed = false;
        for next in probe {
            length = length.checked_add(1)?;
            if previous == b'%' && next == b'}' {
                closed = true;
                break;
            }
            previous = next;
        }
        if !closed {
            self.opaque = true;
            return Some(value);
        }
        if length < 3 {
            self.copy = length;
            return Some(value);
        }
        let body = self.input.clone().skip(1).take(length - 3);
        if let Some((prefix, word, replacement)) = classify(body) {
            self.copy = 1 + prefix;
            self.skip = word;
            self.replacement = replacement.iter();
        } else {
            self.copy = length;
        }
        Some(value)
    }
}
fn character(input: &mut impl Iterator<Item = u8>) -> Option<Result<char, ()>> {
    let first = input.next()?;
    let count = match first {
        0..=127 => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return Some(Err(())),
    };
    let mut bytes = [0u8; 4];
    bytes[0] = first;
    for byte in &mut bytes[1..count] {
        let Some(value) = input.next() else {
            return Some(Err(()));
        };
        *byte = value;
    }
    Some(
        std::str::from_utf8(&bytes[..count])
            .ok()
            .and_then(|s| s.chars().next())
            .ok_or(()),
    )
}
fn classify(mut input: impl Iterator<Item = u8>) -> Option<(usize, usize, &'static [u8])> {
    // trim().trim_matches('-').trim(), followed by an exact identifier.
    let (mut phase, mut prefix, mut used) = (0u8, 0usize, 0usize);
    let mut word = [0u8; 13];
    while let Some(value) = character(&mut input) {
        let value = value.ok()?;
        let space = value.is_whitespace();
        match phase {
            0 if space => {
                prefix += value.len_utf8();
                continue;
            }
            0 if value == '-' => {
                phase = 1;
                prefix += 1;
                continue;
            }
            1 if value == '-' => {
                prefix += 1;
                continue;
            }
            1 | 2 if space => {
                phase = 2;
                prefix += value.len_utf8();
                continue;
            }
            0..=2 => phase = 3,
            _ => {}
        }
        match phase {
            3 if space => phase = 4,
            3 if value == '-' => phase = 5,
            3 if value.is_ascii_lowercase() && used < word.len() => {
                word[used] = value as u8;
                used += 1;
            }
            4 if space => {}
            4 if value == '-' => phase = 5,
            5 if value == '-' => {}
            5 | 6 if space => phase = 6,
            _ => return None,
        }
    }
    match &word[..used] {
        b"generation" => Some((prefix, used, b"if true")),
        b"endgeneration" => Some((prefix, used, b"endif")),
        _ => None,
    }
}
pub(crate) fn control_bytes<I>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Bytes<I>>(),
        size_of::<I>(),
        size_of::<I>(),
        size_of::<std::iter::Take<std::iter::Skip<I>>>(),
        size_of::<[u8; 13]>(),
        size_of::<[u8; 4]>(),
        size_of::<[usize; 6]>(),
        size_of::<[u8; 4]>(),
        size_of::<Option<Result<char, ()>>>(),
        size_of::<Option<(usize, usize, &'static [u8])>>(),
        size_of::<(&mut I, Option<u8>)>(),
        size_of::<bool>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

#[cfg(test)]
mod tests {
    use super::Bytes;
    #[test]
    fn normalization_preserves_raw_scan_unicode_and_cloned_cursor() {
        for (source,expected) in [
            ("{% generation %}é{% endgeneration %}", "{% if true %}é{% endif %}"),
            ("{%\u{2003}generation\u{2003}%}界{%endgeneration%}", "{%\u{2003}if true\u{2003}%}界{%endif%}"),
            ("{# {% generation %} #}", "{# {% if true %} #}"),
            ("{% generation", "{% generation"),
            ("{% other '{% generation %}' %}", "{% other '{% generation %}' %}"),
            ("{%- -- generation -%}", "{%- -- generation -%}"),
            ("{%}", "{%}"),
        ] {
            let iterator=Bytes::new(source.bytes());
            assert_eq!(String::from_utf8(iterator.clone().collect()).unwrap(),expected);
            let mut partial=iterator;
            let prefix:Vec<_>=partial.by_ref().take(4).collect();
            let tail=partial.clone().collect::<Vec<_>>();
            assert_eq!(partial.collect::<Vec<_>>(),tail);
            assert_eq!([prefix,tail].concat(),expected.as_bytes());
        }
    }
}
