//! Shared literal decoding with ordinary, counting and fixed-capacity destinations.
#![forbid(unsafe_code)]
use std::char::decode_utf16;
use std::iter::{once, repeat};
use std::str::Chars;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    BadEscape,
    Capacity,
}

pub(crate) enum Output<'a> {
    Ordinary(&'a mut String),
    Count(&'a mut usize),
    Packed {
        bytes: &'a mut Vec<u8>,
        limit: usize,
    },
}
impl Output<'_> {
    fn push(&mut self, c: char) -> Result<(), Failure> {
        match self {
            Self::Ordinary(out) => out.push(c),
            Self::Count(count) => {
                **count = count.checked_add(c.len_utf8()).ok_or(Failure::Capacity)?
            }
            Self::Packed { bytes, limit } => {
                let end = bytes
                    .len()
                    .checked_add(c.len_utf8())
                    .ok_or(Failure::Capacity)?;
                if end > *limit || end > bytes.capacity() {
                    return Err(Failure::Capacity);
                }
                let mut buf = [0; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
        Ok(())
    }
}

pub(crate) fn decode(source: &str, out: Output<'_>) -> Result<(), Failure> {
    Unescaper {
        out,
        pending_surrogate: 0,
    }
    .unescape(source)
}

struct Unescaper<'a> {
    out: Output<'a>,
    pending_surrogate: u16,
}

impl Unescaper<'_> {
    fn unescape(mut self, s: &str) -> Result<(), Failure> {
        let mut char_iter = s.chars();

        while let Some(c) = char_iter.next() {
            if c == '\\' {
                match char_iter.next() {
                    None => return Err(Failure::BadEscape),
                    Some(d) => match d {
                        '"' | '\\' | '/' | '\'' => ok!(self.push_char(d)),
                        'b' => ok!(self.push_char('\x08')),
                        'f' => ok!(self.push_char('\x0C')),
                        'n' => ok!(self.push_char('\n')),
                        'r' => ok!(self.push_char('\r')),
                        't' => ok!(self.push_char('\t')),
                        'u' => {
                            let val = ok!(self.parse_u16(&mut char_iter));
                            ok!(self.push_u16(val));
                        }
                        'x' => {
                            let val = ok!(self.parse_hex_byte(&mut char_iter));
                            ok!(self.push_char(val as char));
                        }
                        '0'..='7' => {
                            let val = ok!(self.parse_octal_byte(d, &mut char_iter));
                            ok!(self.push_char(val as char));
                        }
                        _ => {
                            ok!(self.push_char('\\'));
                            ok!(self.push_char(d));
                        }
                    },
                }
            } else {
                ok!(self.push_char(c));
            }
        }

        if self.pending_surrogate != 0 {
            Err(Failure::BadEscape)
        } else {
            Ok(())
        }
    }

    fn parse_u16(&self, chars: &mut Chars) -> Result<u16, Failure> {
        // Preserve from_str_radix (including leading '+'), Unicode consumption
        // and the ordinary zero-padding of fewer than four source characters.
        let mut bytes = [0; 16];
        let mut len = 0;
        for c in chars.chain(repeat('\0')).take(4) {
            len += c.encode_utf8(&mut bytes[len..]).len();
        }
        let digits = std::str::from_utf8(&bytes[..len]).expect("encoded UTF-8");
        u16::from_str_radix(digits, 16).map_err(|_| Failure::BadEscape)
    }
    fn parse_hex_byte(&self, chars: &mut Chars) -> Result<u8, Failure> {
        let mut bytes = [0; 8];
        let mut len = 0;
        for c in chars.take(2) {
            len += c.encode_utf8(&mut bytes[len..]).len();
        }
        if len != 2 {
            return Err(Failure::BadEscape);
        }
        let digits = std::str::from_utf8(&bytes[..len]).expect("encoded UTF-8");
        u8::from_str_radix(digits, 16).map_err(|_| Failure::BadEscape)
    }
    fn parse_octal_byte(&self, first_digit: char, chars: &mut Chars) -> Result<u8, Failure> {
        let mut bytes = [0; 3];
        bytes[0] = first_digit as u8;
        let mut len = 1;
        for _ in 0..2 {
            if let Some(c @ '0'..='7') = chars.as_str().chars().next() {
                bytes[len] = c as u8;
                len += 1;
                chars.next();
            } else {
                break;
            }
        }
        let digits = std::str::from_utf8(&bytes[..len]).expect("ASCII octal digits");
        u8::from_str_radix(digits, 8).map_err(|_| Failure::BadEscape)
    }

    fn push_u16(&mut self, c: u16) -> Result<(), Failure> {
        match (self.pending_surrogate, (0xD800..=0xDFFF).contains(&c)) {
            (0, false) => match decode_utf16(once(c)).next() {
                Some(Ok(c)) => ok!(self.out.push(c)),
                _ => return Err(Failure::BadEscape),
            },
            (_, false) => return Err(Failure::BadEscape),
            (0, true) => self.pending_surrogate = c,
            (prev, true) => match decode_utf16(once(prev).chain(once(c))).next() {
                Some(Ok(c)) => {
                    ok!(self.out.push(c));
                    self.pending_surrogate = 0;
                }
                _ => return Err(Failure::BadEscape),
            },
        }
        Ok(())
    }

    fn push_char(&mut self, c: char) -> Result<(), Failure> {
        if self.pending_surrogate != 0 {
            Err(Failure::BadEscape)
        } else {
            ok!(self.out.push(c));
            Ok(())
        }
    }
}
