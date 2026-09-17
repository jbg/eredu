// Pre-refactor ordinary escape worker, preserved as the independent test reference.
use std::char::decode_utf16;
use std::iter::{once, repeat};
use std::str::Chars;
use crate::{Error, ErrorKind};
struct Unescaper {
    out: String,
    pending_surrogate: u16,
}

impl Unescaper {
    fn unescape(mut self, s: &str) -> Result<String, Error> {
        let mut char_iter = s.chars();

        while let Some(c) = char_iter.next() {
            if c == '\\' {
                match char_iter.next() {
                    None => return Err(ErrorKind::BadEscape.into()),
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
            Err(ErrorKind::BadEscape.into())
        } else {
            Ok(self.out)
        }
    }

    fn parse_u16(&self, chars: &mut Chars) -> Result<u16, Error> {
        let hexnum = chars.chain(repeat('\0')).take(4).collect::<String>();
        u16::from_str_radix(&hexnum, 16).map_err(|_| ErrorKind::BadEscape.into())
    }

    fn parse_hex_byte(&self, chars: &mut Chars) -> Result<u8, Error> {
        let hexnum = chars.take(2).collect::<String>();
        if hexnum.len() != 2 {
            return Err(ErrorKind::BadEscape.into());
        }
        u8::from_str_radix(&hexnum, 16).map_err(|_| ErrorKind::BadEscape.into())
    }

    fn parse_octal_byte(&self, first_digit: char, chars: &mut Chars) -> Result<u8, Error> {
        let mut octal_str = String::new();
        octal_str.push(first_digit);

        // Collect up to 2 more octal digits (0-7)
        for _ in 0..2 {
            let next_char = chars.as_str().chars().next();
            if let Some(c) = next_char {
                if ('0'..='7').contains(&c) {
                    octal_str.push(c);
                    chars.next(); // consume the character
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        u8::from_str_radix(&octal_str, 8).map_err(|_| ErrorKind::BadEscape.into())
    }

    fn push_u16(&mut self, c: u16) -> Result<(), Error> {
        match (self.pending_surrogate, (0xD800..=0xDFFF).contains(&c)) {
            (0, false) => match decode_utf16(once(c)).next() {
                Some(Ok(c)) => self.out.push(c),
                _ => return Err(ErrorKind::BadEscape.into()),
            },
            (_, false) => return Err(ErrorKind::BadEscape.into()),
            (0, true) => self.pending_surrogate = c,
            (prev, true) => match decode_utf16(once(prev).chain(once(c))).next() {
                Some(Ok(c)) => {
                    self.out.push(c);
                    self.pending_surrogate = 0;
                }
                _ => return Err(ErrorKind::BadEscape.into()),
            },
        }
        Ok(())
    }

    fn push_char(&mut self, c: char) -> Result<(), Error> {
        if self.pending_surrogate != 0 {
            Err(ErrorKind::BadEscape.into())
        } else {
            self.out.push(c);
            Ok(())
        }
    }
}

/// Un-escape a string, following Jinja-compatible string escape rules.
pub fn unescape(s: &str) -> Result<String, Error> {
    Unescaper {
        out: String::new(),
        pending_surrogate: 0,
    }
    .unescape(s)
}

