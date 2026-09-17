//! Fixed-stack validation and borrowed strings. No serde Value or owned key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    InvalidJson,
    DepthLimit,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Error {
    pub kind: Kind,
    pub offset: usize,
}

pub(crate) const DEPTH: usize = 128;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Span {
    pub start: usize,
    pub end: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Text<'a> {
    pub raw: &'a str,
    pub offset: usize,
}
impl<'a> Text<'a> {
    pub fn bytes(self) -> Bytes<'a> {
        Bytes {
            chars: self.raw.chars(),
            buffer: [0; 4],
            at: 0,
            len: 0,
        }
    }
    pub fn is(self, expected: &str) -> bool {
        self.bytes().eq(expected.bytes())
    }
    pub fn len(self) -> usize {
        self.bytes().count()
    }
}
#[derive(Clone)]
pub(crate) struct Bytes<'a> {
    chars: std::str::Chars<'a>,
    buffer: [u8; 4],
    at: usize,
    len: usize,
}
impl Iterator for Bytes<'_> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        if self.at < self.len {
            let b = self.buffer[self.at];
            self.at += 1;
            return Some(b);
        }
        let c = match self.chars.next()? {
            '\\' => match self.chars.next().expect("validated escape") {
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                'b' => '\u{8}',
                'f' => '\u{c}',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                'u' => {
                    let mut read = || {
                        (0..4).fold(0, |n, _| {
                            n * 16 + self.chars.next().expect("hex").to_digit(16).expect("hex")
                        })
                    };
                    let high = read();
                    let scalar = if (0xd800..=0xdbff).contains(&high) {
                        self.chars.next();
                        self.chars.next();
                        let low = (0..4).fold(0, |n, _| {
                            n * 16 + self.chars.next().expect("hex").to_digit(16).expect("hex")
                        });
                        0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
                    } else {
                        high
                    };
                    char::from_u32(scalar).expect("validated scalar")
                }
                _ => unreachable!("validated escape"),
            },
            c => c,
        };
        self.len = c.encode_utf8(&mut self.buffer).len();
        self.at = 1;
        Some(self.buffer[0])
    }
}

#[derive(Clone, Copy)]
enum Frame {
    ObjectFirst,
    ObjectKey,
    Colon,
    ObjectValue,
    ObjectComma,
    ArrayFirst,
    ArrayValue,
    ArrayComma,
}
pub(crate) struct Reader<'a> {
    pub input: &'a str,
    pub pos: usize,
    end: usize,
}
impl<'a> Reader<'a> {
    pub fn new(input: &'a str, span: Span) -> Self {
        Self {
            input,
            pos: span.start,
            end: span.end,
        }
    }
    fn err(&self, kind: Kind) -> Error {
        Error {
            kind,
            offset: self.pos,
        }
    }
    fn peek(&self) -> Option<u8> {
        self.input
            .as_bytes()
            .get(self.pos)
            .copied()
            .filter(|_| self.pos < self.end)
    }
    pub fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }
    pub fn byte(&mut self, b: u8) -> Result<(), Error> {
        self.ws();
        if self.peek() != Some(b) {
            return Err(self.err(Kind::InvalidJson));
        }
        self.pos += 1;
        Ok(())
    }
    pub fn finish(&mut self) -> Result<(), Error> {
        self.ws();
        if self.pos == self.end {
            Ok(())
        } else {
            Err(self.err(Kind::InvalidJson))
        }
    }
    fn hex(&mut self) -> Result<u32, Error> {
        let mut n = 0;
        for _ in 0..4 {
            let b = self.peek().ok_or_else(|| self.err(Kind::InvalidJson))?;
            let d = char::from(b)
                .to_digit(16)
                .ok_or_else(|| self.err(Kind::InvalidJson))?;
            n = n * 16 + d;
            self.pos += 1;
        }
        Ok(n)
    }
    pub fn string(&mut self) -> Result<Text<'a>, Error> {
        self.byte(b'"')?;
        let start = self.pos;
        loop {
            match self.peek().ok_or_else(|| self.err(Kind::InvalidJson))? {
                b'"' => {
                    let end = self.pos;
                    self.pos += 1;
                    return Ok(Text {
                        raw: &self.input[start..end],
                        offset: start,
                    });
                }
                0..=31 => return Err(self.err(Kind::InvalidJson)),
                b'\\' => {
                    self.pos += 1;
                    match self.peek().ok_or_else(|| self.err(Kind::InvalidJson))? {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => self.pos += 1,
                        b'u' => {
                            self.pos += 1;
                            let high = self.hex()?;
                            if (0xd800..=0xdbff).contains(&high) {
                                if self.peek() != Some(b'\\') {
                                    return Err(self.err(Kind::InvalidJson));
                                }
                                self.pos += 1;
                                if self.peek() != Some(b'u') {
                                    return Err(self.err(Kind::InvalidJson));
                                }
                                self.pos += 1;
                                if !(0xdc00..=0xdfff).contains(&self.hex()?) {
                                    return Err(self.err(Kind::InvalidJson));
                                }
                            } else if (0xdc00..=0xdfff).contains(&high) {
                                return Err(self.err(Kind::InvalidJson));
                            }
                        }
                        _ => return Err(self.err(Kind::InvalidJson)),
                    }
                }
                _ => self.pos += 1,
            }
        }
    }
    fn number(&mut self) -> Result<(), Error> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.err(Kind::InvalidJson)),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if start == self.pos {
                return Err(self.err(Kind::InvalidJson));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if start == self.pos {
                return Err(self.err(Kind::InvalidJson));
            }
        }
        Ok(())
    }
    fn atom(&mut self, stack: &mut [Frame; DEPTH], depth: &mut usize) -> Result<(), Error> {
        self.ws();
        match self.peek() {
            Some(b'{' | b'[') => {
                if *depth == DEPTH {
                    return Err(self.err(Kind::DepthLimit));
                }
                stack[*depth] = if self.peek() == Some(b'{') {
                    Frame::ObjectFirst
                } else {
                    Frame::ArrayFirst
                };
                *depth += 1;
                self.pos += 1;
            }
            Some(b'"') => {
                self.string()?;
            }
            Some(b'-' | b'0'..=b'9') => self.number()?,
            Some(b't' | b'f' | b'n') => {
                let expected = match self.peek() {
                    Some(b't') => "true",
                    Some(b'f') => "false",
                    _ => "null",
                };
                if !self.input.as_bytes()[self.pos..self.end].starts_with(expected.as_bytes()) {
                    return Err(self.err(Kind::InvalidJson));
                }
                self.pos += expected.len();
            }
            _ => return Err(self.err(Kind::InvalidJson)),
        }
        Ok(())
    }
    pub fn value(&mut self) -> Result<Span, Error> {
        self.ws();
        let start = self.pos;
        let mut stack = [Frame::ArrayFirst; DEPTH];
        let mut depth = 0;
        self.atom(&mut stack, &mut depth)?;
        while depth > 0 {
            self.ws();
            let i = depth - 1;
            match stack[i] {
                Frame::ObjectFirst if self.peek() == Some(b'}') => {
                    self.pos += 1;
                    depth -= 1;
                }
                Frame::ObjectFirst | Frame::ObjectKey => {
                    self.string()?;
                    stack[i] = Frame::Colon;
                }
                Frame::Colon => {
                    self.byte(b':')?;
                    stack[i] = Frame::ObjectValue;
                }
                Frame::ObjectValue => {
                    stack[i] = Frame::ObjectComma;
                    self.atom(&mut stack, &mut depth)?;
                }
                Frame::ObjectComma => {
                    if self.peek() == Some(b'}') {
                        self.pos += 1;
                        depth -= 1;
                    } else {
                        self.byte(b',')?;
                        stack[i] = Frame::ObjectKey;
                    }
                }
                Frame::ArrayFirst if self.peek() == Some(b']') => {
                    self.pos += 1;
                    depth -= 1;
                }
                Frame::ArrayFirst | Frame::ArrayValue => {
                    stack[i] = Frame::ArrayComma;
                    self.atom(&mut stack, &mut depth)?;
                }
                Frame::ArrayComma => {
                    if self.peek() == Some(b']') {
                        self.pos += 1;
                        depth -= 1;
                    } else {
                        self.byte(b',')?;
                        stack[i] = Frame::ArrayValue;
                    }
                }
            }
        }
        Ok(Span {
            start,
            end: self.pos,
        })
    }
    pub fn array(self) -> Result<Array<'a>, Error> {
        let mut reader = self;
        reader.byte(b'[')?;
        Ok(Array {
            reader,
            first: true,
            done: false,
        })
    }
    pub fn object(self) -> Result<Object<'a>, Error> {
        let mut reader = self;
        reader.byte(b'{')?;
        Ok(Object {
            reader,
            first: true,
            done: false,
        })
    }
}
pub(crate) struct Array<'a> {
    reader: Reader<'a>,
    first: bool,
    done: bool,
}
impl Array<'_> {
    pub fn next(&mut self) -> Result<Option<Span>, Error> {
        if self.done {
            return Ok(None);
        }
        self.reader.ws();
        if self.reader.peek() == Some(b']') {
            self.reader.pos += 1;
            self.reader.finish()?;
            self.done = true;
            return Ok(None);
        }
        if !self.first {
            self.reader.byte(b',')?;
        }
        self.first = false;
        self.reader.value().map(Some)
    }
}
pub(crate) struct Object<'a> {
    reader: Reader<'a>,
    first: bool,
    done: bool,
}
impl<'a> Object<'a> {
    pub fn next(&mut self) -> Result<Option<(Text<'a>, Span)>, Error> {
        if self.done {
            return Ok(None);
        }
        self.reader.ws();
        if self.reader.peek() == Some(b'}') {
            self.reader.pos += 1;
            self.reader.finish()?;
            self.done = true;
            return Ok(None);
        }
        if !self.first {
            self.reader.byte(b',')?;
        }
        self.first = false;
        let key = self.reader.string()?;
        self.reader.byte(b':')?;
        let value = self.reader.value()?;
        Ok(Some((key, value)))
    }
}
pub(crate) fn stack_bytes() -> usize {
    std::mem::size_of::<[Frame; DEPTH]>()
}
